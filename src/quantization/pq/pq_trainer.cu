#include "quantization/pq/pq_trainer.cuh"

#include "clustering/kmeans/kmeans.cuh"

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <stdexcept>

namespace ivf {


namespace {

void validate_plain_pq_train_config(const ivf::PlainPQTrainConfig& cfg) {
    if (cfg.n <= 0 || cfg.dim <= 0 || cfg.n_subspace <= 0 || cfg.n_codes <= 0 || cfg.n_codes > 256) {
        throw std::invalid_argument("train_plain_pq_codebooks_gpu: invalid shape");
    }
    if (cfg.dim % cfg.n_subspace != 0) {
        throw std::invalid_argument("train_plain_pq_codebooks_gpu: dim must be divisible by n_subspace");
    }
    if (cfg.n < cfg.n_codes) {
        throw std::invalid_argument("train_plain_pq_codebooks_gpu: n must be >= n_codes");
    }
    if (cfg.iters <= 0) {
        throw std::invalid_argument("train_plain_pq_codebooks_gpu: iters must be positive");
    }
}

template <typename T>
void train_plain_pq_codebooks_gpu_impl(
    const T* h_data,
    const ivf::PlainPQTrainConfig& cfg,
    std::vector<float>& codebooks)
{
    if (!h_data) throw std::invalid_argument("train_plain_pq_codebooks_gpu: null data");
    validate_plain_pq_train_config(cfg);

    const int sub_dim = cfg.dim / cfg.n_subspace;
    const int train_dim = (sub_dim + 3) / 4 * 4;
    codebooks.assign((size_t)cfg.n_subspace * cfg.n_codes * sub_dim, 0.0f);

    std::vector<float> subspace((size_t)cfg.n * train_dim, 0.0f);
    idx_t* d_assign = nullptr;
    float* d_centroids = nullptr;
    try {
        CUDA_CHECK(cudaMalloc(&d_assign, (size_t)cfg.n * sizeof(idx_t)));
        CUDA_CHECK(cudaMalloc(&d_centroids, (size_t)cfg.n_codes * train_dim * sizeof(float)));
        for (int m = 0; m < cfg.n_subspace; ++m) {
            for (int i = 0; i < cfg.n; ++i) {
                const T* src = h_data + (size_t)i * cfg.dim + (size_t)m * sub_dim;
                float* dst = subspace.data() + (size_t)i * train_dim;
                for (int d = 0; d < sub_dim; ++d) dst[d] = static_cast<float>(src[d]);
            }

            KMeansCase kmeans_cfg;
            kmeans_cfg.n = cfg.n;
            kmeans_cfg.dim = train_dim;
            kmeans_cfg.k = cfg.n_codes;
            kmeans_cfg.iters = cfg.iters;
            kmeans_cfg.minibatch_iters = cfg.iters;
            kmeans_cfg.seed = (int)(cfg.seed + (unsigned)m * 104729u);
            kmeans_cfg.dist = L2_DISTANCE;
            kmeans_cfg.precision = KMeansComputePrecision::Tf32;
            kmeans_cfg.init_policy = KMEANS_INIT_AUTO_THRESHOLD;
            kmeans_cfg.use_flash_assign = false;
            kmeans_cfg.collect_timing = false;

            std::vector<float> init_centroids((size_t)cfg.n_codes * train_dim);
            init_centroids_by_sampling(kmeans_cfg, subspace.data(), init_centroids.data());
            CUDA_CHECK(cudaMemcpy(
                d_centroids,
                init_centroids.data(),
                init_centroids.size() * sizeof(float),
                cudaMemcpyHostToDevice));

            float objective = 0.0f;
            gpu_kmeans_lloyd(kmeans_cfg, subspace.data(), d_assign, d_centroids, &objective);
            CUDA_CHECK(cudaMemcpy(
                init_centroids.data(),
                d_centroids,
                (size_t)cfg.n_codes * train_dim * sizeof(float),
                cudaMemcpyDeviceToHost));
            for (int c = 0; c < cfg.n_codes; ++c)
                std::copy_n(init_centroids.data() + (size_t)c * train_dim, sub_dim,
                    codebooks.data() + ((size_t)m * cfg.n_codes + c) * sub_dim);
        }
        CUDA_CHECK(cudaFree(d_assign));
        CUDA_CHECK(cudaFree(d_centroids));
    } catch (...) {
        if (d_assign) cudaFree(d_assign);
        if (d_centroids) cudaFree(d_centroids);
        throw;
    }
}

}  // namespace

void train_plain_pq_codebooks_gpu(
    const float* h_data,
    const PlainPQTrainConfig& cfg,
    std::vector<float>& codebooks)
{
    train_plain_pq_codebooks_gpu_impl(h_data, cfg, codebooks);
}

void train_plain_pq_codebooks_gpu(
    const uint8_t* h_data,
    const PlainPQTrainConfig& cfg,
    std::vector<float>& codebooks)
{
    train_plain_pq_codebooks_gpu_impl(h_data, cfg, codebooks);
}

}  // namespace ivf
