#ifndef PARASELECT_QUANTIZATION_PQ_RESIDUAL_PQ_TRAINER_CUH
#define PARASELECT_QUANTIZATION_PQ_RESIDUAL_PQ_TRAINER_CUH

#include "cpu/utils/cpu_affinity.h"
#include "quantization/pq/pq_kmeans.h"
#include "utils/pch.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>
#include <cublas_v2.h>

namespace paraselect {
namespace quantization {
namespace pq {

namespace detail {

inline void pq_cublas_check(cublasStatus_t st, const char* msg) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "ResidualPQTrainerGPU cuBLAS error " << (int)st
                  << " at " << msg << std::endl;
        std::abort();
    }
}

inline int choose_pq_gpu_count(int requested_max) {
    int n_gpus = 0;
    cudaError_t err = cudaGetDeviceCount(&n_gpus);
    if (err != cudaSuccess || n_gpus <= 0) return 1;
    if (requested_max > 0) n_gpus = std::min(n_gpus, requested_max);
    return std::max(1, n_gpus);
}

static __global__ void build_cluster_ids_from_offsets_kernel(
    const idx_t* __restrict__ cluster_offsets,
    int nlist,
    int base,
    int n,
    int* __restrict__ cluster_ids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const idx_t global = (idx_t)base + i;
    int lo = 0;
    int hi = nlist;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) >> 1;
        if (cluster_offsets[mid] <= global) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    cluster_ids[i] = lo;
}

static __global__ void build_full_residual_kernel(
    const float* __restrict__ data,
    const float* __restrict__ coarse_centroids,
    const int* __restrict__ cluster_ids,
    int n,
    int dim,
    float* __restrict__ residual)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * dim;
    if (idx >= total) return;
    const int row = idx / dim;
    const int d = idx - row * dim;
    const int cid = cluster_ids[row];
    residual[idx] = data[idx] - coarse_centroids[(size_t)cid * dim + d];
}

static __global__ void compute_strided_l2_norm_kernel(
    const float* __restrict__ data,
    int n,
    int stride,
    int offset,
    int dim,
    float* __restrict__ norm2)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= n) return;

    float sum = 0.0f;
    const float* x = data + (size_t)row * stride + offset;
    for (int d = tid; d < dim; d += blockDim.x) {
        const float v = x[d];
        sum += v * v;
    }

    __shared__ float smem[256];
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) norm2[row] = smem[0];
}

static __global__ void compute_centroid_l2_norm_kernel(
    const float* __restrict__ centroids,
    int k,
    int dim,
    float* __restrict__ norm2)
{
    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    if (c >= k) return;

    float sum = 0.0f;
    const float* x = centroids + (size_t)c * dim;
    for (int d = tid; d < dim; d += blockDim.x) {
        const float v = x[d];
        sum += v * v;
    }

    __shared__ float smem[256];
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) norm2[c] = smem[0];
}

static __global__ void init_best_kernel(
    float* __restrict__ best_dist2,
    idx_t* __restrict__ best_idx,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        best_dist2[i] = 3.402823e38f;
        best_idx[i] = 0;
    }
}

static __global__ void update_best_from_dot_kernel(
    const float* __restrict__ dot,
    const float* __restrict__ xnorm2,
    const float* __restrict__ cnorm2,
    int n,
    int k,
    idx_t* __restrict__ best_idx,
    float* __restrict__ best_dist2)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float xn = xnorm2[i];
    float best = best_dist2[i];
    idx_t best_id = best_idx[i];
    for (int c = 0; c < k; ++c) {
        const float dist = xn + cnorm2[c] - 2.0f * dot[i + (size_t)c * n];
        if (dist < best) {
            best = dist;
            best_id = c;
        }
    }
    best_dist2[i] = best;
    best_idx[i] = best_id;
}

static __global__ void accum_strided_subspace_kernel(
    const float* __restrict__ residual,
    const idx_t* __restrict__ assign,
    int n,
    int full_dim,
    int subspace_offset,
    int d_sub,
    int k,
    float* __restrict__ accum,
    int* __restrict__ counts)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * d_sub;
    if (idx >= total) return;

    const int row = idx / d_sub;
    const int d = idx - row * d_sub;
    const int cid = (int)assign[row];
    if (cid < 0 || cid >= k) return;
    atomicAdd(accum + (size_t)cid * d_sub + d,
              residual[(size_t)row * full_dim + subspace_offset + d]);
    if (d == 0) {
        atomicAdd(counts + cid, 1);
    }
}

static __global__ void update_centroids_kernel(
    float* __restrict__ centroids,
    const float* __restrict__ accum,
    const int* __restrict__ counts,
    int k,
    int dim)
{
    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    if (c >= k) return;

    const int cnt = counts[c];
    if (cnt <= 0) return;
    const float inv = 1.0f / (float)cnt;
    for (int d = tid; d < dim; d += blockDim.x) {
        centroids[(size_t)c * dim + d] =
            accum[(size_t)c * dim + d] * inv;
    }
}

}  // namespace detail

struct ResidualPQCase {
    int n = 0;
    int dim = 0;
    int nlist = 0;
    int m = 16;
    int k = 256;
    int iters = 25;
    int num_threads = 0;
    int device_id = 0;
    int max_gpus = 8;
    float train_sample_ratio = 1.0f;
    int train_max_samples = 0;
};

struct ResidualPQInput {
    const float* data = nullptr;               // [n, dim], cluster-reordered
    const float* coarse_centroids = nullptr;   // [nlist, dim]
    const idx_t* cluster_offsets = nullptr;    // [nlist + 1]
    const int* cluster_counts = nullptr;       // [nlist]
};

struct ResidualPQResult {
    std::vector<uint8_t> codes;    // [n, m]
    std::vector<float> codebook;   // [m, k, d_sub]
    int m = 0;
    int k = 0;
    int d_sub = 0;
    double fit_ms = 0.0;
    double encode_ms = 0.0;
    double total_ms = 0.0;
};

void train_residual_pq_codebook_multigpu(
    const ResidualPQCase& cfg,
    const ResidualPQInput& input,
    float* codebook);

class ResidualPQTrainerCPU {
public:
    explicit ResidualPQTrainerCPU(ResidualPQCase cfg)
        : cfg_(cfg) {}

    ResidualPQResult fit_encode(const ResidualPQInput& input) const {
        validate(input);

        const auto total_t0 = std::chrono::high_resolution_clock::now();
        const int d_sub = cfg_.dim / cfg_.m;
        const int threads = resolve_num_threads(cfg_.num_threads);

        ResidualPQResult ret;
        ret.m = cfg_.m;
        ret.k = cfg_.k;
        ret.d_sub = d_sub;
        ret.codebook.resize((size_t)cfg_.m * (size_t)cfg_.k * (size_t)d_sub);
        ret.codes.resize((size_t)cfg_.n * (size_t)cfg_.m);

        const auto fit_t0 = std::chrono::high_resolution_clock::now();
        fit(input, ret.codebook.data());
        const auto fit_t1 = std::chrono::high_resolution_clock::now();
        ret.fit_ms = std::chrono::duration<double, std::milli>(fit_t1 - fit_t0).count();

        const auto enc_t0 = std::chrono::high_resolution_clock::now();
        encode(input, ret.codebook.data(), ret.codes.data());
        const auto enc_t1 = std::chrono::high_resolution_clock::now();
        ret.encode_ms = std::chrono::duration<double, std::milli>(enc_t1 - enc_t0).count();

        const auto total_t1 = std::chrono::high_resolution_clock::now();
        ret.total_ms = std::chrono::duration<double, std::milli>(total_t1 - total_t0).count();
        (void)threads;
        return ret;
    }

    void fit(const ResidualPQInput& input, float* out_codebook) const {
        validate(input);
        if (!out_codebook) {
            throw std::invalid_argument("ResidualPQTrainerCPU::fit: out_codebook is null");
        }
        train_codebook_scheduler(input, resolve_num_threads(cfg_.num_threads), out_codebook);
    }

    void encode(
        const ResidualPQInput& input,
        const float* codebook,
        uint8_t* out_codes) const
    {
        validate(input);
        if (!codebook || !out_codes) {
            throw std::invalid_argument("ResidualPQTrainerCPU::encode: null output/input pointer");
        }
        encode_scheduler(input, resolve_num_threads(cfg_.num_threads), codebook, out_codes);
    }

private:
    ResidualPQCase cfg_;

    static int resolve_num_threads(int requested) {
        if (requested > 0) return requested;
        return paraselect::resolve_cpu_fine_thread_count();
    }

    void validate(const ResidualPQInput& input) const {
        if (!input.data || !input.coarse_centroids ||
            !input.cluster_offsets || !input.cluster_counts) {
            throw std::invalid_argument("ResidualPQTrainerCPU: null input pointers");
        }
        if (cfg_.n <= 0 || cfg_.dim <= 0 || cfg_.nlist <= 0 ||
            cfg_.m <= 0 || cfg_.k <= 0 || cfg_.iters <= 0) {
            throw std::invalid_argument("ResidualPQTrainerCPU: invalid case shape");
        }
        if ((cfg_.dim % cfg_.m) != 0) {
            throw std::invalid_argument("ResidualPQTrainerCPU: dim must be divisible by m");
        }
        if (cfg_.k > 256) {
            throw std::invalid_argument("ResidualPQTrainerCPU: k must be <= 256");
        }
    }

    void train_codebook_scheduler(
        const ResidualPQInput& input,
        int num_threads,
        float* codebook) const
    {
        const int d_sub = cfg_.dim / cfg_.m;
        std::atomic<int> next_subspace(0);
        const int subspace_workers = std::max(1, std::min(cfg_.m, num_threads));
        std::vector<std::thread> train_workers;
        train_workers.reserve((size_t)subspace_workers);
        for (int w = 0; w < subspace_workers; ++w) {
            train_workers.emplace_back([&]() {
                std::vector<float> local_subspace_resid((size_t)cfg_.n * (size_t)d_sub);
                while (true) {
                    const int m = next_subspace.fetch_add(1, std::memory_order_relaxed);
                    if (m >= cfg_.m) break;

                    fill_subspace_residuals_kernel(input, d_sub, m, local_subspace_resid.data());

                    float* out_codebook =
                        codebook + (size_t)m * (size_t)cfg_.k * (size_t)d_sub;
                    ivf::pq_kmeans_subspace(
                        local_subspace_resid.data(),
                        cfg_.n,
                        d_sub,
                        cfg_.k,
                        cfg_.iters,
                        42u + (unsigned)m,
                        out_codebook);
                }
            });
        }
        for (auto& th : train_workers) th.join();
    }

    void fill_subspace_residuals_kernel(
        const ResidualPQInput& input,
        int d_sub,
        int subspace,
        float* out) const
    {
        for (int cid = 0; cid < cfg_.nlist; ++cid) {
            const size_t off = (size_t)input.cluster_offsets[cid];
            const int cnt = input.cluster_counts[cid];
            const float* cen = input.coarse_centroids +
                (size_t)cid * cfg_.dim + (size_t)subspace * d_sub;
            for (int j = 0; j < cnt; ++j) {
                const size_t ridx = off + (size_t)j;
                const float* vec =
                    input.data + ridx * cfg_.dim + (size_t)subspace * d_sub;
                float* dst = out + ridx * (size_t)d_sub;
                for (int d = 0; d < d_sub; ++d) {
                    dst[d] = vec[d] - cen[d];
                }
            }
        }
    }

    void encode_scheduler(
        const ResidualPQInput& input,
        int num_threads,
        const float* codebook,
        uint8_t* codes) const
    {
        const int d_sub = cfg_.dim / cfg_.m;
        std::atomic<int> next_cluster(0);
        std::vector<std::thread> workers;
        workers.reserve((size_t)num_threads);
        for (int t = 0; t < num_threads; ++t) {
            workers.emplace_back([&]() {
                while (true) {
                    const int cid = next_cluster.fetch_add(1, std::memory_order_relaxed);
                    if (cid >= cfg_.nlist) break;
                    encode_cluster_kernel(input, cid, d_sub, codebook, codes);
                }
            });
        }
        for (auto& th : workers) th.join();
    }

    void encode_cluster_kernel(
        const ResidualPQInput& input,
        int cid,
        int d_sub,
        const float* codebook,
        uint8_t* codes) const
    {
        const size_t off = (size_t)input.cluster_offsets[cid];
        const int cnt = input.cluster_counts[cid];
        const float* cen = input.coarse_centroids + (size_t)cid * cfg_.dim;
        for (int j = 0; j < cnt; ++j) {
            const size_t ridx = off + (size_t)j;
            const float* vec = input.data + ridx * cfg_.dim;
            uint8_t* code_out = codes + ridx * (size_t)cfg_.m;
            encode_one_kernel(vec, cen, codebook, d_sub, code_out);
        }
    }

    void encode_one_kernel(
        const float* vec,
        const float* centroid,
        const float* codebook,
        int d_sub,
        uint8_t* code_out) const
    {
        for (int m = 0; m < cfg_.m; ++m) {
            const int base = m * d_sub;
            int best_k = 0;
            float best_dist = std::numeric_limits<float>::max();
            const float* cb_m = codebook + (size_t)m * (size_t)cfg_.k * (size_t)d_sub;
            for (int k = 0; k < cfg_.k; ++k) {
                const float* cb = cb_m + (size_t)k * (size_t)d_sub;
                float dist = 0.0f;
                for (int d = 0; d < d_sub; ++d) {
                    const float r = (vec[base + d] - centroid[base + d]) - cb[d];
                    dist += r * r;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_k = k;
                }
            }
            code_out[m] = (uint8_t)best_k;
        }
    }
};

class ResidualPQTrainerGPU {
public:
    explicit ResidualPQTrainerGPU(ResidualPQCase cfg)
        : cfg_(cfg) {}

    ResidualPQResult fit_encode(const ResidualPQInput& input) const {
        validate(input);

        const auto total_t0 = std::chrono::high_resolution_clock::now();
        const int d_sub = cfg_.dim / cfg_.m;

        ResidualPQResult ret;
        ret.m = cfg_.m;
        ret.k = cfg_.k;
        ret.d_sub = d_sub;
        ret.codebook.resize((size_t)cfg_.m * (size_t)cfg_.k * (size_t)d_sub);
        ret.codes.resize((size_t)cfg_.n * (size_t)cfg_.m);

        const auto fit_t0 = std::chrono::high_resolution_clock::now();
        fit(input, ret.codebook.data());
        const auto fit_t1 = std::chrono::high_resolution_clock::now();
        ret.fit_ms = std::chrono::duration<double, std::milli>(fit_t1 - fit_t0).count();

        const auto enc_t0 = std::chrono::high_resolution_clock::now();
        ResidualPQTrainerCPU cpu_encoder(cfg_);
        cpu_encoder.encode(input, ret.codebook.data(), ret.codes.data());
        const auto enc_t1 = std::chrono::high_resolution_clock::now();
        ret.encode_ms = std::chrono::duration<double, std::milli>(enc_t1 - enc_t0).count();

        const auto total_t1 = std::chrono::high_resolution_clock::now();
        ret.total_ms = std::chrono::duration<double, std::milli>(total_t1 - total_t0).count();
        return ret;
    }

    void fit(const ResidualPQInput& input, float* out_codebook) const {
        validate(input);
        if (!out_codebook) {
            throw std::invalid_argument("ResidualPQTrainerGPU::fit: out_codebook is null");
        }
        train_codebook_gpu(input, out_codebook);
    }

    void encode(
        const ResidualPQInput& input,
        const float* codebook,
        uint8_t* out_codes) const
    {
        ResidualPQTrainerCPU cpu_encoder(cfg_);
        cpu_encoder.encode(input, codebook, out_codes);
    }

private:
    ResidualPQCase cfg_;

    void validate(const ResidualPQInput& input) const {
        if (!input.data || !input.coarse_centroids ||
            !input.cluster_offsets || !input.cluster_counts) {
            throw std::invalid_argument("ResidualPQTrainerGPU: null input pointers");
        }
        if (cfg_.n <= 0 || cfg_.dim <= 0 || cfg_.nlist <= 0 ||
            cfg_.m <= 0 || cfg_.k <= 0 || cfg_.iters <= 0) {
            throw std::invalid_argument("ResidualPQTrainerGPU: invalid case shape");
        }
        if ((cfg_.dim % cfg_.m) != 0) {
            throw std::invalid_argument("ResidualPQTrainerGPU: dim must be divisible by m");
        }
        if (cfg_.k > 256) {
            throw std::invalid_argument("ResidualPQTrainerGPU: k must be <= 256");
        }
    }

    void sample_initial_residual_centroids(
        const ResidualPQInput& input,
        int subspace,
        int d_sub,
        float* out) const
    {
        std::mt19937 rng((unsigned)(42 + subspace));
        std::uniform_int_distribution<int> dist(0, cfg_.n - 1);
        for (int k = 0; k < cfg_.k; ++k) {
            const int ridx = dist(rng);
            int cid = (int)(std::upper_bound(
                input.cluster_offsets,
                input.cluster_offsets + cfg_.nlist + 1,
                (idx_t)ridx) - input.cluster_offsets) - 1;
            if (cid < 0) cid = 0;
            if (cid >= cfg_.nlist) cid = cfg_.nlist - 1;
            const float* vec = input.data +
                (size_t)ridx * (size_t)cfg_.dim +
                (size_t)subspace * (size_t)d_sub;
            const float* cen = input.coarse_centroids +
                (size_t)cid * (size_t)cfg_.dim +
                (size_t)subspace * (size_t)d_sub;
            float* dst = out + (size_t)k * (size_t)d_sub;
            for (int d = 0; d < d_sub; ++d) {
                dst[d] = vec[d] - cen[d];
            }
        }
    }

    void train_codebook_gpu(
        const ResidualPQInput& input,
        float* codebook) const
    {
        if (cfg_.max_gpus != 1 && detail::choose_pq_gpu_count(cfg_.max_gpus) > 1) {
            train_residual_pq_codebook_multigpu(cfg_, input, codebook);
            return;
        }
        CUDA_CHECK(cudaSetDevice(cfg_.device_id));
        const int d_sub = cfg_.dim / cfg_.m;
        const int batch_size = 1 << 15;
        const int threads = 256;
        const int total_batches = (cfg_.n + batch_size - 1) / batch_size;
        const float alpha = 1.0f;
        const float beta = 0.0f;

        std::cerr
            << "[ResidualPQTrainerGPU] fit begin"
            << " n=" << cfg_.n
            << " dim=" << cfg_.dim
            << " nlist=" << cfg_.nlist
            << " M=" << cfg_.m
            << " K=" << cfg_.k
            << " d_sub=" << d_sub
            << " iters=" << cfg_.iters
            << " batch_size=" << batch_size
            << " total_batches=" << total_batches
            << std::endl;

        cublasHandle_t handle = nullptr;
        detail::pq_cublas_check(cublasCreate(&handle), "cublasCreate");
        detail::pq_cublas_check(
            cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH),
            "cublasSetMathMode");

        float* d_data = nullptr;
        float* d_residual = nullptr;
        float* d_coarse_centroids = nullptr;
        idx_t* d_cluster_offsets = nullptr;
        int* d_cluster_ids = nullptr;
        float* d_centroids = nullptr;
        float* d_accum = nullptr;
        int* d_counts = nullptr;
        float* d_xnorm2 = nullptr;
        float* d_cnorm2 = nullptr;
        float* d_best_dist2 = nullptr;
        float* d_dot = nullptr;
        idx_t* d_assign = nullptr;

        CUDA_CHECK(cudaMalloc(&d_data, sizeof(float) * (size_t)batch_size * (size_t)cfg_.dim));
        CUDA_CHECK(cudaMalloc(&d_residual, sizeof(float) * (size_t)batch_size * (size_t)cfg_.dim));
        CUDA_CHECK(cudaMalloc(&d_coarse_centroids, sizeof(float) * (size_t)cfg_.nlist * (size_t)cfg_.dim));
        CUDA_CHECK(cudaMalloc(&d_cluster_offsets, sizeof(idx_t) * (size_t)(cfg_.nlist + 1)));
        CUDA_CHECK(cudaMalloc(&d_cluster_ids, sizeof(int) * (size_t)batch_size));
        CUDA_CHECK(cudaMalloc(&d_centroids, sizeof(float) * (size_t)cfg_.m * (size_t)cfg_.k * (size_t)d_sub));
        CUDA_CHECK(cudaMalloc(&d_accum, sizeof(float) * (size_t)cfg_.m * (size_t)cfg_.k * (size_t)d_sub));
        CUDA_CHECK(cudaMalloc(&d_counts, sizeof(int) * (size_t)cfg_.m * (size_t)cfg_.k));
        CUDA_CHECK(cudaMalloc(&d_xnorm2, sizeof(float) * (size_t)batch_size));
        CUDA_CHECK(cudaMalloc(&d_cnorm2, sizeof(float) * (size_t)cfg_.k));
        CUDA_CHECK(cudaMalloc(&d_best_dist2, sizeof(float) * (size_t)batch_size));
        CUDA_CHECK(cudaMalloc(&d_dot, sizeof(float) * (size_t)batch_size * (size_t)cfg_.k));
        CUDA_CHECK(cudaMalloc(&d_assign, sizeof(idx_t) * (size_t)batch_size));

        CUDA_CHECK(cudaMemcpy(
            d_coarse_centroids,
            input.coarse_centroids,
            sizeof(float) * (size_t)cfg_.nlist * (size_t)cfg_.dim,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_cluster_offsets,
            input.cluster_offsets,
            sizeof(idx_t) * (size_t)(cfg_.nlist + 1),
            cudaMemcpyHostToDevice));

        std::vector<float> h_init((size_t)cfg_.m * (size_t)cfg_.k * (size_t)d_sub);
        for (int subspace = 0; subspace < cfg_.m; ++subspace) {
            sample_initial_residual_centroids(
                input,
                subspace,
                d_sub,
                h_init.data() + (size_t)subspace * (size_t)cfg_.k * (size_t)d_sub);
        }
        CUDA_CHECK(cudaMemcpy(
            d_centroids,
            h_init.data(),
            sizeof(float) * (size_t)cfg_.m * (size_t)cfg_.k * (size_t)d_sub,
            cudaMemcpyHostToDevice));

        for (int it = 0; it < cfg_.iters; ++it) {
            const auto iter_t0 = std::chrono::high_resolution_clock::now();
            std::cerr
                << "[ResidualPQTrainerGPU] iter "
                << (it + 1) << "/" << cfg_.iters
                << " begin"
                << std::endl;
            CUDA_CHECK(cudaMemset(
                d_accum,
                0,
                sizeof(float) * (size_t)cfg_.m * (size_t)cfg_.k * (size_t)d_sub));
            CUDA_CHECK(cudaMemset(
                d_counts,
                0,
                sizeof(int) * (size_t)cfg_.m * (size_t)cfg_.k));

            for (int base = 0; base < cfg_.n; base += batch_size) {
                const int curB = std::min(batch_size, cfg_.n - base);
                CUDA_CHECK(cudaMemcpy(
                    d_data,
                    input.data + (size_t)base * (size_t)cfg_.dim,
                    sizeof(float) * (size_t)curB * (size_t)cfg_.dim,
                    cudaMemcpyHostToDevice));

                detail::build_cluster_ids_from_offsets_kernel<<<
                    (curB + threads - 1) / threads,
                    threads>>>(
                        d_cluster_offsets,
                        cfg_.nlist,
                        base,
                        curB,
                        d_cluster_ids);
                CUDA_LAUNCH_CHECK("build_cluster_ids_from_offsets_kernel");

                detail::build_full_residual_kernel<<<
                    (curB * cfg_.dim + threads - 1) / threads,
                    threads>>>(
                        d_data,
                        d_coarse_centroids,
                        d_cluster_ids,
                        curB,
                        cfg_.dim,
                        d_residual);
                CUDA_LAUNCH_CHECK("build_full_residual_kernel");

                for (int subspace = 0; subspace < cfg_.m; ++subspace) {
                    const int offset = subspace * d_sub;
                    float* d_centroids_m =
                        d_centroids + (size_t)subspace * (size_t)cfg_.k * (size_t)d_sub;
                    float* d_accum_m =
                        d_accum + (size_t)subspace * (size_t)cfg_.k * (size_t)d_sub;
                    int* d_counts_m =
                        d_counts + (size_t)subspace * (size_t)cfg_.k;

                    detail::compute_strided_l2_norm_kernel<<<curB, 256>>>(
                        d_residual,
                        curB,
                        cfg_.dim,
                        offset,
                        d_sub,
                        d_xnorm2);
                    CUDA_LAUNCH_CHECK("compute_strided_l2_norm_kernel");

                    detail::compute_centroid_l2_norm_kernel<<<cfg_.k, 256>>>(
                        d_centroids_m,
                        cfg_.k,
                        d_sub,
                        d_cnorm2);
                    CUDA_LAUNCH_CHECK("compute_centroid_l2_norm_kernel");

                    detail::init_best_kernel<<<
                        (curB + threads - 1) / threads,
                        threads>>>(
                            d_best_dist2,
                            d_assign,
                            curB);
                    CUDA_LAUNCH_CHECK("init_best_kernel");

                    detail::pq_cublas_check(
                        cublasSgemm(
                            handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            curB,
                            cfg_.k,
                            d_sub,
                            &alpha,
                            d_residual + offset,
                            cfg_.dim,
                            d_centroids_m,
                            d_sub,
                            &beta,
                            d_dot,
                            curB),
                        "cublasSgemm residual strided subspace");

                    detail::update_best_from_dot_kernel<<<
                        (curB + threads - 1) / threads,
                        threads>>>(
                            d_dot,
                            d_xnorm2,
                            d_cnorm2,
                            curB,
                            cfg_.k,
                            d_assign,
                            d_best_dist2);
                    CUDA_LAUNCH_CHECK("update_best_from_dot_kernel");

                    detail::accum_strided_subspace_kernel<<<
                        (curB * d_sub + threads - 1) / threads,
                        threads>>>(
                            d_residual,
                            d_assign,
                            curB,
                            cfg_.dim,
                            offset,
                            d_sub,
                            cfg_.k,
                            d_accum_m,
                            d_counts_m);
                    CUDA_LAUNCH_CHECK("accum_strided_subspace_kernel");
                }
            }

            for (int subspace = 0; subspace < cfg_.m; ++subspace) {
                float* d_centroids_m =
                    d_centroids + (size_t)subspace * (size_t)cfg_.k * (size_t)d_sub;
                float* d_accum_m =
                    d_accum + (size_t)subspace * (size_t)cfg_.k * (size_t)d_sub;
                int* d_counts_m =
                    d_counts + (size_t)subspace * (size_t)cfg_.k;
                detail::update_centroids_kernel<<<cfg_.k, 256>>>(
                    d_centroids_m,
                    d_accum_m,
                    d_counts_m,
                    cfg_.k,
                    d_sub);
                CUDA_LAUNCH_CHECK("update_centroids_kernel");
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto iter_t1 = std::chrono::high_resolution_clock::now();
            const double iter_ms =
                std::chrono::duration<double, std::milli>(iter_t1 - iter_t0).count();
            std::cerr
                << "[ResidualPQTrainerGPU] iter "
                << (it + 1) << "/" << cfg_.iters
                << " end ms=" << iter_ms
                << std::endl;
        }

        CUDA_CHECK(cudaMemcpy(
            codebook,
            d_centroids,
            sizeof(float) * (size_t)cfg_.m * (size_t)cfg_.k * (size_t)d_sub,
            cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_assign));
        CUDA_CHECK(cudaFree(d_dot));
        CUDA_CHECK(cudaFree(d_best_dist2));
        CUDA_CHECK(cudaFree(d_cnorm2));
        CUDA_CHECK(cudaFree(d_xnorm2));
        CUDA_CHECK(cudaFree(d_counts));
        CUDA_CHECK(cudaFree(d_accum));
        CUDA_CHECK(cudaFree(d_centroids));
        CUDA_CHECK(cudaFree(d_cluster_ids));
        CUDA_CHECK(cudaFree(d_cluster_offsets));
        CUDA_CHECK(cudaFree(d_coarse_centroids));
        CUDA_CHECK(cudaFree(d_residual));
        CUDA_CHECK(cudaFree(d_data));
        detail::pq_cublas_check(cublasDestroy(handle), "cublasDestroy");
        std::cerr << "[ResidualPQTrainerGPU] fit end" << std::endl;
    }
};

}  // namespace pq
}  // namespace quantization
}  // namespace paraselect

#endif
