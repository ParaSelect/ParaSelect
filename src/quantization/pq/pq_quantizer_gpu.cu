/**
 * PQQuantizerGPU: device,encode/decode  device , linear_fp16.
 * train  GPU PQ trainer , device.
 */
#include "quantization/pq/pq_quantizer.cuh"
#include "quantization/pq/pq_trainer.cuh"
#include <cuda_runtime.h>
#include <cstring>
#include <vector>

namespace ivf {

namespace {

__global__ void pq_encode_kernel(
    const float* __restrict__ d_data,
    const float* __restrict__ d_codebooks,
    int n, int dim,
    int n_subspace, int n_codes, int sub_dim,
    uint8_t* __restrict__ d_codes
) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= n) return;
    const float* vec = d_data + (size_t)vec_idx * dim;
    uint8_t* out = d_codes + (size_t)vec_idx * n_subspace;

    for (int m = 0; m < n_subspace; ++m) {
        const float* sub = vec + (size_t)m * sub_dim;
        const float* cb = d_codebooks + (size_t)m * n_codes * sub_dim;
        int best = 0;
        float best_sq = 0.f;
        for (int d = 0; d < sub_dim; ++d) {
            float v = sub[d] - cb[d];
            best_sq += v * v;
        }
        for (int k = 1; k < n_codes; ++k) {
            float sq = 0.f;
            const float* cen = cb + (size_t)k * sub_dim;
            for (int d = 0; d < sub_dim; ++d) {
                float v = sub[d] - cen[d];
                sq += v * v;
            }
            if (sq < best_sq) {
                best_sq = sq;
                best = k;
            }
        }
        out[m] = (uint8_t)best;
    }
}

__global__ void pq_decode_kernel(
    const uint8_t* __restrict__ d_codes,
    const float* __restrict__ d_codebooks,
    int n, int dim,
    int n_subspace, int n_codes, int sub_dim,
    float* __restrict__ d_data
) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= n) return;
    const uint8_t* codes = d_codes + (size_t)vec_idx * n_subspace;
    float* out = d_data + (size_t)vec_idx * dim;

    for (int m = 0; m < n_subspace; ++m) {
        int k = (int)codes[m];
        if (k >= n_codes) k = 0;
        const float* cen = d_codebooks + (size_t)m * n_codes * sub_dim + (size_t)k * sub_dim;
        for (int d = 0; d < sub_dim; ++d)
            out[(size_t)m * sub_dim + d] = cen[d];
    }
}

} // namespace

PQQuantizerGPU::PQQuantizerGPU(int n_subspace, int n_codes)
    : n_subspace_(n_subspace), n_codes_(n_codes), sub_dim_(0), d_codebooks_(nullptr) {}

PQQuantizerGPU::~PQQuantizerGPU() {
    if (d_codebooks_) {
        cudaFree(d_codebooks_);
        d_codebooks_ = nullptr;
    }
}

void PQQuantizerGPU::train(const float* h_data, int n, int dim) {
    if (!h_data || n <= 0 || dim <= 0 || n_subspace_ <= 0 || n_codes_ <= 0) return;
    if (dim % n_subspace_ != 0) return;
    if (n_codes_ > 256) return;

    if (d_codebooks_) {
        cudaFree(d_codebooks_);
        d_codebooks_ = nullptr;
    }
    sub_dim_ = dim / n_subspace_;
    const size_t codebook_bytes = (size_t)n_subspace_ * n_codes_ * sub_dim_ * sizeof(float);
    std::vector<float> h_codebooks;
    PlainPQTrainConfig cfg;
    cfg.n = n;
    cfg.dim = dim;
    cfg.n_subspace = n_subspace_;
    cfg.n_codes = n_codes_;
    cfg.iters = 25;
    cfg.seed = 42u;
    train_plain_pq_codebooks_gpu(h_data, cfg, h_codebooks);

    if (cudaMalloc(&d_codebooks_, codebook_bytes) != cudaSuccess) return;
    cudaMemcpy(d_codebooks_, h_codebooks.data(), codebook_bytes, cudaMemcpyHostToDevice);
}

void PQQuantizerGPU::encode(const float* d_data, int n, int dim, void* d_codes) {
    if (!d_data || !d_codes || !d_codebooks_ || (dim / n_subspace_) != sub_dim_) return;
    if (n <= 0) return;
    int blk = 256;
    int grid = (n + blk - 1) / blk;
    pq_encode_kernel<<<grid, blk>>>(d_data, d_codebooks_, n, dim,
        n_subspace_, n_codes_, sub_dim_, (uint8_t*)d_codes);
}

void PQQuantizerGPU::decode(const void* d_codes, int n, int dim, float* d_data) {
    if (!d_codes || !d_data || !d_codebooks_ || (dim / n_subspace_) != sub_dim_) return;
    if (n <= 0) return;
    int blk = 256;
    int grid = (n + blk - 1) / blk;
    pq_decode_kernel<<<grid, blk>>>((const uint8_t*)d_codes, d_codebooks_, n, dim,
        n_subspace_, n_codes_, sub_dim_, d_data);
}

int PQQuantizerGPU::code_size_bytes(int n, int dim) const {
    (void)dim;
    return n * n_subspace_;
}

} // namespace ivf
