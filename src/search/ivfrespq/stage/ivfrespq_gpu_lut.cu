#include "search/ivfrespq/stage/ivfrespq_gpu_lut.cuh"
#include "utils/check_cuda_error.cuh"

#include <algorithm>

namespace {

__global__ void ivfrespq_build_lut_m16_k256_d8_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ centroids,
    const idx_t* __restrict__ top_nprobe,
    const float* __restrict__ codebook,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    float* __restrict__ lut)
{
    const int q = blockIdx.x;
    const int p = blockIdx.y;
    const int m = blockIdx.z;
    const int k = threadIdx.x;
    if (q >= n_query || p >= n_probes || m >= 16 || k >= 256) return;

    const idx_t cid_idx = top_nprobe[(size_t)q * n_probes + p];
    const size_t out_base = (((size_t)q * n_probes + p) * 16 + m) * 256 + k;
    if (cid_idx < 0 || cid_idx >= n_total_clusters) {
        lut[out_base] = 3.402823466e+38f;
        return;
    }

    const int cid = static_cast<int>(cid_idx);
    const float* qv = queries + (size_t)q * dim + (size_t)m * 8;
    const float* cv = centroids + (size_t)cid * dim + (size_t)m * 8;
    const float* cb = codebook + ((size_t)m * 256 + k) * 8;

    float s = 0.0f;
#pragma unroll
    for (int d = 0; d < 8; ++d) {
        const float r = qv[d] - cv[d] - cb[d];
        s = fmaf(r, r, s);
    }
    lut[out_base] = s;
}

}  // namespace

void ivfrespq_build_lut_gpu(
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const float* d_codebook,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    int pq_M,
    int pq_K,
    float* d_lut,
    cudaStream_t stream)
{
    if (!d_queries || !d_centroids || !d_top_nprobe || !d_codebook || !d_lut) return;
    if (n_query <= 0 || n_probes <= 0) return;
    if (dim == 128 && pq_M == 16 && pq_K == 256) {
        dim3 block(256);
        dim3 grid(n_query, n_probes, 16);
        ivfrespq_build_lut_m16_k256_d8_kernel<<<grid, block, 0, stream>>>(
            d_queries,
            d_centroids,
            d_top_nprobe,
            d_codebook,
            n_query,
            dim,
            n_total_clusters,
            n_probes,
            d_lut);
        CUDA_LAUNCH_CHECK("ivfrespq_build_lut_m16_k256_d8_kernel");
        return;
    }
}
