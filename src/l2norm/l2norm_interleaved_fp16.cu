/**
 * Interleaved FP16  L2 ( float, GEMV )
 *
 *  interleaved  [block][dim][lane], __half.
 *  [n_batch] float, compute_l2_norm_gpu_interleaved .
 */
#include "l2norm.cuh"
#include "layout/ivf_interleaved_layout.cuh"
#include "utils/pch.h"
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void l2_norm_kernel_interleaved_fp16(
    const __half* __restrict__ interleaved,
    float* __restrict__ vector_l2_norm,
    int n_batch,
    int n_dim
) {
    const int vec_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_id >= n_batch) return;

    constexpr int kGroup = paraselect::kInterleavedGroupSize;
    const int block_idx = vec_id / kGroup;
    const int lane = vec_id % kGroup;
    const size_t words_per_block = (size_t)kGroup * n_dim;
    const __half* block_base = interleaved + (size_t)block_idx * words_per_block;

    float sum = 0.0f;
#pragma unroll 8
    for (int d = 0; d < n_dim; ++d) {
        float v = __half2float(__ldg(&block_base[(size_t)d * kGroup + lane]));
        sum = fmaf(v, v, sum);
    }
    vector_l2_norm[vec_id] = sqrtf(sum);
}

void compute_l2_norm_gpu_interleaved_fp16(
    const __half* interleaved,
    float* vector_l2_norm,
    int n_batch,
    int n_dim,
    cudaStream_t stream
) {
    if (!interleaved || !vector_l2_norm || n_batch <= 0 || n_dim <= 0) return;
    const int block_size = 256;
    const int grid_size = (n_batch + block_size - 1) / block_size;
    if (stream) {
        l2_norm_kernel_interleaved_fp16<<<grid_size, block_size, 0, stream>>>(
            interleaved, vector_l2_norm, n_batch, n_dim);
    } else {
        l2_norm_kernel_interleaved_fp16<<<grid_size, block_size>>>(
            interleaved, vector_l2_norm, n_batch, n_dim);
    }
}
