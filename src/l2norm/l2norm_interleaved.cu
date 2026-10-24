/**
 * Interleaved  L2
 *
 *  interleaved  [block][dim][lane], (v0 d0)(v1 d0)...(v31 d0)...
 *  row-major  l2norm.cu ,.
 *
 * : 4 , float4 ( dim  4  lane )
 *  float4 ,.
 */

#include "l2norm.cuh"
#include "layout/ivf_interleaved_layout.cuh"
#include "utils/pch.h"
#include <cmath>
#include <cuda_runtime.h>

__global__ void l2_norm_kernel_interleaved(
    const float* __restrict__ interleaved,
    float* __restrict__ vector_l2_norm,
    int n_batch,
    int n_dim
) {
    constexpr int kGroup = paraselect::kInterleavedGroupSize;
    const size_t words_per_block = (size_t)kGroup * n_dim;

    /*  4 ,vec_base  4  4  interleaved block  */
    const int vec_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (vec_base >= n_batch) return;

    const int block_idx = vec_base / kGroup;
    const int lane_start = vec_base % kGroup;  /* 0,4,8,...,28, 4  */
    const float* block_base = interleaved + (size_t)block_idx * words_per_block;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    /* ILP: 2  float4,, */
#pragma unroll 4
    for (int d = 0; d < n_dim; d += 2) {
        float4 v0 = __ldg(reinterpret_cast<const float4*>(&block_base[(size_t)d * kGroup + lane_start]));
        float4 v1 = (d + 1 < n_dim)
            ? __ldg(reinterpret_cast<const float4*>(&block_base[(size_t)(d + 1) * kGroup + lane_start]))
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        sum0 = fmaf(v0.x, v0.x, sum0);
        sum1 = fmaf(v0.y, v0.y, sum1);
        sum2 = fmaf(v0.z, v0.z, sum2);
        sum3 = fmaf(v0.w, v0.w, sum3);

        sum0 = fmaf(v1.x, v1.x, sum0);
        sum1 = fmaf(v1.y, v1.y, sum1);
        sum2 = fmaf(v1.z, v1.z, sum2);
        sum3 = fmaf(v1.w, v1.w, sum3);
    }

    /* float4 ( 4 ) */
    float4 out;
    out.x = sqrtf(sum0);
    out.y = sqrtf(sum1);
    out.z = sqrtf(sum2);
    out.w = sqrtf(sum3);
    if (vec_base + 4 <= n_batch) {
        *reinterpret_cast<float4*>(&vector_l2_norm[vec_base]) = out;
    } else {
        vector_l2_norm[vec_base] = out.x;
        if (vec_base + 1 < n_batch) vector_l2_norm[vec_base + 1] = out.y;
        if (vec_base + 2 < n_batch) vector_l2_norm[vec_base + 2] = out.z;
        if (vec_base + 3 < n_batch) vector_l2_norm[vec_base + 3] = out.w;
    }
}

/* :n_dim  4  */
__global__ void l2_norm_kernel_interleaved_scalar(
    const float* __restrict__ interleaved,
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
    const float* block_base = interleaved + (size_t)block_idx * words_per_block;

    float sum = 0.0f;
#pragma unroll 8
    for (int d = 0; d < n_dim; ++d) {
        float v = __ldg(&block_base[(size_t)d * kGroup + lane]);
        sum = fmaf(v, v, sum);
    }
    vector_l2_norm[vec_id] = sqrtf(sum);
}

void compute_l2_norm_gpu_interleaved(
    const float* interleaved,
    float* vector_l2_norm,
    int n_batch,
    int n_dim,
    cudaStream_t stream
) {
    const int block_size = 256;
    if (n_batch >= 4 && n_dim >= 4) {
        /* float4 : 4 ,float4 / */
        const int n_threads = (n_batch + 3) / 4;
        const int grid_size = (n_threads + block_size - 1) / block_size;
        l2_norm_kernel_interleaved<<<grid_size, block_size, 0, stream>>>(
            interleaved, vector_l2_norm, n_batch, n_dim);
    } else {
        /*  */
        const int grid_size = (n_batch + block_size - 1) / block_size;
        l2_norm_kernel_interleaved_scalar<<<grid_size, block_size, 0, stream>>>(
            interleaved, vector_l2_norm, n_batch, n_dim);
    }
}
