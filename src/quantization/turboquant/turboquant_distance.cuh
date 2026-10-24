#pragma once
#include <cuda_runtime.h>
#include "search/ivf_common/quantized/residual_codec_view.h"

namespace paraselect::search::turboquant {
#ifdef __CUDACC__
__device__ inline void prepare_residual(SignResidualDeviceView view, int query, int cluster,
                                       int dim, float* shared) {
    // Accumulate the residual norm while the residual is still in registers.
    // The previous single-thread 128-FMA chain parked the whole block at the
    // barrier for ~500 cycles per probed list.
    float partial = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        const float x = view.queries[(size_t)query * dim + d] -
                        view.centroids[(size_t)cluster * dim + d];
        shared[d] = x;
        partial = fmaf(x, x, partial);
    }
    __syncthreads();
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffffu, partial, offset);
    }
    __shared__ float warp_norm_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_norm_sums[warp] = partial;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        const int warps = (blockDim.x + 31) >> 5;
        for (int w = 0; w < warps; ++w) total += warp_norm_sums[w];
        shared[dim] = total;
    }
    __syncthreads();
}
__device__ inline float signed_dot_word(unsigned signs, const float* residual) {
    float sum = 0.0f;
#pragma unroll 4
    for (int bit = 0; bit < 32; ++bit) {
        const unsigned bits = __float_as_uint(residual[bit]);
        sum += __uint_as_float(bits ^ (((signs >> bit) & 1u) ? 0u : 0x80000000u));
    }
    return sum;
}
__device__ inline float reconstructed_distance(float norm, float scale, float dot, int dim) {
    return norm + dim * scale * scale - 2.0f * scale * dot;
}
// Register-resident code is shared by every visitor in list-major scanning.
__device__ inline float distance(uint4 code, float scale, const float* residual) {
    const float dot = (signed_dot_word(code.x, residual) +
                       signed_dot_word(code.y, residual + 32)) +
                      (signed_dot_word(code.z, residual + 64) +
                       signed_dot_word(code.w, residual + 96));
    return reconstructed_distance(residual[128], scale, dot, 128);
}
__device__ inline float distance(SignResidualDeviceView view, const float* residual,
                                 int row) {
    // Host preparation currently accepts SIFT128 only. Both seed and full
    // scan use this same score and summation order.
    const uint4 code = reinterpret_cast<const uint4*>(view.codes)[row];
    return distance(code, view.scales[row], residual);
}
#endif
} // namespace paraselect::search::turboquant
