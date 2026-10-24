/**
 *  fp16 :cos topk warpsort, __half.
 */
#include <limits>
#include <type_traits>
#include <math_constants.h>
#include <cfloat>
#include <cuda_fp16.h>
#include "utils/utils.cuh"
#include "fusion_dist_topk.cuh"
#include "utils/pch.h"
#include "search/kernels/topk/warpsort/warpsort_utils.cuh"
#include "search/kernels/topk/warpsort/warpsort.cuh"

namespace paraselect {
namespace fusion_dist_topk_warpsort {

using namespace warpsort_utils;
using namespace warpsort;

template<int Capacity, bool Ascending>
__global__ void fusion_cos_topk_warpsort_kernel_fp16(
    const __half* __restrict__ d_query_norm,
    const __half* __restrict__ d_data_norm,
    const __half* __restrict__ d_inner_product,
    int batch_size,
    int len,
    int k,
    __half* __restrict__ output_vals,
    idx_t* __restrict__ output_idx)
{
    const int row = blockIdx.x;
    if (row >= batch_size) return;
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = laneId();
    const int n_warps = blockDim.x / kWarpSize;
    WarpSortFiltered<Capacity, Ascending, __half, idx_t> queue(k);
    float query_norm = __half2float(d_query_norm[row]);
    using BaseWarpSort = WarpSort<Capacity, Ascending, __half, int>;
    const __half dummy_val = BaseWarpSort::kDummy();
    const __half* row_inner_product = d_inner_product + row * len;
    __syncwarp();
    int max_iter = (len + n_warps * kWarpSize - 1) / (n_warps * kWarpSize);
    for (int iter = 0; iter < max_iter; iter++) {
        __syncwarp();
        int i = warp_id * kWarpSize + lane + iter * n_warps * kWarpSize;
        bool has_data = (i < len);
        if (has_data) {
            float data_norm = __half2float(d_data_norm[i]);
            float inner_product = __half2float(row_inner_product[i]);
            int index = i;  //
            bool is_valid_index = (index >= 0);
            if (is_valid_index) {
                float cos_similarity = inner_product / (query_norm * data_norm);
                float cos_distance = 1.0f - cos_similarity;
                queue.add(__float2half_rn(cos_distance), index);
            } else {
                queue.add(dummy_val, 0);
            }
        } else {
            queue.add(dummy_val, 0);
        }
    }
    __syncwarp();
    queue.done();
    if (warp_id == 0) {
        __half* row_out_vals = output_vals + row * k;
        idx_t* row_out_idx = output_idx + row * k;
        queue.store(row_out_vals, row_out_idx);
    }
}

cudaError_t fusion_cos_topk_warpsort_fp16(
    const __half* d_query_norm,
    const __half* d_data_norm,
    const __half* d_inner_product,
    int batch_size,
    int len,
    int k,
    __half* output_vals,
    idx_t* output_idx,
    bool select_min,
    cudaStream_t stream) {
    if (k > kMaxCapacity) return cudaErrorInvalidValue;
    int capacity = 32;
    while (capacity < k) capacity <<= 1;
    dim3 block(32);
    dim3 grid(batch_size);
    if (select_min) {
        if (capacity <= 32)
            fusion_cos_topk_warpsort_kernel_fp16<64, true><<<grid, block, 0, stream>>>(d_query_norm, d_data_norm, d_inner_product, batch_size, len, k, output_vals, output_idx);
        else if (capacity <= 64)
            fusion_cos_topk_warpsort_kernel_fp16<128, true><<<grid, block, 0, stream>>>(d_query_norm, d_data_norm, d_inner_product, batch_size, len, k, output_vals, output_idx);
        else
            fusion_cos_topk_warpsort_kernel_fp16<256, true><<<grid, block, 0, stream>>>(d_query_norm, d_data_norm, d_inner_product, batch_size, len, k, output_vals, output_idx);
    } else {
        if (capacity <= 32)
            fusion_cos_topk_warpsort_kernel_fp16<64, false><<<grid, block, 0, stream>>>(d_query_norm, d_data_norm, d_inner_product, batch_size, len, k, output_vals, output_idx);
        else if (capacity <= 64)
            fusion_cos_topk_warpsort_kernel_fp16<128, false><<<grid, block, 0, stream>>>(d_query_norm, d_data_norm, d_inner_product, batch_size, len, k, output_vals, output_idx);
        else
            fusion_cos_topk_warpsort_kernel_fp16<256, false><<<grid, block, 0, stream>>>(d_query_norm, d_data_norm, d_inner_product, batch_size, len, k, output_vals, output_idx);
    }
    return cudaGetLastError();
}

} // namespace fusion_dist_topk_warpsort
} // namespace paraselect
