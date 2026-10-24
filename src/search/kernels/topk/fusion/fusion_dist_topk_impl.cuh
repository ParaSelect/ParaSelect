/**
 * Fusion Distance Top-K Implementation - Template Kernels
 *
 * This header contains the template implementations for fused distance
 * computation and top-k selection.
 *
 * :
 * 1. warpsort  -  kernel, topk  kernel
 * 2. cuVS  -  kernel,, cuVS select_k
 */

#ifndef FUSION_DIST_TOPK_IMPL_CUH
#define FUSION_DIST_TOPK_IMPL_CUH

#include <limits>
#include <type_traits>
#include <math_constants.h>
#include <cfloat>
#include "utils/utils.cuh"
#include "utils/pch.h"
#include "search/kernels/topk/warpsort/warpsort_utils.cuh"
#include "search/kernels/topk/warpsort/warpsort.cuh"

namespace paraselect {
namespace fusion_dist_topk_warpsort {

using namespace warpsort_utils;
using namespace warpsort;

/* ============================================================================
 *  warpsort  -  kernel
 * ============================================================================ */

/**
 * Cosine Distance Top-K Kernel (warpsort )
 *
 * Fused kernel that computes cosine distance and selects top-k in one pass.
 * Each CUDA block processes one query row.
 */
template<int Capacity, bool Ascending, typename T, typename IdxT>
__global__ void fusion_cos_topk_warpsort_kernel(
    const T* __restrict__ d_query_norm,
    const T* __restrict__ d_data_norm,
    const T* __restrict__ d_inner_product,
    int batch_size,
    int len,
    int k,
    T* __restrict__ output_vals,
    IdxT* __restrict__ output_idx)
{
    const int row = blockIdx.x;
    if (row >= batch_size) return;

    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = laneId();
    const int n_warps = blockDim.x / kWarpSize;

    WarpSortFiltered<Capacity, Ascending, T, IdxT> queue(k);

    float query_norm = d_query_norm[row];

    using BaseWarpSort = WarpSort<Capacity, Ascending, T, IdxT>;
    const T dummy_val = BaseWarpSort::kDummy();

    const T* row_inner_product = d_inner_product + row * len;

    __syncwarp();

    int max_iter = (len + n_warps * kWarpSize - 1) / (n_warps * kWarpSize);

    for (int iter = 0; iter < max_iter; iter++) {
        __syncwarp();

        int i = warp_id * kWarpSize + lane + iter * n_warps * kWarpSize;
        bool has_data = (i < len);

        if (has_data) {
            float data_norm = d_data_norm[i];
            float inner_product = row_inner_product[i];
            IdxT index = static_cast<IdxT>(i);

            float cos_similarity = inner_product / (query_norm * data_norm);
            float cos_distance = 1.0f - cos_similarity;
            queue.add(cos_distance, index);
        } else {
            queue.add(dummy_val, IdxT{});
        }
    }

    __syncwarp();
    queue.done();

    if (warp_id == 0) {
        T* row_out_val = output_vals + row * k;
        IdxT* row_out_idx = output_idx + row * k;
        queue.store(row_out_val, row_out_idx);
    }
}

/**
 * L2 Distance Top-K Kernel (warpsort )
 *
 * Fused kernel that computes L2 distance and selects top-k in one pass.
 * L2 distance: ||q - v||^2 = ||q||^2 + ||v||^2 - 2 * <q, v>
 */
template<int Capacity, bool Ascending, typename T, typename IdxT>
__global__ void fusion_l2_topk_warpsort_kernel(
    const T* __restrict__ d_query_norm,
    const T* __restrict__ d_data_norm,
    const T* __restrict__ d_inner_product,
    int batch_size,
    int len,
    int k,
    T* __restrict__ output_vals,
    IdxT* __restrict__ output_idx)
{
    const int row = blockIdx.x;
    if (row >= batch_size) return;

    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = laneId();
    const int n_warps = blockDim.x / kWarpSize;

    WarpSortFiltered<Capacity, Ascending, T, IdxT> queue(k);

    float query_norm_sq = d_query_norm[row] * d_query_norm[row];

    using BaseWarpSort = WarpSort<Capacity, Ascending, T, IdxT>;
    const T dummy_val = BaseWarpSort::kDummy();

    const T* row_inner_product = d_inner_product + row * len;

    __syncwarp();

    int max_iter = (len + n_warps * kWarpSize - 1) / (n_warps * kWarpSize);

    for (int iter = 0; iter < max_iter; iter++) {
        __syncwarp();

        int i = warp_id * kWarpSize + lane + iter * n_warps * kWarpSize;
        bool has_data = (i < len);

        if (has_data) {
            float data_norm = d_data_norm[i];
            float inner_product = row_inner_product[i];
            IdxT index = static_cast<IdxT>(i);

            float data_norm_sq = data_norm * data_norm;
            float l2_dist = query_norm_sq + data_norm_sq - 2.0f * inner_product;
            queue.add(l2_dist, index);
        } else {
            queue.add(dummy_val, IdxT{});
        }
    }

    __syncwarp();
    queue.done();

    if (warp_id == 0) {
        T* row_out_val = output_vals + row * k;
        IdxT* row_out_idx = output_idx + row * k;
        queue.store(row_out_val, row_out_idx);
    }
}

} // namespace fusion_dist_topk_warpsort
} // namespace paraselect

/* ============================================================================
 *  Kernel:L2 (cuVS )
 * , warpsort  cuvs
 * ============================================================================ */
template <typename T>
__global__ void l2_distance_from_inner_product_kernel(
    const T* __restrict__ d_query_norm,
    const T* __restrict__ d_data_norm,
    const T* __restrict__ d_inner_product,
    T* __restrict__ d_dist,
    int batch_size,
    int len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * len;
    if (tid >= total) return;

    int row = tid / len;
    int col = tid % len;

    T qn = d_query_norm[row];
    T dn = d_data_norm[col];
    T ip = d_inner_product[tid];

    // L2^2 = ||q||^2 + ||v||^2 - 2*<q,v>
    d_dist[tid] = qn * qn + dn * dn - (T)2.0 * ip;
}

namespace paraselect {
namespace fusion_dist_topk_warpsort {

/* ============================================================================
 * cuVS  - ,topk  cuVS
 * ============================================================================ */

/**
 * Cosine Distance Kernel (cuVS )
 *
 * , topk. d_output_dist, cuVS select_k.
 * Cosine distance = 1 - <q,v>/(||q||*||v||)
 */
template<typename T>
__global__ void fusion_cos_distance_kernel(
    const T* __restrict__ d_query_norm,
    const T* __restrict__ d_data_norm,
    const T* __restrict__ d_inner_product,
    int batch_size,
    int len,
    T* __restrict__ d_output_dist)
{
    const int row = blockIdx.x;
    if (row >= batch_size) return;

    const int tid = threadIdx.x;
    const int row_offset = row * len;

    float query_norm = d_query_norm[row];

    for (int i = tid; i < len; i += blockDim.x) {
        float data_norm = d_data_norm[i];
        float inner_product = d_inner_product[row_offset + i];

        float denom = query_norm * data_norm;
        if (denom < 1e-12f) denom = 1e-12f;

        d_output_dist[row_offset + i] = static_cast<T>(1.0f - inner_product / denom);
    }
}

/**
 * L2 Distance Kernel (cuVS )
 *
 *  L2 , topk. d_output_dist, cuVS select_k.
 * L2 distance: ||q - v||^2 = ||q||^2 + ||v||^2 - 2 * <q, v>
 */
template<typename T>
__global__ void fusion_l2_distance_kernel(
    const T* __restrict__ d_query_norm,
    const T* __restrict__ d_data_norm,
    const T* __restrict__ d_inner_product,
    int batch_size,
    int len,
    T* __restrict__ d_output_dist)
{
    const int row = blockIdx.x;
    if (row >= batch_size) return;

    const int tid = threadIdx.x;
    const int row_offset = row * len;

    float query_norm = d_query_norm[row];
    float query_norm_sq = query_norm * query_norm;

    for (int i = tid; i < len; i += blockDim.x) {
        float data_norm = d_data_norm[i];
        float inner_product = d_inner_product[row_offset + i];

        float data_norm_sq = data_norm * data_norm;
        float l2_dist = query_norm_sq + data_norm_sq - 2.0f * inner_product;
        if (l2_dist < 0) l2_dist = 0; // Numerical error correction

        d_output_dist[row_offset + i] = static_cast<T>(l2_dist);
    }
}

} // namespace fusion_dist_topk_warpsort
} // namespace paraselect

#endif // FUSION_DIST_TOPK_IMPL_CUH
