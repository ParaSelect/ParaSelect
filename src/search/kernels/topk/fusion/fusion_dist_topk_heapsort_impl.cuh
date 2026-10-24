/**
 * Fusion Distance Top-K Implementation - HeapSort Kernels
 *
 * This header contains the HeapSort-based template implementations for fused
 * distance computation and top-k selection for large k values (256 < k <= 1024).
 */

#ifndef FUSION_DIST_TOPK_HEAPSORT_IMPL_CUH
#define FUSION_DIST_TOPK_HEAPSORT_IMPL_CUH

#include <limits>
#include <type_traits>
#include <math_constants.h>
#include <cfloat>
#include "utils/utils.cuh"
#include "utils/pch.h"

namespace paraselect {
namespace fusion_dist_topk_heapsort {

constexpr int kBlockThreads = 128;
constexpr int kWarpSize_ = 32;
constexpr int kWarpsPerBlock = kBlockThreads / kWarpSize_;

template <typename T>
__host__ __device__ inline T lower_bound_val() {
    return std::numeric_limits<T>::lowest();
}

template <typename T>
__host__ __device__ inline T upper_bound_val() {
    return std::numeric_limits<T>::max();
}

template <typename T>
__device__ inline bool lt_dev(T a, T b) {
    return a < b;
}

template <typename T>
__device__ inline bool gt_dev(T a, T b) {
    return a > b;
}

template <typename T, typename IdxT, bool Ascending>
__device__ inline bool better(T va, IdxT ia, T vb, IdxT ib) {
    if constexpr (Ascending) {
        if (lt_dev(va, vb)) return true;
        if (gt_dev(va, vb)) return false;
        return ia < ib;
    } else {
        if (gt_dev(va, vb)) return true;
        if (lt_dev(va, vb)) return false;
        return ia < ib;
    }
}

template <typename T, typename IdxT, bool Ascending>
__device__ inline bool worse(T va, IdxT ia, T vb, IdxT ib) {
    return better<T, IdxT, !Ascending>(va, ia, vb, ib);
}

template <typename T, typename IdxT>
__device__ inline void swap_pair(T* vals, IdxT* idx, int a, int b) {
    T tv = vals[a];
    vals[a] = vals[b];
    vals[b] = tv;

    IdxT ti = idx[a];
    idx[a] = idx[b];
    idx[b] = ti;
}

template <typename T, typename IdxT, bool Ascending>
__device__ inline void sift_down(T* vals, IdxT* idx, int n, int pos) {
    while (true) {
        int l = pos * 2 + 1;
        if (l >= n) break;
        int r = l + 1;
        int c = l;
        if (r < n && worse<T, IdxT, Ascending>(vals[r], idx[r], vals[l], idx[l])) {
            c = r;
        }
        if (!worse<T, IdxT, Ascending>(vals[c], idx[c], vals[pos], idx[pos])) break;
        swap_pair(vals, idx, pos, c);
        pos = c;
    }
}

template <typename T, typename IdxT, bool Ascending>
__device__ inline void build_heap(T* vals, IdxT* idx, int n) {
    for (int i = n / 2 - 1; i >= 0; --i) {
        sift_down<T, IdxT, Ascending>(vals, idx, n, i);
    }
}

template <typename T, typename IdxT, bool Ascending>
__device__ inline void consider(
    T* heap_vals,
    IdxT* heap_idx,
    int k,
    T v,
    IdxT i)
{
    if (better<T, IdxT, Ascending>(v, i, heap_vals[0], heap_idx[0])) {
        heap_vals[0] = v;
        heap_idx[0] = i;
        sift_down<T, IdxT, Ascending>(heap_vals, heap_idx, k, 0);
    }
}

template <typename T, typename IdxT, bool Ascending>
__device__ inline void heap_sort_selected(T* vals, IdxT* idx, int k) {
    for (int end = k - 1; end > 0; --end) {
        swap_pair(vals, idx, 0, end);
        sift_down<T, IdxT, Ascending>(vals, idx, end, 0);
    }
}

/**
 * Cosine Distance Top-K Kernel using HeapSort
 *
 * Fused kernel that computes cosine distance and selects top-k using heap.
 * Each CUDA block processes one query row.
 * Supports k up to 1024.
 */
template<bool Ascending, typename T, typename IdxT>
__global__ void fusion_cos_topk_heapsort_kernel(
    const T* __restrict__ d_query_norm,
    const T* __restrict__ d_data_norm,
    const T* __restrict__ d_inner_product,
    int batch_size,
    int len,
    int k,
    T* __restrict__ output_vals,
    IdxT* __restrict__ output_idx)
{
    int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ unsigned char smem[];
    unsigned char* p = smem;

    T* warp_heap_vals = reinterpret_cast<T*>(p);
    p += sizeof(T) * kWarpsPerBlock * k;

    IdxT* warp_heap_idx = reinterpret_cast<IdxT*>(p);
    p += sizeof(IdxT) * kWarpsPerBlock * k;

    T* stage_vals = reinterpret_cast<T*>(p);
    p += sizeof(T) * kWarpsPerBlock * kWarpSize_;

    IdxT* stage_idx = reinterpret_cast<IdxT*>(p);

    int tid = threadIdx.x;
    int warp_id = tid / kWarpSize_;
    int lane = tid % kWarpSize_;

    T* my_heap_vals = warp_heap_vals + warp_id * k;
    IdxT* my_heap_idx = warp_heap_idx + warp_id * k;
    T* my_stage_vals = stage_vals + warp_id * kWarpSize_;
    IdxT* my_stage_idx = stage_idx + warp_id * kWarpSize_;

    float query_norm = d_query_norm[row];
    const T* row_inner_product = d_inner_product + row * len;

    int chunk = (len + kWarpsPerBlock - 1) / kWarpsPerBlock;
    int begin = warp_id * chunk;
    int end = begin + chunk;
    if (end > len) end = len;

    int local_len = max(0, end - begin);
    int init_count = min(k, local_len);

    T dummy = Ascending ? upper_bound_val<T>() : lower_bound_val<T>();

    for (int j = lane; j < k; j += kWarpSize_) {
        my_heap_vals[j] = dummy;
        my_heap_idx[j] = static_cast<IdxT>(-1);
    }
    __syncthreads();

    // Initialize heap with first k elements
    for (int j = lane; j < init_count; j += kWarpSize_) {
        int idx = begin + j;
        float data_norm = d_data_norm[idx];
        float inner_product = row_inner_product[idx];
        IdxT index = static_cast<IdxT>(idx);  //

        bool is_valid_index = true;
        if (std::is_signed<IdxT>::value) {
            is_valid_index = (index >= 0);
        }

        if (is_valid_index) {
            float cos_similarity = inner_product / (query_norm * data_norm);
            float cos_distance = 1.0f - cos_similarity;
            my_heap_vals[j] = cos_distance;
            my_heap_idx[j] = index;
        } else {
            my_heap_vals[j] = dummy;
            my_heap_idx[j] = static_cast<IdxT>(-1);
        }
    }
    __syncwarp();

    if (lane == 0) {
        build_heap<T, IdxT, Ascending>(my_heap_vals, my_heap_idx, k);
    }
    __syncwarp();

    // Process remaining elements
    for (int base = begin + init_count; base < end; base += kWarpSize_) {
        int idx = base + lane;
        if (idx < end) {
            float data_norm = d_data_norm[idx];
            float inner_product = row_inner_product[idx];
            IdxT index = static_cast<IdxT>(idx);  //

            bool is_valid_index = true;
            if (std::is_signed<IdxT>::value) {
                is_valid_index = (index >= 0);
            }

            if (is_valid_index) {
                float cos_similarity = inner_product / (query_norm * data_norm);
                float cos_distance = 1.0f - cos_similarity;
                my_stage_vals[lane] = cos_distance;
                my_stage_idx[lane] = index;
            } else {
                my_stage_vals[lane] = dummy;
                my_stage_idx[lane] = static_cast<IdxT>(-1);
            }
        }
        __syncwarp();

        unsigned mask = __ballot_sync(0xffffffffu, idx < end);

        if (lane == 0) {
            while (mask) {
                int src_lane = __ffs(mask) - 1;
                if (my_stage_idx[src_lane] != static_cast<IdxT>(-1)) {
                    consider<T, IdxT, Ascending>(
                        my_heap_vals, my_heap_idx, k,
                        my_stage_vals[src_lane], my_stage_idx[src_lane]);
                }
                mask &= (mask - 1);
            }
        }
        __syncwarp();
    }

    __syncthreads();

    // Merge warps and output
    if (warp_id == 0 && lane == 0) {
        T* final_vals = warp_heap_vals;
        IdxT* final_idx = warp_heap_idx;

        for (int src_warp = 1; src_warp < kWarpsPerBlock; ++src_warp) {
            T* src_vals = warp_heap_vals + src_warp * k;
            IdxT* src_idx = warp_heap_idx + src_warp * k;
            for (int j = 0; j < k; ++j) {
                if (src_idx[j] != static_cast<IdxT>(-1)) {
                    consider<T, IdxT, Ascending>(
                        final_vals, final_idx, k, src_vals[j], src_idx[j]);
                }
            }
        }

        heap_sort_selected<T, IdxT, Ascending>(final_vals, final_idx, k);

        T* out_v = output_vals + static_cast<size_t>(row) * k;
        IdxT* out_i = output_idx + static_cast<size_t>(row) * k;
        for (int j = 0; j < k; ++j) {
            out_v[j] = final_vals[j];
            out_i[j] = final_idx[j];
        }
    }
}

/**
 * L2 Distance Top-K Kernel using HeapSort
 *
 * Fused kernel that computes L2 distance and selects top-k using heap.
 * L2 distance: ||q - v||^2 = ||q||^2 + ||v||^2 - 2 * <q, v>
 */
template<bool Ascending, typename T, typename IdxT>
__global__ void fusion_l2_topk_heapsort_kernel(
    const T* __restrict__ d_query_norm,
    const T* __restrict__ d_data_norm,
    const T* __restrict__ d_inner_product,
    int batch_size,
    int len,
    int k,
    T* __restrict__ output_vals,
    IdxT* __restrict__ output_idx)
{
    int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ unsigned char smem[];
    unsigned char* p = smem;

    T* warp_heap_vals = reinterpret_cast<T*>(p);
    p += sizeof(T) * kWarpsPerBlock * k;

    IdxT* warp_heap_idx = reinterpret_cast<IdxT*>(p);
    p += sizeof(IdxT) * kWarpsPerBlock * k;

    T* stage_vals = reinterpret_cast<T*>(p);
    p += sizeof(T) * kWarpsPerBlock * kWarpSize_;

    IdxT* stage_idx = reinterpret_cast<IdxT*>(p);

    int tid = threadIdx.x;
    int warp_id = tid / kWarpSize_;
    int lane = tid % kWarpSize_;

    T* my_heap_vals = warp_heap_vals + warp_id * k;
    IdxT* my_heap_idx = warp_heap_idx + warp_id * k;
    T* my_stage_vals = stage_vals + warp_id * kWarpSize_;
    IdxT* my_stage_idx = stage_idx + warp_id * kWarpSize_;

    float query_norm_sq = d_query_norm[row] * d_query_norm[row];
    const T* row_inner_product = d_inner_product + row * len;

    int chunk = (len + kWarpsPerBlock - 1) / kWarpsPerBlock;
    int begin = warp_id * chunk;
    int end = begin + chunk;
    if (end > len) end = len;

    int local_len = max(0, end - begin);
    int init_count = min(k, local_len);

    T dummy = Ascending ? upper_bound_val<T>() : lower_bound_val<T>();

    for (int j = lane; j < k; j += kWarpSize_) {
        my_heap_vals[j] = dummy;
        my_heap_idx[j] = static_cast<IdxT>(-1);
    }
    __syncthreads();

    // Initialize heap with first k elements
    for (int j = lane; j < init_count; j += kWarpSize_) {
        int idx = begin + j;
        float data_norm = d_data_norm[idx];
        float inner_product = row_inner_product[idx];
        IdxT index = static_cast<IdxT>(idx);  //

        bool is_valid_index = true;
        if (std::is_signed<IdxT>::value) {
            is_valid_index = (index >= 0);
        }

        if (is_valid_index) {
            float data_norm_sq = data_norm * data_norm;
            float l2_dist = query_norm_sq + data_norm_sq - 2.0f * inner_product;
            if (l2_dist < 0) l2_dist = 0;
            my_heap_vals[j] = l2_dist;
            my_heap_idx[j] = index;
        } else {
            my_heap_vals[j] = dummy;
            my_heap_idx[j] = static_cast<IdxT>(-1);
        }
    }
    __syncwarp();

    if (lane == 0) {
        build_heap<T, IdxT, Ascending>(my_heap_vals, my_heap_idx, k);
    }
    __syncwarp();

    // Process remaining elements
    for (int base = begin + init_count; base < end; base += kWarpSize_) {
        int idx = base + lane;
        if (idx < end) {
            float data_norm = d_data_norm[idx];
            float inner_product = row_inner_product[idx];
            IdxT index = static_cast<IdxT>(idx);  //

            bool is_valid_index = true;
            if (std::is_signed<IdxT>::value) {
                is_valid_index = (index >= 0);
            }

            if (is_valid_index) {
                float data_norm_sq = data_norm * data_norm;
                float l2_dist = query_norm_sq + data_norm_sq - 2.0f * inner_product;
                if (l2_dist < 0) l2_dist = 0;
                my_stage_vals[lane] = l2_dist;
                my_stage_idx[lane] = index;
            } else {
                my_stage_vals[lane] = dummy;
                my_stage_idx[lane] = static_cast<IdxT>(-1);
            }
        }
        __syncwarp();

        unsigned mask = __ballot_sync(0xffffffffu, idx < end);

        if (lane == 0) {
            while (mask) {
                int src_lane = __ffs(mask) - 1;
                if (my_stage_idx[src_lane] != static_cast<IdxT>(-1)) {
                    consider<T, IdxT, Ascending>(
                        my_heap_vals, my_heap_idx, k,
                        my_stage_vals[src_lane], my_stage_idx[src_lane]);
                }
                mask &= (mask - 1);
            }
        }
        __syncwarp();
    }

    __syncthreads();

    // Merge warps and output
    if (warp_id == 0 && lane == 0) {
        T* final_vals = warp_heap_vals;
        IdxT* final_idx = warp_heap_idx;

        for (int src_warp = 1; src_warp < kWarpsPerBlock; ++src_warp) {
            T* src_vals = warp_heap_vals + src_warp * k;
            IdxT* src_idx = warp_heap_idx + src_warp * k;
            for (int j = 0; j < k; ++j) {
                if (src_idx[j] != static_cast<IdxT>(-1)) {
                    consider<T, IdxT, Ascending>(
                        final_vals, final_idx, k, src_vals[j], src_idx[j]);
                }
            }
        }

        heap_sort_selected<T, IdxT, Ascending>(final_vals, final_idx, k);

        T* out_v = output_vals + static_cast<size_t>(row) * k;
        IdxT* out_i = output_idx + static_cast<size_t>(row) * k;
        for (int j = 0; j < k; ++j) {
            out_v[j] = final_vals[j];
            out_i[j] = final_idx[j];
        }
    }
}

} // namespace fusion_dist_topk_heapsort
} // namespace paraselect

#endif // FUSION_DIST_TOPK_HEAPSORT_IMPL_CUH
