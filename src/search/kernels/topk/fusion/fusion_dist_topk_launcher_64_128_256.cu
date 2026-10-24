/**
 * Fusion Distance Top-K Launcher - Capacity 64/128/256
 */

#include "fusion_dist_topk_impl.cuh"
#include "fusion_dist_topk_launcher.cuh"

namespace paraselect {
namespace fusion_dist_topk_warpsort {

// Explicit instantiations for kernel templates
template __global__ void fusion_cos_topk_warpsort_kernel<64, true, float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*);
template __global__ void fusion_cos_topk_warpsort_kernel<64, false, float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*);
template __global__ void fusion_cos_topk_warpsort_kernel<64, true, float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*);
template __global__ void fusion_cos_topk_warpsort_kernel<64, false, float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*);
template __global__ void fusion_cos_topk_warpsort_kernel<64, true, float, idx_t>(
    const float*, const float*, const float*, int, int, int, float*, idx_t*);
template __global__ void fusion_cos_topk_warpsort_kernel<64, false, float, idx_t>(
    const float*, const float*, const float*, int, int, int, float*, idx_t*);

template __global__ void fusion_cos_topk_warpsort_kernel<128, true, float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*);
template __global__ void fusion_cos_topk_warpsort_kernel<128, false, float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*);
template __global__ void fusion_cos_topk_warpsort_kernel<128, true, float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*);
template __global__ void fusion_cos_topk_warpsort_kernel<128, false, float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*);
template __global__ void fusion_cos_topk_warpsort_kernel<128, true, float, idx_t>(
    const float*, const float*, const float*, int, int, int, float*, idx_t*);
template __global__ void fusion_cos_topk_warpsort_kernel<128, false, float, idx_t>(
    const float*, const float*, const float*, int, int, int, float*, idx_t*);

template __global__ void fusion_cos_topk_warpsort_kernel<256, true, float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*);
template __global__ void fusion_cos_topk_warpsort_kernel<256, false, float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*);
template __global__ void fusion_cos_topk_warpsort_kernel<256, true, float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*);
template __global__ void fusion_cos_topk_warpsort_kernel<256, false, float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*);
template __global__ void fusion_cos_topk_warpsort_kernel<256, true, float, idx_t>(
    const float*, const float*, const float*, int, int, int, float*, idx_t*);
template __global__ void fusion_cos_topk_warpsort_kernel<256, false, float, idx_t>(
    const float*, const float*, const float*, int, int, int, float*, idx_t*);

// Capacity 512 instantiations
template __global__ void fusion_cos_topk_warpsort_kernel<512, true, float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*);
template __global__ void fusion_cos_topk_warpsort_kernel<512, false, float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*);
template __global__ void fusion_cos_topk_warpsort_kernel<512, true, float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*);
template __global__ void fusion_cos_topk_warpsort_kernel<512, false, float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*);
template __global__ void fusion_cos_topk_warpsort_kernel<512, true, float, idx_t>(
    const float*, const float*, const float*, int, int, int, float*, idx_t*);
template __global__ void fusion_cos_topk_warpsort_kernel<512, false, float, idx_t>(
    const float*, const float*, const float*, int, int, int, float*, idx_t*);

template __global__ void fusion_l2_topk_warpsort_kernel<64, true, float, int>(
    const float*, const float*, const float*,  int, int, int, float*, int*);
template __global__ void fusion_l2_topk_warpsort_kernel<64, false, float, int>(
    const float*, const float*, const float*,  int, int, int, float*, int*);
template __global__ void fusion_l2_topk_warpsort_kernel<64, true, float, uint32_t>(
    const float*, const float*, const float*,  int, int, int, float*, uint32_t*);
template __global__ void fusion_l2_topk_warpsort_kernel<64, false, float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*);
template __global__ void fusion_l2_topk_warpsort_kernel<64, true, float, idx_t>(
    const float*, const float*, const float*,  int, int, int, float*, idx_t*);
template __global__ void fusion_l2_topk_warpsort_kernel<64, false, float, idx_t>(
    const float*, const float*, const float*,  int, int, int, float*, idx_t*);

template __global__ void fusion_l2_topk_warpsort_kernel<128, true, float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*);
template __global__ void fusion_l2_topk_warpsort_kernel<128, false, float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*);
template __global__ void fusion_l2_topk_warpsort_kernel<128, true, float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*);
template __global__ void fusion_l2_topk_warpsort_kernel<128, false, float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*);
template __global__ void fusion_l2_topk_warpsort_kernel<128, true, float, idx_t>(
    const float*, const float*, const float*, int, int, int, float*, idx_t*);
template __global__ void fusion_l2_topk_warpsort_kernel<128, false, float, idx_t>(
    const float*, const float*, const float*, int, int, int, float*, idx_t*);

template __global__ void fusion_l2_topk_warpsort_kernel<256, true, float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*);
template __global__ void fusion_l2_topk_warpsort_kernel<256, false, float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*);
template __global__ void fusion_l2_topk_warpsort_kernel<256, true, float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*);
template __global__ void fusion_l2_topk_warpsort_kernel<256, false, float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*);
template __global__ void fusion_l2_topk_warpsort_kernel<256, true, float, idx_t>(
    const float*, const float*, const float*, int, int, int, float*, idx_t*);
template __global__ void fusion_l2_topk_warpsort_kernel<256, false, float, idx_t>(
    const float*, const float*, const float*, int, int, int, float*, idx_t*);

// Capacity 512 instantiations for L2
template __global__ void fusion_l2_topk_warpsort_kernel<512, true, float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*);
template __global__ void fusion_l2_topk_warpsort_kernel<512, false, float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*);
template __global__ void fusion_l2_topk_warpsort_kernel<512, true, float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*);
template __global__ void fusion_l2_topk_warpsort_kernel<512, false, float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*);
template __global__ void fusion_l2_topk_warpsort_kernel<512, true, float, idx_t>(
    const float*, const float*, const float*, int, int, int, float*, idx_t*);
template __global__ void fusion_l2_topk_warpsort_kernel<512, false, float, idx_t>(
    const float*, const float*, const float*, int, int, int, float*, idx_t*);

// Host launch functions for k <= 256
template<typename T, typename IdxT>
cudaError_t fusion_cos_topk_warpsort_small(
    const T* d_query_norm, const T* d_data_norm, const T* d_inner_product,
    int batch_size, int len, int k,
    T* output_vals, IdxT* output_idx,
    bool select_min,
    cudaStream_t stream)
{
    // WarpSortFiltered keeps both the selected lanes and a staging buffer.
    // Match IVFTensor's rule: one extra 32-lane segment beyond ceil(k / 32).
    int capacity = 64;
    const int lanes_needed = ((k + 31) / 32 + 1) * 32;
    while (capacity < lanes_needed) capacity <<= 1;

    dim3 block(32);
    dim3 grid(batch_size);

    if (select_min) {
        if (capacity <= 64) {
            fusion_cos_topk_warpsort_kernel<64, true, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product,
                batch_size, len, k, output_vals, output_idx);
        } else if (capacity <= 128) {
            fusion_cos_topk_warpsort_kernel<128, true, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product,
                batch_size, len, k, output_vals, output_idx);
        } else if (capacity <= 256) {
            fusion_cos_topk_warpsort_kernel<256, true, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product,
                batch_size, len, k, output_vals, output_idx);
        } else {
            fusion_cos_topk_warpsort_kernel<512, true, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product,
                batch_size, len, k, output_vals, output_idx);
        }
    } else {
        if (capacity <= 64) {
            fusion_cos_topk_warpsort_kernel<64, false, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product,
                batch_size, len, k, output_vals, output_idx);
        } else if (capacity <= 128) {
            fusion_cos_topk_warpsort_kernel<128, false, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product,
                batch_size, len, k, output_vals, output_idx);
        } else if (capacity <= 256) {
            fusion_cos_topk_warpsort_kernel<256, false, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product,
                batch_size, len, k, output_vals, output_idx);
        } else {
            fusion_cos_topk_warpsort_kernel<512, false, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product,
                batch_size, len, k, output_vals, output_idx);
        }
    }
    return cudaGetLastError();
}

template<typename T, typename IdxT>
cudaError_t fusion_l2_topk_warpsort_small(
    const T* d_query_norm, const T* d_data_norm, const T* d_inner_product,
    int batch_size, int len, int k,
    T* output_vals, IdxT* output_idx,
    bool select_min,
    cudaStream_t stream)
{
    // WarpSortFiltered keeps both the selected lanes and a staging buffer.
    // Match IVFTensor's rule: one extra 32-lane segment beyond ceil(k / 32).
    int capacity = 64;
    const int lanes_needed = ((k + 31) / 32 + 1) * 32;
    while (capacity < lanes_needed) capacity <<= 1;

    dim3 block(32);
    dim3 grid(batch_size);

    if (select_min) {
        if (capacity <= 64) {
            fusion_l2_topk_warpsort_kernel<64, true, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product,
                batch_size, len, k, output_vals, output_idx);
        } else if (capacity <= 128) {
            fusion_l2_topk_warpsort_kernel<128, true, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product,
                batch_size, len, k, output_vals, output_idx);
        } else if (capacity <= 256) {
            fusion_l2_topk_warpsort_kernel<256, true, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product,
                batch_size, len, k, output_vals, output_idx);
        } else {
            fusion_l2_topk_warpsort_kernel<512, true, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product,
                batch_size, len, k, output_vals, output_idx);
        }
    } else {
        if (capacity <= 64) {
            fusion_l2_topk_warpsort_kernel<64, false, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product,
                batch_size, len, k, output_vals, output_idx);
        } else if (capacity <= 128) {
            fusion_l2_topk_warpsort_kernel<128, false, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product,
                batch_size, len, k, output_vals, output_idx);
        } else if (capacity <= 256) {
            fusion_l2_topk_warpsort_kernel<256, false, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product,
                batch_size, len, k, output_vals, output_idx);
        } else {
            fusion_l2_topk_warpsort_kernel<512, false, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product,
                batch_size, len, k, output_vals, output_idx);
        }
    }
    return cudaGetLastError();
}

// Explicit instantiations for host functions
template cudaError_t fusion_cos_topk_warpsort_small<float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*, bool, cudaStream_t);
template cudaError_t fusion_cos_topk_warpsort_small<float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*, bool, cudaStream_t);
template cudaError_t fusion_cos_topk_warpsort_small<float, idx_t>(
    const float*, const float*, const float*, int, int, int, float*, idx_t*, bool, cudaStream_t);

template cudaError_t fusion_l2_topk_warpsort_small<float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*, bool, cudaStream_t);
template cudaError_t fusion_l2_topk_warpsort_small<float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*, bool, cudaStream_t);
template cudaError_t fusion_l2_topk_warpsort_small<float, idx_t>(
    const float*, const float*, const float*, int, int, int, float*, idx_t*, bool, cudaStream_t);

} // namespace fusion_dist_topk_warpsort
} // namespace paraselect
