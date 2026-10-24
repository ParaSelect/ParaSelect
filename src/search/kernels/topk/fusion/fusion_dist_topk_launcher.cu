/**
 * Fusion Distance Top-K Launcher - Main Entry Point
 *
 * Thin launcher for the fused WarpSort kernels.
 *
 * This file does not choose cuVS or tiled top-k. Higher-level coarse search
 * policy selects those paths explicitly before calling into this launcher.
 */

#include "fusion_dist_topk_launcher.cuh"
#include "fusion_dist_topk.cuh"
#include "fusion_dist_topk_impl.cuh"

// Forward declaration for cuVS resource
namespace raft { class resources; }

namespace paraselect {
namespace fusion_dist_topk_warpsort {

// Forward declarations from split files
//  k  WarpSort  kernel
template<typename T, typename IdxT>
cudaError_t fusion_cos_topk_warpsort_small(
    const T* d_query_norm, const T* d_data_norm, const T* d_inner_product,
    int batch_size, int len, int k,
    T* output_vals, IdxT* output_idx,
    bool select_min,
    cudaStream_t stream);

template<typename T, typename IdxT>
cudaError_t fusion_l2_topk_warpsort_small(
    const T* d_query_norm, const T* d_data_norm, const T* d_inner_product,
    int batch_size, int len, int k,
    T* output_vals, IdxT* output_idx,
    bool select_min,
    cudaStream_t stream);

// Main entry point for cosine distance top-k
template<typename T, typename IdxT>
cudaError_t fusion_cos_topk_warpsort(
    const T* d_query_norm, const T* d_data_norm, const T* d_inner_product,
    int batch_size, int len, int k,
    T* output_vals, IdxT* output_idx,
    bool select_min,
    cudaStream_t stream,
    raft::resources* cuvs_res)
{
    (void)cuvs_res;
    return fusion_cos_topk_warpsort_small(
        d_query_norm, d_data_norm, d_inner_product,
        batch_size, len, k, output_vals, output_idx,
        select_min, stream);
}

// Main entry point for L2 distance top-k
template<typename T, typename IdxT>
cudaError_t fusion_l2_topk_warpsort(
    const T* d_query_norm, const T* d_data_norm, const T* d_inner_product,
    int batch_size, int len, int k,
    T* output_vals, IdxT* output_idx,
    bool select_min,
    cudaStream_t stream,
    raft::resources* cuvs_res)
{
    (void)cuvs_res;
    return fusion_l2_topk_warpsort_small(
        d_query_norm, d_data_norm, d_inner_product,
        batch_size, len, k, output_vals, output_idx,
        select_min, stream);
}

template cudaError_t fusion_cos_topk_warpsort<float, idx_t>(
    const float*, const float*, const float*, int, int, int, float*, idx_t*, bool, cudaStream_t, raft::resources*);


template cudaError_t fusion_l2_topk_warpsort<float, idx_t>(
    const float*, const float*, const float*, int, int, int, float*, idx_t*, bool, cudaStream_t, raft::resources*);

} // namespace fusion_dist_topk_warpsort
} // namespace paraselect
