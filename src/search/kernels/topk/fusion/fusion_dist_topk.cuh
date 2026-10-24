#ifndef FUSION_DIST_TOPK_CUH
#define FUSION_DIST_TOPK_CUH

#include "utils/pch.h"
#include "fusion_dist_topk_launcher.cuh"

/**
 * :+  topk
 **/
__global__ void fusion_cos_topk_warpsort_kernel(
    float* d_query_norm,
    float* d_data_norm,
    float* d_inner_product,
    int* d_index,
    int* topk_index,
    float* topk_dist,
    int n_query,
    int n_batch,
    int k
);

__global__ void fusion_l2_topk_warpsort_kernel(
    float* d_query_norm,
    float* d_data_norm,
    float* d_inner_product,
    int* d_index,
    int* topk_index,
    float* topk_dist,
    int n_query,
    int n_batch,
    int k
);

/**
 * :paraselect::fusion_dist_topk_warpsort
 * ,GPU
 */
// Forward declaration
namespace raft { class resources; }

namespace paraselect {
namespace fusion_dist_topk_cuvs {

template<typename T, typename IdxT>
cudaError_t fusion_cos_topk_cuvs(
    const T* d_query_norm,
    const T* d_data_norm,
    const T* d_inner_product,
    const IdxT* d_index,
    int batch_size,
    int len,
    int k,
    T* output_vals,
    IdxT* output_idx,
    bool select_min,
    cudaStream_t stream = 0,
    raft::resources* res = nullptr
);

template<typename T, typename IdxT>
cudaError_t fusion_l2_topk_cuvs(
    const T* d_query_norm,
    const T* d_data_norm,
    const T* d_inner_product,
    const IdxT* d_index,
    int batch_size,
    int len,
    int k,
    T* output_vals,
    IdxT* output_idx,
    bool select_min,
    cudaStream_t stream = 0,
    raft::resources* res = nullptr
);

template<typename T, typename IdxT>
cudaError_t select_k_cuvs(
    const T* input,
    int batch_size,
    int len,
    int k,
    T* output_vals,
    IdxT* output_idx,
    bool select_min,
    cudaStream_t stream = 0,
    raft::resources* res = nullptr
);

} // namespace fusion_dist_topk_cuvs
} // namespace paraselect

// Note: fusion_cos_topk_warpsort and fusion_l2_topk_warpsort are declared in fusion_dist_topk_launcher.cuh

/**
 * (kernel)
 **/
inline size_t get_shared_memory_size(int k) {
    return (k * sizeof(float) + k * sizeof(int));  /*  +  */
}
#endif
