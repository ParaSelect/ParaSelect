/**
 * Fusion Distance Top-K Launcher
 *
 * Host-side launch functions for fused distance computation and top-k selection.
 */

#ifndef FUSION_DIST_TOPK_LAUNCHER_CUH
#define FUSION_DIST_TOPK_LAUNCHER_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include "utils/pch.h"

// Forward declaration for cuVS resource
namespace raft { class resources; }

namespace paraselect {
namespace fusion_dist_topk_warpsort {

// Maximum capacity for top-k selection (WarpSort for k<=256, HeapSort for 256<k<=2048)
constexpr int kMaxCapacity = 2048;

// Cosine distance top-k
template<typename T, typename IdxT>
cudaError_t fusion_cos_topk_warpsort(
    const T* d_query_norm,
    const T* d_data_norm,
    const T* d_inner_product,
    int batch_size,
    int len,
    int k,
    T* output_vals,
    IdxT* output_idx,
    bool select_min,
    cudaStream_t stream = 0,
    raft::resources* cuvs_res = nullptr);

// L2 distance top-k
template<typename T, typename IdxT>
cudaError_t fusion_l2_topk_warpsort(
    const T* d_query_norm,
    const T* d_data_norm,
    const T* d_inner_product,
    int batch_size,
    int len,
    int k,
    T* output_vals,
    IdxT* output_idx,
    bool select_min,
    cudaStream_t stream = 0,
    raft::resources* cuvs_res = nullptr);

} // namespace fusion_dist_topk_warpsort
} // namespace paraselect

#endif // FUSION_DIST_TOPK_LAUNCHER_CUH
