#ifndef PARASELECT_CUDA_CLUSTERING_KMEANS_FLASH_KMEANS_ASSIGN_CUH
#define PARASELECT_CUDA_CLUSTERING_KMEANS_FLASH_KMEANS_ASSIGN_CUH

#include "clustering/flash_kmeans/flash_kmeans_assign_config.cuh"
#include "utils/pch.h"

namespace paraselect {
namespace clustering {
namespace kmeans {

struct FlashKMeansAssignResult {
    float ms = 0.0f;
    float norm_ms = 0.0f;
    float assign_ms = 0.0f;
    float init_ms = 0.0f;
    float cutlass_gemm_ms = 0.0f;
    float reduce_ms = 0.0f;
    float partial_reduce_ms = 0.0f;
    float final_reduce_ms = 0.0f;
    int variant = 0;
    int point_tile = 0;
    int centroid_tile = 0;
};

struct FlashKMeansAssignWorkspace {
    float* d_x_sq = nullptr;
    float* d_c_sq = nullptr;
    unsigned long long* d_partial_packed = nullptr;
    int n_capacity = 0;
    int k_capacity = 0;
    int partial_tiles_capacity = 0;
};

void reserve_flash_kmeans_assign_workspace(
    FlashKMeansAssignWorkspace& workspace,
    int n,
    int k,
    const FlashKMeansAssignConfig& config);

void reserve_flash_kmeans_assign_scratch_workspace(
    FlashKMeansAssignWorkspace& workspace,
    int n,
    int k,
    const FlashKMeansAssignConfig& config);

void release_flash_kmeans_assign_workspace(FlashKMeansAssignWorkspace& workspace);

/**
 * Experimental materialization-free L2 assignment.
 *
 * Computes, for each row in d_data [n, dim], the nearest centroid in
 * d_centroids [k, dim]. This prototype is assignment-only and does not update
 * centroids or write a distance matrix.
 */
FlashKMeansAssignResult flash_kmeans_assign_l2(
    const float* d_data,
    const float* d_centroids,
    int n,
    int k,
    int dim,
    idx_t* d_best_idx,
    float* d_best_dist,
    const FlashKMeansAssignConfig& config,
    FlashKMeansAssignWorkspace& workspace,
    cudaStream_t stream = 0);

/**
 * Assignment path for callers that already own norm buffers.
 *
 * d_xnorm2 and d_cnorm2 are external inputs. The workspace is only used for
 * Flash-private scratch, such as packed tile partial minima.
 */
FlashKMeansAssignResult flash_kmeans_assign_l2_precomputed_norms(
    const float* d_data,
    const float* d_centroids,
    const float* d_xnorm2,
    const float* d_cnorm2,
    int n,
    int k,
    int dim,
    idx_t* d_best_idx,
    float* d_best_dist,
    const FlashKMeansAssignConfig& config,
    FlashKMeansAssignWorkspace& workspace,
    cudaStream_t stream = 0);

}  // namespace kmeans
}  // namespace clustering
}  // namespace paraselect

#endif  // PARASELECT_CUDA_CLUSTERING_KMEANS_FLASH_KMEANS_ASSIGN_CUH
