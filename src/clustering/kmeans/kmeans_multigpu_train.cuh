#ifndef PARASELECT_CUDA_CLUSTERING_KMEANS_MULTIGPU_TRAIN_CUH
#define PARASELECT_CUDA_CLUSTERING_KMEANS_MULTIGPU_TRAIN_CUH

#include "clustering/kmeans/kmeans_final_assign.cuh"

// Shared by production Lloyd training and its benchmark. Accumulation uses
// caller-owned buffers and device assignments; sums/counts must be zeroed.
cudaError_t kmeans_accumulate_host_vectors(
    const float* data, int n, int dim, int batch, const idx_t* assignments,
    float* data_buffer, float* sums, int* counts);
void kmeans_update_centroids_host(
    const std::vector<float>& sums, const std::vector<int>& counts,
    int k, int dim, std::vector<float>& centers);

bool kmeans_train_centroids_multigpu(
    const GpuCentroidAssignmentConfig& cfg,
    const KMeansCase& kmeans_cfg,
    const float* h_data,
    float* d_centroids,
    float* h_objective);

#endif  // PARASELECT_CUDA_CLUSTERING_KMEANS_MULTIGPU_TRAIN_CUH
