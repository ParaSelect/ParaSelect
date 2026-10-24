#ifndef PARASELECT_CUDA_CLUSTERING_KMEANS_MULTIGPU_ASSIGN_CUH
#define PARASELECT_CUDA_CLUSTERING_KMEANS_MULTIGPU_ASSIGN_CUH

#include "clustering/kmeans/kmeans_final_assign.cuh"

bool assign_vectors_to_centroids_gpu_multi_device(
    const GpuCentroidAssignmentConfig& cfg,
    const KMeansCase& kmeans_cfg,
    const float* h_data,
    const float* d_centroids,
    idx_t* d_assign);

#endif  // PARASELECT_CUDA_CLUSTERING_KMEANS_MULTIGPU_ASSIGN_CUH
