#ifndef PARASELECT_CUDA_CLUSTERING_KMEANS_PROGRESSIVE_PREFIX_CUH
#define PARASELECT_CUDA_CLUSTERING_KMEANS_PROGRESSIVE_PREFIX_CUH

#include "clustering/clustering.cuh"
#include "clustering/kmeans/kmeans.cuh"

void initialize_progressive_prefix_centroids(
    const float* h_data,
    int n,
    int dim,
    int k,
    float* h_centroids);

bool run_progressive_prefix_kmeans_inplace(
    const ClusteringConfig& cfg,
    const KMeansCase& base_kmeans_cfg,
    float* h_data,
    int n,
    int dim,
    int k,
    float* d_centroids,
    idx_t* reordered_indices,
    ClusterInfo* cluster_info);

#endif  // PARASELECT_CUDA_CLUSTERING_KMEANS_PROGRESSIVE_PREFIX_CUH
