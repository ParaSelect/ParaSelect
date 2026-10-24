#ifndef PARASELECT_CUDA_CLUSTERING_KMEANS_CLUSTER_ONLY_CUH
#define PARASELECT_CUDA_CLUSTERING_KMEANS_CLUSTER_ONLY_CUH

#include "clustering/kmeans/kmeans.cuh"
#include <vector>

struct KMeansClusterOnlyConfig {
    int n = 0;
    int dim = 0;
    int k = 0;
    int iters = 10;
    int seed = 1234;
    int train_samples = 50000000;
    int device_id = 0;
    DistanceType dist = L2_DISTANCE;
    KMeansComputePrecision precision = KMeansComputePrecision::Tf32;
    KMeansFinalAssignStrategy final_assign_strategy = KMeansFinalAssignStrategy::Exact;
};

struct KMeansClusterOnlyResult {
    std::vector<float> centroids;
    std::vector<int> counts;
    std::vector<idx_t> offsets;
    float objective = 0.0f;
};

bool kmeans_fast_ivf_cluster_only(
    const KMeansClusterOnlyConfig& cfg,
    const float* h_data,
    KMeansClusterOnlyResult* out);

#endif  // PARASELECT_CUDA_CLUSTERING_KMEANS_CLUSTER_ONLY_CUH
