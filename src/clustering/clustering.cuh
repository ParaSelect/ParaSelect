#ifndef IVF_CLUSTERING_CUH
#define IVF_CLUSTERING_CUH

#include "utils/pch.h"
#include "clustering/kmeans/kmeans_policy.cuh"
#include <cstddef>

/**
 * ()
 *  KMeans .
 */

// ============================================================
// : IVF  ClusterInfo
// ============================================================
struct ClusterInfo {
    idx_t* offsets;   // [k + 1]  cluster ()
    int* counts;          // [k]  cluster
    int k;                // cluster
};

// ============================================================
//
// ============================================================
struct ClusteringConfig {
    int n_clusters;
    int max_iters;
    DistanceType distance_mode;
    unsigned int seed;
    bool use_minibatch;
    int batch_size;
    int device_id;
    KMeansAssignOperator assign_operator = KMeansAssignOperator::Kmeans;
    KMeansTrainStrategy train_strategy = KMeansTrainStrategy::FullLloyd;
    KMeansComputePrecision compute_precision = KMeansComputePrecision::Tf32;
    /** :(sum dist^2); obj */
    float* h_objective = nullptr;
    /**
     * :fit .
     *  KMeans,actual == n_clusters;,actual >= n_clusters.
     *  nullptr .
     */
    int* actual_n_clusters = nullptr;
    /**
     * : K-means .
     *  - 1.0:()
     *  - 0.1: 10%
     * ,.
     */
    float train_sample_ratio = 1.0f;
    /**
     * :.
     *  - 0:()
     *  - >0:
     *  train_sample_ratio ,.
     */
    int train_max_samples = 0;
    int progressive_stage1_samples = 0;
    int progressive_stage2_samples = 0;
    int fast_ivf_train_samples = 0;
    KMeansFinalAssignStrategy final_assign_strategy = KMeansFinalAssignStrategy::Exact;
    KMeansReorderPolicy reorder_policy = KMeansReorderPolicy::Standard;
    bool cluster_only = false;
};

// ============================================================
// (fit  get_result() )
// ,.
// ============================================================
struct ClusteringResult {
    float* centroids;           // [k, dim]
    int* assignments;            // [n]  cluster
    int* cluster_sizes;          // [k], ClusterInfo.counts
    float* reordered_data;       // [n, dim]  cluster (IVF )
    idx_t* reordered_indices;      // [n] (IVF )
    ClusterInfo* cluster_info;   // offsets / counts / k
    int n_vectors;
    int dim;
    bool reordered_data_owned;   // true: reordered_data ,;false:

    ClusteringResult()
        : centroids(nullptr),
          assignments(nullptr),
          cluster_sizes(nullptr),
          reordered_data(nullptr),
          reordered_indices(nullptr),
          cluster_info(nullptr),
          n_vectors(0),
          dim(0),
          reordered_data_owned(false) {}
};

// ============================================================
//
// ============================================================
class ClusteringInterface {
public:
    virtual void fit(const float* h_data, int n, int dim, const ClusteringConfig& cfg) = 0;
    virtual ClusteringResult get_result() = 0;
    /**
     *  reordered_data .
     *  reordered_data , nullptr.
     * (reordered_data_owned=true).
     */
    virtual float* release_reordered_data() = 0;
    virtual ~ClusteringInterface() = default;
};

// In-place (, n*dim )
class InPlaceClusteringInterface {
public:
    virtual void fit(float* h_data, int n, int dim, const ClusteringConfig& cfg) = 0;
    virtual ClusteringResult get_result() = 0;
    /**
     *  reordered_data .
     *  In-place , fit .
     * (malloc/free  cudaMallocHost/cudaFreeHost).
     */
    virtual float* release_reordered_data() = 0;
    virtual ~InPlaceClusteringInterface() = default;
};

/**
 * KMeans : ivf_kmeans, ClusteringResult.
 */
class KMeansClustering : public ClusteringInterface {
public:
    KMeansClustering() = default;
    void fit(const float* h_data, int n, int dim, const ClusteringConfig& cfg) override;
    ClusteringResult get_result() override;
    float* release_reordered_data() override;
    ~KMeansClustering() override;

private:
    ClusteringResult result_{};
    ClusterInfo cluster_info_{};
    float* h_centroids_ = nullptr;
    bool fitted_ = false;
};

/**
 * KMeans In-place :, n*dim .
 * : release_reordered_data() .
 */
class KMeansClusteringInPlace : public InPlaceClusteringInterface {
public:
    KMeansClusteringInPlace() = default;
    void fit(float* h_data, int n, int dim, const ClusteringConfig& cfg) override;
    ClusteringResult get_result() override;
    float* release_reordered_data() override;
    ~KMeansClusteringInPlace() override;

private:
    ClusteringResult result_{};
    ClusterInfo cluster_info_{};
    float* h_centroids_ = nullptr;
    float* h_data_ = nullptr;  // ,
    bool fitted_ = false;
};

#endif // IVF_CLUSTERING_CUH
