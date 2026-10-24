#ifndef IVF_HIERARCHICAL_CUH
#define IVF_HIERARCHICAL_CUH

#include "clustering/clustering.cuh"

/**
 * Balanced Hierarchical Clustering ()
 *
 * ( design.md ):
 * 1.  n_train = alpha * n
 * 2.  KMeans(K1 = sqrt(M))
 * 3. : n'_i > s
 * 4. : M'
 */
class BalancedHierarchicalClustering : public ClusteringInterface {
public:
    explicit BalancedHierarchicalClustering(float alpha = 0.1f);
    ~BalancedHierarchicalClustering() override;

    void fit(const float* h_data, int n, int dim, const ClusteringConfig& cfg) override;
    ClusteringResult get_result() override;
    float* release_reordered_data() override;

private:
    void release();

    float alpha_;
    bool fitted_;

    ClusteringResult result_;
    ClusterInfo cluster_info_;

    float* h_centroids_;
    float* h_reordered_data_;
    idx_t* h_reordered_indices_;
};

/**
 * Basic Hierarchical Clustering ()
 *
 * :
 * 1. :1 ()
 * 2. :, K
 * 3.  C
 * 4. :
 */
class BasicHierarchicalClustering : public ClusteringInterface {
public:
    explicit BasicHierarchicalClustering(int split_k = 2);
    ~BasicHierarchicalClustering() override;

    void fit(const float* h_data, int n, int dim, const ClusteringConfig& cfg) override;
    ClusteringResult get_result() override;
    float* release_reordered_data() override;

private:
    void release();

    int split_k_;
    bool fitted_;

    ClusteringResult result_;
    ClusterInfo cluster_info_;

    float* h_centroids_;
    float* h_reordered_data_;
    idx_t* h_reordered_indices_;
};

#endif // IVF_HIERARCHICAL_CUH
