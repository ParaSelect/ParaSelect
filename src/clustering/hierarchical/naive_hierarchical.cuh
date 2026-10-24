#ifndef IVF_NAIVE_HIERARCHICAL_CUH
#define IVF_NAIVE_HIERARCHICAL_CUH

#include "clustering/clustering.cuh"

/**
 * Naive Hierarchical Clustering()
 *
 *  Balanced  KMeans(K1 = sqrt(M));
 *  k_i  M( n'_i ),().
 */
class NaiveHierarchicalClustering : public ClusteringInterface {
public:
    explicit NaiveHierarchicalClustering(float alpha = 0.1f);
    ~NaiveHierarchicalClustering() override;

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

#endif  // IVF_NAIVE_HIERARCHICAL_CUH
