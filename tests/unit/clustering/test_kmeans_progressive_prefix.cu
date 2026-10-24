#include "src/clustering/clustering.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

void check_or_die(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[test_kmeans_progressive_prefix] FAIL: " << msg << std::endl;
        std::exit(1);
    }
}

std::vector<float> make_two_cluster_data(int n_per_cluster, int dim) {
    std::vector<float> data((size_t)n_per_cluster * 2 * dim, 0.0f);
    for (int i = 0; i < n_per_cluster; ++i) {
        data[(size_t)i * dim + 0] = 0.01f * static_cast<float>(i % 7);
        data[(size_t)i * dim + 1] = 0.01f * static_cast<float>(i % 5);
    }
    for (int i = 0; i < n_per_cluster; ++i) {
        const int row = n_per_cluster + i;
        data[(size_t)row * dim + 0] = 100.0f + 0.01f * static_cast<float>(i % 7);
        data[(size_t)row * dim + 1] = 100.0f + 0.01f * static_cast<float>(i % 5);
    }
    return data;
}

void verify_permutation(const idx_t* reordered, int n) {
    std::vector<int> seen(n, 0);
    for (int i = 0; i < n; ++i) {
        check_or_die(reordered[i] >= 0 && reordered[i] < n, "reordered index out of range");
        seen[(int)reordered[i]]++;
    }
    for (int i = 0; i < n; ++i) {
        check_or_die(seen[i] == 1, "reordered indices must be a permutation");
    }
}

}  // namespace

int main() {
    constexpr int n_per_cluster = 32;
    constexpr int n = n_per_cluster * 2;
    constexpr int dim = 2;
    constexpr int k = 2;

    std::vector<float> data = make_two_cluster_data(n_per_cluster, dim);

    ClusteringConfig cfg{};
    cfg.n_clusters = k;
    cfg.max_iters = 10;
    cfg.distance_mode = L2_DISTANCE;
    cfg.seed = 1234;
    cfg.use_minibatch = false;
    cfg.batch_size = 1 << 10;
    cfg.device_id = 0;
    cfg.train_strategy = KMeansTrainStrategy::ProgressivePrefix;
    cfg.compute_precision = KMeansComputePrecision::Tf32;
    cfg.progressive_stage1_samples = 16;
    cfg.progressive_stage2_samples = 24;

    KMeansClusteringInPlace clusterer;
    clusterer.fit(data.data(), n, dim, cfg);
    ClusteringResult result = clusterer.get_result();

    check_or_die(result.cluster_info != nullptr, "cluster_info should be populated");
    check_or_die(result.cluster_info->k == k, "cluster count should match");
    check_or_die(result.centroids != nullptr, "centroids should be populated");
    check_or_die(result.reordered_indices != nullptr, "reordered indices should be populated");
    check_or_die(result.reordered_data == data.data(), "in-place clusterer should keep input buffer");

    int total_count = 0;
    for (int c = 0; c < k; ++c) {
        check_or_die(result.cluster_info->counts[c] >= 0, "cluster count should be non-negative");
        total_count += result.cluster_info->counts[c];
        check_or_die(std::isfinite(result.centroids[(size_t)c * dim + 0]), "centroid x should be finite");
        check_or_die(std::isfinite(result.centroids[(size_t)c * dim + 1]), "centroid y should be finite");
    }
    check_or_die(total_count == n, "cluster counts should sum to n");
    verify_permutation(result.reordered_indices, n);

    std::cout << "[test_kmeans_progressive_prefix] PASS n=" << n
              << " k=" << k
              << " counts=" << result.cluster_info->counts[0]
              << "," << result.cluster_info->counts[1]
              << std::endl;
    return 0;
}
