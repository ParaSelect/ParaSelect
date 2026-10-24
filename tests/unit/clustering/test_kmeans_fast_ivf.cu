#include "src/clustering/clustering.cuh"
#include "src/clustering/kmeans/kmeans_cagra_train.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void check_or_die(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[test_kmeans_fast_ivf] FAIL: " << msg << std::endl;
        std::exit(1);
    }
}

std::vector<float> make_four_blob_data(int n_per_cluster, int dim) {
    std::vector<float> data((size_t)n_per_cluster * 4 * dim, 0.0f);
    const float centers[4][2] = {{0.0f, 0.0f}, {100.0f, 0.0f}, {0.0f, 100.0f}, {100.0f, 100.0f}};
    for (int c = 0; c < 4; ++c) {
        for (int i = 0; i < n_per_cluster; ++i) {
            int row = c * n_per_cluster + i;
            data[(size_t)row * dim + 0] = centers[c][0] + 0.01f * static_cast<float>(i % 7);
            data[(size_t)row * dim + 1] = centers[c][1] + 0.01f * static_cast<float>(i % 5);
        }
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

void run_cagra_lloyd_u8_case() {
    constexpr int n = 512;
    constexpr int dim = 8;
    constexpr int k = 4;
    std::vector<uint8_t> data(static_cast<size_t>(n) * dim);
    for (int row = 0; row < n; ++row) {
        const uint8_t center = static_cast<uint8_t>((row / (n / k)) * 60 + 10);
        for (int d = 0; d < dim; ++d) {
            data[static_cast<size_t>(row) * dim + d] =
                static_cast<uint8_t>(center + ((row + d) % 3));
        }
    }

    KMeansCagraTrainConfig cfg;
    cfg.n = n;
    cfg.dim = dim;
    cfg.k = k;
    cfg.iters = 2;
    cfg.params.train_batch = 128;
    cfg.params.topk_centroids = 2;
    cfg.params.itopk_size = 16;
    cfg.params.graph_degree = 2;
    cfg.params.intermediate_graph_degree = 4;
    cfg.params.build_algo = KMeansCagraBuildAlgo::NnDescent;
    std::vector<float> centroids;
    check_or_die(kmeans_train_centroids_cagra_u8(cfg, data.data(), &centroids),
                 "CAGRA Lloyd U8 training should succeed");
    check_or_die(centroids.size() == static_cast<size_t>(k) * dim,
                 "CAGRA Lloyd centroid shape should match");
    for (float value : centroids) {
        check_or_die(std::isfinite(value), "CAGRA Lloyd centroids should be finite");
    }
}

}  // namespace

void run_case(KMeansFinalAssignStrategy final_assign_strategy) {
    constexpr int n_per_cluster = 32;
    constexpr int n = n_per_cluster * 4;
    constexpr int dim = 2;
    constexpr int k = 4;

    std::vector<float> data = make_four_blob_data(n_per_cluster, dim);

    ClusteringConfig cfg{};
    cfg.n_clusters = k;
    cfg.max_iters = 5;
    cfg.distance_mode = L2_DISTANCE;
    cfg.seed = 1234;
    cfg.use_minibatch = false;
    cfg.batch_size = 1 << 10;
    cfg.device_id = 0;
    cfg.train_strategy = KMeansTrainStrategy::FastIVF;
    cfg.compute_precision = KMeansComputePrecision::Tf32;
    cfg.fast_ivf_train_samples = 96;
    cfg.final_assign_strategy = final_assign_strategy;

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

    std::cout << "[test_kmeans_fast_ivf] PASS n=" << n
              << " k=" << k
              << " train=" << cfg.fast_ivf_train_samples
              << " final=" << kmeans_final_assign_strategy_tag(final_assign_strategy)
              << std::endl;
}

int main() {
    run_case(KMeansFinalAssignStrategy::Exact);
    run_case(KMeansFinalAssignStrategy::Cagra);
    run_cagra_lloyd_u8_case();
    return 0;
}
