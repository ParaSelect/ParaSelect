#pragma once

#include <algorithm>
#include <vector>

namespace paraselect {
namespace numa {

inline std::vector<int> build_cluster_owner_boundaries_by_counts(
    const int* cluster_counts,
    int n_clusters,
    int numa_nodes)
{
    const int clusters = std::max(0, n_clusters);
    const int nodes = std::max(1, numa_nodes);
    std::vector<int> boundaries((size_t)nodes + 1u, clusters);
    boundaries[0] = 0;
    if (!cluster_counts || clusters <= 0 || nodes <= 1) return boundaries;

    long long total_vectors = 0;
    for (int c = 0; c < clusters; ++c) {
        total_vectors += std::max(0, cluster_counts[c]);
    }
    if (total_vectors <= 0) return boundaries;

    const long long target = std::max(1LL, (total_vectors + nodes - 1) / nodes);
    int node = 0;
    long long accum = 0;
    for (int c = 0; c < clusters; ++c) {
        accum += std::max(0, cluster_counts[c]);
        if (node + 1 < nodes && accum >= target) {
            ++node;
            boundaries[(size_t)node] = c + 1;
            accum = 0;
        }
    }
    return boundaries;
}

inline std::vector<int> build_cluster_owner_by_counts(
    const int* cluster_counts,
    int n_clusters,
    int numa_nodes)
{
    std::vector<int> owners((size_t)std::max(0, n_clusters), 0);
    const std::vector<int> boundaries = build_cluster_owner_boundaries_by_counts(
        cluster_counts, n_clusters, numa_nodes);
    const int nodes = std::max(0, (int)boundaries.size() - 1);
    for (int node = 0; node < nodes; ++node) {
        for (int c = boundaries[(size_t)node]; c < boundaries[(size_t)node + 1u]; ++c) {
            owners[(size_t)c] = node;
        }
    }
    return owners;
}

}  // namespace numa
}  // namespace paraselect
