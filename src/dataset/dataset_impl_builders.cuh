#ifndef PARASELECT_DATASET_IMPL_BUILDERS_CUH
#define PARASELECT_DATASET_IMPL_BUILDERS_CUH

#include "dataset/ownership_helper.h"
#include "cpu/l2norm_precompute.h"
#include "cpu/utils/cpu_affinity.h"
#include "cpu/utils/numa_cluster_owner.h"
#include "cpu/utils/numa_page_stats.h"

#include <malloc.h>

inline void ClusterDataset::_compute_vector_l2_norm(
    const float* row_source,
    bool numa_first_touch_row_major) {
    using Owner = paraselect::ownership::BufferOwner;
    using Handle = paraselect::ownership::BufferHandle;

    if (vector_l2_norm) {
        paraselect::ownership::free_raw(vector_l2_norm, Owner::StdMalloc);
        vector_l2_norm = nullptr;
    }
    if (!row_source || n_total_vectors <= 0 || vector_dim <= 0) return;

    const size_t bytes = (size_t)n_total_vectors * sizeof(float);
    Handle l2_buf(memalign(64, bytes), bytes, Owner::StdMalloc);
    if (!l2_buf.data) return;
    float* out = static_cast<float*>(l2_buf.data);
    if (numa_first_touch_row_major && cluster_info.counts && cluster_info.offsets && cluster_info.k > 0) {
        const int total_threads = std::max(1, std::min(paraselect::default_worker_thread_count(), cluster_info.k));
        const int hw_nodes = std::max(1, paraselect::get_numa_node_count());
        const int numa_nodes = std::max(1, std::min(hw_nodes, total_threads));
        const std::vector<int> owners = paraselect::numa::build_cluster_owner_by_counts(
            cluster_info.counts, cluster_info.k, numa_nodes);
        std::vector<std::vector<int>> clusters_by_node((size_t)numa_nodes);
        for (int c = 0; c < cluster_info.k; ++c) {
            const int owner = std::max(0, std::min(owners[(size_t)c], numa_nodes - 1));
            clusters_by_node[(size_t)owner].push_back(c);
        }

        std::vector<std::thread> node_workers;
        node_workers.reserve((size_t)numa_nodes);
        int remaining_threads = total_threads;
        for (int node = 0; node < numa_nodes; ++node) {
            const int nodes_left = numa_nodes - node;
            const int node_threads = std::max(1, remaining_threads / std::max(1, nodes_left));
            remaining_threads -= node_threads;
            node_workers.emplace_back([&, node, node_threads]() {
                const auto& clusters = clusters_by_node[(size_t)node];
                if (clusters.empty()) return;
                std::atomic<int> next{0};
                std::vector<std::thread> workers;
                workers.reserve((size_t)node_threads);
                for (int t = 0; t < node_threads; ++t) {
                    workers.emplace_back([&, t]() {
                        paraselect::pin_thread_to_numa_core(node, t);
                        while (true) {
                            const int pos = next.fetch_add(1, std::memory_order_relaxed);
                            if (pos >= static_cast<int>(clusters.size())) break;
                            const int c = clusters[(size_t)pos];
                            const idx_t begin = cluster_info.offsets[c];
                            const idx_t end = begin + static_cast<idx_t>(std::max(0, cluster_info.counts[c]));
                            for (idx_t row = begin; row < end && row < n_total_vectors; ++row) {
                                const float* vec = row_source + (size_t)row * (size_t)vector_dim;
                                #ifdef PARASELECT_CPU_HAS_AVX2
                                out[row] = paraselect::cpu::norm2_avx2(vec, vector_dim);
                                #else
                                out[row] = paraselect::cpu::norm2_scalar(vec, vector_dim);
                                #endif
                            }
                        }
                    });
                }
                for (auto& worker : workers) worker.join();
            });
        }
        for (auto& worker : node_workers) worker.join();
    } else {
        paraselect::cpu::precompute_l2_norms(
            row_source, n_total_vectors, vector_dim, out, true);
    }
    vector_l2_norm = static_cast<float*>(l2_buf.detach());
    paraselect::numa::print_page_distribution(
        "vector_l2_norm", vector_l2_norm, (size_t)n_total_vectors * sizeof(float));
}

#endif
