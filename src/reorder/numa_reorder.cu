// reorder/numa_reorder.cu
#include "reorder/reorder.cuh"
#include "cpu/utils/cpu_affinity.h"
#include "cpu/utils/numa_cluster_owner.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

struct ClusterRangeTask {
    int cluster_id = -1;
    idx_t begin = 0;
    idx_t end = 0;
};

int clamp_positive(int value, int fallback) {
    return value > 0 ? value : fallback;
}

std::vector<ClusterRangeTask> build_cluster_range_tasks(
    const idx_t* offsets,
    const int* counts,
    int n_clusters,
    int n) {
    std::vector<ClusterRangeTask> tasks;
    tasks.reserve(static_cast<size_t>(std::max(0, n_clusters)));
    for (int c = 0; c < n_clusters; ++c) {
        const idx_t count = static_cast<idx_t>(std::max(0, counts[c]));
        if (count <= 0) continue;
        const idx_t begin = offsets[c];
        const idx_t end = begin + count;
        if (begin < 0 || end <= begin || begin >= n) continue;
        tasks.push_back({c, begin, std::min<idx_t>(end, n)});
    }
    return tasks;
}

std::vector<std::vector<ClusterRangeTask>> assign_tasks_to_nodes(
    const std::vector<ClusterRangeTask>& tasks,
    const int* counts,
    int n_clusters,
    int numa_nodes) {
    std::vector<std::vector<ClusterRangeTask>> per_node(static_cast<size_t>(numa_nodes));
    if (tasks.empty()) return per_node;

    const std::vector<int> owners = paraselect::numa::build_cluster_owner_by_counts(counts, n_clusters, numa_nodes);
    for (const auto& task : tasks) {
        const int owner = (task.cluster_id >= 0 && task.cluster_id < n_clusters)
            ? owners[static_cast<size_t>(task.cluster_id)]
            : 0;
        const int node = std::max(0, std::min(owner, numa_nodes - 1));
        per_node[static_cast<size_t>(node)].push_back(task);
    }
    return per_node;
}

void copy_range_by_permutation(
    const float* h_data_in,
    const idx_t* h_perm,
    float* h_data_out,
    int n,
    int dim,
    idx_t begin,
    idx_t end) {
    for (idx_t p = begin; p < end; ++p) {
        const idx_t src = h_perm[p];
        if (src < 0 || src >= n) continue;
        std::memcpy(
            h_data_out + static_cast<size_t>(p) * static_cast<size_t>(dim),
            h_data_in + static_cast<size_t>(src) * static_cast<size_t>(dim),
            sizeof(float) * static_cast<size_t>(dim));
    }
}

void run_node_group(
    int node_id,
    int node_threads,
    const std::vector<ClusterRangeTask>& tasks,
    const float* h_data_in,
    const idx_t* h_perm,
    float* h_data_out,
    int n,
    int dim,
    bool pin_threads) {
    if (tasks.empty() || node_threads <= 0) return;
    const int actual_threads = std::min<int>(node_threads, static_cast<int>(tasks.size()));
    std::atomic<int> next{0};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(actual_threads));

    for (int t = 0; t < actual_threads; ++t) {
        workers.emplace_back([&, t]() {
            if (pin_threads) {
                paraselect::pin_thread_to_numa_core(node_id, t);
            }
            while (true) {
                const int task_id = next.fetch_add(1, std::memory_order_relaxed);
                if (task_id >= static_cast<int>(tasks.size())) break;
                const auto& task = tasks[static_cast<size_t>(task_id)];
                copy_range_by_permutation(
                    h_data_in,
                    h_perm,
                    h_data_out,
                    n,
                    dim,
                    task.begin,
                    task.end);
            }
        });
    }
    for (auto& worker : workers) worker.join();
}

}  // namespace

void cpu_reorder_vectors_by_cluster_ranges_numa(
    const float* h_data_in,
    const idx_t* h_perm,
    float* h_data_out,
    const idx_t* h_cluster_offsets,
    const int* h_cluster_counts,
    int n,
    int dim,
    int n_clusters,
    const NumaReorderConfig& config) {
    if (!h_data_in || !h_perm || !h_data_out || !h_cluster_offsets || !h_cluster_counts ||
        n <= 0 || dim <= 0 || n_clusters <= 0) {
        std::fprintf(stderr, "[numa_reorder] invalid arguments; falling back to standard reorder\n");
        cpu_reorder_vectors_by_permutation(h_data_in, h_perm, h_data_out, n, dim);
        return;
    }

    const int hw_nodes = std::max(1, paraselect::get_numa_node_count());
    const int numa_nodes = std::max(1, std::min(clamp_positive(config.numa_nodes, hw_nodes), hw_nodes));
    if (numa_nodes <= 1) {
        cpu_reorder_vectors_by_permutation(h_data_in, h_perm, h_data_out, n, dim);
        return;
    }

    const int total_threads = std::max(
        1,
        std::min(clamp_positive(config.threads, paraselect::default_worker_thread_count()), n_clusters));
    const auto tasks = build_cluster_range_tasks(h_cluster_offsets, h_cluster_counts, n_clusters, n);
    if (tasks.empty()) {
        std::fprintf(stderr, "[numa_reorder] no valid cluster range tasks; falling back to standard reorder\n");
        cpu_reorder_vectors_by_permutation(h_data_in, h_perm, h_data_out, n, dim);
        return;
    }

    const auto per_node = assign_tasks_to_nodes(tasks, h_cluster_counts, n_clusters, numa_nodes);
    std::vector<std::thread> node_workers;
    node_workers.reserve(static_cast<size_t>(numa_nodes));

    int remaining_threads = total_threads;
    for (int node = 0; node < numa_nodes; ++node) {
        const int nodes_left = numa_nodes - node;
        const int node_threads = std::max(1, remaining_threads / std::max(1, nodes_left));
        remaining_threads -= node_threads;
        node_workers.emplace_back([&, node, node_threads]() {
            run_node_group(
                node,
                node_threads,
                per_node[static_cast<size_t>(node)],
                h_data_in,
                h_perm,
                h_data_out,
                n,
                dim,
                config.pin_threads);
        });
    }
    for (auto& worker : node_workers) worker.join();
}
