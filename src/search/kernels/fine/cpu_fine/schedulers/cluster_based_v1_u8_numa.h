#ifndef PARASELECT_CLUSTER_BASED_V1_U8_NUMA_H
#define PARASELECT_CLUSTER_BASED_V1_U8_NUMA_H

#include "cpu/utils/adaptive_thread_count.h"
#include "cpu/utils/cpu_affinity.h"
#include "search/kernels/fine/cpu_fine/cpu_fine_common.h"
#include "search/kernels/fine/cpu_fine/schedulers/cluster_based_compact.h"
#include "search/kernels/fine/cpu_fine/schedulers/cluster_based_sharding.h"
#include "search/kernels/fine/cpu_fine/schedulers/thread_stats.h"
#include "search/kernels/fine/cpu_fine/executor/persistent_thread_pool.h"
#include "search/kernels/fine/cpu_fine/topk_local.h"
#include "utils/pch.h"


#include <algorithm>
#include <atomic>
#include <cstddef>
#include <vector>
#include <chrono>

namespace paraselect {
namespace cpu_fine {
namespace cluster_based_v1_u8_numa_detail {

static constexpr int kPrefetchAhead = 8;

struct NumaSchedulerWorkspace {
    std::vector<int> owned_cluster_query_offset;
    std::vector<int> owned_cluster_query_list;
    std::vector<cluster_based_sharding_detail::ClusterTask> tasks;
    std::vector<std::vector<int>> node_task_ids;
    std::vector<int> node_task_counts;
    std::vector<int> node_task_next;
    std::vector<thread_stats_detail::ThreadStat> thread_stats;
    std::vector<long long> thread_fma;
};

inline NumaSchedulerWorkspace& numa_scheduler_workspace() {
    thread_local NumaSchedulerWorkspace workspace;
    return workspace;
}

inline int clamp_numa_nodes(int requested_nodes, int num_threads) {
    const int hw_nodes = std::max(1, paraselect::get_numa_node_count());
    int nodes = requested_nodes > 0 ? requested_nodes : hw_nodes;
    nodes = std::max(1, std::min(nodes, hw_nodes));
    nodes = std::max(1, std::min(nodes, num_threads));
    return nodes;
}

inline int numa_owner_from_boundaries(
    const int* h_numa_cluster_boundaries,
    int numa_cluster_boundary_count,
    int cid)
{
    const int numa_nodes = numa_cluster_boundary_count - 1;
    const int owner = static_cast<int>(
        std::upper_bound(
            h_numa_cluster_boundaries,
            h_numa_cluster_boundaries + numa_cluster_boundary_count,
            cid) - h_numa_cluster_boundaries) - 1;
    return std::max(0, std::min(owner, numa_nodes - 1));
}

inline void build_numa_node_task_ids(
    std::vector<std::vector<int>>* node_task_ids,
    std::vector<int>* node_task_counts,
    const std::vector<cluster_based_sharding_detail::ClusterTask>& tasks,
    const int* h_numa_cluster_boundaries,
    int numa_cluster_boundary_count,
    bool tasks_sorted_by_cluster)
{
    const int numa_nodes = numa_cluster_boundary_count - 1;
    node_task_counts->assign((size_t)numa_nodes, 0);
    if (tasks_sorted_by_cluster) {
        int node = 0;
        for (const auto& task : tasks) {
            const int cid = task.cid;
            while (node + 1 < numa_nodes && cid >= h_numa_cluster_boundaries[node + 1]) {
                ++node;
            }
            (*node_task_counts)[(size_t)node]++;
        }
        for (int n = 0; n < numa_nodes; ++n) {
            (*node_task_ids)[(size_t)n].reserve((size_t)(*node_task_counts)[(size_t)n]);
        }
        node = 0;
        for (int task_id = 0; task_id < static_cast<int>(tasks.size()); ++task_id) {
            const int cid = tasks[(size_t)task_id].cid;
            while (node + 1 < numa_nodes && cid >= h_numa_cluster_boundaries[node + 1]) {
                ++node;
            }
            (*node_task_ids)[(size_t)node].push_back(task_id);
        }
        return;
    }

    for (const auto& task : tasks) {
        const int owner = numa_owner_from_boundaries(
            h_numa_cluster_boundaries, numa_cluster_boundary_count, task.cid);
        (*node_task_counts)[(size_t)owner]++;
    }
    for (int n = 0; n < numa_nodes; ++n) {
        (*node_task_ids)[(size_t)n].reserve((size_t)(*node_task_counts)[(size_t)n]);
    }
    for (int task_id = 0; task_id < static_cast<int>(tasks.size()); ++task_id) {
        const int cid = tasks[(size_t)task_id].cid;
        const int owner = numa_owner_from_boundaries(
            h_numa_cluster_boundaries, numa_cluster_boundary_count, cid);
        (*node_task_ids)[(size_t)owner].push_back(task_id);
    }
}

inline void pin_thread_to_numa_core_cached(int node_id, int core_in_node) {
    thread_local int pinned_node = -1;
    thread_local int pinned_core = -1;
    if (pinned_node == node_id && pinned_core == core_in_node) return;
    paraselect::pin_thread_to_numa_core(node_id, core_in_node);
    pinned_node = node_id;
    pinned_core = core_in_node;
}

inline void build_cluster_query_metadata(
    std::vector<int>* owned_cluster_query_offset,
    std::vector<int>* owned_cluster_query_list,
    const idx_t* h_coarse_cluster_ids,
    int n_query,
    int n_probes,
    int n_total_clusters)
{
    std::vector<int> cluster_query_count(n_total_clusters, 0);
    for (int qi = 0; qi < n_query; ++qi) {
        for (int pi = 0; pi < n_probes; ++pi) {
            const int cid = static_cast<int>(h_coarse_cluster_ids[(size_t)qi * (size_t)n_probes + (size_t)pi]);
            if (cid >= 0 && cid < n_total_clusters) cluster_query_count[cid]++;
        }
    }
    owned_cluster_query_offset->assign((size_t)n_total_clusters + 1u, 0);
    for (int c = 0; c < n_total_clusters; ++c) {
        (*owned_cluster_query_offset)[(size_t)c + 1u] =
            (*owned_cluster_query_offset)[(size_t)c] + cluster_query_count[c];
    }

    owned_cluster_query_list->assign((size_t)(*owned_cluster_query_offset)[(size_t)n_total_clusters], 0);
    std::vector<int> wpos(n_total_clusters, 0);
    for (int qi = 0; qi < n_query; ++qi) {
        for (int pi = 0; pi < n_probes; ++pi) {
            const int cid = static_cast<int>(h_coarse_cluster_ids[(size_t)qi * (size_t)n_probes + (size_t)pi]);
            if (cid >= 0 && cid < n_total_clusters) {
                const int pos = (*owned_cluster_query_offset)[(size_t)cid] + wpos[cid]++;
                (*owned_cluster_query_list)[(size_t)pos] = qi;
            }
        }
    }
}

template <typename Evaluator>
inline long long cluster_based_v1_u8_numa_impl(
    const uint8_t* h_base,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const uint8_t* h_query,
    const idx_t* h_coarse_cluster_ids,
    const int* h_cluster_query_offsets,
    const int* h_cluster_query_list,
    const int* h_active_cluster_ids,
    int active_cluster_count,
    int metadata_compact,
    const int* h_numa_cluster_boundaries,
    int numa_cluster_boundary_count,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    int topk,
    int distance_mode,
    int num_threads,
    int* h_topk_local_idx,
    float* h_topk_dist,
    float* scratch_thread_dists,
    int* scratch_thread_idxs,
    int* scratch_thread_sizes,
    uint8_t* scratch_thread_touched)
{
    (void)distance_mode;
    if (num_threads <= 0) num_threads = paraselect::resolve_cpu_fine_thread_count();
    num_threads = paraselect::resolve_adaptive_thread_count(
        num_threads,
        n_query * std::max(1, n_probes),
        std::max(1, numa_cluster_boundary_count - 1));

    auto now = []() { return std::chrono::high_resolution_clock::now(); };
    auto elapsed_ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto now_s = []() {
        return std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    };
    thread_stats_detail::FineTimingSnapshot timing;

    NumaSchedulerWorkspace& workspace = numa_scheduler_workspace();
    const int* cluster_query_offset_ptr = h_cluster_query_offsets;
    const int* cluster_query_list_ptr = h_cluster_query_list;
    auto metadata_t0 = now();
    if (!cluster_query_offset_ptr || !cluster_query_list_ptr) {
        build_cluster_query_metadata(
            &workspace.owned_cluster_query_offset, &workspace.owned_cluster_query_list,
            h_coarse_cluster_ids, n_query, n_probes, n_total_clusters);
        cluster_query_offset_ptr = workspace.owned_cluster_query_offset.data();
        cluster_query_list_ptr = workspace.owned_cluster_query_list.data();
    }
    timing.metadata_build_ms = elapsed_ms(metadata_t0, now());

    const int T = num_threads;
    const size_t touch_elems = (size_t)T * (size_t)n_query;
    auto scratch_t0 = now();
    std::fill(scratch_thread_touched, scratch_thread_touched + touch_elems, (uint8_t)0);
    timing.scratch_clear_ms = elapsed_ms(scratch_t0, now());

    auto& tasks = workspace.tasks;
    tasks.clear();
    auto task_t0 = now();
    if (metadata_compact) {
        cluster_based_compact_detail::build_cluster_tasks_active_compact(
            &tasks, h_active_cluster_ids, active_cluster_count,
            cluster_query_offset_ptr, h_cluster_counts, n_total_clusters, 16);
    } else {
        cluster_based_sharding_detail::build_cluster_tasks_active(
            &tasks, h_active_cluster_ids, active_cluster_count,
            cluster_query_offset_ptr, h_cluster_counts, n_total_clusters, 16);
    }
    timing.task_build_ms = elapsed_ms(task_t0, now());

    if (!h_numa_cluster_boundaries || numa_cluster_boundary_count < 2 ||
        h_numa_cluster_boundaries[0] != 0 ||
        h_numa_cluster_boundaries[numa_cluster_boundary_count - 1] != n_total_clusters) {
        throw std::runtime_error("u8 NUMA fine requires precomputed cluster boundary metadata");
    }
    const int numa_nodes = numa_cluster_boundary_count - 1;
    if (numa_nodes > T) {
        throw std::runtime_error("u8 NUMA fine has more NUMA boundary ranges than worker threads");
    }
    auto& node_task_ids = workspace.node_task_ids;
    if ((int)node_task_ids.size() != numa_nodes) node_task_ids.resize((size_t)numa_nodes);
    for (auto& ids : node_task_ids) ids.clear();
    build_numa_node_task_ids(
        &node_task_ids, &workspace.node_task_counts, tasks,
        h_numa_cluster_boundaries, numa_cluster_boundary_count,
        !metadata_compact);
    auto& node_task_next = workspace.node_task_next;
    node_task_next.assign((size_t)numa_nodes, 0);

    const bool print_thread_stats = thread_stats_detail::cpu_fine_thread_stats_enabled();
    const bool collect_thread_stats = true;
    auto& thread_stats = workspace.thread_stats;
    thread_stats.assign((size_t)T, thread_stats_detail::ThreadStat{});
    auto& thread_fma = workspace.thread_fma;
    thread_fma.assign((size_t)T, 0);

    auto scan_t0 = now();
    auto& pool = paraselect::cpu_fine::persistent_thread_pool_for_threads(T);
    pool.run([&](int tid) {
        thread_stats_detail::ThreadStat local_stat;
        if (collect_thread_stats) {
            local_stat.start_s = now_s();
        }
        const int thread_node = std::min((tid * numa_nodes) / std::max(1, T), numa_nodes - 1);
        if (collect_thread_stats) {
            local_stat.node_id = thread_node;
        }
        const int node_tid_begin = (thread_node * T) / numa_nodes;
        const int node_local_tid = tid - node_tid_begin;
        pin_thread_to_numa_core_cached(thread_node, node_local_tid);

        float* my_dists = scratch_thread_dists + (size_t)tid * (size_t)n_query * (size_t)topk;
        int* my_idxs = scratch_thread_idxs + (size_t)tid * (size_t)n_query * (size_t)topk;
        int* my_sizes = scratch_thread_sizes + (size_t)tid * (size_t)n_query;
        uint8_t* my_touched = scratch_thread_touched + (size_t)tid * (size_t)n_query;

        auto push_local = [&](int qid, float d, int gidx) {
            LocalTopK h;
            h.dists = my_dists + (size_t)qid * (size_t)topk;
            h.idxs = my_idxs + (size_t)qid * (size_t)topk;
            if (!my_touched[qid]) {
                my_touched[qid] = 1;
                my_sizes[qid] = 0;
            }
            h.size = my_sizes[qid];
            h.push(d, gidx, topk);
            my_sizes[qid] = h.size;
        };

        auto process_task = [&](const cluster_based_sharding_detail::ClusterTask& task) {
            const int cid = task.cid;
            const int qlist_size = task.qlist_end - task.qlist_start;
            const long long base_off = h_cluster_offsets[cid];
            const int shard_begin = task.lane_begin;
            const int shard_end = task.lane_end;
            const int shard_count = shard_end - shard_begin;
            if (shard_count <= 0) return;
            const int* qlist = cluster_query_list_ptr + task.qlist_start;
            if (collect_thread_stats) {
                local_stat.tasks += 1;
                local_stat.query_refs += qlist_size;
                local_stat.vectors += shard_count;
            }
            int qi = 0;
            for (; qi + 4 <= qlist_size; qi += 4) {
                const int q0 = qlist[qi + 0];
                const int q1 = qlist[qi + 1];
                const int q2 = qlist[qi + 2];
                const int q3 = qlist[qi + 3];
                const uint8_t* pq0 = h_query + (size_t)q0 * (size_t)dim;
                const uint8_t* pq1 = h_query + (size_t)q1 * (size_t)dim;
                const uint8_t* pq2 = h_query + (size_t)q2 * (size_t)dim;
                const uint8_t* pq3 = h_query + (size_t)q3 * (size_t)dim;
                const int pf_end = std::min(kPrefetchAhead, shard_count);
                for (int vi = 0; vi < pf_end; ++vi) {
                    Evaluator::prefetch_candidate(h_base + ((size_t)base_off + shard_begin + vi) * (size_t)dim, dim);
                }
                for (int vi = 0; vi < shard_count; ++vi) {
                    const int pf_vi = vi + kPrefetchAhead;
                    if (pf_vi < shard_count) {
                        Evaluator::prefetch_candidate(
                            h_base + ((size_t)base_off + shard_begin + pf_vi) * (size_t)dim, dim);
                    }
                    const int lane = shard_begin + vi;
                    const uint8_t* v = h_base + ((size_t)base_off + lane) * (size_t)dim;
                    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    Evaluator::distance_q4(pq0, pq1, pq2, pq3, v, dim, d0, d1, d2, d3);
                    const int gidx = static_cast<int>(base_off + lane);
                    push_local(q0, (float)d0, gidx);
                    push_local(q1, (float)d1, gidx);
                    push_local(q2, (float)d2, gidx);
                    push_local(q3, (float)d3, gidx);
                }
                const long long fma = (long long)shard_count * (long long)dim * 4LL;
                thread_fma[(size_t)tid] += fma;
                if (collect_thread_stats) {
                    local_stat.fmas += fma;
                }
            }
            for (; qi < qlist_size; ++qi) {
                const int qid = qlist[qi];
                const uint8_t* q = h_query + (size_t)qid * (size_t)dim;
                const int pf_end = std::min(kPrefetchAhead, shard_count);
                for (int vi = 0; vi < pf_end; ++vi) {
                    Evaluator::prefetch_candidate(h_base + ((size_t)base_off + shard_begin + vi) * (size_t)dim, dim);
                }
                for (int vi = 0; vi < shard_count; ++vi) {
                    const int pf_vi = vi + kPrefetchAhead;
                    if (pf_vi < shard_count) {
                        Evaluator::prefetch_candidate(
                            h_base + ((size_t)base_off + shard_begin + pf_vi) * (size_t)dim, dim);
                    }
                    const int lane = shard_begin + vi;
                    const uint8_t* v = h_base + ((size_t)base_off + lane) * (size_t)dim;
                    const int32_t d = Evaluator::distance_q1(q, v, dim);
                    push_local(qid, (float)d, static_cast<int>(base_off + lane));
                }
                const long long fma = (long long)shard_count * (long long)dim;
                thread_fma[(size_t)tid] += fma;
                if (collect_thread_stats) {
                    local_stat.fmas += fma;
                }
            }
        };

        while (true) {
            const int local_pos = __atomic_fetch_add(&node_task_next[(size_t)thread_node], 1, __ATOMIC_RELAXED);
            if (local_pos >= static_cast<int>(node_task_ids[(size_t)thread_node].size())) break;
            const int task_id = node_task_ids[(size_t)thread_node][(size_t)local_pos];
            process_task(tasks[(size_t)task_id]);
        }
        if (collect_thread_stats) {
            local_stat.end_s = now_s();
            thread_stats[(size_t)tid] = local_stat;
        }
    });
    long long total_fma = 0;
    for (long long fma : thread_fma) total_fma += fma;
    if (print_thread_stats) {
        thread_stats_detail::print_thread_stats_summary("u8_cluster_dynamic_numa", thread_stats);
    }
    timing.scan_ms = elapsed_ms(scan_t0, now());
    thread_stats_detail::record_numa_node_stats_summary("u8_cluster_dynamic_numa", thread_stats, print_thread_stats);


    auto merge_t0 = now();
    std::atomic<int> next_query(0);
    pool.run([&](int /*tid*/) {
        while (true) {
            const int q_begin = next_query.fetch_add(64, std::memory_order_relaxed);
            if (q_begin >= n_query) break;
            const int q_end = std::min(n_query, q_begin + 64);
            for (int qi = q_begin; qi < q_end; ++qi) {
                float out_dist[1024];
                int out_idx[1024];
                TopKHeap g;
                g.init(topk, out_dist, out_idx);
                for (int t = 0; t < T; ++t) {
                    if (!scratch_thread_touched[(size_t)t * (size_t)n_query + (size_t)qi]) continue;
                    const int sz = scratch_thread_sizes[(size_t)t * (size_t)n_query + (size_t)qi];
                    const float* td = scratch_thread_dists + ((size_t)t * (size_t)n_query + (size_t)qi) * (size_t)topk;
                    const int* ti = scratch_thread_idxs + ((size_t)t * (size_t)n_query + (size_t)qi) * (size_t)topk;
                    for (int j = 0; j < sz; ++j) g.push(td[j], ti[j]);
                }
                finalize_topk(
                    g,
                    h_topk_dist + (size_t)qi * (size_t)topk,
                    h_topk_local_idx + (size_t)qi * (size_t)topk,
                    topk);
            }
        }
    });
    timing.merge_ms = elapsed_ms(merge_t0, now());
    auto outer_timing = thread_stats_detail::get_latest_fine_timing();
    outer_timing.metadata_build_ms += timing.metadata_build_ms;
    outer_timing.scratch_clear_ms += timing.scratch_clear_ms;
    outer_timing.task_build_ms += timing.task_build_ms;
    outer_timing.scan_ms += timing.scan_ms;
    outer_timing.merge_ms += timing.merge_ms;
    thread_stats_detail::record_latest_fine_timing(outer_timing);
    return total_fma;
}

template <typename Evaluator>
inline long long cluster_based_v1_u8_numa_with_scratch(
    const uint8_t* h_base,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const uint8_t* h_query,
    const idx_t* h_coarse_cluster_ids,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    int topk,
    int distance_mode,
    int num_threads,
    int* h_topk_local_idx,
    float* h_topk_dist,
    float* scratch_thread_dists,
    int* scratch_thread_idxs,
    int* scratch_thread_sizes,
    uint8_t* scratch_thread_touched)
{
    return cluster_based_v1_u8_numa_impl<Evaluator>(
        h_base, h_cluster_offsets, h_cluster_counts, h_query, h_coarse_cluster_ids,
        nullptr, nullptr, nullptr, -1, 0, nullptr, 0, n_query, dim, n_total_clusters, n_probes, topk,
        distance_mode, num_threads, h_topk_local_idx, h_topk_dist,
        scratch_thread_dists, scratch_thread_idxs, scratch_thread_sizes, scratch_thread_touched);
}

template <typename Evaluator>
inline long long cluster_based_v1_u8_numa_with_metadata(
    const uint8_t* h_base,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const uint8_t* h_query,
    const int* h_cluster_query_offsets,
    const int* h_cluster_query_list,
    const int* h_active_cluster_ids,
    int active_cluster_count,
    const int* h_numa_cluster_boundaries,
    int numa_cluster_boundary_count,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    int topk,
    int distance_mode,
    int num_threads,
    int* h_topk_local_idx,
    float* h_topk_dist,
    float* scratch_thread_dists,
    int* scratch_thread_idxs,
    int* scratch_thread_sizes,
    uint8_t* scratch_thread_touched)
{
    return cluster_based_v1_u8_numa_impl<Evaluator>(
        h_base, h_cluster_offsets, h_cluster_counts, h_query, nullptr,
        h_cluster_query_offsets, h_cluster_query_list, h_active_cluster_ids, active_cluster_count, 0, h_numa_cluster_boundaries, numa_cluster_boundary_count, n_query, dim,
        n_total_clusters, n_probes, topk, distance_mode, num_threads,
        h_topk_local_idx, h_topk_dist, scratch_thread_dists, scratch_thread_idxs,
        scratch_thread_sizes, scratch_thread_touched);
}

template <typename Evaluator>
inline long long cluster_based_v1_u8_numa_with_compact_metadata(
    const uint8_t* h_base,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const uint8_t* h_query,
    const int* h_compact_query_offsets,
    const int* h_cluster_query_list,
    const int* h_active_cluster_ids,
    int active_cluster_count,
    const int* h_numa_cluster_boundaries,
    int numa_cluster_boundary_count,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    int topk,
    int distance_mode,
    int num_threads,
    int* h_topk_local_idx,
    float* h_topk_dist,
    float* scratch_thread_dists,
    int* scratch_thread_idxs,
    int* scratch_thread_sizes,
    uint8_t* scratch_thread_touched)
{
    return cluster_based_v1_u8_numa_impl<Evaluator>(
        h_base, h_cluster_offsets, h_cluster_counts, h_query, nullptr,
        h_compact_query_offsets, h_cluster_query_list, h_active_cluster_ids, active_cluster_count, 1, h_numa_cluster_boundaries, numa_cluster_boundary_count, n_query, dim,
        n_total_clusters, n_probes, topk, distance_mode, num_threads,
        h_topk_local_idx, h_topk_dist, scratch_thread_dists, scratch_thread_idxs,
        scratch_thread_sizes, scratch_thread_touched);
}

}  // namespace cluster_based_v1_u8_numa_detail
}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_CLUSTER_BASED_V1_U8_NUMA_H
