#ifndef PARASELECT_CLUSTER_BASED_V1_U8_H
#define PARASELECT_CLUSTER_BASED_V1_U8_H

#include "search/kernels/fine/cpu_fine/schedulers/cluster_based_v1_common.h"
#include "utils/pch.h"

#include <algorithm>
#include <atomic>
#include <cstddef>


namespace paraselect {
namespace cpu_fine {
namespace cluster_based_v1_u8_detail {

static constexpr int kPrefetchAhead = 8;

inline cluster_based_v1_common_detail::SchedulerWorkspace& u8_scheduler_workspace() {
    thread_local cluster_based_v1_common_detail::SchedulerWorkspace workspace;
    return workspace;
}

template <typename Evaluator>
inline long long cluster_based_v1_u8_impl(
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
    const int* h_numa_cluster_owner,
    int numa_cluster_owner_count,
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
    (void)h_numa_cluster_owner;
    (void)numa_cluster_owner_count;
    if (num_threads <= 0) num_threads = paraselect::resolve_cpu_fine_thread_count();
    num_threads = paraselect::resolve_adaptive_thread_count(
        num_threads, n_query * std::max(1, n_probes));

    auto now = cluster_based_v1_common_detail::scheduler_now;
    auto elapsed_ms = [](auto a, auto b) {
        return cluster_based_v1_common_detail::scheduler_elapsed_ms(a, b);
    };
    auto now_s = cluster_based_v1_common_detail::scheduler_now_s;
    thread_stats_detail::FineTimingSnapshot timing;

    auto& workspace = u8_scheduler_workspace();
    const int* cluster_query_offset_ptr = h_cluster_query_offsets;
    const int* cluster_query_list_ptr = h_cluster_query_list;
    auto metadata_t0 = now();
    cluster_based_v1_common_detail::prepare_cluster_query_metadata(
        &workspace, h_coarse_cluster_ids, n_query, n_probes, n_total_clusters,
        &cluster_query_offset_ptr, &cluster_query_list_ptr);
    timing.metadata_build_ms = elapsed_ms(metadata_t0, now());

    const int T = num_threads;
    const size_t touch_elems = (size_t)T * (size_t)n_query;
    auto scratch_t0 = now();
    std::fill(scratch_thread_touched, scratch_thread_touched + touch_elems, (uint8_t)0);
    timing.scratch_clear_ms = elapsed_ms(scratch_t0, now());

    auto& tasks = workspace.tasks;
    auto task_t0 = now();
    cluster_based_v1_common_detail::build_cluster_tasks_for_metadata(
        &tasks, metadata_compact, h_active_cluster_ids, active_cluster_count,
        cluster_query_offset_ptr, h_cluster_counts, n_total_clusters, 16);
    timing.task_build_ms = elapsed_ms(task_t0, now());

    static bool printed_scheduler_debug = false;
    if (!printed_scheduler_debug && std::getenv("PARASELECT_DEBUG_CPU_FINE_TASKS")) {
        long long query_refs = 0;
        long long vectors = 0;
        int nonempty_clusters = 0;
        int max_cluster_queries = 0;
        int max_cluster_count = 0;
        for (int c = 0; c < n_total_clusters; ++c) {
            const int qrefs = cluster_query_offset_ptr[c + 1] - cluster_query_offset_ptr[c];
            if (qrefs > 0) ++nonempty_clusters;
            query_refs += qrefs;
            max_cluster_queries = std::max(max_cluster_queries, qrefs);
            max_cluster_count = std::max(max_cluster_count, h_cluster_counts[c]);
        }
        for (const auto& task : tasks) {
            vectors += (long long)(task.lane_end - task.lane_begin);
        }
        fprintf(stderr,
                "[u8_scheduler:debug] T=%d n_query=%d nprobe=%d ncluster=%d active_cluster_count=%d metadata_compact=%d task_count=%zu nonempty_clusters=%d query_refs=%lld vectors=%lld max_cluster_queries=%d max_cluster_count=%d first_offsets=%d,%d,%d,%d\n",
                T, n_query, n_probes, n_total_clusters, active_cluster_count, metadata_compact,
                tasks.size(), nonempty_clusters, query_refs, vectors, max_cluster_queries, max_cluster_count,
                cluster_query_offset_ptr[0],
                n_total_clusters >= 1 ? cluster_query_offset_ptr[1] : -1,
                n_total_clusters >= 2 ? cluster_query_offset_ptr[2] : -1,
                n_total_clusters >= 3 ? cluster_query_offset_ptr[3] : -1);
        printed_scheduler_debug = true;
    }

    const bool print_thread_stats = thread_stats_detail::cpu_fine_thread_stats_enabled();
    const bool collect_thread_stats = true;
    auto& thread_stats = workspace.thread_stats;
    thread_stats.assign((size_t)T, thread_stats_detail::ThreadStat{});
    auto& thread_fma = workspace.thread_fma;
    thread_fma.assign((size_t)T, 0);

    auto scan_t0 = now();
    auto& pool = paraselect::cpu_fine::persistent_thread_pool_for_threads(T);
    std::atomic<int> next_task(0);
    pool.run([&](int tid) {
        thread_stats_detail::ThreadStat local_stat;
        if (collect_thread_stats) {
            local_stat.start_s = now_s();
        }
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
                int valid_count = 0;
                for (int vi = 0; vi < shard_count; ++vi) {
                    const int pf_vi = vi + kPrefetchAhead;
                    if (pf_vi < shard_count) {
                        Evaluator::prefetch_candidate(
                            h_base + ((size_t)base_off + shard_begin + pf_vi) * (size_t)dim, dim);
                    }
                    const int lane = shard_begin + vi;
                    const int gidx = static_cast<int>(base_off + lane);
                    const uint8_t* v = h_base + ((size_t)base_off + lane) * (size_t)dim;
                    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    Evaluator::distance_q4(pq0, pq1, pq2, pq3, v, dim, d0, d1, d2, d3);
                    push_local(q0, (float)d0, gidx);
                    push_local(q1, (float)d1, gidx);
                    push_local(q2, (float)d2, gidx);
                    push_local(q3, (float)d3, gidx);
                    ++valid_count;
                }
                const long long fma = (long long)valid_count * (long long)dim * 4LL;
                thread_fma[(size_t)tid] += fma;
                if (collect_thread_stats) local_stat.fmas += fma;
            }
            for (; qi < qlist_size; ++qi) {
                const int qid = qlist[qi];
                const uint8_t* q = h_query + (size_t)qid * (size_t)dim;
                const int pf_end = std::min(kPrefetchAhead, shard_count);
                for (int vi = 0; vi < pf_end; ++vi) {
                    Evaluator::prefetch_candidate(h_base + ((size_t)base_off + shard_begin + vi) * (size_t)dim, dim);
                }
                int valid_count = 0;
                for (int vi = 0; vi < shard_count; ++vi) {
                    const int pf_vi = vi + kPrefetchAhead;
                    if (pf_vi < shard_count) {
                        Evaluator::prefetch_candidate(
                            h_base + ((size_t)base_off + shard_begin + pf_vi) * (size_t)dim, dim);
                    }
                    const int lane = shard_begin + vi;
                    const int gidx = static_cast<int>(base_off + lane);
                    const uint8_t* v = h_base + ((size_t)base_off + lane) * (size_t)dim;
                    const int32_t d = Evaluator::distance_q1(q, v, dim);
                    push_local(qid, (float)d, gidx);
                    ++valid_count;
                }
                const long long fma = (long long)valid_count * (long long)dim;
                thread_fma[(size_t)tid] += fma;
                if (collect_thread_stats) local_stat.fmas += fma;
            }
        };

        const int task_count = static_cast<int>(tasks.size());
        while (true) {
            const int task_id = next_task.fetch_add(1, std::memory_order_relaxed);
            if (task_id >= task_count) break;
            process_task(tasks[(size_t)task_id]);
        }
        if (collect_thread_stats) {
            local_stat.end_s = now_s();
            thread_stats[(size_t)tid] = local_stat;
        }
    });
    long long total_fma = 0;
    for (long long fma : thread_fma) total_fma += fma;
    timing.scan_ms = elapsed_ms(scan_t0, now());
    if (print_thread_stats) {
        thread_stats_detail::print_thread_stats_summary("u8_cluster_dynamic", thread_stats);
    }

    auto merge_t0 = now();
    cluster_based_v1_common_detail::merge_thread_topk(
        pool, T, n_query, topk,
        scratch_thread_dists, scratch_thread_idxs, scratch_thread_sizes, scratch_thread_touched,
        h_topk_dist, h_topk_local_idx);
    timing.merge_ms = elapsed_ms(merge_t0, now());
    cluster_based_v1_common_detail::accumulate_latest_fine_timing(timing);
    return total_fma;
}

template <typename Evaluator>
inline long long cluster_based_v1_u8_with_scratch(
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
    return cluster_based_v1_u8_impl<Evaluator>(
        h_base, h_cluster_offsets, h_cluster_counts, h_query, h_coarse_cluster_ids,
        nullptr, nullptr, nullptr, -1, 0, nullptr, 0, n_query, dim, n_total_clusters, n_probes, topk, distance_mode,
        num_threads, h_topk_local_idx, h_topk_dist,
        scratch_thread_dists, scratch_thread_idxs, scratch_thread_sizes, scratch_thread_touched);
}

template <typename Evaluator>
inline long long cluster_based_v1_u8_with_metadata(
    const uint8_t* h_base,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const uint8_t* h_query,
    const int* h_cluster_query_offsets,
    const int* h_cluster_query_list,
    const int* h_active_cluster_ids,
    int active_cluster_count,
    const int* h_numa_cluster_owner,
    int numa_cluster_owner_count,
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
    return cluster_based_v1_u8_impl<Evaluator>(
        h_base, h_cluster_offsets, h_cluster_counts, h_query, nullptr,
        h_cluster_query_offsets, h_cluster_query_list, h_active_cluster_ids, active_cluster_count, 0,
        h_numa_cluster_owner, numa_cluster_owner_count,
        n_query, dim, n_total_clusters, n_probes, topk, distance_mode,
        num_threads, h_topk_local_idx, h_topk_dist,
        scratch_thread_dists, scratch_thread_idxs, scratch_thread_sizes, scratch_thread_touched);
}

template <typename Evaluator>
inline long long cluster_based_v1_u8_with_compact_metadata(
    const uint8_t* h_base,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const uint8_t* h_query,
    const int* h_compact_query_offsets,
    const int* h_cluster_query_list,
    const int* h_active_cluster_ids,
    int active_cluster_count,
    const int* h_numa_cluster_owner,
    int numa_cluster_owner_count,
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
    return cluster_based_v1_u8_impl<Evaluator>(
        h_base, h_cluster_offsets, h_cluster_counts, h_query, nullptr,
        h_compact_query_offsets, h_cluster_query_list, h_active_cluster_ids, active_cluster_count, 1,
        h_numa_cluster_owner, numa_cluster_owner_count,
        n_query, dim, n_total_clusters, n_probes, topk, distance_mode,
        num_threads, h_topk_local_idx, h_topk_dist,
        scratch_thread_dists, scratch_thread_idxs, scratch_thread_sizes, scratch_thread_touched);
}
}  // namespace cluster_based_v1_u8_detail
}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_CLUSTER_BASED_V1_U8_H
