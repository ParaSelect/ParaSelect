#ifndef PARASELECT_CLUSTER_BASED_V1_AOSOA_H
#define PARASELECT_CLUSTER_BASED_V1_AOSOA_H

#include "search/kernels/fine/cpu_fine/cpu_fine_common.h"
#include "search/kernels/fine/cpu_fine/schedulers/cluster_based_compact.h"
#include "search/kernels/fine/cpu_fine/schedulers/cluster_based_sharding.h"
#include "search/kernels/fine/cpu_fine/schedulers/thread_stats.h"
#include "search/kernels/fine/cpu_fine/topk_local.h"
#include "utils/pch.h"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace paraselect {
namespace cpu_fine {
namespace cluster_based_v1_aosoa_detail {

template <typename Evaluator, int Tile>
inline long long cluster_based_v1_aosoa_impl(
    const float* h_base_aosoa,
    const long long* h_aosoa_offsets,
    const long long* h_row_major_offsets,
    const int* h_cluster_counts,
    const float* h_query,
    const idx_t* h_coarse_cluster_ids,
    const int* h_cluster_query_offsets,
    const int* h_cluster_query_list,
    const int* h_active_cluster_ids,
    int active_cluster_count,
    int metadata_compact,
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
    if (num_threads <= 0) num_threads = omp_get_max_threads();

    std::vector<int> owned_cluster_query_offset;
    std::vector<int> owned_cluster_query_list;
    const int* cluster_query_offset_ptr = h_cluster_query_offsets;
    const int* cluster_query_list_ptr = h_cluster_query_list;
    if (!cluster_query_offset_ptr || !cluster_query_list_ptr) {
        std::vector<int> cluster_query_count(n_total_clusters, 0);
        for (int qi = 0; qi < n_query; ++qi) {
            for (int pi = 0; pi < n_probes; ++pi) {
                const int cid = static_cast<int>(h_coarse_cluster_ids[(size_t)qi * (size_t)n_probes + (size_t)pi]);
                if (cid >= 0 && cid < n_total_clusters) cluster_query_count[cid]++;
            }
        }
        owned_cluster_query_offset.assign((size_t)n_total_clusters + 1u, 0);
        for (int c = 0; c < n_total_clusters; ++c) {
            owned_cluster_query_offset[(size_t)c + 1u] = owned_cluster_query_offset[(size_t)c] + cluster_query_count[c];
        }
        owned_cluster_query_list.resize((size_t)owned_cluster_query_offset[(size_t)n_total_clusters]);
        std::vector<int> wpos(n_total_clusters, 0);
        for (int qi = 0; qi < n_query; ++qi) {
            for (int pi = 0; pi < n_probes; ++pi) {
                const int cid = static_cast<int>(h_coarse_cluster_ids[(size_t)qi * (size_t)n_probes + (size_t)pi]);
                if (cid >= 0 && cid < n_total_clusters) {
                    const int pos = owned_cluster_query_offset[(size_t)cid] + wpos[cid]++;
                    owned_cluster_query_list[(size_t)pos] = qi;
                }
            }
        }
        cluster_query_offset_ptr = owned_cluster_query_offset.data();
        cluster_query_list_ptr = owned_cluster_query_list.data();
    }

    const int T = num_threads;
    const size_t touch_elems = (size_t)T * (size_t)n_query;
    std::fill(scratch_thread_touched, scratch_thread_touched + touch_elems, (uint8_t)0);

    std::vector<cluster_based_sharding_detail::ClusterTask> tasks;
    if (metadata_compact) {
        cluster_based_compact_detail::build_cluster_tasks_active_compact(
            &tasks, h_active_cluster_ids, active_cluster_count,
            cluster_query_offset_ptr, h_cluster_counts, n_total_clusters, Tile);
    } else if (h_active_cluster_ids && active_cluster_count >= 0) {
        cluster_based_sharding_detail::build_cluster_tasks_active(
            &tasks, h_active_cluster_ids, active_cluster_count,
            cluster_query_offset_ptr, h_cluster_counts, n_total_clusters, Tile);
    } else {
        cluster_based_sharding_detail::build_cluster_tasks(
            &tasks, cluster_query_offset_ptr, h_cluster_counts, n_total_clusters, Tile);
    }

    const bool collect_thread_stats = thread_stats_detail::cpu_fine_thread_stats_enabled();
    std::vector<thread_stats_detail::ThreadStat> thread_stats(
        collect_thread_stats ? (size_t)T : 0);

    long long total_fma = 0;
    #pragma omp parallel num_threads(T) reduction(+:total_fma)
    {
        const int tid = omp_get_thread_num();
        thread_stats_detail::ThreadStat local_stat;
        if (collect_thread_stats) {
            local_stat.start_s = omp_get_wtime();
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

        alignas(64) float d0[Tile];
        alignas(64) float d1[Tile];
        alignas(64) float d2[Tile];
        alignas(64) float d3[Tile];

        const int task_count = static_cast<int>(tasks.size());
        #pragma omp for schedule(dynamic, 1)
        for (int task_id = 0; task_id < task_count; ++task_id) {
            const cluster_based_sharding_detail::ClusterTask& task = tasks[(size_t)task_id];
            const int cid = task.cid;
            const int qlist_size = task.qlist_end - task.qlist_start;
            const int count = h_cluster_counts[cid];
            const long long row_base = h_row_major_offsets[cid];
            const long long aosoa_base = h_aosoa_offsets[cid];
            const int group_begin = task.lane_begin / Tile;
            const int group_end = (task.lane_end + Tile - 1) / Tile;
            if (group_begin >= group_end) continue;
            const int shard_lane_begin = group_begin * Tile;
            const int shard_lane_end = std::min(group_end * Tile, count);
            const int shard_count = shard_lane_end - shard_lane_begin;
            if (shard_count <= 0) continue;
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
                const float* pq0 = h_query + (size_t)q0 * (size_t)dim;
                const float* pq1 = h_query + (size_t)q1 * (size_t)dim;
                const float* pq2 = h_query + (size_t)q2 * (size_t)dim;
                const float* pq3 = h_query + (size_t)q3 * (size_t)dim;

                for (int g = group_begin; g < group_end; ++g) {
                    const float* tile = h_base_aosoa + aosoa_base + (long long)g * (long long)dim * Tile;
                    Evaluator::distance_q4_tile(pq0, pq1, pq2, pq3, tile, dim, d0, d1, d2, d3);
                    const int lane_begin = g * Tile;
                    const int lane_end = std::min(lane_begin + Tile, count);
                    for (int lane = 0; lane < lane_end - lane_begin; ++lane) {
                        const int gidx = (int)(row_base + lane_begin + lane);
                        push_local(q0, d0[lane], gidx);
                        push_local(q1, d1[lane], gidx);
                        push_local(q2, d2[lane], gidx);
                        push_local(q3, d3[lane], gidx);
                    }
                }
                total_fma += (long long)shard_count * (long long)dim * 4LL;
                if (collect_thread_stats) {
                    local_stat.fmas += (long long)shard_count * (long long)dim * 4LL;
                }
            }
            for (; qi < qlist_size; ++qi) {
                const int qid = qlist[qi];
                const float* q = h_query + (size_t)qid * (size_t)dim;
                for (int g = group_begin; g < group_end; ++g) {
                    const float* tile = h_base_aosoa + aosoa_base + (long long)g * (long long)dim * Tile;
                    Evaluator::distance_q1_tile(q, tile, dim, d0);
                    const int lane_begin = g * Tile;
                    const int lane_end = std::min(lane_begin + Tile, count);
                    for (int lane = 0; lane < lane_end - lane_begin; ++lane) {
                        push_local(qid, d0[lane], (int)(row_base + lane_begin + lane));
                    }
                }
                total_fma += (long long)shard_count * (long long)dim;
                if (collect_thread_stats) {
                    local_stat.fmas += (long long)shard_count * (long long)dim;
                }
            }
        }
        if (collect_thread_stats) {
            local_stat.end_s = omp_get_wtime();
            thread_stats[(size_t)tid] = local_stat;
        }
    }
    thread_stats_detail::print_thread_stats_summary("aosoa_cluster_dynamic", thread_stats);

    #pragma omp parallel for schedule(dynamic, 64) num_threads(T)
    for (int qi = 0; qi < n_query; ++qi) {
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
    return total_fma;
}

template <typename Evaluator, int Tile>
inline long long cluster_based_v1_aosoa_with_scratch(
    const float* h_base_aosoa,
    const long long* h_aosoa_offsets,
    const long long* h_row_major_offsets,
    const int* h_cluster_counts,
    const float* h_query,
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
    return cluster_based_v1_aosoa_impl<Evaluator, Tile>(
        h_base_aosoa, h_aosoa_offsets, h_row_major_offsets, h_cluster_counts,
        h_query, h_coarse_cluster_ids, nullptr, nullptr, nullptr, -1, 0,
        n_query, dim, n_total_clusters, n_probes, topk, distance_mode, num_threads,
        h_topk_local_idx, h_topk_dist,
        scratch_thread_dists, scratch_thread_idxs, scratch_thread_sizes, scratch_thread_touched);
}

template <typename Evaluator, int Tile>
inline long long cluster_based_v1_aosoa_with_metadata(
    const float* h_base_aosoa,
    const long long* h_aosoa_offsets,
    const long long* h_row_major_offsets,
    const int* h_cluster_counts,
    const float* h_query,
    const int* h_cluster_query_offsets,
    const int* h_cluster_query_list,
    const int* h_active_cluster_ids,
    int active_cluster_count,
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
    return cluster_based_v1_aosoa_impl<Evaluator, Tile>(
        h_base_aosoa, h_aosoa_offsets, h_row_major_offsets, h_cluster_counts,
        h_query, nullptr, h_cluster_query_offsets, h_cluster_query_list,
        h_active_cluster_ids, active_cluster_count, 0,
        n_query, dim, n_total_clusters, n_probes, topk, distance_mode, num_threads,
        h_topk_local_idx, h_topk_dist,
        scratch_thread_dists, scratch_thread_idxs, scratch_thread_sizes, scratch_thread_touched);
}

template <typename Evaluator, int Tile>
inline long long cluster_based_v1_aosoa_with_compact_metadata(
    const float* h_base_aosoa,
    const long long* h_aosoa_offsets,
    const long long* h_row_major_offsets,
    const int* h_cluster_counts,
    const float* h_query,
    const int* h_compact_query_offsets,
    const int* h_cluster_query_list,
    const int* h_active_cluster_ids,
    int active_cluster_count,
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
    return cluster_based_v1_aosoa_impl<Evaluator, Tile>(
        h_base_aosoa, h_aosoa_offsets, h_row_major_offsets, h_cluster_counts,
        h_query, nullptr, h_compact_query_offsets, h_cluster_query_list,
        h_active_cluster_ids, active_cluster_count, 1,
        n_query, dim, n_total_clusters, n_probes, topk, distance_mode, num_threads,
        h_topk_local_idx, h_topk_dist,
        scratch_thread_dists, scratch_thread_idxs, scratch_thread_sizes, scratch_thread_touched);
}

}  // namespace cluster_based_v1_aosoa_detail
}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_CLUSTER_BASED_V1_AOSOA_H
