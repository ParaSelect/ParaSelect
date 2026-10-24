#ifndef PARASELECT_CLUSTER_BASED_V1_COMMON_H
#define PARASELECT_CLUSTER_BASED_V1_COMMON_H

#include "cpu/utils/adaptive_thread_count.h"
#include "cpu/utils/cpu_affinity.h"
#include "search/kernels/fine/cpu_fine/cpu_fine_common.h"
#include "search/kernels/fine/cpu_fine/executor/persistent_thread_pool.h"
#include "search/kernels/fine/cpu_fine/schedulers/cluster_based_compact.h"
#include "search/kernels/fine/cpu_fine/schedulers/cluster_based_sharding.h"
#include "search/kernels/fine/cpu_fine/schedulers/thread_stats.h"
#include "search/kernels/fine/cpu_fine/topk_local.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <vector>

namespace paraselect {
namespace cpu_fine {
namespace cluster_based_v1_common_detail {

struct SchedulerWorkspace {
    std::vector<int> owned_cluster_query_offset;
    std::vector<int> owned_cluster_query_list;
    std::vector<cluster_based_sharding_detail::ClusterTask> tasks;
    std::vector<thread_stats_detail::ThreadStat> thread_stats;
    std::vector<long long> thread_fma;
};

inline auto scheduler_now() {
    return std::chrono::high_resolution_clock::now();
}

template <typename TimePoint>
inline double scheduler_elapsed_ms(TimePoint a, TimePoint b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

inline double scheduler_now_s() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
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
            if (cid >= 0 && cid < n_total_clusters) cluster_query_count[(size_t)cid]++;
        }
    }
    owned_cluster_query_offset->assign((size_t)n_total_clusters + 1u, 0);
    for (int c = 0; c < n_total_clusters; ++c) {
        (*owned_cluster_query_offset)[(size_t)c + 1u] =
            (*owned_cluster_query_offset)[(size_t)c] + cluster_query_count[(size_t)c];
    }

    owned_cluster_query_list->assign((size_t)(*owned_cluster_query_offset)[(size_t)n_total_clusters], 0);
    std::vector<int> wpos(n_total_clusters, 0);
    for (int qi = 0; qi < n_query; ++qi) {
        for (int pi = 0; pi < n_probes; ++pi) {
            const int cid = static_cast<int>(h_coarse_cluster_ids[(size_t)qi * (size_t)n_probes + (size_t)pi]);
            if (cid >= 0 && cid < n_total_clusters) {
                const int pos = (*owned_cluster_query_offset)[(size_t)cid] + wpos[(size_t)cid]++;
                (*owned_cluster_query_list)[(size_t)pos] = qi;
            }
        }
    }
}

inline void prepare_cluster_query_metadata(
    SchedulerWorkspace* workspace,
    const idx_t* h_coarse_cluster_ids,
    int n_query,
    int n_probes,
    int n_total_clusters,
    const int** cluster_query_offset_ptr,
    const int** cluster_query_list_ptr)
{
    if (*cluster_query_offset_ptr && *cluster_query_list_ptr) return;
    build_cluster_query_metadata(
        &workspace->owned_cluster_query_offset,
        &workspace->owned_cluster_query_list,
        h_coarse_cluster_ids,
        n_query,
        n_probes,
        n_total_clusters);
    *cluster_query_offset_ptr = workspace->owned_cluster_query_offset.data();
    *cluster_query_list_ptr = workspace->owned_cluster_query_list.data();
}

inline void build_cluster_tasks_for_metadata(
    std::vector<cluster_based_sharding_detail::ClusterTask>* tasks,
    int metadata_compact,
    const int* h_active_cluster_ids,
    int active_cluster_count,
    const int* cluster_query_offset_ptr,
    const int* h_cluster_counts,
    int n_total_clusters,
    int shard_tile)
{
    tasks->clear();
    if (metadata_compact) {
        cluster_based_compact_detail::build_cluster_tasks_active_compact(
            tasks, h_active_cluster_ids, active_cluster_count,
            cluster_query_offset_ptr, h_cluster_counts, n_total_clusters, shard_tile);
    } else if (h_active_cluster_ids && active_cluster_count >= 0) {
        cluster_based_sharding_detail::build_cluster_tasks_active(
            tasks, h_active_cluster_ids, active_cluster_count,
            cluster_query_offset_ptr, h_cluster_counts, n_total_clusters, shard_tile);
    } else {
        cluster_based_sharding_detail::build_cluster_tasks(
            tasks, cluster_query_offset_ptr, h_cluster_counts, n_total_clusters, shard_tile);
    }
}

struct MergeThreadTopkJob {
    std::atomic<int> next_query{0};
    int T = 0;
    int n_query = 0;
    int topk = 0;
    const float* scratch_thread_dists = nullptr;
    const int* scratch_thread_idxs = nullptr;
    const int* scratch_thread_sizes = nullptr;
    const uint8_t* scratch_thread_touched = nullptr;
    float* h_topk_dist = nullptr;
    int* h_topk_local_idx = nullptr;
};

static inline void merge_thread_topk_worker(void* opaque, int /*tid*/) {
    auto* job = static_cast<MergeThreadTopkJob*>(opaque);
    while (true) {
        const int q_begin = job->next_query.fetch_add(64, std::memory_order_relaxed);
        if (q_begin >= job->n_query) break;
        const int q_end = std::min(job->n_query, q_begin + 64);
        for (int qi = q_begin; qi < q_end; ++qi) {
            float out_dist[1024];
            int out_idx[1024];
            TopKHeap g;
            g.init(job->topk, out_dist, out_idx);
            for (int t = 0; t < job->T; ++t) {
                if (!job->scratch_thread_touched[(size_t)t * (size_t)job->n_query + (size_t)qi]) continue;
                const int sz = job->scratch_thread_sizes[(size_t)t * (size_t)job->n_query + (size_t)qi];
                const float* td = job->scratch_thread_dists +
                    ((size_t)t * (size_t)job->n_query + (size_t)qi) * (size_t)job->topk;
                const int* ti = job->scratch_thread_idxs +
                    ((size_t)t * (size_t)job->n_query + (size_t)qi) * (size_t)job->topk;
                for (int j = 0; j < sz; ++j) g.push(td[j], ti[j]);
            }
            finalize_topk(
                g,
                job->h_topk_dist + (size_t)qi * (size_t)job->topk,
                job->h_topk_local_idx + (size_t)qi * (size_t)job->topk,
                job->topk);
        }
    }
}

inline void merge_thread_topk(
    PersistentThreadPool& pool,
    int T,
    int n_query,
    int topk,
    const float* scratch_thread_dists,
    const int* scratch_thread_idxs,
    const int* scratch_thread_sizes,
    const uint8_t* scratch_thread_touched,
    float* h_topk_dist,
    int* h_topk_local_idx)
{
    MergeThreadTopkJob job;
    job.T = T;
    job.n_query = n_query;
    job.topk = topk;
    job.scratch_thread_dists = scratch_thread_dists;
    job.scratch_thread_idxs = scratch_thread_idxs;
    job.scratch_thread_sizes = scratch_thread_sizes;
    job.scratch_thread_touched = scratch_thread_touched;
    job.h_topk_dist = h_topk_dist;
    job.h_topk_local_idx = h_topk_local_idx;
    pool.run_raw_job(&job, merge_thread_topk_worker);
}

inline void accumulate_latest_fine_timing(const thread_stats_detail::FineTimingSnapshot& timing) {
    auto outer_timing = thread_stats_detail::get_latest_fine_timing();
    outer_timing.metadata_build_ms += timing.metadata_build_ms;
    outer_timing.scratch_clear_ms += timing.scratch_clear_ms;
    outer_timing.task_build_ms += timing.task_build_ms;
    outer_timing.scan_ms += timing.scan_ms;
    outer_timing.merge_ms += timing.merge_ms;
    thread_stats_detail::record_latest_fine_timing(outer_timing);
}

}  // namespace cluster_based_v1_common_detail
}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_CLUSTER_BASED_V1_COMMON_H
