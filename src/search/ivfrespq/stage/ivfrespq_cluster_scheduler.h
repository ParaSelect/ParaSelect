#ifndef PARASELECT_IVFRESPQ_CLUSTER_SCHEDULER_H
#define PARASELECT_IVFRESPQ_CLUSTER_SCHEDULER_H

#include "search/kernels/fine/cpu_fine/cpu_fine_common.h"
#include "search/kernels/fine/cpu_fine/schedulers/cluster_based_sharding.h"
#include "utils/pch.h"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace paraselect {
namespace cpu_fine {
namespace cluster_based_scheduler_pq_detail {

struct ClusterBasedPQScheduleStats {
    long long adc_lookups = 0;
    long long scanned_codes = 0;
};

inline void build_cluster_query_metadata(
    const idx_t* h_coarse_cluster_ids,
    int n_query,
    int n_total_clusters,
    int n_probes,
    std::vector<int>* cluster_query_offsets,
    std::vector<int>* cluster_query_list)
{
    cluster_query_offsets->assign((size_t)n_total_clusters + 1, 0);
    if (!h_coarse_cluster_ids || n_query <= 0 || n_total_clusters <= 0 || n_probes <= 0) {
        cluster_query_list->clear();
        return;
    }

    std::vector<int> cluster_query_count((size_t)n_total_clusters, 0);
    for (int qi = 0; qi < n_query; ++qi) {
        for (int pi = 0; pi < n_probes; ++pi) {
            const int cid = static_cast<int>(h_coarse_cluster_ids[(size_t)qi * n_probes + pi]);
            if (cid >= 0 && cid < n_total_clusters) ++cluster_query_count[(size_t)cid];
        }
    }

    for (int cid = 0; cid < n_total_clusters; ++cid) {
        (*cluster_query_offsets)[(size_t)cid + 1] =
            (*cluster_query_offsets)[(size_t)cid] + cluster_query_count[(size_t)cid];
    }

    cluster_query_list->assign((size_t)(*cluster_query_offsets)[(size_t)n_total_clusters], 0);
    std::vector<int> write_pos((size_t)n_total_clusters, 0);
    for (int qi = 0; qi < n_query; ++qi) {
        for (int pi = 0; pi < n_probes; ++pi) {
            const int cid = static_cast<int>(h_coarse_cluster_ids[(size_t)qi * n_probes + pi]);
            if (cid < 0 || cid >= n_total_clusters) continue;
            const int pos = (*cluster_query_offsets)[(size_t)cid] + write_pos[(size_t)cid]++;
            (*cluster_query_list)[(size_t)pos] = qi;
        }
    }
}

inline void merge_heap_into_query(
    const float* local_dists,
    const int* local_idxs,
    int local_size,
    int global_k,
    float* global_dists,
    int* global_idxs,
    int* global_size)
{
    TopKHeap heap;
    heap.k = global_k;
    heap.size = *global_size;
    heap.dists = global_dists;
    heap.idxs = global_idxs;
    for (int i = 0; i < local_size; ++i) {
        heap.push(local_dists[i], local_idxs[i]);
    }
    *global_size = heap.size;
}

template <typename Ops>
inline ClusterBasedPQScheduleStats run_cluster_based_scheduler_pq(
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const uint8_t* h_pq_codes,
    const float* h_codebook,
    const float* h_centroids,
    const float* h_query_fp32,
    const idx_t* h_coarse_cluster_ids,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    int global_k,
    int local_k,
    int pq_M,
    int pq_K,
    int d_sub,
    int num_threads,
    float* stage1_dists,
    int* stage1_idxs,
    int* stage1_sizes)
{
    ClusterBasedPQScheduleStats stats;
    if (num_threads <= 0) num_threads = omp_get_max_threads();
    if (global_k <= 0 || local_k <= 0 || n_query <= 0 || n_total_clusters <= 0) return stats;
    local_k = std::min(local_k, global_k);

    std::fill(stage1_sizes, stage1_sizes + (size_t)n_query, 0);

    std::vector<int> cluster_query_offsets;
    std::vector<int> cluster_query_list;
    build_cluster_query_metadata(
        h_coarse_cluster_ids,
        n_query,
        n_total_clusters,
        n_probes,
        &cluster_query_offsets,
        &cluster_query_list);

    std::vector<cluster_based_sharding_detail::ClusterTask> tasks;
    cluster_based_sharding_detail::build_cluster_tasks(
        &tasks,
        cluster_query_offsets.data(),
        h_cluster_counts,
        n_total_clusters,
        16);

    std::vector<omp_lock_t> query_locks((size_t)n_query);
    for (int qi = 0; qi < n_query; ++qi) omp_init_lock(&query_locks[(size_t)qi]);

    long long total_lookups = 0;
    #pragma omp parallel num_threads(num_threads) reduction(+:total_lookups)
    {
        std::vector<float> lut((size_t)pq_M * (size_t)pq_K);
        std::vector<float> q_resid((size_t)dim);
        std::vector<float> local_dists((size_t)local_k);
        std::vector<int> local_idxs((size_t)local_k);

        const int task_count = static_cast<int>(tasks.size());
        #pragma omp for schedule(dynamic, 1)
        for (int task_id = 0; task_id < task_count; ++task_id) {
            const cluster_based_sharding_detail::ClusterTask& task = tasks[(size_t)task_id];
            const int cid = task.cid;
            const int query_begin = task.qlist_start;
            const int query_end = task.qlist_end;
            const int lane_begin = task.lane_begin;
            const int lane_end = task.lane_end;
            const int lane_count = lane_end - lane_begin;
            if (lane_count <= 0 || query_begin >= query_end) continue;

            const long long base_off = h_cluster_offsets[cid];
            const uint8_t* pq_base = h_pq_codes + ((size_t)base_off + (size_t)lane_begin) * pq_M;
            const float* centroid = h_centroids + (size_t)cid * dim;

            for (int qpos = query_begin; qpos < query_end; ++qpos) {
                const int qi = cluster_query_list[(size_t)qpos];
                const float* q_fp32 = h_query_fp32 + (size_t)qi * dim;
                Ops::build_query_residual(q_fp32, centroid, dim, q_resid.data());
                Ops::build_lut(q_resid.data(), h_codebook, pq_M, pq_K, d_sub, lut.data());

                TopKHeap local_heap;
                local_heap.init(local_k, local_dists.data(), local_idxs.data());
                total_lookups += Ops::scan_pq_cluster(
                    pq_base,
                    base_off + lane_begin,
                    lane_count,
                    lut.data(),
                    pq_M,
                    pq_K,
                    local_heap);

                if (local_heap.size <= 0) continue;
                omp_set_lock(&query_locks[(size_t)qi]);
                merge_heap_into_query(
                    local_dists.data(),
                    local_idxs.data(),
                    local_heap.size,
                    global_k,
                    stage1_dists + (size_t)qi * global_k,
                    stage1_idxs + (size_t)qi * global_k,
                    stage1_sizes + (size_t)qi);
                omp_unset_lock(&query_locks[(size_t)qi]);
            }
        }
    }

    for (int qi = 0; qi < n_query; ++qi) omp_destroy_lock(&query_locks[(size_t)qi]);

    stats.adc_lookups = total_lookups;
    stats.scanned_codes = pq_M > 0 ? total_lookups / pq_M : 0;
    return stats;
}

}  // namespace cluster_based_scheduler_pq_detail
}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_CLUSTER_BASED_SCHEDULER_PQ_H
