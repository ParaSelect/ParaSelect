#ifndef PARASELECT_IVFRESPQ_QUERY_SCHEDULER_H
#define PARASELECT_IVFRESPQ_QUERY_SCHEDULER_H

#include "search/kernels/fine/cpu_fine/cpu_fine_common.h"
#include "search/ivfrespq/stage/ivfrespq_pq_storage.h"
#include "utils/pch.h"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace paraselect {
namespace cpu_fine {
namespace query_balanced_scheduler_pq_detail {

struct QueryBalancedPQScheduleStats {
    long long adc_lookups = 0;
    long long scanned_codes = 0;
    long long heap_pushes = 0;
    double residual_ms = 0.0;
    double lut_build_ms = 0.0;
    double adc_lookup_ms = 0.0;
    double heap_push_ms = 0.0;
};

struct QueryCost {
    int qi = 0;
    long long cost = 0;
};

inline std::vector<std::vector<int>> build_query_buckets(
    const int* h_cluster_counts,
    const idx_t* h_coarse_cluster_ids,
    int n_query,
    int n_total_clusters,
    int n_probes,
    int num_threads)
{
    std::vector<QueryCost> query_costs((size_t)n_query);
    for (int qi = 0; qi < n_query; ++qi) {
        long long cost = 0;
        for (int pi = 0; pi < n_probes; ++pi) {
            const int cid = static_cast<int>(h_coarse_cluster_ids[(size_t)qi * n_probes + pi]);
            if (cid >= 0 && cid < n_total_clusters) cost += h_cluster_counts[cid];
        }
        query_costs[(size_t)qi] = {qi, cost};
    }

    std::sort(query_costs.begin(), query_costs.end(), [](const QueryCost& a, const QueryCost& b) {
        if (a.cost != b.cost) return a.cost > b.cost;
        return a.qi < b.qi;
    });

    std::vector<std::vector<int>> buckets((size_t)num_threads);
    std::vector<long long> bucket_loads((size_t)num_threads, 0);
    for (const QueryCost& qc : query_costs) {
        const int tid = (int)(std::min_element(bucket_loads.begin(), bucket_loads.end()) - bucket_loads.begin());
        buckets[(size_t)tid].push_back(qc.qi);
        bucket_loads[(size_t)tid] += qc.cost;
    }
    return buckets;
}

inline int resolve_pq_lut_tile(int pq_M, int pq_K, int d_sub, int requested) {
    if (pq_M != 16 || pq_K != 256 || d_sub != 8) return 1;
    if (requested != 1 && requested != 2 && requested != 4) {
        throw std::invalid_argument("PQ LUT tile must be 1, 2, or 4");
    }
    return requested;
}

template <typename Ops>
inline QueryBalancedPQScheduleStats run_query_balanced_scheduler_pq(
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const CpuPqCodeStorageView& pq_storage,
    const float* h_codebook,
    const float* h_centroids,
    const float* h_query_fp32,
    const idx_t* h_coarse_cluster_ids,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    int stage1_k,
    int pq_M,
    int pq_K,
    int d_sub,
    const float* precomputed_lut,
    int num_threads,
    float* stage1_dists,
    int* stage1_idxs,
    int* stage1_sizes,
    bool profile_breakdown = false,
    int requested_lut_tile = 1)
{
    QueryBalancedPQScheduleStats stats;
    if (num_threads <= 0) num_threads = omp_get_max_threads();
    if (num_threads <= 0 || stage1_k <= 0 || n_query <= 0 || n_total_clusters <= 0) return stats;

    std::fill(stage1_sizes, stage1_sizes + (size_t)n_query, 0);
    const int lut_tile = resolve_pq_lut_tile(
        pq_M, pq_K, d_sub, requested_lut_tile);
    std::vector<std::vector<int>> buckets = build_query_buckets(
        h_cluster_counts,
        h_coarse_cluster_ids,
        n_query,
        n_total_clusters,
        n_probes,
        num_threads);

    long long total_lookups = 0;
    long long total_heap_pushes = 0;
    double max_residual_ms = 0.0;
    double max_lut_build_ms = 0.0;
    double max_adc_lookup_ms = 0.0;
    double max_heap_push_ms = 0.0;
    #pragma omp parallel num_threads(num_threads) reduction(+:total_lookups,total_heap_pushes) reduction(max:max_residual_ms,max_lut_build_ms,max_adc_lookup_ms,max_heap_push_ms)
    {
        const int tid = omp_get_thread_num();
        std::vector<float> lut((size_t)pq_M * (size_t)pq_K);
        std::vector<float> lut_tile_buf((size_t)lut_tile * (size_t)pq_M * (size_t)pq_K);
        std::vector<float> q_resid((size_t)dim);
        std::vector<float> q_resid_tile((size_t)lut_tile * (size_t)dim);
        std::vector<float> heap_dists((size_t)stage1_k);
        std::vector<int> heap_idxs((size_t)stage1_k);
        std::vector<int> tile_cids((size_t)lut_tile);
        std::vector<int> tile_probe_slots((size_t)lut_tile);
        std::vector<int> tile_counts((size_t)lut_tile);
        std::vector<long long> tile_base_offsets((size_t)lut_tile);
        std::vector<const float*> tile_centroids((size_t)lut_tile);
        std::vector<const float*> tile_q_resids((size_t)lut_tile);
        double thread_residual_ms = 0.0;
        double thread_lut_build_ms = 0.0;
        double thread_adc_lookup_ms = 0.0;
        double thread_heap_push_ms = 0.0;

        const std::vector<int>& my_queries = buckets[(size_t)tid];
        for (int qi : my_queries) {
            TopKHeap heap;
            heap.init(stage1_k, heap_dists.data(), heap_idxs.data());

            const float* q_fp32 = h_query_fp32 + (size_t)qi * dim;
            const idx_t* cids = h_coarse_cluster_ids + (size_t)qi * n_probes;
            int pi = 0;
            while (pi < n_probes) {
                int valid = 0;
                while (pi < n_probes && valid < lut_tile) {
                    const int probe_slot = pi;
                    const int cid = static_cast<int>(cids[pi++]);
                    if (cid < 0 || cid >= n_total_clusters) continue;
                    const int count = h_cluster_counts[cid];
                    if (count <= 0) continue;
                    tile_cids[(size_t)valid] = cid;
                    tile_probe_slots[(size_t)valid] = probe_slot;
                    tile_counts[(size_t)valid] = count;
                    tile_base_offsets[(size_t)valid] = h_cluster_offsets[cid];
                    tile_centroids[(size_t)valid] = h_centroids + (size_t)cid * dim;
                    tile_q_resids[(size_t)valid] = q_resid_tile.data() + (size_t)valid * dim;
                    ++valid;
                }
                if (valid <= 0) continue;

                if (!precomputed_lut) {
                    const double residual_t0 = profile_breakdown ? Ops::now_ms() : 0.0;
                    for (int t = 0; t < valid; ++t) {
                        Ops::build_query_residual(
                            q_fp32,
                            tile_centroids[(size_t)t],
                            dim,
                            q_resid_tile.data() + (size_t)t * dim);
                    }
                    if (profile_breakdown) thread_residual_ms += Ops::now_ms() - residual_t0;

                    const double lut_t0 = profile_breakdown ? Ops::now_ms() : 0.0;
                    if (lut_tile > 1) {
                        Ops::build_lut_tile(
                            tile_q_resids.data(),
                            valid,
                            h_codebook,
                            pq_M,
                            pq_K,
                            d_sub,
                            lut_tile_buf.data());
                    } else {
                        Ops::build_lut(
                            tile_q_resids[0],
                            h_codebook,
                            pq_M,
                            pq_K,
                            d_sub,
                            lut.data());
                    }
                    if (profile_breakdown) thread_lut_build_ms += Ops::now_ms() - lut_t0;
                }

                for (int t = 0; t < valid; ++t) {
                    typename Ops::ScanBreakdown scan_breakdown;
                    typename Ops::ScanBreakdown* scan_breakdown_ptr =
                        profile_breakdown ? &scan_breakdown : nullptr;
                    const int cid = tile_cids[(size_t)t];
                    const int count = tile_counts[(size_t)t];
                    const long long base_off = tile_base_offsets[(size_t)t];
                    const float* lut_ptr = precomputed_lut
                        ? (precomputed_lut + ((size_t)qi * n_probes + (size_t)tile_probe_slots[(size_t)t]) * pq_M * pq_K)
                        : ((lut_tile > 1)
                            ? (lut_tile_buf.data() + (size_t)t * pq_M * pq_K)
                            : lut.data());
                    if (pq_storage.layout == CpuPqCodeLayout::Aosoa &&
                        pq_storage.aosoa &&
                        pq_storage.aosoa_offsets &&
                        pq_storage.aosoa_tile > 0) {
                        total_lookups += Ops::scan_pq_cluster_aosoa(
                            pq_storage.aosoa + pq_storage.aosoa_offsets[cid],
                            base_off,
                            count,
                            lut_ptr,
                            pq_M,
                            pq_K,
                            pq_storage.aosoa_tile,
                            heap,
                            scan_breakdown_ptr);
                    } else {
                        total_lookups += Ops::scan_pq_cluster(
                            pq_storage.row_major + (size_t)base_off * pq_M,
                            base_off,
                            count,
                            lut_ptr,
                            pq_M,
                            pq_K,
                            heap,
                            scan_breakdown_ptr);
                    }
                    if (profile_breakdown) {
                        thread_adc_lookup_ms += scan_breakdown.adc_lookup_ms;
                        thread_heap_push_ms += scan_breakdown.heap_push_ms;
                        total_heap_pushes += scan_breakdown.heap_pushes;
                    }
                }
            }

            stage1_sizes[(size_t)qi] = heap.size;
            std::memcpy(
                stage1_dists + (size_t)qi * stage1_k,
                heap_dists.data(),
                (size_t)heap.size * sizeof(float));
            std::memcpy(
                stage1_idxs + (size_t)qi * stage1_k,
                heap_idxs.data(),
                (size_t)heap.size * sizeof(int));
        }
        if (profile_breakdown) {
            max_residual_ms = std::max(max_residual_ms, thread_residual_ms);
            max_lut_build_ms = std::max(max_lut_build_ms, thread_lut_build_ms);
            max_adc_lookup_ms = std::max(max_adc_lookup_ms, thread_adc_lookup_ms);
            max_heap_push_ms = std::max(max_heap_push_ms, thread_heap_push_ms);
        }
    }

    stats.adc_lookups = total_lookups;
    stats.scanned_codes = pq_M > 0 ? total_lookups / pq_M : 0;
    stats.heap_pushes = total_heap_pushes;
    stats.residual_ms = max_residual_ms;
    stats.lut_build_ms = max_lut_build_ms;
    stats.adc_lookup_ms = max_adc_lookup_ms;
    stats.heap_push_ms = max_heap_push_ms;
    return stats;
}

}  // namespace query_balanced_scheduler_pq_detail
}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_QUERY_BALANCED_SCHEDULER_PQ_H
