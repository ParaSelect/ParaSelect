#ifndef PARASELECT_FP32_ROWMAJOR_NUMA_RUNNER_IMPL_H
#define PARASELECT_FP32_ROWMAJOR_NUMA_RUNNER_IMPL_H

#include "search/kernels/fine/cpu_fine/executor/cpu_fine_batch_view.h"
#include "search/kernels/fine/cpu_fine/schedulers/cluster_based_v1_rowmajor_numa.h"

#include <stdexcept>

namespace paraselect {
namespace cpu_fine {
namespace fp32_rowmajor_numa_runner_impl {

inline void require_fp32_rowmajor_storage(const CpuIvfStorageView* storage) {
    if (!storage || storage->layout != CPU_FINE_LAYOUT_FP32_ROW_MAJOR ||
        !storage->fp32_row_major || !storage->row_major_offsets ||
        !storage->cluster_counts) {
        throw std::invalid_argument("CPU fine runner requires fp32 row-major storage");
    }
}


template <typename Evaluator>
long long run_scratch(const CpuFineRunArgs& args)
{
    require_fp32_rowmajor_storage(args.storage);
    const CpuFineBatchView* batch = args.batch;
    return cluster_based_v1_rowmajor_numa_detail::cluster_based_v1_rowmajor_numa_with_scratch<Evaluator>(
        args.storage->fp32_row_major,
        args.storage->row_major_offsets,
        args.storage->cluster_counts,
        args.query,
        args.top_nprobe,
        batch->n_query,
        args.storage->dim,
        args.storage->n_total_clusters,
        args.n_probes,
        args.topk,
        args.distance_mode,
        args.num_threads,
        batch->h_topk_local_idx_batch,
        batch->h_topk_dist_out,
        batch->h_thread_dists,
        batch->h_thread_idxs,
        batch->h_thread_sizes,
        batch->h_thread_touched);
}

template <typename Evaluator>
long long run_metadata(const CpuFineRunArgs& args)
{
    require_fp32_rowmajor_storage(args.storage);
    const CpuFineBatchView* batch = args.batch;
    return cluster_based_v1_rowmajor_numa_detail::cluster_based_v1_rowmajor_numa_with_metadata<Evaluator>(
        args.storage->fp32_row_major,
        args.storage->row_major_offsets,
        args.storage->cluster_counts,
        args.query,
        batch->h_cluster_query_offset_batch,
        batch->h_cluster_query_data_batch,
        batch->h_active_cluster_ids_batch,
        batch->active_cluster_count,
        args.numa_cluster_boundaries,
        args.numa_cluster_boundary_count,
        batch->n_query,
        args.storage->dim,
        args.storage->n_total_clusters,
        args.n_probes,
        args.topk,
        args.distance_mode,
        args.num_threads,
        batch->h_topk_local_idx_batch,
        batch->h_topk_dist_out,
        batch->h_thread_dists,
        batch->h_thread_idxs,
        batch->h_thread_sizes,
        batch->h_thread_touched);
}

template <typename Evaluator>
long long run_compact_metadata(const CpuFineRunArgs& args)
{
    require_fp32_rowmajor_storage(args.storage);
    const CpuFineBatchView* batch = args.batch;
    return cluster_based_v1_rowmajor_numa_detail::cluster_based_v1_rowmajor_numa_with_compact_metadata<Evaluator>(
        args.storage->fp32_row_major,
        args.storage->row_major_offsets,
        args.storage->cluster_counts,
        args.query,
        batch->h_cluster_query_offset_batch,
        batch->h_cluster_query_data_batch,
        batch->h_active_cluster_ids_batch,
        batch->active_cluster_count,
        args.numa_cluster_boundaries,
        args.numa_cluster_boundary_count,
        batch->n_query,
        args.storage->dim,
        args.storage->n_total_clusters,
        args.n_probes,
        args.topk,
        args.distance_mode,
        args.num_threads,
        batch->h_topk_local_idx_batch,
        batch->h_topk_dist_out,
        batch->h_thread_dists,
        batch->h_thread_idxs,
        batch->h_thread_sizes,
        batch->h_thread_touched);
}

}  // namespace fp32_rowmajor_numa_runner_impl
}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_FP32_ROWMAJOR_NUMA_RUNNER_IMPL_H
