#include "search/kernels/fine/cpu_fine/runners/fp32_aosoa_runners.h"

#include "search/kernels/fine/cpu_fine/executor/cpu_fine_batch_view.h"
#include "search/kernels/fine/cpu_fine/kernels/kernel_aosoa.h"
#include "search/kernels/fine/cpu_fine/kernels/kernel_aosoa_v1.h"
#include "search/kernels/fine/cpu_fine/schedulers/cluster_based_v1_aosoa.h"

#include <stdexcept>

namespace paraselect {
namespace cpu_fine {
namespace {

static constexpr int kAosoaTile = Fp32AosoaTile16Kernel::kTile;

void require_fp32_aosoa_storage(const CpuIvfStorageView* storage) {
    if (!storage || storage->layout != CPU_FINE_LAYOUT_FP32_AOSOA ||
        !storage->fp32_aosoa || !storage->aosoa_offsets ||
        !storage->row_major_offsets || !storage->cluster_counts) {
        throw std::invalid_argument("CPU fine runner requires fp32 AoSoA storage");
    }
    if (storage->aosoa_tile != kAosoaTile) {
        throw std::invalid_argument("CPU fine AoSoA runner currently requires tile=16");
    }
}

using Fp32AosoaTile16EvaluatorV0 = Fp32AosoaTile16Kernel;
using Fp32AosoaTile16EvaluatorV1 = Fp32AosoaTile16KernelV1;

template <typename Evaluator>
long long run_cluster_scratch(const CpuFineRunArgs& args) {
    require_fp32_aosoa_storage(args.storage);
    const CpuFineBatchView* batch = args.batch;
    return cluster_based_v1_aosoa_detail::cluster_based_v1_aosoa_with_scratch<
        Evaluator, kAosoaTile>(
        args.storage->fp32_aosoa,
        args.storage->aosoa_offsets,
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
long long run_cluster_metadata(const CpuFineRunArgs& args) {
    require_fp32_aosoa_storage(args.storage);
    const CpuFineBatchView* batch = args.batch;
    return cluster_based_v1_aosoa_detail::cluster_based_v1_aosoa_with_metadata<
        Evaluator, kAosoaTile>(
        args.storage->fp32_aosoa,
        args.storage->aosoa_offsets,
        args.storage->row_major_offsets,
        args.storage->cluster_counts,
        args.query,
        batch->h_cluster_query_offset_batch,
        batch->h_cluster_query_data_batch,
        batch->h_active_cluster_ids_batch,
        batch->active_cluster_count,
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
long long run_cluster_compact_metadata(const CpuFineRunArgs& args) {
    require_fp32_aosoa_storage(args.storage);
    const CpuFineBatchView* batch = args.batch;
    return cluster_based_v1_aosoa_detail::cluster_based_v1_aosoa_with_compact_metadata<
        Evaluator, kAosoaTile>(
        args.storage->fp32_aosoa,
        args.storage->aosoa_offsets,
        args.storage->row_major_offsets,
        args.storage->cluster_counts,
        args.query,
        batch->h_cluster_query_offset_batch,
        batch->h_cluster_query_data_batch,
        batch->h_active_cluster_ids_batch,
        batch->active_cluster_count,
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

}  // namespace

long long fp32_aosoa_v0_cluster_dynamic_scratch(const CpuFineRunArgs& args) {
    return run_cluster_scratch<Fp32AosoaTile16EvaluatorV0>(args);
}

long long fp32_aosoa_v0_cluster_dynamic_metadata(const CpuFineRunArgs& args) {
    return run_cluster_metadata<Fp32AosoaTile16EvaluatorV0>(args);
}

long long fp32_aosoa_v0_cluster_dynamic_compact_metadata(const CpuFineRunArgs& args) {
    return run_cluster_compact_metadata<Fp32AosoaTile16EvaluatorV0>(args);
}

long long fp32_aosoa_v1_cluster_dynamic_scratch(const CpuFineRunArgs& args) {
    return run_cluster_scratch<Fp32AosoaTile16EvaluatorV1>(args);
}

long long fp32_aosoa_v1_cluster_dynamic_metadata(const CpuFineRunArgs& args) {
    return run_cluster_metadata<Fp32AosoaTile16EvaluatorV1>(args);
}

long long fp32_aosoa_v1_cluster_dynamic_compact_metadata(const CpuFineRunArgs& args) {
    return run_cluster_compact_metadata<Fp32AosoaTile16EvaluatorV1>(args);
}

}  // namespace cpu_fine
}  // namespace paraselect
