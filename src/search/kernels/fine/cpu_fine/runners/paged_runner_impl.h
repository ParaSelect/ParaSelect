#ifndef PARASELECT_PAGED_RUNNER_IMPL_H
#define PARASELECT_PAGED_RUNNER_IMPL_H

#include "search/kernels/fine/cpu_fine/executor/cpu_fine_batch_view.h"
#include "search/kernels/fine/cpu_fine/schedulers/cluster_based_v1_paged.h"

#include <stdexcept>

namespace paraselect {
namespace cpu_fine {
namespace paged_runner_impl {

template <typename Scalar>
struct PagedStorageAccess;

template <>
struct PagedStorageAccess<uint8_t> {
    static constexpr CpuFineVectorLayout layout = CPU_FINE_LAYOUT_U8_PAGED;
    static const uint8_t* pages(const CpuIvfStorageView& s) { return s.u8_paged; }
    static const uint8_t* alive(const CpuIvfStorageView& s) { return s.u8_paged_alive; }
    static const long long* cluster_offsets(const CpuIvfStorageView& s) { return s.u8_paged_cluster_offsets; }
    static const int* cluster_page_offsets(const CpuIvfStorageView& s) { return s.u8_paged_cluster_page_offsets; }
    static const int* cluster_page_counts(const CpuIvfStorageView& s) { return s.u8_paged_cluster_page_counts; }
    static const int* page_ids(const CpuIvfStorageView& s) { return s.u8_paged_page_ids; }
    static int page_vecs(const CpuIvfStorageView& s) { return s.u8_paged_page_vecs; }
};

template <>
struct PagedStorageAccess<float> {
    static constexpr CpuFineVectorLayout layout = CPU_FINE_LAYOUT_FP32_PAGED;
    static const float* pages(const CpuIvfStorageView& s) { return s.fp32_paged; }
    static const uint8_t* alive(const CpuIvfStorageView& s) { return s.fp32_paged_alive; }
    static const long long* cluster_offsets(const CpuIvfStorageView& s) { return s.fp32_paged_cluster_offsets; }
    static const int* cluster_page_offsets(const CpuIvfStorageView& s) { return s.fp32_paged_cluster_page_offsets; }
    static const int* cluster_page_counts(const CpuIvfStorageView& s) { return s.fp32_paged_cluster_page_counts; }
    static const int* page_ids(const CpuIvfStorageView& s) { return s.fp32_paged_page_ids; }
    static int page_vecs(const CpuIvfStorageView& s) { return s.fp32_paged_page_vecs; }
};

template <typename Scalar>
inline void require_paged_storage(const CpuIvfStorageView* storage) {
    using Access = PagedStorageAccess<Scalar>;
    if (!storage || storage->layout != Access::layout || !Access::pages(*storage) ||
        !Access::cluster_offsets(*storage) || !Access::cluster_page_offsets(*storage) ||
        !Access::cluster_page_counts(*storage) || !Access::page_ids(*storage) ||
        Access::page_vecs(*storage) <= 0 || !storage->cluster_counts) {
        throw std::invalid_argument("CPU fine runner requires matching paged storage");
    }
}

template <typename Scalar>
inline const Scalar* typed_query(const CpuFineRunArgs& args) {
    return reinterpret_cast<const Scalar*>(args.query);
}

template <typename Scalar, typename Evaluator>
long long run_scratch(const CpuFineRunArgs& args) {
    require_paged_storage<Scalar>(args.storage);
    const CpuFineBatchView* batch = args.batch;
    return cluster_based_v1_paged_detail::cluster_based_v1_paged_with_scratch<Scalar, Evaluator>(
        PagedStorageAccess<Scalar>::pages(*args.storage),
        PagedStorageAccess<Scalar>::alive(*args.storage),
        PagedStorageAccess<Scalar>::cluster_offsets(*args.storage),
        args.storage->cluster_counts,
        PagedStorageAccess<Scalar>::cluster_page_offsets(*args.storage),
        PagedStorageAccess<Scalar>::cluster_page_counts(*args.storage),
        PagedStorageAccess<Scalar>::page_ids(*args.storage),
        PagedStorageAccess<Scalar>::page_vecs(*args.storage),
        typed_query<Scalar>(args),
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

template <typename Scalar, typename Evaluator>
long long run_metadata(const CpuFineRunArgs& args) {
    require_paged_storage<Scalar>(args.storage);
    const CpuFineBatchView* batch = args.batch;
    return cluster_based_v1_paged_detail::cluster_based_v1_paged_with_metadata<Scalar, Evaluator>(
        PagedStorageAccess<Scalar>::pages(*args.storage),
        PagedStorageAccess<Scalar>::alive(*args.storage),
        PagedStorageAccess<Scalar>::cluster_offsets(*args.storage),
        args.storage->cluster_counts,
        PagedStorageAccess<Scalar>::cluster_page_offsets(*args.storage),
        PagedStorageAccess<Scalar>::cluster_page_counts(*args.storage),
        PagedStorageAccess<Scalar>::page_ids(*args.storage),
        PagedStorageAccess<Scalar>::page_vecs(*args.storage),
        typed_query<Scalar>(args),
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

template <typename Scalar, typename Evaluator>
long long run_compact_metadata(const CpuFineRunArgs& args) {
    require_paged_storage<Scalar>(args.storage);
    const CpuFineBatchView* batch = args.batch;
    return cluster_based_v1_paged_detail::cluster_based_v1_paged_with_compact_metadata<Scalar, Evaluator>(
        PagedStorageAccess<Scalar>::pages(*args.storage),
        PagedStorageAccess<Scalar>::alive(*args.storage),
        PagedStorageAccess<Scalar>::cluster_offsets(*args.storage),
        args.storage->cluster_counts,
        PagedStorageAccess<Scalar>::cluster_page_offsets(*args.storage),
        PagedStorageAccess<Scalar>::cluster_page_counts(*args.storage),
        PagedStorageAccess<Scalar>::page_ids(*args.storage),
        PagedStorageAccess<Scalar>::page_vecs(*args.storage),
        typed_query<Scalar>(args),
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

} // namespace paged_runner_impl
} // namespace cpu_fine
} // namespace paraselect

#endif // PARASELECT_PAGED_RUNNER_IMPL_H
