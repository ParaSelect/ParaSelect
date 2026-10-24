#ifndef PARASELECT_DATASET_AOSOA_CUH
#define PARASELECT_DATASET_AOSOA_CUH

#include "dataset/ownership_helper.h"
#include "search/kernels/fine/cpu_fine/cpu_fine_layout.h"
#include "search/kernels/fine/cpu_fine/cpu_fine_layout_aosoa.h"
#include "utils/pch.h"

#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <stdexcept>
#include <vector>

namespace paraselect {
namespace dataset {

struct DatasetAosoaStorage {
    float* data = nullptr;
    long long* offsets = nullptr;
    int tile = 0;
    paraselect::ownership::BufferOwner data_owner = paraselect::ownership::BufferOwner::None;
    paraselect::ownership::BufferOwner offsets_owner = paraselect::ownership::BufferOwner::None;

    bool is_valid() const {
        return data != nullptr && offsets != nullptr && tile > 0;
    }

    size_t size(int n_clusters) const {
        if (!offsets || n_clusters <= 0) return 0;
        return (size_t)offsets[n_clusters];
    }

    void release() {
        if (data) {
            paraselect::ownership::free_raw(data, data_owner);
            data = nullptr;
        }
        if (offsets) {
            paraselect::ownership::free_raw(offsets, offsets_owner);
            offsets = nullptr;
        }
        tile = 0;
        data_owner = paraselect::ownership::BufferOwner::None;
        offsets_owner = paraselect::ownership::BufferOwner::None;
    }
};

inline void build_direct_aosoa_from_reordered_indices(
    DatasetAosoaStorage* storage,
    const float* h_base_rowmajor,
    const idx_t* h_reordered_indices,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    int n_clusters,
    int dim,
    int tile,
    int num_threads)
{
    if (!storage || !h_base_rowmajor || !h_reordered_indices ||
        !h_cluster_offsets || !h_cluster_counts || n_clusters <= 0 ||
        dim <= 0 || tile <= 0) {
        throw std::invalid_argument("build_direct_aosoa_from_reordered_indices: invalid arguments");
    }

    using Owner = paraselect::ownership::BufferOwner;
    using Handle = paraselect::ownership::BufferHandle;

    const size_t offsets_bytes = (size_t)(n_clusters + 1) * sizeof(long long);
    Handle offsets_buf(memalign(64, offsets_bytes), offsets_bytes, Owner::StdMalloc);
    if (!offsets_buf.data) throw std::bad_alloc();

    const long long total = cpu_fine::plan_aosoa_layout(
        h_cluster_counts, n_clusters, dim, tile, static_cast<long long*>(offsets_buf.data));

    const size_t data_bytes = (size_t)total * sizeof(float);
    Handle data_buf(memalign(64, data_bytes), data_bytes, Owner::StdMalloc);
    if (!data_buf.data) throw std::bad_alloc();

    cpu_fine::reorder_base_to_aosoa(
        h_base_rowmajor,
        h_reordered_indices,
        h_cluster_offsets,
        h_cluster_counts,
        static_cast<long long*>(offsets_buf.data),
        n_clusters,
        dim,
        tile,
        static_cast<float*>(data_buf.data),
        num_threads);
    storage->release();
    paraselect::ownership::take_to(offsets_buf, storage->offsets, storage->offsets_owner);
    paraselect::ownership::take_to(data_buf, storage->data, storage->data_owner);
    storage->tile = tile;
}

inline void build_aosoa_from_reordered_rowmajor(
    DatasetAosoaStorage* storage,
    const float* reordered_rowmajor,
    const idx_t* cluster_offsets,
    const int* cluster_counts,
    int n_clusters,
    int dim,
    int tile,
    int num_threads)
{
    if (!storage || !reordered_rowmajor || !cluster_offsets || !cluster_counts ||
        n_clusters <= 0 || dim <= 0 || tile <= 0) {
        throw std::invalid_argument("build_aosoa_from_reordered_rowmajor: invalid arguments");
    }
    std::vector<long long> offsets_ll((size_t)n_clusters + 1u);
    offsets_ll[0] = static_cast<long long>(cluster_offsets[0]);
    for (int i = 0; i < n_clusters; ++i) {
        offsets_ll[(size_t)i] = static_cast<long long>(cluster_offsets[i]);
    }
    offsets_ll[(size_t)n_clusters] = static_cast<long long>(cluster_offsets[n_clusters]);

    using Owner = paraselect::ownership::BufferOwner;
    using Handle = paraselect::ownership::BufferHandle;
    const size_t offsets_bytes = (size_t)(n_clusters + 1) * sizeof(long long);
    Handle offsets_buf(memalign(64, offsets_bytes), offsets_bytes, Owner::StdMalloc);
    if (!offsets_buf.data) throw std::bad_alloc();
    const long long total = cpu_fine::plan_aosoa_layout(
        cluster_counts, n_clusters, dim, tile, static_cast<long long*>(offsets_buf.data));
    const size_t data_bytes = (size_t)total * sizeof(float);
    Handle data_buf(memalign(64, data_bytes), data_bytes, Owner::StdMalloc);
    if (!data_buf.data) throw std::bad_alloc();
    cpu_fine::convert_rowmajor_to_aosoa(
        reordered_rowmajor,
        offsets_ll.data(),
        cluster_counts,
        static_cast<long long*>(offsets_buf.data),
        n_clusters,
        dim,
        tile,
        static_cast<float*>(data_buf.data),
        num_threads);
    storage->release();
    paraselect::ownership::take_to(offsets_buf, storage->offsets, storage->offsets_owner);
    paraselect::ownership::take_to(data_buf, storage->data, storage->data_owner);
    storage->tile = tile;
}

inline void build_direct_aosoa_from_reordered_indices(
    DatasetAosoaStorage* storage,
    const float* h_base_rowmajor,
    const idx_t* h_reordered_indices,
    const idx_t* h_cluster_offsets,
    const int* h_cluster_counts,
    int n_clusters,
    int dim,
    int tile,
    int num_threads)
{
    std::vector<long long> offsets_ll((size_t)n_clusters + 1u);
    for (int i = 0; i <= n_clusters; ++i) {
        offsets_ll[(size_t)i] = static_cast<long long>(h_cluster_offsets[i]);
    }
    build_direct_aosoa_from_reordered_indices(
        storage,
        h_base_rowmajor,
        h_reordered_indices,
        offsets_ll.data(),
        h_cluster_counts,
        n_clusters,
        dim,
        tile,
        num_threads);
}

}  // namespace dataset
}  // namespace paraselect

#endif  // PARASELECT_DATASET_AOSOA_CUH
