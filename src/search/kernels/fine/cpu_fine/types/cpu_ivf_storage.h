#ifndef PARASELECT_CPU_IVF_STORAGE_H
#define PARASELECT_CPU_IVF_STORAGE_H

#include "utils/pch.h"

namespace paraselect {
namespace cpu_fine {

struct CpuIvfStorageView {
    CpuFineVectorLayout layout = CPU_FINE_LAYOUT_UNSPECIFIED;
    int dim = 0;
    int n_total_clusters = 0;
    int n_total_vectors = 0;

    const float* fp32_row_major = nullptr;
    const long long* row_major_offsets = nullptr;

    const float* fp32_aosoa = nullptr;
    const long long* aosoa_offsets = nullptr;
    int aosoa_tile = 0;

    const uint8_t* u8_row_major = nullptr;
    const long long* u8_row_major_offsets = nullptr;

    const float* fp32_paged = nullptr;
    const uint8_t* fp32_paged_alive = nullptr;
    const long long* fp32_paged_cluster_offsets = nullptr;
    const int* fp32_paged_cluster_page_offsets = nullptr;
    const int* fp32_paged_cluster_page_counts = nullptr;
    const int* fp32_paged_page_ids = nullptr;
    int fp32_paged_page_vecs = 0;

    const uint8_t* u8_paged = nullptr;
    const uint8_t* u8_paged_alive = nullptr;
    const long long* u8_paged_cluster_offsets = nullptr;
    const int* u8_paged_cluster_page_offsets = nullptr;
    const int* u8_paged_cluster_page_counts = nullptr;
    const int* u8_paged_page_ids = nullptr;
    int u8_paged_page_vecs = 0;

    const int* cluster_counts = nullptr;
};

inline CpuIvfStorageView make_fp32_row_major_storage(
    const float* data,
    const long long* offsets,
    const int* counts,
    int dim,
    int n_total_clusters,
    int n_total_vectors) {
    CpuIvfStorageView storage{};
    storage.layout = CPU_FINE_LAYOUT_FP32_ROW_MAJOR;
    storage.dim = dim;
    storage.n_total_clusters = n_total_clusters;
    storage.n_total_vectors = n_total_vectors;
    storage.fp32_row_major = data;
    storage.row_major_offsets = offsets;
    storage.cluster_counts = counts;
    return storage;
}

inline CpuIvfStorageView make_u8_row_major_storage(
    const uint8_t* data,
    const long long* offsets,
    const int* counts,
    int dim,
    int n_total_clusters,
    int n_total_vectors) {
    CpuIvfStorageView storage{};
    storage.layout = CPU_FINE_LAYOUT_U8_ROW_MAJOR;
    storage.dim = dim;
    storage.n_total_clusters = n_total_clusters;
    storage.n_total_vectors = n_total_vectors;
    storage.u8_row_major = data;
    storage.u8_row_major_offsets = offsets;
    storage.cluster_counts = counts;
    return storage;
}

inline CpuIvfStorageView make_fp32_paged_storage(
    const float* data,
    const long long* cluster_offsets,
    const int* cluster_page_offsets,
    const int* cluster_page_counts,
    const int* page_ids,
    int page_vecs,
    const int* counts,
    int dim,
    int n_total_clusters,
    int n_total_vectors,
    const uint8_t* alive = nullptr) {
    CpuIvfStorageView storage{};
    storage.layout = CPU_FINE_LAYOUT_FP32_PAGED;
    storage.dim = dim;
    storage.n_total_clusters = n_total_clusters;
    storage.n_total_vectors = n_total_vectors;
    storage.fp32_paged = data;
    storage.fp32_paged_alive = alive;
    storage.fp32_paged_cluster_offsets = cluster_offsets;
    storage.fp32_paged_cluster_page_offsets = cluster_page_offsets;
    storage.fp32_paged_cluster_page_counts = cluster_page_counts;
    storage.fp32_paged_page_ids = page_ids;
    storage.fp32_paged_page_vecs = page_vecs;
    storage.cluster_counts = counts;
    return storage;
}

inline CpuIvfStorageView make_u8_paged_storage(
    const uint8_t* data,
    const long long* cluster_offsets,
    const int* cluster_page_offsets,
    const int* cluster_page_counts,
    const int* page_ids,
    int page_vecs,
    const int* counts,
    int dim,
    int n_total_clusters,
    int n_total_vectors,
    const uint8_t* alive = nullptr) {
    CpuIvfStorageView storage{};
    storage.layout = CPU_FINE_LAYOUT_U8_PAGED;
    storage.dim = dim;
    storage.n_total_clusters = n_total_clusters;
    storage.n_total_vectors = n_total_vectors;
    storage.u8_paged = data;
    storage.u8_paged_alive = alive;
    storage.u8_paged_cluster_offsets = cluster_offsets;
    storage.u8_paged_cluster_page_offsets = cluster_page_offsets;
    storage.u8_paged_cluster_page_counts = cluster_page_counts;
    storage.u8_paged_page_ids = page_ids;
    storage.u8_paged_page_vecs = page_vecs;
    storage.cluster_counts = counts;
    return storage;
}

} // namespace cpu_fine
} // namespace paraselect

#endif // PARASELECT_CPU_IVF_STORAGE_H
