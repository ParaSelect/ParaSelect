#ifndef PARASELECT_IVFRESPQ_PQ_STORAGE_H
#define PARASELECT_IVFRESPQ_PQ_STORAGE_H

#include "utils/pch.h"

namespace paraselect {
namespace cpu_fine {

enum class CpuPqCodeLayout {
    RowMajor,
    Aosoa,
};

struct CpuPqCodeStorageView {
    CpuPqCodeLayout layout = CpuPqCodeLayout::RowMajor;
    int pq_m = 0;
    int n_total_clusters = 0;
    int n_total_vectors = 0;

    const uint8_t* row_major = nullptr;
    const long long* row_major_offsets = nullptr;

    const uint8_t* aosoa = nullptr;
    const long long* aosoa_offsets = nullptr;
    int aosoa_tile = 0;

    const int* cluster_counts = nullptr;
};

inline CpuPqCodeStorageView make_pq_row_major_storage(
    const uint8_t* codes,
    const long long* offsets,
    const int* counts,
    int pq_m,
    int n_total_clusters,
    int n_total_vectors)
{
    CpuPqCodeStorageView storage{};
    storage.layout = CpuPqCodeLayout::RowMajor;
    storage.pq_m = pq_m;
    storage.n_total_clusters = n_total_clusters;
    storage.n_total_vectors = n_total_vectors;
    storage.row_major = codes;
    storage.row_major_offsets = offsets;
    storage.cluster_counts = counts;
    return storage;
}

inline CpuPqCodeStorageView make_pq_aosoa_storage(
    const uint8_t* aosoa,
    const long long* aosoa_offsets,
    const uint8_t* row_major,
    const long long* row_major_offsets,
    const int* counts,
    int tile,
    int pq_m,
    int n_total_clusters,
    int n_total_vectors)
{
    CpuPqCodeStorageView storage{};
    storage.layout = CpuPqCodeLayout::Aosoa;
    storage.pq_m = pq_m;
    storage.n_total_clusters = n_total_clusters;
    storage.n_total_vectors = n_total_vectors;
    storage.aosoa = aosoa;
    storage.aosoa_offsets = aosoa_offsets;
    storage.aosoa_tile = tile;
    storage.row_major = row_major;
    storage.row_major_offsets = row_major_offsets;
    storage.cluster_counts = counts;
    return storage;
}

}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_CPU_PQ_STORAGE_H
