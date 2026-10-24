#ifndef PARASELECT_CPU_IVF_STORAGE_AOSOA_H
#define PARASELECT_CPU_IVF_STORAGE_AOSOA_H

#include "search/kernels/fine/cpu_fine/types/cpu_ivf_storage.h"

namespace paraselect {
namespace cpu_fine {

inline CpuIvfStorageView make_fp32_aosoa_storage(
    const float* data,
    const long long* aosoa_offsets,
    const long long* row_major_offsets,
    const int* counts,
    int tile,
    int dim,
    int n_total_clusters,
    int n_total_vectors)
{
    CpuIvfStorageView storage{};
    storage.layout = CPU_FINE_LAYOUT_FP32_AOSOA;
    storage.dim = dim;
    storage.n_total_clusters = n_total_clusters;
    storage.n_total_vectors = n_total_vectors;
    storage.fp32_aosoa = data;
    storage.aosoa_offsets = aosoa_offsets;
    storage.aosoa_tile = tile;
    storage.row_major_offsets = row_major_offsets;
    storage.cluster_counts = counts;
    return storage;
}

}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_CPU_IVF_STORAGE_AOSOA_H
