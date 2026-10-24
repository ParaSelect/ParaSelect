#ifndef PARASELECT_CPU_FINE_LAYOUT_AOSOA_H
#define PARASELECT_CPU_FINE_LAYOUT_AOSOA_H

#include "utils/pch.h"

namespace paraselect {
namespace cpu_fine {

void reorder_base_to_aosoa(
    const float* h_base_rowmajor,
    const idx_t* h_reordered_indices,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const long long* h_aosoa_offsets,
    int n_total_clusters,
    int dim,
    int tile,
    float* h_base_aosoa_out,
    int num_threads);

}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_CPU_FINE_LAYOUT_AOSOA_H
