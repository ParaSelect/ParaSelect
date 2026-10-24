#include "search/kernels/fine/cpu_fine/cpu_fine_layout_aosoa.h"

#include <omp.h>

#include <algorithm>
#include <cstring>

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
    int num_threads)
{
    if (num_threads <= 0) num_threads = omp_get_max_threads();

    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int cid = 0; cid < n_total_clusters; ++cid) {
        const long long begin = h_aosoa_offsets[cid];
        const long long end = h_aosoa_offsets[cid + 1];
        if (end > begin) {
            std::memset(h_base_aosoa_out + begin, 0, (size_t)(end - begin) * sizeof(float));
        }
    }

    #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
    for (int cid = 0; cid < n_total_clusters; ++cid) {
        const int count = h_cluster_counts[cid];
        if (count <= 0) continue;

        const long long row_base = h_cluster_offsets[cid];
        float* dst_cluster = h_base_aosoa_out + h_aosoa_offsets[cid];
        const int groups = (count + tile - 1) / tile;
        for (int g = 0; g < groups; ++g) {
            const int local_begin = g * tile;
            const int local_end = std::min(local_begin + tile, count);
            float* dst_group = dst_cluster + (long long)g * dim * tile;
            for (int lane_idx = local_begin; lane_idx < local_end; ++lane_idx) {
                const int lane = lane_idx - local_begin;
                const idx_t src_idx = h_reordered_indices[row_base + lane_idx];
                const float* src = h_base_rowmajor + (size_t)src_idx * dim;
                for (int d = 0; d < dim; ++d) {
                    dst_group[(long long)d * tile + lane] = src[d];
                }
            }
        }
    }
}

}  // namespace cpu_fine
}  // namespace paraselect
