// reorder/standard_reorder.cu
#include "reorder/reorder.cuh"
#include "clustering/utils.cuh"
#include "utils/utils.cuh"
#include "cpu/utils/cpu_affinity.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ============================================================
// CPU Reorder Functions Implementation (Out-of-Place)
// ============================================================

void cpu_reorder_vectors_by_permutation(
    const float* h_data_in,
    const idx_t*   h_perm,
    float*       h_data_out,
    int n, int dim
) {
    if (!h_data_in || !h_perm || !h_data_out || n <= 0 || dim <= 0) {
        fprintf(stderr, "[reorder] ERROR: Invalid parameters for cpu_reorder_vectors_by_permutation\n");
        return;
    }

    paraselect::parallel_chunk_on_physical_cores(n,
        [=](int start, int end, int core_id) {
            (void)core_id;
            for (int p = start; p < end; ++p) {
                int i = h_perm[p];
                if (i < 0 || i >= n) continue;
                std::memcpy(h_data_out + (size_t)p * dim,
                            h_data_in  + (size_t)i * dim,
                            sizeof(float) * (size_t)dim);
            }
        });
}

void cpu_reorder_indices_by_permutation(
    const idx_t* h_indices_in,
    const idx_t* h_perm,
    idx_t*       h_indices_out,
    int n
) {
    if (!h_indices_in || !h_perm || !h_indices_out || n <= 0) {
        fprintf(stderr, "[reorder] ERROR: Invalid parameters for cpu_reorder_indices_by_permutation\n");
        return;
    }

    paraselect::parallel_chunk_on_physical_cores(n,
        [=](int start, int end, int core_id) {
            (void)core_id;
            for (int p = start; p < end; ++p) {
                int i = h_perm[p];
                if (i < 0 || i >= n) continue;
                h_indices_out[p] = h_indices_in[i];
            }
        });
}
