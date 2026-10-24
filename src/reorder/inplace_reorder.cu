// reorder/inplace_reorder.cu
#include "reorder/reorder.cuh"
#include "clustering/utils.cuh"
#include "utils/utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ============================================================
// In-Place Reorder Functions Implementation (O(1) extra space)
// ============================================================

void cpu_reorder_vectors_inplace(
    float* h_data,
    const idx_t* h_perm,
    int n, int dim
) {
    if (!h_data || !h_perm || n <= 0 || dim <= 0) {
        fprintf(stderr, "[reorder] ERROR: Invalid parameters for cpu_reorder_vectors_inplace\n");
        return;
    }

    std::vector<char> visited(n, 0);

    for (int i = 0; i < n; ++i) {
        if (visited[i]) continue;

        int cur = i;

        if (h_perm[cur] == cur) {
            visited[cur] = 1;
            continue;
        }

        std::vector<float> temp(dim);
        std::memcpy(temp.data(), h_data + (size_t)cur * dim, sizeof(float) * dim);

        int start = cur;
        while (!visited[cur]) {
            visited[cur] = 1;
            int next = h_perm[cur];

            if (next == start) {
                std::memcpy(h_data + (size_t)cur * dim, temp.data(), sizeof(float) * dim);
                break;
            }

            std::memcpy(h_data + (size_t)cur * dim,
                       h_data + (size_t)next * dim,
                       sizeof(float) * dim);
            cur = next;
        }
    }
}

void cpu_reorder_indices_inplace(
    int* h_indices,
    const idx_t* h_perm,
    int n
) {
    if (!h_indices || !h_perm || n <= 0) {
        fprintf(stderr, "[reorder] ERROR: Invalid parameters for cpu_reorder_indices_inplace\n");
        return;
    }

    std::vector<char> visited(n, 0);

    for (int i = 0; i < n; ++i) {
        if (visited[i]) continue;

        int cur = i;

        if (h_perm[cur] == cur) {
            visited[cur] = 1;
            continue;
        }

        int temp = h_indices[cur];
        int start = cur;

        while (!visited[cur]) {
            visited[cur] = 1;
            int next = h_perm[cur];

            if (next == start) {
                h_indices[cur] = temp;
                break;
            }

            h_indices[cur] = h_indices[next];
            cur = next;
        }
    }
}
