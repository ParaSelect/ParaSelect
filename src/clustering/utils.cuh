#ifndef IVF_CLUSTERING_UTILS_CUH
#define IVF_CLUSTERING_UTILS_CUH

#include "clustering/clustering.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ============================================================
// cuBLAS
// ============================================================

static inline void cublas_check(cublasStatus_t st, const char* msg) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublas error %d at %s\n", (int)st, msg);
        std::abort();
    }
}

// ============================================================
// GPU Kernels
// ============================================================

__global__ void kernel_count_clusters(const idx_t* __restrict__ assign,
                                      int* __restrict__ counts,
                                      int n, int k,
                                      unsigned int* __restrict__ invalid_counter);

__global__ void kernel_compute_positions(const idx_t* __restrict__ assign,
                                         idx_t* __restrict__ write_ptr,
                                         idx_t* __restrict__ out_pos,
                                         int n, int k,
                                         unsigned int* __restrict__ invalid_counter);

__global__ void kernel_scatter_perm(const idx_t* __restrict__ pos,
                                   idx_t* __restrict__ perm,
                                   int base, int curB, idx_t n,
                                   unsigned int* __restrict__ oob_counter);

// ============================================================
// GPU Permutation Builder
// ============================================================

void gpu_build_permutation_by_cluster(
    int n,
    int k,
    const idx_t* d_assign,
    idx_t* h_perm_out,
    ClusterInfo* h_cluster_info,
    int device_id = 0,
    int B = 1 << 20,
    cudaStream_t stream = 0
);

// ============================================================
// Host  offsets
// ============================================================

static inline void build_offsets_host(const std::vector<int>& counts,
                                      std::vector<idx_t>& offsets,
                                      idx_t& total) {
    const int k = (int)counts.size();
    offsets.resize((size_t)k + 1u);
    total = 0;
    for (int c = 0; c < k; ++c) {
        offsets[c] = total;
        total += counts[c];
    }
    offsets[(size_t)k] = total;
}

// ============================================================
// GPU ClusterInfo Builder (no perm)
// ============================================================

/**
 * GPU  ClusterInfo(counts + offsets), permutation .
 *
 *  gpu_build_permutation_by_cluster:
 * -  d_perm[n], n×sizeof(int)
 * -  perm, ClusterInfo
 * -  CUB DeviceScan::ExclusiveSum  offsets
 *
 * @param n
 * @param k            cluster
 * @param d_assign      assign  [n]
 * @param h_cluster_info :host  ClusterInfo( malloc counts/offsets)
 * @param device_id    GPU  ID
 * @param stream       CUDA stream
 */
void gpu_build_cluster_info(
    int n,
    int k,
    const idx_t* d_assign,
    ClusterInfo* h_cluster_info,
    int device_id = 0,
    cudaStream_t stream = 0
);

// ============================================================
// Utility Functions
// ============================================================

void free_cluster_info(ClusterInfo* info, bool is_device = false);

#endif // IVF_CLUSTERING_UTILS_CUH
