#ifndef IVF_REORDER_REORDER_CUH
#define IVF_REORDER_REORDER_CUH

#include "clustering/clustering.cuh"
#include "clustering/utils.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

// ============================================================
// CPU Reorder Functions
// ============================================================

/**
 *  permutation ( CPU )
 * out[p] = in[perm[p]]
 */
void cpu_reorder_vectors_by_permutation(
    const float* h_data_in,
    const idx_t*   h_perm,
    float*       h_data_out,
    int n, int dim
);

/**
 *  permutation ( CPU )
 * out[p] = in[perm[p]]
 */
void cpu_reorder_indices_by_permutation(
    const idx_t* h_indices_in,
    const idx_t* h_perm,
    idx_t*       h_indices_out,
    int n
);

struct NumaReorderConfig {
    int numa_nodes = 0;      // 0: auto-detect
    int threads = 0;         // 0: default worker thread count
    bool pin_threads = true;
};

/**
 * NUMA-aware out-of-place vector reorder.
 *
 * This path writes cluster-contiguous output ranges from threads pinned to the
 * owning NUMA node, so Linux first-touch places output pages on that node.
 * It only reorders vector data; reordered indices should keep using
 * cpu_reorder_indices_by_permutation.
 */
void cpu_reorder_vectors_by_cluster_ranges_numa(
    const float* h_data_in,
    const idx_t* h_perm,
    float* h_data_out,
    const idx_t* h_cluster_offsets,
    const int* h_cluster_counts,
    int n,
    int dim,
    int n_clusters,
    const NumaReorderConfig& config = {}
);

// ============================================================
// In-Place Reorder Functions (O(1) extra space)
// ============================================================

/**
 * In-place (Cycle Leader ,O(1) )
 *
 * :
 * 1.  i, perm[i] != i,
 * 2. : data[i], data[i] = data[perm[i]],
 * 3. ,
 *
 * @param h_data  [in/out] ,
 * @param h_perm  [in] permutation ,
 *                h_perm[dst] = src  dst  src
 * @param n
 * @param dim
 */
void cpu_reorder_vectors_inplace(
    float* h_data,
    const idx_t* h_perm,
    int n, int dim
);

/**
 * In-place (Cycle Leader ,O(1) )
 *
 * @param h_indices  [in/out] ,
 * @param h_perm     [in] permutation
 * @param n
 */
void cpu_reorder_indices_inplace(
    int* h_indices,
    const idx_t* h_perm,
    int n
);

// ============================================================
// Cluster-based In-Place Reorder (American Flag / Bucket Permutation)
// ============================================================

/**
 *  Cluster (American Flag / In-Place Bucket Permutation)
 *
 * :
 * -  in-place, perm[n]
 * -  visited[n] flags
 * -  O(k),k  cluster
 * -  " cluster "
 *
 * :
 * 1.  counts[c]: cluster
 * 2.  offsets[c]: cluster
 * 3.  next_free[c]: cluster
 * 4.  swap
 *
 * @param h_data     [in/out] ,
 * @param h_cluster_id [in]  cluster_id [n]
 * @param n
 * @param dim
 * @param k          cluster
 */
void cpu_cluster_reorder_vectors_inplace(
    float* h_data,
    idx_t* h_cluster_id,
    int n, int dim, int k
);

// Native SIFT storage uses the same permutation without expanding to FP32.
void cpu_cluster_reorder_vectors_inplace(
    uint8_t* h_data, idx_t* h_cluster_id, int n, int dim, int k);

/**
 *  Cluster ()
 *
 * @param h_indices  [in/out] ,
 * @param h_cluster_id [in/out]  cluster_id [n],
 * @param n
 * @param k          cluster
 */
void cpu_cluster_reorder_indices_inplace(
    idx_t* h_indices,
    idx_t* h_cluster_id,
    int n, int k
);

#endif // IVF_REORDER_REORDER_CUH
