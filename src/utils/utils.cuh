#ifndef PARASELECT_CUDA_UTILS_CUH
#define PARASELECT_CUDA_UTILS_CUH

#include "pch.h"
#include "utils/cuda_fill.cuh"
#include "utils/cuda_scan.cuh"

/**
 * Kernel: clusterquery
 *
 * :
 * - gridDim.x = n_query (blockquery)
 * - blockDim.x = k (block,querykprobe)
 * - queryprobe
 *
 * clusterquery
 */
__global__ void count_cluster_queries_kernel(
    const idx_t* d_topk_index,  // [n_query * k] :querytopk cluster
    int* d_cluster_query_count,  // [n_total_clusters] :clusterquery
    int n_query,
    int k,
    int n_total_clusters
);

/**
 * Kernel: cluster-queryCSR
 *
 * :
 * - gridDim.x = n_query (blockquery)
 * - blockDim.x = k (block,querykprobe)
 * - queryprobe
 *
 * cluster-query
 */
__global__ void build_cluster_query_mapping_kernel(
    const idx_t* d_topk_index,  // [n_query * k]
    const int* d_cluster_query_offset,  // [n_total_clusters + 1] CSRoffset
    int* d_cluster_query_data,  // [total_entries] CSRdata:query_id
    int* d_cluster_query_probe_indices,  // [total_entries] probequery
    int* d_cluster_write_pos,  // [n_total_clusters] cluster()
    int n_query,
    int k,
    int n_total_clusters
);

__global__ void build_active_cluster_ids_from_offsets_kernel(
    const int* d_cluster_query_offset,
    int* d_active_cluster_ids,
    int* d_active_cluster_count,
    int n_total_clusters
);

__global__ void build_active_cluster_compact_kernel(
    const int* d_cluster_query_count,
    int* d_active_cluster_ids,
    int* d_active_cluster_count,
    int* d_cluster_to_active_pos,
    int* d_compact_cluster_query_count,
    int n_total_clusters
);

__global__ void build_compact_cluster_query_mapping_kernel(
    const idx_t* d_topk_index,
    const int* d_cluster_to_active_pos,
    const int* d_compact_cluster_query_offset,
    int* d_compact_cluster_query_data,
    int* d_compact_cluster_write_pos,
    int n_query,
    int k,
    int n_total_clusters
);

/**
 * Kernel: (FLT_MAX  -1)
 *
 *  top-k , FLT_MAX, -1
 *
 * :
 * -  grid,
 * - total_size = n_query * n_probes * k()
 */
__global__ void init_invalid_values_kernel(
    float* __restrict__ d_topk_dist_probe,  // [total_size] - , FLT_MAX
    idx_t* __restrict__ d_topk_index_probe,  // [total_size] - , -1
    int total_size  //
);

/**
 * Kernel:  entry  slot  base_offset(producer-consumer )
 *  >= 0  offset,-1
 */
__global__ void add_base_offset_to_block_slots_kernel(
    const int* d_entry_queries,
    const int* d_entry_probe_indices,
    const int* d_entry_query_start,
    const int* d_entry_query_count,
    int entry_start,
    int entry_count,
    idx_t base_offset,
    idx_t* d_topk_index,
    int n_probe_slots,
    int k
);

/**
 * Kernel:
 *
 *  select_k
 *
 * :
 * -  grid, top-k
 * - total = n_query * k
 */
__global__ void map_candidate_indices_kernel(
    const idx_t* __restrict__ d_candidate_indices,  // [n_query][n_probes * k]
    idx_t* __restrict__ d_topk_index,  // [n_query][k] - ,
    int n_query,
    int n_probes,
    int k
);

/**
 * Kernel: clusterentry(:entry)
 *
 * :
 * - gridDim.x = n_total_clusters (blockcluster)
 * - blockDim.x = 1 ()
 */
__global__ void count_entries_per_cluster_kernel(
    const int* d_cluster_query_offset,  // [n_total_clusters + 1] CSRoffset
    int* d_entry_count_per_cluster,  // [n_total_clusters] :clusterentry
    int n_total_clusters,
    int kQueriesPerBlock);

/**
 * Kernel: entry(:entry)
 *
 * :
 * - gridDim.x = n_total_clusters (blockcluster)
 * - blockDim.x = 1 ()
 */
__global__ void build_entry_data_kernel(
    const int* d_cluster_query_offset,  // [n_total_clusters + 1] CSRoffset
    const int* d_cluster_query_data,  // [total_entries] CSRdata:query_id
    const int* d_cluster_query_probe_indices,  // [total_entries] probequery
    const int* d_entry_offset,  // [n_total_clusters + 1] entryoffset(CSR)
    const int* d_entry_query_offset,  // [n_total_clusters + 1] clusterd_entry_queries
    int* d_entry_cluster_id,  // [n_entry] :entrycluster_id
    int* d_entry_query_start,  // [n_entry] :entryquery(d_entry_queries)
    int* d_entry_query_count,  // [n_entry] :entryquery
    int* d_entry_queries,  // [total_queries_in_entries] :entryquery
    int* d_entry_probe_indices,  // [total_queries_in_entries] :entryprobe_indices
    int n_total_clusters,
    int kQueriesPerBlock);

/**
 * Kernel:  -
 *
 * :
 * -  grid, top-k
 * - total = n_query * k
 *
 * @param d_reordered_indices  device : [n_total_vectors]
 * @param d_reordered_index_in  device : [n_query * k]
 * @param d_original_index_out  device : [n_query * k]
 * @param n_query
 * @param k                     top-k
 */
__global__ void lookup_original_indices_kernel(
    const idx_t* __restrict__ d_reordered_indices,
    const idx_t* __restrict__ d_reordered_index_in,
    idx_t* __restrict__ d_original_index_out,
    int n_query,
    int k,
    int n_total_vectors
);

/**
 * :(, ivf_search )
 *
 * @param d_reordered_indices  device : [n_total_vectors]
 * @param d_reordered_index_in  device : [n_query * k]
 * @param d_original_index_out  device : [n_query * k]
 * @param n_query
 * @param k                     top-k
 * @param n_total_vectors       ()
 */
void ivf_lookup_reordered_to_original(
    const idx_t* d_reordered_indices,
    const idx_t* d_reordered_index_in,
    idx_t* d_original_index_out,
    int n_query,
    int k,
    int n_total_vectors
);

#endif // PARASELECT_CUDA_UTILS_CUH
