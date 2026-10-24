/**
 * CUDA Utility Kernels Implementation
 *
 * Common utility kernels used across multiple modules in paraselect.
 */

#include "pch.h"
#include "utils.cuh"
#include <cfloat>
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include <vector>

#if IVF_DEBUG_INDEX_RANGE
// , false; IVF_DEBUG_INDEX_RANGE=1
bool g_ivf_debug_index_range = ([]() -> bool {
    const char* env = std::getenv("IVF_DEBUG_INDEX_RANGE");
    return env && env[0] == '1';
})();
#endif

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
    int n_total_clusters)
{
    // 2D grid :blockIdx.x = query_id, blockIdx.y = probe_block_id
    const int query_id = blockIdx.x;
    const int probe_block_id = blockIdx.y;
    const int probe_local_id = probe_block_id * blockDim.x + threadIdx.x;

    if (query_id >= n_query || probe_local_id >= k) return;

    const int idx = query_id * k + probe_local_id;
    idx_t cluster_id = d_topk_index[idx];

    // :cluster
    if (cluster_id >= 0 && cluster_id < n_total_clusters) {
        atomicAdd(&d_cluster_query_count[static_cast<int>(cluster_id)], 1);
    }
}

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
    int n_total_clusters)
{
    //  2D grid :blockIdx.x = query, blockIdx.y = probe block, threadIdx.x = probe in block
    const int query_id = blockIdx.x;
    if (query_id >= n_query) return;

    const int rank = blockIdx.y * blockDim.x + threadIdx.x;
    if (rank >= k) return;

    idx_t cluster_id = d_topk_index[query_id * k + rank];

    // :cluster
    if (cluster_id >= 0 && cluster_id < n_total_clusters) {
        //
        int write_pos = atomicAdd(&d_cluster_write_pos[static_cast<int>(cluster_id)], 1);
        d_cluster_query_data[write_pos] = query_id;
        if (d_cluster_query_probe_indices) {
            d_cluster_query_probe_indices[write_pos] = rank;  // rankprobe_index_in_query
        }
    }
}

__global__ void build_active_cluster_ids_from_offsets_kernel(
    const int* d_cluster_query_offset,
    int* d_active_cluster_ids,
    int* d_active_cluster_count,
    int n_total_clusters)
{
    const int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= n_total_clusters) return;
    if (d_cluster_query_offset[cid] == d_cluster_query_offset[cid + 1]) return;
    const int pos = atomicAdd(d_active_cluster_count, 1);
    d_active_cluster_ids[pos] = cid;
}

__global__ void build_active_cluster_compact_kernel(
    const int* d_cluster_query_count,
    int* d_active_cluster_ids,
    int* d_active_cluster_count,
    int* d_cluster_to_active_pos,
    int* d_compact_cluster_query_count,
    int n_total_clusters)
{
    const int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= n_total_clusters) return;
    const int count = d_cluster_query_count[cid];
    if (count <= 0) return;
    const int pos = atomicAdd(d_active_cluster_count, 1);
    d_active_cluster_ids[pos] = cid;
    d_cluster_to_active_pos[cid] = pos;
    d_compact_cluster_query_count[pos] = count;
}

__global__ void build_compact_cluster_query_mapping_kernel(
    const idx_t* d_topk_index,
    const int* d_cluster_to_active_pos,
    const int* d_compact_cluster_query_offset,
    int* d_compact_cluster_query_data,
    int* d_compact_cluster_write_pos,
    int n_query,
    int k,
    int n_total_clusters)
{
    const int query_id = blockIdx.x;
    if (query_id >= n_query) return;
    const int rank = blockIdx.y * blockDim.x + threadIdx.x;
    if (rank >= k) return;
    const idx_t cluster_id_raw = d_topk_index[(size_t)query_id * (size_t)k + (size_t)rank];
    if (cluster_id_raw < 0 || cluster_id_raw >= n_total_clusters) return;
    const int cid = static_cast<int>(cluster_id_raw);
    const int active_pos = d_cluster_to_active_pos[cid];
    if (active_pos < 0) return;
    const int write_pos = atomicAdd(&d_compact_cluster_write_pos[active_pos], 1);
    d_compact_cluster_query_data[write_pos] = query_id;
}

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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    d_topk_dist_probe[idx] = FLT_MAX;
    d_topk_index_probe[idx] = static_cast<idx_t>(-1);
}

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
) {
    const int entry_id = blockIdx.x;
    if (entry_id >= entry_count) return;
    int q_start = d_entry_query_start[entry_start + entry_id];
    int q_count = d_entry_query_count[entry_start + entry_id];
    for (int i = threadIdx.x; i < q_count * k; i += blockDim.x) {
        int qi = i / k, ki = i % k;
        int eq_idx = q_start + qi;
        int query_id = d_entry_queries[eq_idx];
        int probe_idx = d_entry_probe_indices[eq_idx];
        int pos = (query_id * n_probe_slots + probe_idx) * k + ki;
        idx_t v = d_topk_index[pos];
        if (v >= 0) d_topk_index[pos] = v + base_offset;
    }
}

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
) {
    int max_candidates_per_query = n_probes * k;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_query * k;
    if (idx >= total) return;

    int query_id = idx / k;

    idx_t candidate_pos = d_topk_index[idx];
    if (candidate_pos >= 0 && candidate_pos < max_candidates_per_query) {
        idx_t original_idx = d_candidate_indices[query_id * max_candidates_per_query + static_cast<int>(candidate_pos)];
#if IVF_DEBUG_INDEX_RANGE
        if (original_idx < -1 || original_idx > 200000000) {
            printf("[map_candidate DEBUG] idx=%d query=%d rank=%d candidate_pos=%lld -> original_idx=%lld (SUSPICIOUS)\n",
                idx, query_id, idx % k, (long long)candidate_pos, (long long)original_idx);
        }
#endif
        d_topk_index[idx] = original_idx;
    } else {
        d_topk_index[idx] = static_cast<idx_t>(-1);
    }
}

/**
 * Kernel: clusterentry(:entry)
 *
 * :
 * - gridDim.x = n_total_clusters (blockcluster)
 * - blockDim.x = 1 ()
 *
 * cluster,query,entry(kQueriesPerBlockquery)
 */
__global__ void count_entries_per_cluster_kernel(
    const int* d_cluster_query_offset,  // [n_total_clusters + 1] CSRoffset
    int* d_entry_count_per_cluster,  // [n_total_clusters] :clusterentry
    int n_total_clusters,
    int kQueriesPerBlock)
{
    const int cluster_id = blockIdx.x;
    if (cluster_id >= n_total_clusters) return;

    int query_start = d_cluster_query_offset[cluster_id];
    int query_end = d_cluster_query_offset[cluster_id + 1];
    int n_queries = query_end - query_start;

    // clusterentry(kQueriesPerBlockquery)
    int n_entries = (n_queries > 0) ? (n_queries + kQueriesPerBlock - 1) / kQueriesPerBlock : 0;
    d_entry_count_per_cluster[cluster_id] = n_entries;
}

/**
 * Kernel: entry(:entry)
 *
 * :
 * - gridDim.x = n_total_clusters (blockcluster)
 * - blockDim.x = 1 ()
 *
 * cluster,queryentry,entry
 *
 * :d_entry_queries , entry  query( entry )
 *  cluster  d_entry_queries
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
    int kQueriesPerBlock)
{
    const int cluster_id = blockIdx.x;
    if (cluster_id >= n_total_clusters) return;

    int query_start = d_cluster_query_offset[cluster_id];
    int query_end = d_cluster_query_offset[cluster_id + 1];
    int n_queries = query_end - query_start;

    if (n_queries == 0) return;  // querycluster

    int entry_start = d_entry_offset[cluster_id];
    int entry_idx = entry_start;

    // clusterentry queries
    int cluster_entry_query_start = d_entry_query_offset[cluster_id];
    int current_query_offset = cluster_entry_query_start;

    // queryentry(kQueriesPerBlock)
    for (int batch_start = 0; batch_start < n_queries; batch_start += kQueriesPerBlock) {
        int batch_size = min(kQueriesPerBlock, n_queries - batch_start);

        // entry
        d_entry_cluster_id[entry_idx] = cluster_id;
        d_entry_query_start[entry_idx] = current_query_offset;
        d_entry_query_count[entry_idx] = batch_size;

        // entryqueryprobe_indices
        for (int i = 0; i < batch_size; ++i) {
            int query_idx = query_start + batch_start + i;
            d_entry_queries[current_query_offset + i] = d_cluster_query_data[query_idx];
            d_entry_probe_indices[current_query_offset + i] = d_cluster_query_probe_indices[query_idx];
        }

        current_query_offset += batch_size;
        entry_idx++;
    }
}

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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_query * k;

    if (idx < total) {
        idx_t reordered_idx = d_reordered_index_in[idx];
        if (reordered_idx >= 0 && reordered_idx < n_total_vectors) {
            d_original_index_out[idx] = d_reordered_indices[static_cast<int>(reordered_idx)];
        } else {
            d_original_index_out[idx] = static_cast<idx_t>(-1);
        }
    }
}

/**
 * :()
 */
void ivf_lookup_reordered_to_original(
    const idx_t* d_reordered_indices,
    const idx_t* d_reordered_index_in,
    idx_t* d_original_index_out,
    int n_query,
    int k,
    int n_total_vectors
) {
    if (!d_reordered_indices || !d_reordered_index_in || !d_original_index_out) {
        fprintf(stderr, "[ERROR] ivf_lookup_reordered_to_original: null\n");
        throw std::invalid_argument("input pointers must not be null");
    }

    if (n_query <= 0 || k <= 0) {
        fprintf(stderr, "[ERROR] ivf_lookup_reordered_to_original:  - n_query=%d, k=%d\n", n_query, k);
        throw std::invalid_argument("invalid parameters");
    }

    //  GPU kernel
    dim3 block_size(256);
    dim3 grid_size((n_query * k + block_size.x - 1) / block_size.x);

    lookup_original_indices_kernel<<<grid_size, block_size>>>(
        d_reordered_indices,
        d_reordered_index_in,
        d_original_index_out,
        n_query,
        k,
        n_total_vectors
    );

    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;
}
