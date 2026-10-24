#include "search/kernels/fine/gpu_fine/gemv.cuh"
#include "search/kernels/fine/gpu_fine/inner_product_utils.cuh"
#include "utils/pch.h"
#include "search/kernels/topk/warpsort/warpsort_utils.cuh"
#include "search/kernels/topk/warpsort/warpsort.cuh"
#include <cfloat>
#include <stdint.h>
#include <vector>
#include <algorithm>

using namespace paraselect::warpsort_utils;
using namespace paraselect::warpsort;

/**
 * v5:Entry-based
 * entryclusterquery(84)
 * gridn_entry,blockentry
 */

/**
 * Entry:GPUentry
 * d_entry_cluster_id: entrycluster_id [n_entry]
 * d_entry_query_start: entryquery [n_entry]
 * d_entry_query_count: entryquery [n_entry]
 * d_entry_queries: entryquery()[total_queries_in_entries]
 * d_entry_probe_indices: entry-queryprobequery [total_queries_in_entries]
 */
template<int Capacity, bool Ascending, int QueriesPerBlock>
__global__ __launch_bounds__(256, 1)
void gemv_with_l2_topk_kernel(
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_probe_vector_offset,
    int* __restrict__ d_probe_vector_count,
    int* __restrict__ d_entry_cluster_id,  // [n_entry]
    int* __restrict__ d_entry_query_start,  // [n_entry]
    int* __restrict__ d_entry_query_count,  // [n_entry]
    int* __restrict__ d_entry_queries,  // [total_queries_in_entries]
    int* __restrict__ d_entry_probe_indices,  // [total_queries_in_entries]
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_entry,  // entry
    int n_probes,  // queryprobe(probe_index_in_query)
    int n_dim,
    int k,
    float* __restrict__ d_topk_dist,  // [n_query][n_probes][k]
    idx_t* __restrict__ d_topk_index    // [n_query][n_probes][k]
) {
    __shared__ float s_query_norm[QueriesPerBlock];

    const int entry_id = blockIdx.x;
    if (entry_id >= n_entry) return;

    const int cluster_id = d_entry_cluster_id[entry_id];
    const int vector_offset = d_probe_vector_offset[cluster_id];
    const int vector_count = d_probe_vector_count[cluster_id];

    const int entry_query_start = d_entry_query_start[entry_id];
    const int entry_query_count = d_entry_query_count[entry_id];
    // const int entry_query_end = entry_query_start + entry_query_count;

    const int local_query_idx = threadIdx.x / kWarpSize;
    const int lane = laneId();

    const bool active_warp = (local_query_idx < entry_query_count);
    int query_global_id = -1;
    int probe_index_in_query = -1;
    if (active_warp) {
        const int entry_query_idx = entry_query_start + local_query_idx;
        // : d_entry_queries / d_entry_probe_indices ,entry_query_idx
        query_global_id = d_entry_queries[entry_query_idx];
        probe_index_in_query = d_entry_probe_indices[entry_query_idx];
        if (lane == 0) {
            s_query_norm[local_query_idx] = d_query_norm[query_global_id];
        }
    }
    __syncthreads();

    if (!active_warp) return;

    const float* query_global_ptr = d_query_group + query_global_id * n_dim;
    const bool query_ptr_aligned = (reinterpret_cast<uintptr_t>(query_global_ptr) & (sizeof(float4) - 1)) == 0;
    const bool data_ptr_aligned = (reinterpret_cast<uintptr_t>(d_cluster_vector) & (sizeof(float4) - 1)) == 0;
    const bool prefer_vec4 = query_ptr_aligned && data_ptr_aligned && ((n_dim & 3) == 0);

    using WarpSortBase = paraselect::warpsort::WarpSort<Capacity, Ascending, float, idx_t>;
    const float dummy_val = WarpSortBase::kDummy();

    WarpSortFiltered<Capacity, Ascending, float, idx_t> queue(k);

    int max_iterations = (vector_count + kWarpSize - 1) / kWarpSize;

    if (prefer_vec4) {
        for (int iter = 0; iter < max_iterations; ++iter) {
            int vec_idx = vector_offset + iter * kWarpSize + lane;
            bool has_valid_vec = (vec_idx < vector_offset + vector_count);

            if (!has_valid_vec) {
                queue.add(dummy_val, -1);
            } else {
                const float* vec_ptr = d_cluster_vector + vec_idx * n_dim;
                float dot_product = dot_product_vec4_aligned(query_global_ptr, vec_ptr, n_dim);

                float data_norm = d_cluster_vector_norm[vec_idx];
                float query_norm = s_query_norm[local_query_idx];
                float l2_distance = query_norm*query_norm + data_norm*data_norm - 2.0f * dot_product;
                queue.add(l2_distance, vec_idx);
            }
        }
    } else {
        for (int iter = 0; iter < max_iterations; ++iter) {
            int vec_idx = vector_offset + iter * kWarpSize + lane;
            bool has_valid_vec = (vec_idx < vector_offset + vector_count);

            if (!has_valid_vec) {
                queue.add(dummy_val, -1);
            } else {
                const float* vec_ptr = d_cluster_vector + vec_idx * n_dim;
                float dot_product = dot_product_accumulate(query_global_ptr, vec_ptr, n_dim);

                float data_norm = d_cluster_vector_norm[vec_idx];
                float query_norm = s_query_norm[local_query_idx];
                float l2_distance = query_norm*query_norm + data_norm*data_norm - 2.0f * dot_product;
                queue.add(l2_distance, vec_idx);
            }
        }
    }
    __syncwarp();

    queue.done();
    __syncwarp();

    // :[n_query][n_probes][k]
    // probe,
    float* row_dist = d_topk_dist + (query_global_id * n_probes + probe_index_in_query) * k;
    idx_t* row_idx = d_topk_index + (query_global_id * n_probes + probe_index_in_query) * k;
    queue.store(row_dist, row_idx);
}

/**
 * Launch :entry-based
 */
template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_gemv_with_l2_topk_kernel(
    dim3 block,
    int n_dim,
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_probe_vector_offset,
    int* __restrict__ d_probe_vector_count,
    int* __restrict__ d_entry_cluster_id,  // [n_entry]
    int* __restrict__ d_entry_query_start,  // [n_entry]
    int* __restrict__ d_entry_query_count,  // [n_entry]
    int* __restrict__ d_entry_queries,  // [total_queries_in_entries]
    int* __restrict__ d_entry_probe_indices,  // [total_queries_in_entries]
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_entry,  // entry
    int n_probes,  // queryprobe
    int k,
    float* __restrict__ d_topk_dist,
    idx_t* __restrict__ d_topk_index,
    cudaStream_t stream) {

    dim3 grid(n_entry, 1, 1);

    //  generic ,
    gemv_with_l2_topk_kernel<Capacity, Ascending, QueriesPerBlock>
        <<<grid, block, 0, stream>>>(
        d_query_group,
        d_cluster_vector,
        d_probe_vector_offset,
        d_probe_vector_count,
        d_entry_cluster_id,
        d_entry_query_start,
        d_entry_query_count,
        d_entry_queries,
        d_entry_probe_indices,
        d_query_norm,
        d_cluster_vector_norm,
        n_entry,
        n_probes,
        n_dim,
        k,
        d_topk_dist,
        d_topk_index
    );
}

#define LAUNCH_GEMV_WITH_L2_TOPK_KERNEL(Capacity, QueriesPerBlock) \
template void launch_gemv_with_l2_topk_kernel<Capacity, true, QueriesPerBlock>( \
    dim3, int, float*, float*, int*, int*, int*, int*, int*, int*, int*, float*, float*, int, int, int, float*, idx_t*, cudaStream_t);

LAUNCH_GEMV_WITH_L2_TOPK_KERNEL(64, 1)
LAUNCH_GEMV_WITH_L2_TOPK_KERNEL(128, 1)
LAUNCH_GEMV_WITH_L2_TOPK_KERNEL(256, 1)
LAUNCH_GEMV_WITH_L2_TOPK_KERNEL(64, 8)
LAUNCH_GEMV_WITH_L2_TOPK_KERNEL(128, 8)
LAUNCH_GEMV_WITH_L2_TOPK_KERNEL(256, 8)


