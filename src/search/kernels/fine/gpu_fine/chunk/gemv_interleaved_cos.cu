/**
 * Interleaved  × Query  /
 *
 * 1. gemm_interleaved_cos_kernel: ();query row-major, interleaved
 * 2. gemv_interleaved_cos_topk_kernel: entry-based,
 *
 * :[block][dim][lane], (v0 d0)(v1 d0)...(v31 d0)(v0 d1)...
 */

#include "search/kernels/fine/gpu_fine/inner_product_utils.cuh"
#include "layout/ivf_interleaved_layout.cuh"
#include "search/kernels/topk/warpsort/warpsort_utils.cuh"
#include "search/kernels/topk/warpsort/warpsort.cuh"
#include "utils/pch.h"
#include <cstdint>
#include <cfloat>

using namespace paraselect::warpsort_utils;
using namespace paraselect::warpsort;

#ifndef kWarpSize
#define kWarpSize 32
#endif

__device__ __forceinline__ int laneId_interleaved() {
    return threadIdx.x % kWarpSize;
}

template<int Capacity, bool Ascending, int QueriesPerBlock>
__global__ __launch_bounds__(256, 1)
void gemv_interleaved_cos_topk_chunk_kernel(
    float* __restrict__ d_query_group,
    float* __restrict__ d_chunk_vector,
    float* __restrict__ d_chunk_norm,
    int* __restrict__ d_probe_vector_offset,
    int* __restrict__ d_probe_vector_offset_compact,
    int* __restrict__ d_probe_vector_count,
    int* __restrict__ d_entry_cluster_id,
    int* __restrict__ d_entry_query_start,
    int* __restrict__ d_entry_query_count,
    int* __restrict__ d_entry_queries,
    int* __restrict__ d_entry_probe_indices,
    float* __restrict__ d_query_norm,
    int n_entry,
    int n_probes,
    int n_dim,
    int k,
    int chunk_first_block_in_cluster,
    int chunk_num_blocks,
    float* __restrict__ d_topk_dist,
    idx_t* __restrict__ d_topk_index
) {
    __shared__ float s_query_norm[QueriesPerBlock];

    const int entry_id = blockIdx.x;
    if (entry_id >= n_entry) return;

    const int cluster_id = d_entry_cluster_id[entry_id];
    const int vector_offset = d_probe_vector_offset[cluster_id];
    const int compact_offset = d_probe_vector_offset_compact[cluster_id];
    const idx_t vector_count = d_probe_vector_count[cluster_id];

    const int entry_query_start = d_entry_query_start[entry_id];
    const int entry_query_count = d_entry_query_count[entry_id];

    const int local_query_idx = threadIdx.x / kWarpSize;
    const int lane = laneId_interleaved();
    const bool active_warp = (local_query_idx < entry_query_count);

    idx_t query_global_id = -1;
    int probe_index_in_query = -1;
    if (active_warp) {
        const int entry_query_idx = entry_query_start + local_query_idx;
        query_global_id = d_entry_queries[entry_query_idx];
        probe_index_in_query = d_entry_probe_indices[entry_query_idx];
        if (lane == 0) {
            s_query_norm[local_query_idx] = d_query_norm[query_global_id];
        }
    }
    __syncthreads();

    if (!active_warp) return;

    const float* query_global_ptr = d_query_group + query_global_id * n_dim;

    constexpr int kGroup = paraselect::kInterleavedGroupSize;
    const size_t words_per_block = (size_t)kGroup * n_dim;
    const float* cluster_chunk_base = d_chunk_vector;

    using WarpSortBase = paraselect::warpsort::WarpSort<Capacity, Ascending, float, idx_t>;
    const float dummy_val = WarpSortBase::kDummy();
    WarpSortFiltered<Capacity, Ascending, float, idx_t> queue(k);

    float* row_dist = d_topk_dist + (query_global_id * n_probes + probe_index_in_query) * k;
    idx_t* row_idx = d_topk_index + (query_global_id * n_probes + probe_index_in_query) * k;

    for (int base = 0; base < k; base += kWarpSize) {
        float v = dummy_val;
        int id = -1;
        const int si = base + lane;
        if (si < k && row_idx[si] >= 0) {
            v = row_dist[si];
            id = row_idx[si];
        }
        queue.add(v, id);
    }

    for (int iter = 0; iter < chunk_num_blocks; ++iter) {
        idx_t vec_idx_padded = vector_offset + (chunk_first_block_in_cluster + iter) * kWarpSize + lane;
        bool has_valid_vec = (vec_idx_padded < vector_offset + vector_count);

        const float* data = cluster_chunk_base + (size_t)iter * words_per_block;
        float dot = 0.0f;
        const int dimBlocks = (n_dim / kWarpSize) * kWarpSize;

        for (int dBase = 0; dBase < dimBlocks; dBase += kWarpSize) {
            const float qReg = query_global_ptr[dBase + lane];
            for (int i = 0; i < kWarpSize / 4; ++i, data += 4 * kGroup) {
                float v0 = data[0 * kGroup + lane];
                float v1 = data[1 * kGroup + lane];
                float v2 = data[2 * kGroup + lane];
                float v3 = data[3 * kGroup + lane];
                dot += __shfl_sync(0xffffffff, qReg, i * 4 + 0) * v0;
                dot += __shfl_sync(0xffffffff, qReg, i * 4 + 1) * v1;
                dot += __shfl_sync(0xffffffff, qReg, i * 4 + 2) * v2;
                dot += __shfl_sync(0xffffffff, qReg, i * 4 + 3) * v3;
            }
        }
        if (dimBlocks < n_dim) {
            const int dLoad = dimBlocks + lane;
            const float qReg = (dLoad < n_dim) ? query_global_ptr[dLoad] : 0.0f;
            for (int dSub = 0; dSub < n_dim - dimBlocks; ++dSub, data += kGroup) {
                dot += __shfl_sync(0xffffffff, qReg, dSub) * data[lane];
            }
        }

        if (!has_valid_vec) {
            queue.add(dummy_val, -1);
        } else {
            float data_norm = d_chunk_norm[(size_t)iter * kWarpSize + lane];
            if (vec_idx_padded < 0 || compact_offset < 0) {
                queue.add(dummy_val, -1);
            } else {
                float cos_sim = dot / (s_query_norm[local_query_idx] * data_norm);
                float cos_distance = 1.0f - cos_sim;
                idx_t local = vec_idx_padded - vector_offset;
                idx_t vec_idx_compact = compact_offset + local;
                queue.add(cos_distance, vec_idx_compact);
            }
        }
    }
    __syncwarp();
    queue.done();
    __syncwarp();

    queue.store(row_dist, row_idx);
}

template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_gemv_interleaved_cos_topk_chunk_kernel(
    dim3 block,
    int n_dim,
    float* d_query_group,
    float* d_chunk_vector,
    float* d_chunk_norm,
    int* d_probe_vector_offset,
    int* d_probe_vector_offset_compact,
    int* d_probe_vector_count,
    int* d_entry_cluster_id,
    int* d_entry_query_start,
    int* d_entry_query_count,
    int* d_entry_queries,
    int* d_entry_probe_indices,
    float* d_query_norm,
    int n_entry,
    int n_probes,
    int k,
    int chunk_first_block_in_cluster,
    int chunk_num_blocks,
    float* d_topk_dist,
    idx_t* d_topk_index,
    cudaStream_t stream
) {
    dim3 grid(n_entry, 1, 1);
    gemv_interleaved_cos_topk_chunk_kernel<Capacity, Ascending, QueriesPerBlock>
        <<<grid, block, 0, stream>>>(
            d_query_group, d_chunk_vector, d_chunk_norm,
            d_probe_vector_offset, d_probe_vector_offset_compact, d_probe_vector_count,
            d_entry_cluster_id, d_entry_query_start, d_entry_query_count,
            d_entry_queries, d_entry_probe_indices,
            d_query_norm,
            n_entry, n_probes, n_dim, k,
            chunk_first_block_in_cluster, chunk_num_blocks,
            d_topk_dist, d_topk_index);
}

#define TEMPLATE_GEMV_INTERLEAVED_COS_TOPK_CHUNK_KERNEL(Capacity, QueriesPerBlock) \
template void launch_gemv_interleaved_cos_topk_chunk_kernel<Capacity, true, QueriesPerBlock>( \
    dim3, int, float*, float*, float*, int*, int*, int*, int*, int*, int*, int*, int*, float*, int, int, int, int, int, float*, idx_t*, cudaStream_t)

TEMPLATE_GEMV_INTERLEAVED_COS_TOPK_CHUNK_KERNEL(64, 8);
TEMPLATE_GEMV_INTERLEAVED_COS_TOPK_CHUNK_KERNEL(128, 8);
TEMPLATE_GEMV_INTERLEAVED_COS_TOPK_CHUNK_KERNEL(256, 8);

#define LAUNCH_GEMV_INTERLEAVED_COS_TOPK_CHUNK_KERNEL(Capacity, QueriesPerBlock) \
launch_gemv_interleaved_cos_topk_chunk_kernel<Capacity, true, QueriesPerBlock>( \
    block, n_dim, d_query_group, d_chunk_vector, d_chunk_norm, \
    d_probe_vector_offset, d_probe_vector_offset_compact, d_probe_vector_count, \
    d_entry_cluster_id, d_entry_query_start, d_entry_query_count, \
    d_entry_queries, d_entry_probe_indices, d_query_norm, \
    n_entry, n_probes, k, chunk_first_block_in_cluster, chunk_num_blocks, \
    d_topk_dist, d_topk_index, stream)

void launch_gemv_interleaved_cos_topk_chunk(
    dim3 block,
    int n_dim,
    float* d_query_group,
    float* d_chunk_vector,
    float* d_chunk_norm,
    int* d_probe_vector_offset,
    int* d_probe_vector_offset_compact,
    int* d_probe_vector_count,
    int* d_entry_cluster_id,
    int* d_entry_query_start,
    int* d_entry_query_count,
    int* d_entry_queries,
    int* d_entry_probe_indices,
    float* d_query_norm,
    int n_entry,
    int n_probes,
    int k,
    int chunk_first_block_in_cluster,
    int chunk_num_blocks,
    float* d_topk_dist,
    idx_t* d_topk_index,
    cudaStream_t stream) {
    int cap = 32;
    while (cap < k)
        cap <<= 1;
    if (cap > kMaxCapacity)
        cap = kMaxCapacity;
    constexpr int QB = 8;
    if (cap <= 32) { LAUNCH_GEMV_INTERLEAVED_COS_TOPK_CHUNK_KERNEL(64, QB);
    } else if (cap <= 64) { LAUNCH_GEMV_INTERLEAVED_COS_TOPK_CHUNK_KERNEL(128, QB);
    } else { LAUNCH_GEMV_INTERLEAVED_COS_TOPK_CHUNK_KERNEL(256, QB);
    }
}
