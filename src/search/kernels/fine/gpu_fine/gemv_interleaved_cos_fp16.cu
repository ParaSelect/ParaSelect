/**
 * Interleaved FP16  × Query(FP32) /
 *  gemv_lib( fine/chunk ); chunk ., chunk .
 *
 * Fork from gemv_interleaved_cos.cu,cluster  __half,query  float.
 *
 * 1. gemm_interleaved_cos_fp16_kernel: ,d_interleaved  __half*
 * 2. gemv_interleaved_cos_topk_fp16_kernel: entry-based,d_cluster_vector  __half*
 *
 * :[block][dim][lane]  FP32, __half.
 */

#include "inner_product_utils.cuh"
#include "layout/ivf_interleaved_layout.cuh"
#include "search/kernels/topk/warpsort/warpsort_utils.cuh"
#include "search/kernels/topk/warpsort/warpsort.cuh"
#include "utils/pch.h"
#include <cuda_fp16.h>
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

/**
 * Query(FP32)× Interleaved(FP16):
 */
__global__ void gemm_interleaved_cos_fp16_kernel(
    const float* __restrict__ d_query,       // [n_query, n_dim]
    const __half* __restrict__ d_interleaved, // interleaved [n_vec, n_dim] FP16
    const float* __restrict__ d_query_norm,
    const float* __restrict__ d_vec_norm,
    int n_query,
    int n_vec,
    int n_dim,
    float* __restrict__ d_out
) {
    const int query_id = blockIdx.x;
    const int vec_block = blockIdx.y;
    const int lane = laneId_interleaved();

    const int vec_idx = vec_block * kWarpSize + lane;
    if (query_id >= n_query || vec_idx >= n_vec) return;

    const float* query_row = d_query + (size_t)query_id * n_dim;

    // shared memory  query:32  query, global
    extern __shared__ char s_mem[];
    float* s_query = reinterpret_cast<float*>(s_mem);

    /* :query  16B , misaligned float4 */
    for (int d = lane; d < n_dim; d += kWarpSize) {
        s_query[d] = query_row[d];
    }
    __syncthreads();

    constexpr int kGroup = paraselect::kInterleavedGroupSize;
    const size_t words_per_block = (size_t)kGroup * n_dim;
    const __half* block_base = d_interleaved + vec_block * words_per_block;

    float dot = 0.0f;
#pragma unroll 8
    for (int d = 0; d < n_dim; ++d) {
        float q = s_query[d];
        float v = __half2float(__ldg(&block_base[(size_t)d * kGroup + lane]));
        dot += q * v;
    }

    // :
    if (d_query_norm && d_vec_norm) {
        float qn = d_query_norm[query_id];
        float vn = d_vec_norm[vec_idx];
        float cos_sim = dot / (qn * vn);
        dot = 1.0f - cos_sim;  //
    }

    d_out[(size_t)query_id * n_vec + vec_idx] = dot;
}

/* ---------- Entry-based interleaved cos topk FP16(d_cluster_vector  __half*) ---------- */
template<int Capacity, bool Ascending, int QueriesPerBlock>
__global__ __launch_bounds__(256, 1)
void gemv_interleaved_cos_topk_fp16_kernel(
    float* __restrict__ d_query_group,
    __half* __restrict__ d_cluster_vector,  /* interleaved  FP16 */
    int* __restrict__ d_probe_vector_offset,
    int* __restrict__ d_probe_vector_offset_compact,
    int* __restrict__ d_probe_vector_count,
    int* __restrict__ d_entry_cluster_id,
    int* __restrict__ d_entry_query_start,
    int* __restrict__ d_entry_query_count,
    int* __restrict__ d_entry_queries,
    int* __restrict__ d_entry_probe_indices,
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_entry,
    int n_probes,
    int n_dim,
    int k,
    float* __restrict__ d_topk_dist,
    idx_t* __restrict__ d_topk_index
) {
    __shared__ float s_query_norm[QueriesPerBlock];

    const int entry_id = blockIdx.x;
    if (entry_id >= n_entry) return;

    const int cluster_id = d_entry_cluster_id[entry_id];
    const int vector_offset = d_probe_vector_offset[cluster_id];
    const int compact_offset = d_probe_vector_offset_compact[cluster_id];
    const int vector_count = d_probe_vector_count[cluster_id];

    const int entry_query_start = d_entry_query_start[entry_id];
    const int entry_query_count = d_entry_query_count[entry_id];

    const int local_query_idx = threadIdx.x / kWarpSize;
    const int lane = laneId_interleaved();

    const bool active_warp = (local_query_idx < entry_query_count);
    int query_global_id = -1;
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
    const __half* cluster_interleaved_base = d_cluster_vector + (size_t)(vector_offset / kGroup) * words_per_block;

    using WarpSortBase = paraselect::warpsort::WarpSort<Capacity, Ascending, float, idx_t>;
    const float dummy_val = WarpSortBase::kDummy();
    WarpSortFiltered<Capacity, Ascending, float, idx_t> queue(k);

    int max_iterations = (vector_count + kWarpSize - 1) / kWarpSize;

    for (int iter = 0; iter < max_iterations; ++iter) {
        int vec_idx_padded = vector_offset + iter * kWarpSize + lane;
        bool has_valid_vec = (vec_idx_padded < vector_offset + vector_count);

        const __half* data = cluster_interleaved_base + (size_t)iter * words_per_block;
        float dot = 0.0f;
        float sum_v2 = 0.0f;
        const int dimBlocks = (n_dim / kWarpSize) * kWarpSize;

        for (int dBase = 0; dBase < dimBlocks; dBase += kWarpSize) {
            const float qReg = query_global_ptr[dBase + lane];
            for (int i = 0; i < kWarpSize / 4; ++i, data += 4 * kGroup) {
                float v0 = __half2float(__ldg(&data[0 * kGroup + lane]));
                float v1 = __half2float(__ldg(&data[1 * kGroup + lane]));
                float v2 = __half2float(__ldg(&data[2 * kGroup + lane]));
                float v3 = __half2float(__ldg(&data[3 * kGroup + lane]));
                float q0 = __shfl_sync(0xffffffff, qReg, i * 4 + 0);
                float q1 = __shfl_sync(0xffffffff, qReg, i * 4 + 1);
                float q2 = __shfl_sync(0xffffffff, qReg, i * 4 + 2);
                float q3 = __shfl_sync(0xffffffff, qReg, i * 4 + 3);
                dot += q0 * v0 + q1 * v1 + q2 * v2 + q3 * v3;
                if (!d_cluster_vector_norm) sum_v2 += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
            }
        }
        if (dimBlocks < n_dim) {
            const int dLoad = dimBlocks + lane;
            const float qReg = (dLoad < n_dim) ? query_global_ptr[dLoad] : 0.0f;
            for (int dSub = 0; dSub < n_dim - dimBlocks; ++dSub, data += kGroup) {
                float v = __half2float(__ldg(&data[lane]));
                dot += __shfl_sync(0xffffffff, qReg, dSub) * v;
                if (!d_cluster_vector_norm) sum_v2 += v * v;
            }
        }

        if (!has_valid_vec) {
            queue.add(dummy_val, -1);
        } else {
            float data_norm = d_cluster_vector_norm ? d_cluster_vector_norm[vec_idx_padded] : sqrtf(sum_v2);
            if (vec_idx_padded < 0) {
                queue.add(dummy_val, -1);
            } else {
                float cos_sim = dot / (s_query_norm[local_query_idx] * data_norm);
                float cos_distance = 1.0f - cos_sim;
                int local = vec_idx_padded - vector_offset;
                int vec_idx_compact = compact_offset + local;
                queue.add(cos_distance, vec_idx_compact);
            }
        }
    }
    __syncwarp();
    queue.done();
    __syncwarp();

    float* row_dist = d_topk_dist + (query_global_id * n_probes + probe_index_in_query) * k;
    idx_t* row_idx = d_topk_index + (query_global_id * n_probes + probe_index_in_query) * k;
    queue.store(row_dist, row_idx);
}

void launch_gemm_interleaved_cos_fp16(
    const float* d_query,
    const __half* d_interleaved,
    const float* d_query_norm,
    const float* d_vec_norm,
    int n_query,
    int n_vec,
    int n_dim,
    float* d_out,
    cudaStream_t stream
) {
    constexpr int kGroup = paraselect::kInterleavedGroupSize;
    int vec_blocks = (n_vec + kGroup - 1) / kGroup;
    dim3 block(kGroup, 1, 1);
    dim3 grid(n_query, vec_blocks, 1);
    size_t shmem = (size_t)n_dim * sizeof(float);
    gemm_interleaved_cos_fp16_kernel<<<grid, block, shmem, stream>>>(
        d_query, d_interleaved, d_query_norm, d_vec_norm,
        n_query, n_vec, n_dim, d_out);
}

template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_gemv_interleaved_cos_topk_fp16_kernel(
    dim3 block,
    int n_dim,
    float* d_query_group,
    __half* d_cluster_vector,
    int* d_probe_vector_offset,
    int* d_probe_vector_offset_compact,
    int* d_probe_vector_count,
    int* d_entry_cluster_id,
    int* d_entry_query_start,
    int* d_entry_query_count,
    int* d_entry_queries,
    int* d_entry_probe_indices,
    float* d_query_norm,
    float* d_cluster_vector_norm,
    int n_entry,
    int n_probes,
    int k,
    float* d_topk_dist,
    idx_t* d_topk_index,
    cudaStream_t stream
) {
    dim3 grid(n_entry, 1, 1);
    gemv_interleaved_cos_topk_fp16_kernel<Capacity, Ascending, QueriesPerBlock>
        <<<grid, block, 0, stream>>>(
        d_query_group, d_cluster_vector,
        d_probe_vector_offset, d_probe_vector_offset_compact, d_probe_vector_count,
        d_entry_cluster_id, d_entry_query_start, d_entry_query_count,
        d_entry_queries, d_entry_probe_indices,
        d_query_norm, d_cluster_vector_norm,
        n_entry, n_probes, n_dim, k,
        d_topk_dist, d_topk_index);
}

#define LAUNCH_GEMV_INTERLEAVED_COS_TOPK_FP16(Capacity) \
template void launch_gemv_interleaved_cos_topk_fp16_kernel<Capacity, true, 8>( \
    dim3, int, float*, __half*, int*, int*, int*, int*, int*, int*, int*, int*, float*, float*, int, int, int, float*, idx_t*, cudaStream_t);

LAUNCH_GEMV_INTERLEAVED_COS_TOPK_FP16(64)
LAUNCH_GEMV_INTERLEAVED_COS_TOPK_FP16(128)
LAUNCH_GEMV_INTERLEAVED_COS_TOPK_FP16(256)
