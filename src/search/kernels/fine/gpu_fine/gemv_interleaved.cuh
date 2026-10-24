/**
 * Interleaved ( 4 )
 *
 * - **Query**:row-major `[n_query, n_dim]`.
 * - ** / **:interleaved(`[block][dim][lane]`).
 *
 * 1. `launch_gemm_interleaved_cos` / `launch_gemm_interleaved_l2`:Query× interleaved (** / **).
 * 2. `launch_gemv_interleaved_cos_topk_kernel` / `launch_gemv_interleaved_l2_topk_kernel`:entry-based top-k(** IVF **).
 * 3. `launch_gemv_interleaved_*_topk_chunk_kernel`: / (**** `cuda/search/kernels/fine/gpu_fine/chunk/gemv_interleaved_{cos,l2}.cu`).
 */

#ifndef PARASELECT_GEMV_INTERLEAVED_CUH
#define PARASELECT_GEMV_INTERLEAVED_CUH

#include "utils/pch.h"
#include <cuda_runtime.h>

/**
 *  Query × Interleaved :d_out = query @ interleaved^T()
 */
void launch_gemm_interleaved_cos(
    const float* d_query,
    const float* d_interleaved,
    const float* d_query_norm,
    const float* d_vec_norm,
    int n_query,
    int n_vec,
    int n_dim,
    float* d_out,
    cudaStream_t stream = 0);

/**
 *  Query × Interleaved :d_out[q][v] = L2 (||q||^2 + ||v||^2 - 2*q·v)
 */
void launch_gemm_interleaved_l2(
    const float* d_query,
    const float* d_interleaved,
    const float* d_query_norm,
    const float* d_vec_norm,
    int n_query,
    int n_vec,
    int n_dim,
    float* d_out,
    cudaStream_t stream = 0);

/**
 * Entry-based interleaved :
 */
template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_gemv_interleaved_cos_topk_kernel(
    dim3 block,
    int n_dim,
    float* d_query_group,
    float* d_cluster_vector,
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
    cudaStream_t stream);

/**
 * Entry-based interleaved :L2
 */
template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_gemv_interleaved_l2_topk_kernel(
    dim3 block,
    int n_dim,
    float* d_query_group,
    float* d_cluster_vector,
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
    cudaStream_t stream);

/**
 * : cluster  interleaved  d_chunk_vector, d_topk .
 *  kernel  `cuda/search/kernels/fine/gpu_fine/chunk/gemv_interleaved_cos.cu`.
 */
template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_gemv_interleaved_l2_topk_chunk_kernel(
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
    int chunk_first_block,
    int chunk_num_blocks,
    float* d_topk_dist,
    idx_t* d_topk_index,
    cudaStream_t stream);

/** (L2); `cuda/search/kernels/fine/gpu_fine/chunk/gemv_interleaved_l2.cu`. */
template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_gemv_interleaved_l2_topk_chunk_kernel(
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
    cudaStream_t stream);

/**
 *  chunk top-k: `k`  WarpSort (32  `>=k`, `min(..., kMaxCapacity)`),
 *  `launch_gemv_interleaved_*_topk_chunk_kernel<64|128|256, true, 8>`.
 */
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
    cudaStream_t stream);

void launch_gemv_interleaved_l2_topk_chunk(
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
    cudaStream_t stream);

#endif /* PARASELECT_GEMV_INTERLEAVED_CUH */
