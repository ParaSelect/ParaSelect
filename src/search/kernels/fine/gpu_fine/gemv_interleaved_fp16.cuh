/**
 * Interleaved FP16
 * Query  FP32,cluster/interleaved  __half*.
 *
 * ****:`gemv_lib`  `fine/chunk/gemv_interleaved_{cos,l2}_fp16.cu`( topk + ** chunk** `launch_*_chunk_fp16_kernel`).
 * ****:`fine/gemv_interleaved_*_fp16.cu` , chunk .
 */
#ifndef PARASELECT_INDEXED_GEMM_INTERLEAVED_FP16_CUH
#define PARASELECT_INDEXED_GEMM_INTERLEAVED_FP16_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "layout/ivf_interleaved_layout.cuh"

/** FP16 interleaved ( FP32 , __half) */
inline size_t interleaved_storage_half_elements(int n_vec, int n_dim) {
    int num_blocks = (n_vec + paraselect::kInterleavedGroupSize - 1) / paraselect::kInterleavedGroupSize;
    return (size_t)num_blocks * paraselect::kInterleavedGroupSize * n_dim;
}

/** FP16 interleaved :Query × Interleaved(FP16)→ / */
void launch_gemm_interleaved_cos_fp16(
    const float* d_query,
    const __half* d_interleaved,
    const float* d_query_norm,
    const float* d_vec_norm,
    int n_query,
    int n_vec,
    int n_dim,
    float* d_out,
    cudaStream_t stream = 0);

/** FP16 interleaved :Query × Interleaved(FP16)→ L2  */
void launch_gemm_interleaved_l2_fp16(
    const float* d_query,
    const __half* d_interleaved,
    const float* d_query_norm,
    const float* d_vec_norm,
    int n_query,
    int n_vec,
    int n_dim,
    float* d_out,
    cudaStream_t stream = 0);

/** Entry-based  FP16 */
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
    cudaStream_t stream);

/** Entry-based L2  FP16 */
template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_gemv_interleaved_l2_topk_fp16_kernel(
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
    cudaStream_t stream);

/**  chunk:; `fine/chunk/gemv_interleaved_cos_fp16.cu` */
template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_gemv_interleaved_l2_topk_chunk_fp16_kernel(
    dim3 block,
    int n_dim,
    float* d_query_group,
    const __half* d_chunk_vector,
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

/**  chunk:L2; `fine/chunk/gemv_interleaved_l2_fp16.cu` */
template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_gemv_interleaved_l2_topk_chunk_fp16_kernel(
    dim3 block,
    int n_dim,
    float* d_query_group,
    const __half* d_chunk_vector,
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

/** FP16 chunk:, `k`  `<64|128|256, true, 8>` . */
void launch_gemv_interleaved_cos_topk_chunk_fp16(
    dim3 block,
    int n_dim,
    float* d_query_group,
    const __half* d_chunk_vector,
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

void launch_gemv_interleaved_l2_topk_chunk_fp16(
    dim3 block,
    int n_dim,
    float* d_query_group,
    const __half* d_chunk_vector,
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

#endif
