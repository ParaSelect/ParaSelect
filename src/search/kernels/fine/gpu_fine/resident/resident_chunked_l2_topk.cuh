#ifndef PARASELECT_RESIDENT_CHUNKED_L2_TOPK_CUH
#define PARASELECT_RESIDENT_CHUNKED_L2_TOPK_CUH

#include "utils/pch.h"
#include <cuda_runtime.h>

void launch_resident_count_row_chunks(
    const int* d_entry_cluster_id,
    const int* d_entry_query_start,
    const int* d_entry_query_count,
    const int* d_entry_queries,
    const int* d_entry_probe_indices,
    const int* d_counts,
    int n_entry,
    int n_probes,
    int chunk_blocks,
    int* d_row_chunk_count,
    cudaStream_t stream);

void launch_resident_fill_row_chunk_tasks(
    const int* d_entry_cluster_id,
    const int* d_entry_query_start,
    const int* d_entry_query_count,
    const int* d_entry_queries,
    const int* d_entry_probe_indices,
    const int* d_counts,
    int n_entry,
    int n_probes,
    int chunk_blocks,
    int max_tasks,
    int* d_row_chunk_write_pos,
    int* d_task_row,
    int* d_task_cluster,
    int* d_task_first_block,
    int* d_task_num_blocks,
    cudaStream_t stream);

void launch_resident_l2_topk_chunk_scan(
    int k,
    dim3 block,
    int dim,
    float* d_queries,
    const float* d_vectors,
    const int* d_padded_offsets,
    const int* d_compact_offsets,
    const int* d_counts,
    const int* d_task_row,
    const int* d_task_cluster,
    const int* d_task_first_block,
    const int* d_task_num_blocks,
    float* d_query_norm,
    const float* d_norms,
    int n_tasks,
    int n_probes,
    float* d_candidate_dist,
    idx_t* d_candidate_index,
    cudaStream_t stream);


void launch_resident_u8_l2_topk_chunk_scan(
    int k,
    dim3 block,
    int dim,
    const uint8_t* d_queries_u8,
    const uint8_t* d_vectors_u8,
    const int* d_padded_offsets,
    const int* d_compact_offsets,
    const int* d_counts,
    const int* d_task_row,
    const int* d_task_cluster,
    const int* d_task_first_block,
    const int* d_task_num_blocks,
    float* d_query_norm,
    const float* d_norms,
    int n_tasks,
    int n_probes,
    float* d_candidate_dist,
    idx_t* d_candidate_index,
    cudaStream_t stream);

void launch_resident_reduce_row_chunks(
    int k,
    dim3 block,
    const int* d_row_chunk_offset,
    const float* d_candidate_dist,
    const idx_t* d_candidate_index,
    int n_rows,
    float* d_probe_dist,
    idx_t* d_probe_index,
    cudaStream_t stream);

#endif  // PARASELECT_RESIDENT_CHUNKED_L2_TOPK_CUH
