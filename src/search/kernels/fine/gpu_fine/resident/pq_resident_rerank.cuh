#ifndef PARASELECT_PQ_RESIDENT_RERANK_CUH
#define PARASELECT_PQ_RESIDENT_RERANK_CUH

#include "utils/pch.h"
#include <cuda_runtime.h>

void launch_pq_resident_rerank_full(
    int k,
    int dim,
    const float* d_queries,
    const int* d_candidate_idx,
    int candidate_count,
    int candidate_row_stride,
    const float* d_interleaved_vectors,
    const uint8_t* d_interleaved_vectors_u8,
    const int* d_compact_to_padded,
    int n_vectors,
    int batch_n,
    float* d_topk_dist,
    int* d_topk_local_idx,
    cudaStream_t stream);

#endif  // PARASELECT_PQ_RESIDENT_RERANK_CUH
