#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace paraselect::search {

// Retain the best n_rerank rows for each query from a global threshold pool.
// The pool is sorted by (query, ordered float distance, row ID) on the GPU;
// no per-query reservation or pre-sort truncation is performed.
void pq_candidate_cap_select(
    int* d_pool_idx,
    const float* d_pool_dist,
    const int* d_pool_query,
    const int* d_pool_counts,
    int total_candidates,
    int n_query,
    int output_stride,
    int n_rerank,
    uint64_t* d_sort_keys_in,
    uint64_t* d_sort_keys_out,
    int* d_sort_values_out,
    int* d_offsets,
    int* d_output_idx,
    float* d_output_dist,
    int* d_output_counts,
    void* d_sort_temp,
    std::size_t sort_temp_bytes,
    cudaStream_t stream);

std::size_t pq_candidate_cap_select_temp_bytes(int max_candidates, int max_queries);

}  // namespace paraselect::search
