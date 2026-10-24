#pragma once

#include "search/kernels/fine/fine_search_strategy.h"

#include <cuda_runtime.h>

namespace paraselect::search {

const char* pq_threshold_policy_name(PQThresholdPolicy policy);

// Generate one cutoff per query from a sorted dense seed result row.
// This is one CUDA thread per query and does not synchronize with the host.
void pq_threshold_generate_gpu(
    const float* d_seed_dist,
    const int* d_seed_idx,
    int n_query,
    int seed_k,
    const PQAdcThresholdParams& params,
    float* d_threshold,
    int* d_status,
    cudaStream_t stream);

// Diagnostic only. Reports every upper-bound violation before applying max.
void pq_histogram_exact_guard_gpu(
    const float* seed_distances, int queries, int seed_rank,
    float* threshold, cudaStream_t stream);

// Convert a query-major raw-distance histogram into a conservative threshold.
void pq_histogram_threshold_generate_gpu(
    const unsigned int* d_histogram,
    int n_query,
    int histogram_bins,
    int rank,
    float log_min,
    float log_step,
    float* d_threshold,
    int* d_status,
    cudaStream_t stream);

}  // namespace paraselect::search
