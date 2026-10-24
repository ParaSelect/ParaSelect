#ifndef PARASELECT_CUDA_SCAN_CUH
#define PARASELECT_CUDA_SCAN_CUH

#include "utils/pch.h"

__global__ void inclusive_scan_block_kernel(
    const int* __restrict__ d_input,
    int* __restrict__ d_output,
    int* __restrict__ d_block_sums,
    int n);

__global__ void inclusive_scan_add_block_sums_kernel(
    int* __restrict__ d_output,
    const int* __restrict__ d_block_prefix_sums,
    int n);

__global__ void inclusive_scan_kernel(
    const int* __restrict__ d_input,
    int* __restrict__ d_output,
    int n);

void compute_prefix_sum(const idx_t* d_count, int* d_offset, int n, cudaStream_t stream = 0);
void compute_prefix_sum(const int* d_count, int* d_offset, int n, cudaStream_t stream = 0);

// Scratch-based variant: d_block_sums must hold at least ceil(n/256) ints.
// Unlike compute_prefix_sum it never calls cudaMalloc/cudaFree, so it stays
// fully asynchronous on the caller's stream (required by overlapped pipelines).
void compute_prefix_sum_scratch(const int* d_count, int* d_offset, int n,
                                int* d_block_sums, cudaStream_t stream = 0);

#endif  // PARASELECT_CUDA_SCAN_CUH
