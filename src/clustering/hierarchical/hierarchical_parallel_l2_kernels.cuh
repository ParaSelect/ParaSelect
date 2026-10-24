#pragma once

#include <cuda_runtime.h>

namespace paraselect::hier_parallel_l2 {

void launch_grouped_kmeans_assign(
    const float* d_data,
    const float* d_centroids,
    const int* d_offsets_data,
    const int* d_offsets_centroids,
    const int* d_k_array,
    int* d_assign,
    int total_n,
    int num_tasks,
    int dim,
    cudaStream_t stream);

void launch_grouped_kmeans_accum(
    const float* d_data,
    const int* d_assign,
    const int* d_offsets_data,
    const int* d_offsets_centroids,
    float* d_new_centroids,
    int* d_counts,
    int total_n,
    int num_tasks,
    int dim,
    cudaStream_t stream);

void launch_grouped_kmeans_divide(
    float* d_new_centroids,
    const float* d_old_centroids,
    const int* d_counts,
    int total_k,
    int dim,
    cudaStream_t stream);

}  // namespace paraselect::hier_parallel_l2
