#include "hierarchical_parallel_l2_kernels.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>

namespace paraselect::hier_parallel_l2 {

namespace {

__device__ inline int find_task_id(const int* offsets, int num_tasks, int pt_idx) {
    int left = 0;
    int right = num_tasks - 1;
    int ans = 0;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (offsets[mid] <= pt_idx) {
            ans = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return ans;
}

__global__ void grouped_kmeans_assign_kernel(
    const float* __restrict__ data,
    const float* __restrict__ centroids,
    const int* __restrict__ offsets_data,
    const int* __restrict__ offsets_centroids,
    const int* __restrict__ k_array,
    int* __restrict__ assign,
    int total_n,
    int num_tasks,
    int dim) {
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= total_n) {
        return;
    }

    int task_id = find_task_id(offsets_data, num_tasks, pt_idx);
    int k = k_array[task_id];
    int c_off = offsets_centroids[task_id];

    const float* my_pt = data + static_cast<size_t>(pt_idx) * dim;
    const float* my_cents = centroids + static_cast<size_t>(c_off) * dim;

    float min_dist = FLT_MAX;
    int best_k = 0;

    for (int j = 0; j < k; ++j) {
        float dist = 0.0f;
        const float* cj = my_cents + static_cast<size_t>(j) * dim;
#pragma unroll 4
        for (int d = 0; d < dim; ++d) {
            float diff = my_pt[d] - cj[d];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_k = j;
        }
    }
    assign[pt_idx] = best_k;
}

__global__ void grouped_kmeans_update_accum_kernel(
    const float* __restrict__ data,
    const int* __restrict__ assign,
    const int* __restrict__ offsets_data,
    const int* __restrict__ offsets_centroids,
    float* __restrict__ new_centroids,
    int* __restrict__ counts,
    int total_n,
    int num_tasks,
    int dim) {
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= total_n) {
        return;
    }

    int task_id = find_task_id(offsets_data, num_tasks, pt_idx);
    int c_off = offsets_centroids[task_id];
    int k_idx = assign[pt_idx];
    int global_k_idx = c_off + k_idx;

    atomicAdd(&counts[global_k_idx], 1);

    const float* my_pt = data + static_cast<size_t>(pt_idx) * dim;
    float* my_new_cent = new_centroids + static_cast<size_t>(global_k_idx) * dim;

    for (int d = 0; d < dim; ++d) {
        atomicAdd(&my_new_cent[d], my_pt[d]);
    }
}

__global__ void grouped_kmeans_update_divide_kernel(
    float* __restrict__ new_centroids,
    const float* __restrict__ old_centroids,
    const int* __restrict__ counts,
    int total_k,
    int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_k * dim;
    if (idx >= total) {
        return;
    }

    int k_idx = idx / dim;
    int count = counts[k_idx];

    if (count > 0) {
        new_centroids[idx] /= static_cast<float>(count);
    } else {
        new_centroids[idx] = old_centroids[idx];
    }
}

}  // namespace

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
    cudaStream_t stream) {
    int threads = 256;
    int blocks = (total_n + threads - 1) / threads;
    grouped_kmeans_assign_kernel<<<blocks, threads, 0, stream>>>(
        d_data,
        d_centroids,
        d_offsets_data,
        d_offsets_centroids,
        d_k_array,
        d_assign,
        total_n,
        num_tasks,
        dim);
}

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
    cudaStream_t stream) {
    int threads = 256;
    int blocks = (total_n + threads - 1) / threads;
    grouped_kmeans_update_accum_kernel<<<blocks, threads, 0, stream>>>(
        d_data,
        d_assign,
        d_offsets_data,
        d_offsets_centroids,
        d_new_centroids,
        d_counts,
        total_n,
        num_tasks,
        dim);
}

void launch_grouped_kmeans_divide(
    float* d_new_centroids,
    const float* d_old_centroids,
    const int* d_counts,
    int total_k,
    int dim,
    cudaStream_t stream) {
    int threads = 256;
    int total = total_k * dim;
    int blocks = (total + threads - 1) / threads;
    grouped_kmeans_update_divide_kernel<<<blocks, threads, 0, stream>>>(
        d_new_centroids, d_old_centroids, d_counts, total_k, dim);
}

}  // namespace paraselect::hier_parallel_l2
