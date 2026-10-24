#include "clustering/flash_kmeans/flash_kmeans_assign.cuh"
#include "clustering/flash_kmeans/flash_kmeans_assign_v8.cuh"
#include "utils/check_cuda_error.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include <iostream>
#include <limits>

namespace paraselect {
namespace clustering {
namespace kmeans {

namespace {

__global__ void flash_assign_l2_norm_kernel(
    const float* __restrict__ values,
    int rows,
    int dim,
    float* __restrict__ norm2)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;

    __shared__ float smem[256];
    float sum = 0.0f;
    const float* v = values + (size_t)row * dim;
    for (int d = tid; d < dim; d += blockDim.x) {
        const float x = v[d];
        sum += x * x;
    }
    smem[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) norm2[row] = smem[0];
}

__global__ void init_packed_partials_kernel(
    unsigned long long* partial,
    size_t count)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        partial[i] = 0x7f7fffffull << 32;
    }
}

__global__ void final_min_from_packed_partials_kernel(
    const unsigned long long* __restrict__ partial,
    int n,
    int n_tiles,
    idx_t* __restrict__ best_idx,
    float* __restrict__ best_dist)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= n) return;

    unsigned long long best = 0x7f7fffffull << 32;
    for (int tile = tid; tile < n_tiles; tile += blockDim.x) {
        const unsigned long long candidate = partial[(size_t)row * n_tiles + tile];
        if (candidate < best) best = candidate;
    }

    __shared__ unsigned long long s_pack[256];
    s_pack[tid] = best;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && s_pack[tid + stride] < s_pack[tid]) {
            s_pack[tid] = s_pack[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const unsigned long long packed = s_pack[0];
        best_dist[row] = __uint_as_float((unsigned int)(packed >> 32));
        best_idx[row] = (idx_t)(packed & 0xffffffffull);
    }
}

__global__ void final_min_from_packed_partials_small_kernel(
    const unsigned long long* __restrict__ partial,
    int n,
    int n_tiles,
    idx_t* __restrict__ best_idx,
    float* __restrict__ best_dist)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    unsigned long long best = 0x7f7fffffull << 32;
    const unsigned long long* row_partial = partial + (size_t)row * n_tiles;
    for (int tile = 0; tile < n_tiles; ++tile) {
        const unsigned long long candidate = row_partial[tile];
        if (candidate < best) best = candidate;
    }

    best_dist[row] = __uint_as_float((unsigned int)(best >> 32));
    best_idx[row] = (idx_t)(best & 0xffffffffull);
}

}  // namespace

void reserve_flash_kmeans_assign_workspace(
    FlashKMeansAssignWorkspace& workspace,
    int n,
    int k,
    const FlashKMeansAssignConfig& config)
{
    if (n <= 0 || k <= 0) return;
    const bool needs_norm =
        config.variant == 8;
    if (needs_norm && workspace.n_capacity < n) {
        if (workspace.d_x_sq) CUDA_CHECK(cudaFree(workspace.d_x_sq));
        CUDA_CHECK(cudaMalloc(&workspace.d_x_sq, sizeof(float) * (size_t)n));
        workspace.n_capacity = n;
    }
    if (needs_norm && workspace.k_capacity < k) {
        if (workspace.d_c_sq) CUDA_CHECK(cudaFree(workspace.d_c_sq));
        CUDA_CHECK(cudaMalloc(&workspace.d_c_sq, sizeof(float) * (size_t)k));
        workspace.k_capacity = k;
    }
    if (config.variant == 8) {
        const int v6_ctile = (config.centroid_tile == 128) ? 128 : 256;
        const int partial_tiles = (k + v6_ctile - 1) / v6_ctile;
        const size_t required = (size_t)n * (size_t)partial_tiles;
        const size_t capacity =
            (size_t)workspace.n_capacity * (size_t)workspace.partial_tiles_capacity;
        if (workspace.d_partial_packed == nullptr || capacity < required) {
            if (workspace.d_partial_packed) CUDA_CHECK(cudaFree(workspace.d_partial_packed));
            CUDA_CHECK(cudaMalloc(
                &workspace.d_partial_packed,
                sizeof(unsigned long long) * required));
            workspace.partial_tiles_capacity = partial_tiles;
            if (workspace.n_capacity < n) workspace.n_capacity = n;
        }
    }
}

void reserve_flash_kmeans_assign_scratch_workspace(
    FlashKMeansAssignWorkspace& workspace,
    int n,
    int k,
    const FlashKMeansAssignConfig& config)
{
    if (n <= 0 || k <= 0) return;
    if (config.variant != 8) {
        return;
    }
    const int v6_ctile = (config.centroid_tile == 128) ? 128 : 256;
    const int partial_tiles = (k + v6_ctile - 1) / v6_ctile;
    const size_t required = (size_t)n * (size_t)partial_tiles;
    const size_t capacity =
        (size_t)workspace.n_capacity * (size_t)workspace.partial_tiles_capacity;
    if (workspace.d_partial_packed == nullptr || capacity < required) {
        if (workspace.d_partial_packed) CUDA_CHECK(cudaFree(workspace.d_partial_packed));
        CUDA_CHECK(cudaMalloc(
            &workspace.d_partial_packed,
            sizeof(unsigned long long) * required));
        workspace.n_capacity = n;
        workspace.partial_tiles_capacity = partial_tiles;
    }
}

void release_flash_kmeans_assign_workspace(FlashKMeansAssignWorkspace& workspace)
{
    if (workspace.d_partial_packed) CUDA_CHECK(cudaFree(workspace.d_partial_packed));
    if (workspace.d_c_sq) CUDA_CHECK(cudaFree(workspace.d_c_sq));
    if (workspace.d_x_sq) CUDA_CHECK(cudaFree(workspace.d_x_sq));
    workspace = FlashKMeansAssignWorkspace{};
}

FlashKMeansAssignResult flash_kmeans_assign_l2_precomputed_norms(
    const float* d_data,
    const float* d_centroids,
    const float* d_xnorm2,
    const float* d_cnorm2,
    int n,
    int k,
    int dim,
    idx_t* d_best_idx,
    float* d_best_dist,
    const FlashKMeansAssignConfig& config,
    FlashKMeansAssignWorkspace& workspace,
    cudaStream_t stream)
{
    FlashKMeansAssignResult result;
    if (config.variant != 8) {
        //  CUTLASS  +  partial-min(variant 8),
        CUDA_CHECK(cudaErrorInvalidValue);
    }
    if (!d_data || !d_centroids || !d_xnorm2 || !d_cnorm2 ||
        !d_best_idx || !d_best_dist ||
        n <= 0 || k <= 0 || dim <= 0) {
        return result;
    }

    const int point_tile = config.point_tile;
    const int centroid_tile = config.centroid_tile;
    result.variant = config.variant;
    result.point_tile = point_tile;
    result.centroid_tile = centroid_tile;

    unsigned long long* d_partial_packed_workspace = nullptr;
    const int v6_ctile = (config.centroid_tile == 128) ? 128 : 256;
    const int v7_ptile = (config.point_tile == 64) ? 64 : 128;
    const int partial_tiles = (k + v6_ctile - 1) / v6_ctile;
    if (config.variant == 8) {
        if (!workspace.d_partial_packed ||
            workspace.n_capacity < n ||
            workspace.partial_tiles_capacity < partial_tiles) {
            CUDA_CHECK(cudaErrorInvalidValue);
        }
        d_partial_packed_workspace = workspace.d_partial_packed;
    }

    cudaEvent_t start{};
    cudaEvent_t after_norm{};
    cudaEvent_t after_init{};
    cudaEvent_t after_gemm{};
    cudaEvent_t after_partial_reduce{};
    cudaEvent_t stop{};
    if (config.collect_timing) {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&after_norm));
        CUDA_CHECK(cudaEventCreate(&after_init));
        CUDA_CHECK(cudaEventCreate(&after_gemm));
        CUDA_CHECK(cudaEventCreate(&after_partial_reduce));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, stream));
    }

    if (config.variant == 8) {
        result.point_tile = v7_ptile;
        result.centroid_tile = v6_ctile;
        init_packed_partials_kernel<<<
            ((size_t)n * partial_tiles + 255) / 256,
            256,
            0,
            stream>>>(
            d_partial_packed_workspace,
            (size_t)n * partial_tiles);
        CUDA_LAUNCH_CHECK("flash_assign_init_packed_partials_kernel");
        if (config.collect_timing) {
            CUDA_CHECK(cudaEventRecord(after_norm, stream));
            CUDA_CHECK(cudaEventRecord(after_init, stream));
        }
            launch_flash_assign_min_gemm_v8(
                d_data,
                d_centroids,
                d_xnorm2,
                d_cnorm2,
                n,
                k,
                dim,
                partial_tiles,
                v7_ptile,
                v6_ctile,
                config.cutlass_k_tile,
                config.cutlass_stages,
                config.shared_reduce_min_tiles,
                d_partial_packed_workspace,
                stream);
        if (config.collect_timing) {
            CUDA_CHECK(cudaEventRecord(after_gemm, stream));
        }
        if (partial_tiles <= config.small_reduce_max_tiles) {
            final_min_from_packed_partials_small_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
                d_partial_packed_workspace,
                n,
                partial_tiles,
                d_best_idx,
                d_best_dist);
            CUDA_LAUNCH_CHECK("flash_assign_final_min_from_packed_partials_small_kernel");
        } else {
            final_min_from_packed_partials_kernel<<<n, 256, 0, stream>>>(
                d_partial_packed_workspace,
                n,
                partial_tiles,
                d_best_idx,
                d_best_dist);
            CUDA_LAUNCH_CHECK("flash_assign_final_min_from_packed_partials_kernel");
        }
    } else {
        if (config.collect_timing) {
            CUDA_CHECK(cudaEventRecord(after_norm, stream));
        }
    }

    if (config.collect_timing) {
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&result.ms, start, stop));
        CUDA_CHECK(cudaEventElapsedTime(&result.norm_ms, start, after_norm));
        CUDA_CHECK(cudaEventElapsedTime(&result.assign_ms, after_norm, stop));
        if (config.variant == 8) {
            CUDA_CHECK(cudaEventElapsedTime(&result.cutlass_gemm_ms, after_init, after_gemm));
            CUDA_CHECK(cudaEventElapsedTime(&result.reduce_ms, after_gemm, stop));
            result.final_reduce_ms = result.reduce_ms;
        }
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaEventDestroy(after_partial_reduce));
        CUDA_CHECK(cudaEventDestroy(after_gemm));
        CUDA_CHECK(cudaEventDestroy(after_init));
        CUDA_CHECK(cudaEventDestroy(after_norm));
        CUDA_CHECK(cudaEventDestroy(start));
    }
    return result;
}

FlashKMeansAssignResult flash_kmeans_assign_l2(
    const float* d_data,
    const float* d_centroids,
    int n,
    int k,
    int dim,
    idx_t* d_best_idx,
    float* d_best_dist,
    const FlashKMeansAssignConfig& config,
    FlashKMeansAssignWorkspace& workspace,
    cudaStream_t stream)
{
    if (config.variant == 8) {
        if (!workspace.d_x_sq || !workspace.d_c_sq ||
            workspace.n_capacity < n || workspace.k_capacity < k) {
            CUDA_CHECK(cudaErrorInvalidValue);
        }
        flash_assign_l2_norm_kernel<<<n, 256, 0, stream>>>(
            d_data,
            n,
            dim,
            workspace.d_x_sq);
        CUDA_LAUNCH_CHECK("flash_assign_l2_norm_kernel<x>");
        flash_assign_l2_norm_kernel<<<k, 256, 0, stream>>>(
            d_centroids,
            k,
            dim,
            workspace.d_c_sq);
        CUDA_LAUNCH_CHECK("flash_assign_l2_norm_kernel<c>");
    }
    return flash_kmeans_assign_l2_precomputed_norms(
        d_data,
        d_centroids,
        workspace.d_x_sq,
        workspace.d_c_sq,
        n,
        k,
        dim,
        d_best_idx,
        d_best_dist,
        config,
        workspace,
        stream);
}

}  // namespace kmeans
}  // namespace clustering
}  // namespace paraselect
