// clustering/utils.cu
#include "utils.cuh"
#include "utils/utils.cuh"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

// ============================================================
// GPU Kernels Implementation
// ============================================================

//  int64_t ()
// __device__ __forceinline__ idx_t atomicAdd(idx_t* address, idx_t val);

__global__ void kernel_count_clusters(const idx_t* __restrict__ assign,
                                      int* __restrict__ counts,
                                      int n, int k,
                                      unsigned int* __restrict__ invalid_counter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int c = assign[i];
        if (c >= 0 && c < k) {
            atomicAdd(&counts[c], 1);
        } else {
            if (invalid_counter) atomicAdd(invalid_counter, 1u);
        }
    }
}

//  int64_t
__device__ __forceinline__ idx_t atomicAdd(idx_t* address, idx_t val) {
    return atomicAdd(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(val));
}

__global__ void kernel_compute_positions(const idx_t* __restrict__ assign,
                                         idx_t* __restrict__ write_ptr,
                                         idx_t* __restrict__ out_pos,
                                         int n, int k,
                                         unsigned int* __restrict__ invalid_counter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int c = assign[i];
        if (c >= 0 && c < k) {
            idx_t p = atomicAdd(&write_ptr[c], (idx_t)1);
            out_pos[i] = p;
        } else {
            out_pos[i] = ~(idx_t)0;
            if (invalid_counter) atomicAdd(invalid_counter, 1u);
        }
    }
}

__global__ void kernel_scatter_perm(const idx_t* __restrict__ pos,
                                   idx_t* __restrict__ perm,
                                   int base, int curB, idx_t n,
                                   unsigned int* __restrict__ oob_counter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < curB) {
        idx_t p = pos[i];
        if (p < n) {
            perm[p] = base + i;
        } else {
            if (oob_counter) atomicAdd(oob_counter, 1u);
        }
    }
}

// ============================================================
// GPU Permutation Builder Implementation
// ============================================================

void gpu_build_permutation_by_cluster(
    int n,
    int k,
    const idx_t* d_assign,
    idx_t* h_perm_out,
    ClusterInfo* h_cluster_info,
    int device_id,
    int B,
    cudaStream_t stream
) {
    if (n <= 0 || k <= 0) return;

    cudaSetDevice(device_id);
    CHECK_CUDA_ERRORS;

    // Device buffers
    int* d_counts = nullptr;
    idx_t* d_writeptr = nullptr;
    idx_t* d_pos_b[2] = {nullptr, nullptr};
    idx_t* d_perm = nullptr;

    unsigned int* d_invalid = nullptr;
    unsigned int* d_oob = nullptr;

    cudaMalloc(&d_counts, sizeof(int) * (size_t)k);
    cudaMalloc(&d_writeptr, sizeof(idx_t) * (size_t)k);
    cudaMalloc(&d_perm, sizeof(idx_t) * (size_t)n);
    CHECK_CUDA_ERRORS;

    cudaMalloc(&d_invalid, sizeof(unsigned int));
    cudaMalloc(&d_oob, sizeof(unsigned int));
    cudaMemsetAsync(d_invalid, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(d_oob, 0, sizeof(unsigned int), stream);
    CHECK_CUDA_ERRORS;

    for (int t = 0; t < 2; ++t) {
        cudaMalloc(&d_pos_b[t], sizeof(idx_t) * (size_t)B);
    }
    CHECK_CUDA_ERRORS;

    // PASS 1: count clusters
    cudaMemsetAsync(d_counts, 0, sizeof(int) * (size_t)k, stream);
    CHECK_CUDA_ERRORS;

    int cur_buf = 0;
    for (int base = 0; base < n; base += B) {
        int curB = std::min(B, n - base);
        int buf = cur_buf;

        const idx_t* d_assign_batch = d_assign + base;

        int threads = 256;
        int blocks = (curB + threads - 1) / threads;
        kernel_count_clusters<<<blocks, threads, 0, stream>>>(
            d_assign_batch, d_counts, curB, k, d_invalid);
        CHECK_CUDA_ERRORS;

        cur_buf = 1 - cur_buf;
    }

    cudaStreamSynchronize(stream);
    CHECK_CUDA_ERRORS;

    // Build offsets on host
    std::vector<int> h_counts(k);
    cudaMemcpy(h_counts.data(), d_counts, sizeof(int) * (size_t)k,
               cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERRORS;

    std::vector<idx_t> h_offsets;
    idx_t total = 0;
    build_offsets_host(h_counts, h_offsets, total);

    if (total != (idx_t)n) {
        fprintf(stderr,
                "[reorder] WARNING: total counts (%lld) != n (%d). "
                "Some assignments are invalid/out-of-range.\n",
                (long long)total, n);
    }

    // Init write_ptr = offsets
    cudaMemcpy(d_writeptr, h_offsets.data(),
               sizeof(idx_t) * (size_t)k,
               cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS;

    // Optional: output cluster info
    if (h_cluster_info) {
        h_cluster_info->k = k;
        h_cluster_info->counts = (int*)std::malloc(sizeof(int) * (size_t)k);
        h_cluster_info->offsets = (idx_t*)std::malloc(sizeof(idx_t) * ((size_t)k + 1u));
        if (!h_cluster_info->counts || !h_cluster_info->offsets) {
            fprintf(stderr, "[reorder] ERROR: malloc failed for cluster_info.\n");
            std::abort();
        }
        std::memcpy(h_cluster_info->counts, h_counts.data(), sizeof(int) * (size_t)k);
        std::memcpy(h_cluster_info->offsets, h_offsets.data(), sizeof(idx_t) * ((size_t)k + 1u));
    }

    // PASS 2: compute positions + scatter permutation
    cur_buf = 0;
    for (int base = 0; base < n; base += B) {
        int curB = std::min(B, n - base);
        int buf = cur_buf;

        const idx_t* d_assign_batch = d_assign + base;

        int threads = 256;
        int blocks = (curB + threads - 1) / threads;

        kernel_compute_positions<<<blocks, threads, 0, stream>>>(
            d_assign_batch, d_writeptr, d_pos_b[buf], curB, k, d_invalid);
        CHECK_CUDA_ERRORS;

        kernel_scatter_perm<<<blocks, threads, 0, stream>>>(
            d_pos_b[buf], d_perm, base, curB, (idx_t)n, d_oob);
        CHECK_CUDA_ERRORS;

        cur_buf = 1 - cur_buf;
    }

    cudaStreamSynchronize(stream);
    CHECK_CUDA_ERRORS;

    // Check invalid/oob counters
    unsigned int h_invalid = 0, h_oob = 0;
    cudaMemcpy(&h_invalid, d_invalid, sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_oob, d_oob, sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERRORS;
    if (h_invalid != 0 || h_oob != 0) {
        fprintf(stderr, "[reorder] WARNING: invalid_assign=%u, oob_scatter=%u\n",
                h_invalid, h_oob);
    }

    // Copy perm back if requested
    if (h_perm_out) {
        cudaMemcpy(h_perm_out, d_perm, sizeof(idx_t) * (size_t)n,
                   cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERRORS;
    }

    // Cleanup
    for (int t = 0; t < 2; ++t) {
        cudaFree(d_pos_b[t]);
    }
    cudaFree(d_counts);
    cudaFree(d_writeptr);
    cudaFree(d_perm);
    if (d_invalid) cudaFree(d_invalid);
    cudaFree(d_oob);
}

// ============================================================
// GPU ClusterInfo Builder (no perm)
// ============================================================

void gpu_build_cluster_info(
    int n,
    int k,
    const idx_t* d_assign,
    ClusterInfo* h_cluster_info,
    int device_id,
    cudaStream_t stream
) {
    if (n <= 0 || k <= 0 || !d_assign || !h_cluster_info) return;

    cudaSetDevice(device_id);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        fprintf(stderr, "[gpu_build_cluster_info] pre-existing CUDA error after cudaSetDevice: %s\n",
                cudaGetErrorString(status));
    }

    int* d_counts = nullptr;
    idx_t* d_offsets = nullptr;
    unsigned int* d_invalid = nullptr;
    void* d_scan_temp = nullptr;
    size_t scan_temp_bytes = 0;

    if (status == cudaSuccess) status = cudaMalloc(&d_counts, sizeof(int) * (size_t)k);
    if (status == cudaSuccess) status = cudaMalloc(&d_offsets, sizeof(idx_t) * (size_t)k);
    if (status == cudaSuccess) status = cudaMalloc(&d_invalid, sizeof(unsigned int));
    if (status == cudaSuccess) status = cudaMemsetAsync(d_counts, 0, sizeof(int) * (size_t)k, stream);
    if (status == cudaSuccess) status = cudaMemsetAsync(d_invalid, 0, sizeof(unsigned int), stream);

    if (status == cudaSuccess) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        kernel_count_clusters<<<blocks, threads, 0, stream>>>(
            d_assign, d_counts, n, k, d_invalid);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);

    if (status == cudaSuccess) {
        status = cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes,
                                               d_counts, d_offsets, k, stream);
    }
    if (status == cudaSuccess) status = cudaMalloc(&d_scan_temp, scan_temp_bytes);
    if (status == cudaSuccess) {
        status = cub::DeviceScan::ExclusiveSum(d_scan_temp, scan_temp_bytes,
                                               d_counts, d_offsets, k, stream);
    }
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);

    if (status == cudaSuccess) {
        h_cluster_info->k = k;
        h_cluster_info->counts = (int*)std::malloc(sizeof(int) * (size_t)k);
        h_cluster_info->offsets = (idx_t*)std::malloc(sizeof(idx_t) * ((size_t)k + 1u));
        if (!h_cluster_info->counts || !h_cluster_info->offsets) {
            fprintf(stderr, "[gpu_build_cluster_info] ERROR: malloc failed for cluster_info.\n");
            std::abort();
        }
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(h_cluster_info->counts, d_counts, sizeof(int) * (size_t)k,
                            cudaMemcpyDeviceToHost);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(h_cluster_info->offsets, d_offsets, sizeof(idx_t) * (size_t)k,
                            cudaMemcpyDeviceToHost);
        if (status == cudaSuccess) h_cluster_info->offsets[k] = (idx_t)n;
    }

    unsigned int h_invalid = 0;
    if (status == cudaSuccess) {
        status = cudaMemcpy(&h_invalid, d_invalid, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (status == cudaSuccess && h_invalid != 0) {
            fprintf(stderr, "[gpu_build_cluster_info] WARNING: invalid_assign=%u\n", h_invalid);
        }
    }

    if (status != cudaSuccess) {
        fprintf(stderr, "[gpu_build_cluster_info] ERROR exit err=%s\n", cudaGetErrorString(status));
    }

    if (d_counts) cudaFree(d_counts);
    if (d_offsets) cudaFree(d_offsets);
    if (d_invalid) cudaFree(d_invalid);
    if (d_scan_temp) cudaFree(d_scan_temp);
}

// ============================================================
// Utility Functions Implementation
// ============================================================

void free_cluster_info(ClusterInfo* info, bool is_device) {
    if (!info) return;
    if (is_device) {
        if (info->offsets) cudaFree(info->offsets);
        if (info->counts) cudaFree(info->counts);
    } else {
        if (info->offsets) std::free(info->offsets);
        if (info->counts) std::free(info->counts);
    }
    info->offsets = nullptr;
    info->counts = nullptr;
    info->k = 0;
}
