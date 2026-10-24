#include "utils/pch.h"
#include "utils/check_cuda_error.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

__global__ void resident_merge_serial_ref_kernel(
    const float* __restrict__ d_probe_dist,
    const idx_t* __restrict__ d_probe_index,
    const idx_t* __restrict__ d_reordered_indices,
    float* __restrict__ d_final_dist,
    idx_t* __restrict__ d_final_index,
    int n_query,
    int n_probes,
    int k,
    int n_total_vectors)
{
    const int q = blockIdx.x;
    if (q >= n_query || threadIdx.x != 0) return;
    float best_dist[256];
    idx_t best_idx[256];
    for (int i = 0; i < k; ++i) {
        best_dist[i] = FLT_MAX;
        best_idx[i] = (idx_t)-1;
    }
    const int candidates = n_probes * k;
    const size_t row_base = (size_t)q * (size_t)candidates;
    for (int c = 0; c < candidates; ++c) {
        const float dist = d_probe_dist[row_base + (size_t)c];
        const idx_t compact_id = d_probe_index[row_base + (size_t)c];
        if (compact_id < 0 || compact_id >= n_total_vectors) continue;
        int pos = -1;
        for (int r = 0; r < k; ++r) {
            if (dist < best_dist[r]) {
                pos = r;
                break;
            }
        }
        if (pos < 0) continue;
        for (int r = k - 1; r > pos; --r) {
            best_dist[r] = best_dist[r - 1];
            best_idx[r] = best_idx[r - 1];
        }
        best_dist[pos] = dist;
        best_idx[pos] = d_reordered_indices[(int)compact_id];
    }
    for (int r = 0; r < k; ++r) {
        d_final_dist[(size_t)q * (size_t)k + (size_t)r] = best_dist[r];
        d_final_index[(size_t)q * (size_t)k + (size_t)r] = best_idx[r];
    }
}

__device__ __forceinline__ bool resident_pair_less(float da, int ia, float db, int ib) {
    return (da < db) || (da == db && ia < ib);
}

template<int KCap>
__device__ void resident_insert_candidate(float dist, int idx, int k, float* best_dist, int* best_idx) {
    if (idx < 0) return;
    if (!resident_pair_less(dist, idx, best_dist[k - 1], best_idx[k - 1])) return;
    int pos = k - 1;
    while (pos > 0 && resident_pair_less(dist, idx, best_dist[pos - 1], best_idx[pos - 1])) {
        best_dist[pos] = best_dist[pos - 1];
        best_idx[pos] = best_idx[pos - 1];
        --pos;
    }
    best_dist[pos] = dist;
    best_idx[pos] = idx;
}

template<int KCap>
__global__ void resident_merge_pq_like_kernel(
    const float* __restrict__ d_probe_dist,
    const idx_t* __restrict__ d_probe_index,
    const idx_t* __restrict__ d_reordered_indices,
    float* __restrict__ d_final_dist,
    idx_t* __restrict__ d_final_index,
    int n_query,
    int n_probes,
    int k,
    int n_total_vectors)
{
    const int q = blockIdx.x;
    const int lane = threadIdx.x & 31;
    if (q >= n_query) return;

    float local_dist[KCap];
    int local_idx[KCap];
    for (int i = 0; i < k; ++i) {
        local_dist[i] = FLT_MAX;
        local_idx[i] = INT_MAX;
    }

    const int width = n_probes * k;
    const size_t base = (size_t)q * (size_t)width;
    for (int col = lane; col < width; col += 32) {
        const idx_t raw_id = d_probe_index[base + (size_t)col];
        if (raw_id < 0 || raw_id >= n_total_vectors || raw_id > (idx_t)INT_MAX) continue;
        resident_insert_candidate<KCap>(
            d_probe_dist[base + (size_t)col],
            (int)raw_id,
            k,
            local_dist,
            local_idx);
    }

    __shared__ float warp_dist[32 * KCap];
    __shared__ int warp_idx[32 * KCap];
    for (int i = 0; i < k; ++i) {
        warp_dist[(size_t)lane * KCap + i] = local_dist[i];
        warp_idx[(size_t)lane * KCap + i] = local_idx[i];
    }
    __syncwarp();

    if (lane == 0) {
        float final_dist[KCap];
        int final_idx[KCap];
        for (int i = 0; i < k; ++i) {
            final_dist[i] = FLT_MAX;
            final_idx[i] = INT_MAX;
        }
        for (int l = 0; l < 32; ++l) {
            for (int i = 0; i < k; ++i) {
                resident_insert_candidate<KCap>(
                    warp_dist[(size_t)l * KCap + i],
                    warp_idx[(size_t)l * KCap + i],
                    k,
                    final_dist,
                    final_idx);
            }
        }
        for (int i = 0; i < k; ++i) {
            const int compact_id = final_idx[i];
            d_final_dist[(size_t)q * (size_t)k + i] = final_dist[i];
            d_final_index[(size_t)q * (size_t)k + i] =
                compact_id == INT_MAX ? (idx_t)-1 : d_reordered_indices[compact_id];
        }
    }
}

void launch_pq_like(
    const float* d_probe_dist,
    const idx_t* d_probe_index,
    const idx_t* d_reordered_indices,
    float* d_final_dist,
    idx_t* d_final_index,
    int n_query,
    int n_probes,
    int k,
    int n_total_vectors)
{
    if (k <= 64) {
        resident_merge_pq_like_kernel<64><<<n_query, 32>>>(
            d_probe_dist, d_probe_index, d_reordered_indices, d_final_dist, d_final_index,
            n_query, n_probes, k, n_total_vectors);
    } else if (k <= 128) {
        resident_merge_pq_like_kernel<128><<<n_query, 32>>>(
            d_probe_dist, d_probe_index, d_reordered_indices, d_final_dist, d_final_index,
            n_query, n_probes, k, n_total_vectors);
    } else {
        throw std::runtime_error("resident_merge_pq_like test supports k <= 128");
    }
}

void run_case(int n_query, int n_probes, int k)
{
    const int n_total = 2000000;
    const int candidates = n_probes * k;
    std::vector<float> h_dist((size_t)n_query * candidates, FLT_MAX);
    std::vector<idx_t> h_idx((size_t)n_query * candidates, (idx_t)-1);
    std::vector<idx_t> h_reordered(n_total);
    for (int i = 0; i < n_total; ++i) h_reordered[i] = (idx_t)(1000000000LL + i * 3LL);

    for (int q = 0; q < n_query; ++q) {
        for (int p = 0; p < n_probes; ++p) {
            for (int r = 0; r < k; ++r) {
                const size_t pos = ((size_t)q * n_probes + p) * k + r;
                const int compact = (q * 9973 + p * 37 + r * 13) % n_total;
                h_idx[pos] = compact;
                h_dist[pos] = (float)(((q + 1) * 1000003 + p * 97 + r * 7) % 100000) * 0.001f;
            }
        }
        // Force winners to span probes and lanes.
        for (int r = 0; r < k; ++r) {
            const int p = (r * 53 + q * 11) % n_probes;
            const size_t pos = ((size_t)q * n_probes + p) * k + r;
            h_idx[pos] = (idx_t)(q * 1000 + r);
            h_dist[pos] = (float)r * 0.125f + (float)q * 0.01f;
        }
        if (n_probes > 3) {
            const size_t invalid = ((size_t)q * n_probes + 3) * k;
            h_idx[invalid] = -1;
            h_dist[invalid] = -1000.0f;
        }
    }

    float *d_dist = nullptr, *d_ref_dist = nullptr, *d_opt_dist = nullptr;
    idx_t *d_idx = nullptr, *d_reordered = nullptr, *d_ref_idx = nullptr, *d_opt_idx = nullptr;
    const size_t in_n = (size_t)n_query * candidates;
    const size_t out_n = (size_t)n_query * k;
    CUDA_CHECK(cudaMalloc(&d_dist, in_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_idx, in_n * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&d_reordered, (size_t)n_total * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&d_ref_dist, out_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_opt_dist, out_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref_idx, out_n * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&d_opt_idx, out_n * sizeof(idx_t)));
    CUDA_CHECK(cudaMemcpy(d_dist, h_dist.data(), in_n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_idx, h_idx.data(), in_n * sizeof(idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reordered, h_reordered.data(), (size_t)n_total * sizeof(idx_t), cudaMemcpyHostToDevice));

    resident_merge_serial_ref_kernel<<<n_query, 1>>>(
        d_dist, d_idx, d_reordered, d_ref_dist, d_ref_idx, n_query, n_probes, k, n_total);
    CUDA_LAUNCH_CHECK("resident_merge_serial_ref_kernel");
    launch_pq_like(d_dist, d_idx, d_reordered, d_opt_dist, d_opt_idx, n_query, n_probes, k, n_total);
    CUDA_LAUNCH_CHECK("resident_merge_pq_like_kernel");
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> ref_dist(out_n), opt_dist(out_n);
    std::vector<idx_t> ref_idx(out_n), opt_idx(out_n);
    CUDA_CHECK(cudaMemcpy(ref_dist.data(), d_ref_dist, out_n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(opt_dist.data(), d_opt_dist, out_n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref_idx.data(), d_ref_idx, out_n * sizeof(idx_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(opt_idx.data(), d_opt_idx, out_n * sizeof(idx_t), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < out_n; ++i) {
        if (std::fabs(ref_dist[i] - opt_dist[i]) > 1e-5f || ref_idx[i] != opt_idx[i]) {
            std::cerr << "mismatch case nq=" << n_query << " np=" << n_probes << " k=" << k
                      << " slot=" << i << " ref=(" << ref_dist[i] << "," << ref_idx[i]
                      << ") opt=(" << opt_dist[i] << "," << opt_idx[i] << ")" << std::endl;
            throw std::runtime_error("resident merge mismatch");
        }
    }

    cudaFree(d_dist);
    cudaFree(d_idx);
    cudaFree(d_reordered);
    cudaFree(d_ref_dist);
    cudaFree(d_opt_dist);
    cudaFree(d_ref_idx);
    cudaFree(d_opt_idx);
}

}  // namespace

int main()
{
    CUDA_CHECK(cudaSetDevice(0));
    run_case(1, 1, 10);
    run_case(8, 64, 10);
    run_case(8, 512, 10);
    run_case(32, 128, 10);
    run_case(4, 512, 32);
    std::cout << "test_resident_merge passed" << std::endl;
    return 0;
}
