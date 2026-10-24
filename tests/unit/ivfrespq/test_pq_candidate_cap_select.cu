#include "search/kernels/fine/gpu_fine/pq_adc/pq_candidate_cap_select.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void check(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    std::exit(2);
}

}  // namespace

int main() {
    constexpr int kQueries = 2;
    constexpr int kInputCapacity = 2048;
    constexpr int kOutputStride = 512;
    constexpr int kCandidateK = 256;
    constexpr int kAccepted[kQueries] = {1200, 800};
    constexpr int kTotal = kAccepted[0] + kAccepted[1];

    std::vector<int> h_idx(kTotal);
    std::vector<float> h_dist(kTotal);
    std::vector<int> h_query(kTotal);
    std::vector<int> h_counts{kAccepted[0], kAccepted[1]};
    for (int q = 0; q < kQueries; ++q) {
        const int base = q == 0 ? 0 : kAccepted[0];
        for (int rank = 0; rank < kAccepted[q]; ++rank) {
            const int slot = base + rank;
            // A tie group crosses the output boundary; query 1 has only
            // 100 valid rows followed by reserved, invalid seed slots.
            h_idx[slot] = q == 1 && rank >= 100 ? -1 : q * 100000 + kAccepted[q] - rank;
            h_dist[slot] = h_idx[slot] < 0 ? INFINITY : float(rank / 9 - 50);
            h_query[slot] = q;
        }
    }

    int* d_pool_idx = nullptr;
    float* d_pool_dist = nullptr;
    int* d_pool_query = nullptr;
    int* d_pool_counts = nullptr;
    int* d_output_idx = nullptr;
    float* d_output_dist = nullptr;
    int* d_output_counts = nullptr;
    uint64_t* d_sort_keys_in = nullptr;
    uint64_t* d_sort_keys_out = nullptr;
    int* d_sort_values_out = nullptr;
    int* d_offsets = nullptr;
    void* d_sort_temp = nullptr;

    check(cudaMalloc(&d_pool_idx, sizeof(int) * h_idx.size()), "pool idx alloc");
    check(cudaMalloc(&d_pool_dist, sizeof(float) * h_dist.size()), "pool dist alloc");
    check(cudaMalloc(&d_pool_query, sizeof(int) * h_query.size()), "pool query alloc");
    check(cudaMalloc(&d_pool_counts, sizeof(int) * h_counts.size()), "pool counts alloc");
    check(cudaMalloc(&d_output_idx, sizeof(int) * kQueries * kOutputStride), "output idx alloc");
    check(cudaMalloc(&d_output_dist, sizeof(float) * kQueries * kOutputStride), "output dist alloc");
    check(cudaMalloc(&d_output_counts, sizeof(int) * kQueries), "output counts alloc");
    check(cudaMalloc(&d_sort_keys_in, sizeof(uint64_t) * kInputCapacity), "sort input alloc");
    check(cudaMalloc(&d_sort_keys_out, sizeof(uint64_t) * kInputCapacity), "sort output alloc");
    check(cudaMalloc(&d_sort_values_out, sizeof(int) * kInputCapacity), "sort values alloc");
    check(cudaMalloc(&d_offsets, sizeof(int) * (kQueries + 1)), "offset alloc");

    check(cudaMemcpy(d_pool_idx, h_idx.data(), sizeof(int) * h_idx.size(), cudaMemcpyHostToDevice), "copy idx");
    check(cudaMemcpy(d_pool_dist, h_dist.data(), sizeof(float) * h_dist.size(), cudaMemcpyHostToDevice), "copy dist");
    check(cudaMemcpy(d_pool_query, h_query.data(), sizeof(int) * h_query.size(), cudaMemcpyHostToDevice), "copy query");
    check(cudaMemcpy(d_pool_counts, h_counts.data(), sizeof(int) * h_counts.size(), cudaMemcpyHostToDevice), "copy counts");

    const std::size_t sort_temp_bytes =
        paraselect::search::pq_candidate_cap_select_temp_bytes(kInputCapacity, kQueries);
    if (sort_temp_bytes > 0) check(cudaMalloc(&d_sort_temp, sort_temp_bytes), "sort temp alloc");
    paraselect::search::pq_candidate_cap_select(
        d_pool_idx, d_pool_dist, d_pool_query, d_pool_counts, kTotal, kQueries,
        kOutputStride, kCandidateK, d_sort_keys_in, d_sort_keys_out,
        d_sort_values_out, d_offsets, d_output_idx, d_output_dist,
        d_output_counts, d_sort_temp, sort_temp_bytes, 0);
    check(cudaDeviceSynchronize(), "cap-select sync");

    std::vector<int> out_idx(kQueries * kOutputStride);
    std::vector<float> out_dist(kQueries * kOutputStride);
    std::vector<int> out_counts(kQueries);
    check(cudaMemcpy(out_idx.data(), d_output_idx, sizeof(int) * out_idx.size(), cudaMemcpyDeviceToHost), "read idx");
    check(cudaMemcpy(out_dist.data(), d_output_dist, sizeof(float) * out_dist.size(), cudaMemcpyDeviceToHost), "read dist");
    check(cudaMemcpy(out_counts.data(), d_output_counts, sizeof(int) * out_counts.size(), cudaMemcpyDeviceToHost), "read counts");

    for (int q = 0; q < kQueries; ++q) {
        std::vector<std::pair<float, int>> expected;
        const int base = q == 0 ? 0 : kAccepted[0];
        for (int i = 0; i < kAccepted[q]; ++i)
            if (h_idx[base+i] >= 0) expected.emplace_back(h_dist[base+i], h_idx[base+i]);
        std::sort(expected.begin(), expected.end());
        const int expected_count = std::min(int(expected.size()), kCandidateK);
        if (out_counts[q] != expected_count) return 1;
        for (int rank = 0; rank < expected_count; ++rank) {
            const int slot = q * kOutputStride + rank;
            if (out_idx[slot] != expected[rank].second || out_dist[slot] != expected[rank].first) {
                std::fprintf(stderr, "cap-select differs from CPU sort: q=%d rank=%d\n", q, rank);
                return 1;
            }
        }
        for (int rank = expected_count; rank < kOutputStride; ++rank) {
            const int slot = q * kOutputStride + rank;
            if (out_idx[slot] != -1 || !std::isinf(out_dist[slot])) {
                std::fprintf(stderr, "q=%d padding rank=%d was not reset\n", q, rank);
                return 1;
            }
        }
    }

    cudaFree(d_pool_idx);
    cudaFree(d_pool_dist);
    cudaFree(d_pool_query);
    cudaFree(d_pool_counts);
    cudaFree(d_output_idx);
    cudaFree(d_output_dist);
    cudaFree(d_output_counts);
    cudaFree(d_sort_keys_in);
    cudaFree(d_sort_keys_out);
    cudaFree(d_sort_values_out);
    cudaFree(d_offsets);
    cudaFree(d_sort_temp);
    std::puts("PASS threshold cap-select with independent input capacity and output stride");
    return 0;
}
