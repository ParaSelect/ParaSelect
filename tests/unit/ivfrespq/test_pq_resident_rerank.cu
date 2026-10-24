#include "search/kernels/fine/gpu_fine/resident/pq_resident_rerank.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

template <typename T>
T* upload(const std::vector<T>& host) {
    T* device = nullptr;
    if (cudaMalloc(&device, host.size() * sizeof(T)) != cudaSuccess ||
        cudaMemcpy(
            device, host.data(), host.size() * sizeof(T),
            cudaMemcpyHostToDevice) != cudaSuccess) {
        if (device) cudaFree(device);
        throw std::runtime_error("resident rerank test upload failed");
    }
    return device;
}

bool equal_results(
    const std::vector<int>& actual_ids,
    const std::vector<float>& actual_distances,
    const std::vector<int>& expected_ids,
    const std::vector<float>& expected_distances) {
    if (actual_ids != expected_ids) return false;
    for (size_t i = 0; i < actual_distances.size(); ++i) {
        if (std::abs(actual_distances[i] - expected_distances[i]) != 0.0f) return false;
    }
    return true;
}

}  // namespace

int main() {
    constexpr int dim = 4;
    constexpr int n_vectors = 5;
    constexpr int batch = 2;
    constexpr int candidate_count = 5;
    constexpr int topk = 3;
    constexpr int padded_rows = 64;
    const std::vector<int> compact_to_padded{0, 1, 2, 32, 33};
    const std::vector<int> candidates{0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
    const std::vector<float> queries{
        2, 2, 2, 2,
        12, 12, 12, 12,
    };
    std::vector<uint8_t> vectors_u8(static_cast<size_t>(padded_rows) * dim, 0);
    std::vector<float> vectors_fp32(static_cast<size_t>(padded_rows) * dim, 0.0f);
    const uint8_t row_values[n_vectors] = {0, 1, 2, 10, 20};
    for (int row = 0; row < n_vectors; ++row) {
        const int padded = compact_to_padded[row];
        const int group = padded / 32;
        const int lane = padded % 32;
        for (int d = 0; d < dim; ++d) {
            const size_t offset = static_cast<size_t>(group) * 32 * dim + d * 32 + lane;
            vectors_u8[offset] = row_values[row];
            vectors_fp32[offset] = static_cast<float>(row_values[row]);
        }
    }

    float* d_queries = upload(queries);
    int* d_candidates = upload(candidates);
    int* d_compact_to_padded = upload(compact_to_padded);
    uint8_t* d_vectors_u8 = upload(vectors_u8);
    float* d_vectors_fp32 = upload(vectors_fp32);
    int* d_ids = nullptr;
    float* d_distances = nullptr;
    cudaMalloc(&d_ids, batch * topk * sizeof(int));
    cudaMalloc(&d_distances, batch * topk * sizeof(float));

    const std::vector<int> expected_ids{2, 1, 0, 3, 4, 2};
    const std::vector<float> expected_distances{0, 4, 16, 16, 256, 400};
    for (int use_u8 = 0; use_u8 <= 1; ++use_u8) {
        launch_pq_resident_rerank_full(
            topk, dim, d_queries, d_candidates, candidate_count,
            candidate_count,
            use_u8 ? nullptr : d_vectors_fp32,
            use_u8 ? d_vectors_u8 : nullptr,
            d_compact_to_padded, n_vectors, batch, d_distances, d_ids, nullptr);
        std::vector<int> ids(batch * topk);
        std::vector<float> distances(batch * topk);
        cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(
            distances.data(), d_distances, distances.size() * sizeof(float),
            cudaMemcpyDeviceToHost);
        if (!equal_results(ids, distances, expected_ids, expected_distances)) {
            std::cerr << (use_u8 ? "u8" : "fp32")
                      << " resident rerank result mismatch\n";
            return 1;
        }
    }

    cudaFree(d_distances);
    cudaFree(d_ids);
    cudaFree(d_vectors_fp32);
    cudaFree(d_vectors_u8);
    cudaFree(d_compact_to_padded);
    cudaFree(d_candidates);
    cudaFree(d_queries);
    return 0;
}
