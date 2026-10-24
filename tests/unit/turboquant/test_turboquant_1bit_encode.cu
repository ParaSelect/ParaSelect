#include "quantization/turboquant/turboquant_1bit_encode.cuh"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace tq = paraselect::search::turboquant;

int main() {
    constexpr int rows = 3;
    constexpr int dim = 70;
    constexpr int padded_dim = 128;
    std::vector<float> vectors(rows * dim);
    for (int row = 0; row < rows; ++row) {
        for (int d = 0; d < dim; ++d) {
            vectors[row * dim + d] = static_cast<float>(((row + 3) * (d + 5)) % 23 - 11) / 7.0f;
        }
    }

    float* d_vectors = nullptr;
    float* d_rotated = nullptr;
    float* d_norms = nullptr;
    float* d_norm_sq = nullptr;
    std::uint32_t* d_codes = nullptr;
    assert(cudaMalloc(&d_vectors, vectors.size() * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_rotated, rows * padded_dim * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_norms, rows * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_norm_sq, rows * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_codes, rows * (padded_dim / 32) * sizeof(std::uint32_t)) == cudaSuccess);
    assert(cudaMemcpy(d_vectors, vectors.data(), vectors.size() * sizeof(float), cudaMemcpyHostToDevice) ==
           cudaSuccess);

    tq::turboquant_1bit_encode_gpu(
        {d_vectors, rows, dim, padded_dim, 17u},
        {d_rotated, d_codes, d_norms, d_norm_sq},
        nullptr);
    std::vector<float> rotated(rows * padded_dim);
    std::vector<float> norms(rows);
    std::vector<float> norm_sq(rows);
    std::vector<std::uint32_t> codes(rows * padded_dim / 32);
    assert(cudaMemcpy(rotated.data(), d_rotated, rotated.size() * sizeof(float), cudaMemcpyDeviceToHost) ==
           cudaSuccess);
    assert(cudaMemcpy(norms.data(), d_norms, norms.size() * sizeof(float), cudaMemcpyDeviceToHost) ==
           cudaSuccess);
    assert(cudaMemcpy(norm_sq.data(), d_norm_sq, norm_sq.size() * sizeof(float), cudaMemcpyDeviceToHost) ==
           cudaSuccess);
    assert(cudaMemcpy(codes.data(), d_codes, codes.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost) ==
           cudaSuccess);

    for (int row = 0; row < rows; ++row) {
        float expected_norm_sq = 0.0f;
        float rotated_norm_sq = 0.0f;
        for (int d = 0; d < dim; ++d) {
            const float value = vectors[row * dim + d];
            expected_norm_sq += value * value;
        }
        for (int d = 0; d < padded_dim; ++d) {
            const float value = rotated[row * padded_dim + d];
            rotated_norm_sq += value * value;
            const bool encoded_positive =
                ((codes[row * (padded_dim / 32) + d / 32] >> (d & 31)) & 1u) != 0;
            assert(encoded_positive == (value >= 0.0f));
        }
        assert(std::fabs(norm_sq[row] - expected_norm_sq) < 1.0e-3f);
        assert(std::fabs(rotated_norm_sq - expected_norm_sq) < 1.0e-3f);
        assert(std::fabs(norms[row] * norms[row] - expected_norm_sq) < 1.0e-3f);
    }

    // Chunked IVF database encoding must be row-for-row identical to one-shot
    // encoding, including the global row offset used for list lookup/output.
    constexpr int db_rows = 9;
    constexpr int db_dim = 8;
    constexpr int db_padded_dim = 32;
    constexpr int db_lists = 2;
    const std::vector<std::uint8_t> db_vectors = {
        3, 7, 11, 13, 17, 19, 23, 29,
        31, 2, 5, 9, 14, 18, 22, 26,
        1, 4, 8, 12, 16, 20, 24, 28,
        6, 10, 15, 21, 25, 30, 34, 38,
        0, 3, 6, 9, 12, 15, 18, 21,
        40, 35, 30, 25, 20, 15, 10, 5,
        2, 12, 22, 32, 42, 52, 62, 72,
        9, 8, 7, 6, 5, 4, 3, 2,
        14, 24, 34, 44, 54, 64, 74, 84};
    const std::vector<float> db_centroids = {
        4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f,
        20.0f, 18.0f, 16.0f, 14.0f, 12.0f, 10.0f, 8.0f, 6.0f};
    const std::vector<long long> db_offsets = {0, 4, db_rows};
    std::uint8_t* d_db_chunk = nullptr;
    float* d_db_centroids = nullptr;
    long long* d_db_offsets = nullptr;
    std::uint32_t* d_db_codes_one = nullptr;
    std::uint32_t* d_db_codes_chunked = nullptr;
    float* d_db_scales_one = nullptr;
    float* d_db_scales_chunked = nullptr;
    assert(cudaMalloc(&d_db_chunk, db_vectors.size()) == cudaSuccess);
    assert(cudaMalloc(&d_db_centroids, db_centroids.size() * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_db_offsets, db_offsets.size() * sizeof(long long)) == cudaSuccess);
    assert(cudaMalloc(&d_db_codes_one, db_rows * sizeof(std::uint32_t)) == cudaSuccess);
    assert(cudaMalloc(&d_db_codes_chunked, db_rows * sizeof(std::uint32_t)) == cudaSuccess);
    assert(cudaMalloc(&d_db_scales_one, db_rows * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_db_scales_chunked, db_rows * sizeof(float)) == cudaSuccess);
    assert(cudaMemcpy(d_db_centroids, db_centroids.data(), db_centroids.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(d_db_offsets, db_offsets.data(), db_offsets.size() * sizeof(long long), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(d_db_chunk, db_vectors.data(), db_vectors.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    tq::ivf_res_turboquant_encode_database_u8_gpu(
        {d_db_chunk, d_db_centroids, d_db_offsets, db_rows, db_lists,
         db_dim, db_padded_dim, 17u, 0},
        {nullptr, d_db_codes_one, d_db_scales_one, nullptr}, nullptr);
    assert(cudaDeviceSynchronize() == cudaSuccess);

    for (int start = 0; start < db_rows;) {
        const int count = start == 0 ? 3 : (start == 3 ? 1 : db_rows - start);
        assert(cudaMemcpy(d_db_chunk, db_vectors.data() + start * db_dim,
                          count * db_dim, cudaMemcpyHostToDevice) == cudaSuccess);
        tq::ivf_res_turboquant_encode_database_u8_gpu(
            {d_db_chunk, d_db_centroids, d_db_offsets, count, db_lists,
             db_dim, db_padded_dim, 17u, start},
            {nullptr, d_db_codes_chunked, d_db_scales_chunked, nullptr}, nullptr);
        start += count;
    }
    assert(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<std::uint32_t> db_codes_one(db_rows), db_codes_chunked(db_rows);
    std::vector<float> db_scales_one(db_rows), db_scales_chunked(db_rows);
    assert(cudaMemcpy(db_codes_one.data(), d_db_codes_one,
                      db_codes_one.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(db_codes_chunked.data(), d_db_codes_chunked,
                      db_codes_chunked.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(db_scales_one.data(), d_db_scales_one,
                      db_scales_one.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(db_scales_chunked.data(), d_db_scales_chunked,
                      db_scales_chunked.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int row = 0; row < db_rows; ++row) {
        assert(db_codes_one[row] == db_codes_chunked[row]);
        assert(std::fabs(db_scales_one[row] - db_scales_chunked[row]) < 1.0e-6f);
    }
    cudaFree(d_db_chunk);
    cudaFree(d_db_centroids);
    cudaFree(d_db_offsets);
    cudaFree(d_db_codes_one);
    cudaFree(d_db_codes_chunked);
    cudaFree(d_db_scales_one);
    cudaFree(d_db_scales_chunked);

    cudaFree(d_vectors);
    cudaFree(d_rotated);
    cudaFree(d_norms);
    cudaFree(d_norm_sq);
    cudaFree(d_codes);
    std::puts("TurboQuant 1-bit encode: PASS (one-shot/chunked bitwise equivalence)");
    return 0;
}
