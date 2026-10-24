#include "quantization/turboquant/turboquant_1bit_format.h"
#include "quantization/turboquant/turboquant_1bit_scan.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace tq = paraselect::search::turboquant;

template <typename T>
static T* upload(const std::vector<T>& host) {
    T* device = nullptr;
    assert(cudaMalloc(&device, host.size() * sizeof(T)) == cudaSuccess);
    assert(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) ==
           cudaSuccess);
    return device;
}

// Exact 256-row chunks must preserve global top-K, including the wide
// workspace path, partial groups, ties, and both ping-pong parities.
static void test_wide_topk() {
    constexpr int dim = 128, words = 4, rows = 40001;
    std::vector<float> query(dim, 0.125f), norms(rows);
    auto query_code = tq::pack_signs(query.data(), dim);
    std::vector<std::uint32_t> codes(rows * words);
    std::uint32_t state = 17;
    for (auto& code : codes) {
        state = state * 1664525u + 1013904223u;
        code = state;
    }
    for (int r = 0; r < rows; ++r) norms[r] = 0.5f + (r % 31) * 0.03125f;
    std::vector<tq::TurboQuantChunkDesc> chunks;
    for (int r = 0; r < rows; r += 256) chunks.push_back({0, 0, 0, r, std::min(256, rows-r)});
    auto* dq = upload(query);
    auto* dqc = upload(query_code);
    auto* dc = upload(codes);
    auto* dn = upload(norms);
    auto* dqn = upload(std::vector<float>{2.0f});
    auto* dqns = upload(std::vector<float>{2.0f});
    auto* doff = upload(std::vector<long long>{0});
    auto* dchunks = upload(chunks);
    tq::TurboQuant1BitScanInput input{dq, dqc, dqn, dqns, dc, dn, doff, dchunks,
        static_cast<int>(chunks.size()), dim, words, 256};
    for (auto mode : {tq::TurboQuant1BitScoreMode::BinaryQuery, tq::TurboQuant1BitScoreMode::MseFloatQuery}) {
        std::vector<std::pair<float,int>> reference;
        for (int r = 0; r < rows; ++r) {
            float score = mode == tq::TurboQuant1BitScoreMode::BinaryQuery
                ? tq::binary_l2_distance(query_code.data(), codes.data()+r*words, dim, 2.0f, norms[r])
                : tq::sign_scale_l2_distance(query.data(), codes.data()+r*words, dim, norms[r]);
            reference.emplace_back(score, r);
        }
        std::sort(reference.begin(), reference.end());
        for (int k : {512, 2048, 4096, 8192, 16384}) {
            const int group_chunks = k / 256;
            std::vector<tq::TurboQuantGroupDesc> groups;
            for (int first = 0; first < static_cast<int>(chunks.size()); first += group_chunks)
                groups.push_back({first, std::min(group_chunks, static_cast<int>(chunks.size())-first)});
            auto* dg = upload(groups);
            auto* gi = upload(std::vector<int>(groups.size()*k));
            auto* gd = upload(std::vector<float>(groups.size()*k));
            auto* oi = upload(std::vector<int>(k));
            auto* od = upload(std::vector<float>(k));
            auto* starts = upload(std::vector<int>{0});
            auto* counts = upload(std::vector<int>{static_cast<int>(groups.size())});
            tq::turboquant_1bit_grouped_scan_topk_gpu(input, dg, groups.size(), group_chunks, mode, 0.0f, k, gi, gd, nullptr);
            tq::turboquant_merge_query_groups_topk_gpu(gi, gd, starts, counts, 1, k, oi, od, nullptr);
            std::vector<int> got(k);
            std::vector<float> dist(k);
            assert(cudaMemcpy(got.data(), oi, k*sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
            assert(cudaMemcpy(dist.data(), od, k*sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
            for (int j = 0; j < k; ++j) {
                assert(got[j] == reference[j].second);
                assert(std::fabs(dist[j] - reference[j].first) < 1e-4f);
            }
            cudaFree(dg); cudaFree(gi); cudaFree(gd); cudaFree(oi); cudaFree(od); cudaFree(starts); cudaFree(counts);
        }
    }
    cudaFree(dq); cudaFree(dqc); cudaFree(dc); cudaFree(dn); cudaFree(dqn); cudaFree(dqns); cudaFree(doff); cudaFree(dchunks);
}

int main() {
    test_wide_topk();
    constexpr int dim = 70;
    constexpr int rows = 4096;
    constexpr int local_k = 256;
    const int words = static_cast<int>(tq::packed_word_count(dim));

    std::vector<float> query(dim);
    for (int d = 0; d < dim; ++d) query[d] = static_cast<float>((d % 11) - 5) / 8.0f;
    const std::vector<std::uint32_t> query_code = tq::pack_signs(query.data(), dim);

    std::vector<std::uint32_t> codes(static_cast<std::size_t>(rows) * words);
    std::vector<float> row_norms(rows);
    std::vector<float> values(dim);
    for (int row = 0; row < rows; ++row) {
        for (int d = 0; d < dim; ++d) {
            values[d] = ((row * 17 + d * 13) % 19) < 9 ? -1.0f : 1.0f;
        }
        const auto packed = tq::pack_signs(values.data(), dim);
        std::copy(packed.begin(), packed.end(), codes.begin() + static_cast<std::size_t>(row) * words);
        row_norms[row] = 0.5f + static_cast<float>(row % 7) * 0.25f;
    }

    std::vector<tq::TurboQuantChunkDesc> desc{
        {0, 0, 0, 0, 256},
        {0, 0, 0, 0, 2048},
        {0, 0, 0, 2048, 2048},
    };
    std::vector<tq::TurboQuantGroupDesc> groups{{1, 2}};
    std::vector<long long> offsets{0};
    std::vector<float> query_norms{2.0f};
    std::vector<float> query_norm_sq{
        std::inner_product(query.begin(), query.end(), query.begin(), 0.0f)};

    float* d_query = upload(query);
    std::uint32_t* d_query_code = upload(query_code);
    std::uint32_t* d_codes = upload(codes);
    float* d_row_norms = upload(row_norms);
    float* d_query_norms = upload(query_norms);
    float* d_query_norm_sq = upload(query_norm_sq);
    tq::TurboQuantChunkDesc* d_desc = upload(desc);
    tq::TurboQuantGroupDesc* d_groups = upload(groups);
    long long* d_offsets = upload(offsets);
    float* d_dist = nullptr;
    int* d_idx = nullptr;
    assert(cudaMalloc(&d_dist, 512 * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_idx, 512 * sizeof(int)) == cudaSuccess);

    tq::TurboQuant1BitScanInput input{};
    input.rotated_queries = d_query;
    input.packed_query_codes = d_query_code;
    input.query_norms = d_query_norms;
    input.query_norm_sq = d_query_norm_sq;
    input.packed_database_codes = d_codes;
    input.database_row_parameters = d_row_norms;
    input.cluster_offsets = d_offsets;
    input.chunks = d_desc;
    input.n_chunks = 1;
    input.dim = dim;
    input.words_per_code = words;
    input.vector_chunk_size = 256;

    tq::turboquant_1bit_scan_top256_gpu(
        input, tq::TurboQuant1BitScoreMode::MseFloatQuery, 0.25f, d_idx, d_dist, nullptr);
    std::vector<float> got_dist(local_k);
    std::vector<int> got_idx(local_k);
    assert(cudaMemcpy(got_dist.data(), d_dist, local_k * sizeof(float), cudaMemcpyDeviceToHost) ==
           cudaSuccess);
    assert(cudaMemcpy(got_idx.data(), d_idx, local_k * sizeof(int), cudaMemcpyDeviceToHost) ==
           cudaSuccess);
    std::vector<std::pair<float, int>> expected;
    for (int row = 0; row < 256; ++row) {
        expected.emplace_back(
            tq::sign_scale_l2_distance(
                query.data(), codes.data() + static_cast<std::size_t>(row) * words,
                dim, row_norms[row]),
            row);
    }
    std::sort(expected.begin(), expected.end());
    for (int k = 0; k < local_k; ++k) {
        assert(got_idx[k] == expected[k].second);
        assert(std::fabs(got_dist[k] - expected[k].first) < 1.0e-4f);
    }

    input.n_chunks = 3;
    input.vector_chunk_size = 2048;
    tq::turboquant_1bit_grouped_scan_top512_gpu(
        input, d_groups, 1, 2, tq::TurboQuant1BitScoreMode::MseFloatQuery,
        0.25f, d_idx, d_dist, nullptr);
    got_dist.resize(512);
    got_idx.resize(512);
    assert(cudaMemcpy(got_dist.data(), d_dist, 512 * sizeof(float), cudaMemcpyDeviceToHost) ==
           cudaSuccess);
    assert(cudaMemcpy(got_idx.data(), d_idx, 512 * sizeof(int), cudaMemcpyDeviceToHost) ==
           cudaSuccess);
    expected.clear();
    for (int chunk = 0; chunk < 2; ++chunk) {
        const int chunk_start = chunk * 2048;
        for (int lane = 0; lane < 256; ++lane) {
            std::pair<float, int> best{INFINITY, -1};
            for (int item = lane; item < 2048; item += 256) {
                const int row = chunk_start + item;
                const float distance = tq::sign_scale_l2_distance(
                    query.data(), codes.data() + static_cast<std::size_t>(row) * words,
                    dim, row_norms[row]);
                best = std::min(best, std::pair<float, int>{distance, row});
            }
            expected.push_back(best);
        }
    }
    std::sort(expected.begin(), expected.end());
    for (int k = 0; k < 512; ++k) {
        assert(got_idx[k] == expected[k].second);
        assert(std::fabs(got_dist[k] - expected[k].first) < 1.0e-4f);
    }

    input.n_chunks = 1;
    input.vector_chunk_size = 256;
    tq::turboquant_1bit_scan_top256_gpu(
        input, tq::TurboQuant1BitScoreMode::BinaryQuery, 0.0f, d_idx, d_dist, nullptr);
    assert(cudaMemcpy(got_dist.data(), d_dist, local_k * sizeof(float), cudaMemcpyDeviceToHost) ==
           cudaSuccess);
    assert(cudaMemcpy(got_idx.data(), d_idx, local_k * sizeof(int), cudaMemcpyDeviceToHost) ==
           cudaSuccess);
    expected.clear();
    for (int row = 0; row < 256; ++row) {
        expected.emplace_back(
            tq::binary_l2_distance(query_code.data(),
                codes.data() + static_cast<std::size_t>(row) * words,
                dim, query_norms[0], row_norms[row]),
            row);
    }
    std::sort(expected.begin(), expected.end());
    for (int k = 0; k < local_k; ++k) {
        assert(got_idx[k] == expected[k].second);
        assert(std::fabs(got_dist[k] - expected[k].first) < 1.0e-4f);
    }

    input.n_chunks = 3;
    input.vector_chunk_size = 2048;
    tq::turboquant_1bit_grouped_scan_top512_gpu(
        input, d_groups, 1, 2, tq::TurboQuant1BitScoreMode::BinaryQuery,
        0.0f, d_idx, d_dist, nullptr);
    assert(cudaMemcpy(got_dist.data(), d_dist, 512 * sizeof(float), cudaMemcpyDeviceToHost) ==
           cudaSuccess);
    assert(cudaMemcpy(got_idx.data(), d_idx, 512 * sizeof(int), cudaMemcpyDeviceToHost) ==
           cudaSuccess);
    expected.clear();
    for (int chunk = 0; chunk < 2; ++chunk) {
        const int chunk_start = chunk * 2048;
        for (int lane = 0; lane < 256; ++lane) {
            std::pair<float, int> best{INFINITY, -1};
            for (int item = lane; item < 2048; item += 256) {
                const int row = chunk_start + item;
                const float distance = tq::binary_l2_distance(
                    query_code.data(),
                    codes.data() + static_cast<std::size_t>(row) * words,
                    dim, query_norms[0], row_norms[row]);
                best = std::min(best, std::pair<float, int>{distance, row});
            }
            expected.push_back(best);
        }
    }
    std::sort(expected.begin(), expected.end());
    for (int k = 0; k < 512; ++k) {
        assert(got_idx[k] == expected[k].second);
        assert(std::fabs(got_dist[k] - expected[k].first) < 1.0e-4f);
    }

    // Query-level merge must be identical to sorting all per-group candidates.
    constexpr int merge_groups = 3;
    std::vector<float> merge_dist(merge_groups * 512);
    std::vector<int> merge_idx(merge_groups * 512);
    expected.clear();
    for (int group = 0; group < merge_groups; ++group) {
        for (int k = 0; k < 512; ++k) {
            const int idx = group * 10000 + k;
            const float distance = static_cast<float>(k * merge_groups + (2 - group));
            merge_dist[group * 512 + k] = distance;
            merge_idx[group * 512 + k] = idx;
            expected.emplace_back(distance, idx);
        }
    }
    std::sort(expected.begin(), expected.end());
    float* d_merge_dist = upload(merge_dist);
    int* d_merge_idx = upload(merge_idx);
    int* d_query_group_starts = upload(std::vector<int>{0});
    int* d_query_group_counts = upload(std::vector<int>{merge_groups});
    tq::turboquant_merge_query_groups_top512_gpu(
        d_merge_idx, d_merge_dist, d_query_group_starts, d_query_group_counts,
        1, d_idx, d_dist, nullptr);
    assert(cudaMemcpy(got_dist.data(), d_dist, 512 * sizeof(float), cudaMemcpyDeviceToHost) ==
           cudaSuccess);
    assert(cudaMemcpy(got_idx.data(), d_idx, 512 * sizeof(int), cudaMemcpyDeviceToHost) ==
           cudaSuccess);
    for (int k = 0; k < 512; ++k) {
        assert(got_idx[k] == expected[k].second);
        assert(std::fabs(got_dist[k] - expected[k].first) < 1.0e-5f);
    }
    cudaFree(d_merge_dist);
    cudaFree(d_merge_idx);
    cudaFree(d_query_group_starts);
    cudaFree(d_query_group_counts);

    cudaFree(d_query);
    cudaFree(d_query_code);
    cudaFree(d_codes);
    cudaFree(d_row_norms);
    cudaFree(d_query_norms);
    cudaFree(d_query_norm_sq);
    cudaFree(d_desc);
    cudaFree(d_groups);
    cudaFree(d_offsets);
    cudaFree(d_dist);
    cudaFree(d_idx);
    return 0;
}
