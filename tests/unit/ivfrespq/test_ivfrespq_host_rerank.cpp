#include "cpu/distance/distance_dispatch.h"
#include "search/ivf_common/ivf_exact_topk.h"
#include "search/ivf_common/ivf_host_rerank.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

int fail(const std::string& message) {
    std::cerr << message << '\n';
    return 1;
}

template <typename Value>
void exact_rerank_reference(
    const Value* storage,
    const Value* queries,
    int count,
    int dim,
    const int* candidates,
    int batch,
    int candidate_count,
    int candidate_stride,
    int topk,
    int* output_ids,
    float* output_distances) {
    const auto distance = [] {
        if constexpr (std::is_same_v<Value, float>) {
            return paraselect::cpu_distance::select_fp32_l2_fn();
        } else {
            return paraselect::cpu_distance::select_u8_l2_fn();
        }
    }();
    std::vector<int> unique_candidates;
    std::vector<int> heap_ids(topk);
    std::vector<float> heap_distances(topk);
    for (int query = 0; query < batch; ++query) {
        unique_candidates.clear();
        for (int slot = 0; slot < candidate_count; ++slot) {
            const int id = candidates[static_cast<size_t>(query) * candidate_stride + slot];
            if (id >= 0 && id < count) unique_candidates.push_back(id);
        }
        std::sort(unique_candidates.begin(), unique_candidates.end());
        unique_candidates.erase(
            std::unique(unique_candidates.begin(), unique_candidates.end()),
            unique_candidates.end());
        int heap_size = 0;
        for (const int id : unique_candidates) {
            const auto raw_distance = distance(
                queries + static_cast<size_t>(query) * dim,
                storage + static_cast<size_t>(id) * dim,
                dim);
            paraselect::search::ivf_insert_exact_topk(
                static_cast<float>(raw_distance), id, topk,
                heap_distances.data(), heap_ids.data(), heap_size);
        }
        paraselect::search::ivf_finalize_exact_topk(
            heap_distances.data(), heap_ids.data(), heap_size, topk);
        std::copy_n(heap_ids.data(), topk, output_ids + static_cast<size_t>(query) * topk);
        std::copy_n(
            heap_distances.data(), topk,
            output_distances + static_cast<size_t>(query) * topk);
    }
}

template <typename Value>
int test_dtype(bool use_float) {
    constexpr int count = 257;
    constexpr int dim = 96;
    constexpr int batch = 4;
    constexpr int candidate_count = 14;
    constexpr int topk = 6;
    std::vector<Value> storage(static_cast<size_t>(count) * dim);
    std::vector<Value> queries(static_cast<size_t>(batch) * dim);
    for (size_t i = 0; i < storage.size(); ++i) {
        storage[i] = static_cast<Value>((i * 13 + i / dim * 7) % 251);
    }
    for (int query = 0; query < batch; ++query) {
        std::copy_n(
            storage.data() + static_cast<size_t>(query * 31 + 5) * dim,
            dim,
            queries.data() + static_cast<size_t>(query) * dim);
    }
    const std::vector<int> candidates = {
        5, 1, 8, 13, 55, 89, 144, 233, 5, -1, 256, 300, 21, 34,
        36, 2, 9, 14, 56, 90, 145, 234, 36, -1, 255, 301, 22, 35,
        67, 3, 10, 15, 57, 91, 146, 235, 67, -1, 254, 302, 23, 35,
        98, 4, 11, 16, 58, 92, 147, 236, 98, -1, 253, 303, 24, 37,
    };
    constexpr int candidate_stride = candidate_count + 3;
    std::vector<int> strided_candidates(static_cast<size_t>(batch) * candidate_stride, -7);
    for (int query = 0; query < batch; ++query) {
        std::copy_n(
            candidates.data() + static_cast<size_t>(query) * candidate_count,
            candidate_count,
            strided_candidates.data() + static_cast<size_t>(query) * candidate_stride);
    }
    std::vector<int> expected_ids(batch * topk), actual_ids(batch * topk);
    std::vector<float> expected_distances(batch * topk), actual_distances(batch * topk);
    exact_rerank_reference(
        storage.data(), queries.data(), count, dim, strided_candidates.data(), batch,
        candidate_count, candidate_stride, topk,
        expected_ids.data(), expected_distances.data());

    paraselect::search::IVFRerankStats stats;
    paraselect::search::IVFHostStorageView view;
    view.count = count;
    view.dim = dim;
    if constexpr (std::is_same_v<Value, float>) view.fp32 = storage.data();
    else view.u8 = storage.data();
    ivf_rerank_host(
        view,
        use_float ? nullptr : reinterpret_cast<const uint8_t*>(queries.data()),
        use_float ? reinterpret_cast<const float*>(queries.data()) : nullptr,
        use_float,
        strided_candidates.data(), batch, candidate_count, candidate_stride, topk,
        actual_ids.data(), actual_distances.data(), 3, false, 0, &stats);
    if (actual_ids != expected_ids || actual_distances != expected_distances) {
        return fail(use_float ? "fp32 host rerank differs from exact reference"
                              : "u8 host rerank differs from exact reference");
    }
    if (stats.threads <= 0 || stats.avg_valid_candidates <= 0.0) {
        return fail("host rerank did not report execution statistics");
    }
    return 0;
}

}  // namespace

int main() {
    if (const int result = test_dtype<uint8_t>(false); result != 0) return result;
    return test_dtype<float>(true);
}
