#include "search/ivf_common/ivf_host_rerank.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

template<class T>
void check(int dim, int width, int threads, int prefetch) {
    constexpr int rows = 263, batch = 4, topk = 10;
    const int stride = width + 13;
    std::vector<T> base(rows * dim), queries(batch * dim);
    for (int r = 0; r < rows; ++r)
        for (int d = 0; d < dim; ++d)
            base[r * dim + d] = T(((r % 17) * 11 + d * 3) % 251);
    for (int q = 0; q < batch; ++q)
        std::copy_n(base.data() + (q + 2) * dim, dim, queries.data() + q * dim);
    const int counts[batch] = {0, 1, width / 2, width};
    std::vector<int> candidates(batch * stride, 0);
    for (int q = 0; q < batch; ++q)
        for (int i = 0; i < counts[q]; ++i)
            candidates[q * stride + i] = i % 19 == 0 ? -1 :
                (i % 23 == 0 ? rows + 2 : ((width - i) * 17 + q) % rows);
    std::vector<int> expected(batch * topk, -1), actual(batch * topk, -9);
    std::vector<float> expected_d(batch * topk, std::numeric_limits<float>::max());
    std::vector<float> actual_d(batch * topk);
    for (int q = 0; q < batch; ++q) {
        std::vector<bool> seen(rows, false);
        std::vector<std::pair<float, int>> scored;
        for (int i = 0; i < counts[q]; ++i) {
            const int id = candidates[q * stride + i];
            if (id < 0 || id >= rows || seen[id]) continue;
            seen[id] = true;
            float sum = 0;
            for (int d = 0; d < dim; ++d) {
                const float delta = float(queries[q * dim + d]) - float(base[id * dim + d]);
                sum += delta * delta;
            }
            scored.emplace_back(sum, id);
        }
        std::sort(scored.begin(), scored.end());
        for (int k = 0; k < std::min(topk, int(scored.size())); ++k) {
            expected[q * topk + k] = scored[k].second;
            expected_d[q * topk + k] = scored[k].first;
        }
    }
    paraselect::search::IVFHostStorageView storage;
    storage.count = rows;
    storage.dim = dim;
    constexpr bool fp32 = std::is_same_v<T, float>;
    if constexpr (fp32) storage.fp32 = base.data();
    else storage.u8 = base.data();
    ivf_rerank_host(storage,
        fp32 ? nullptr : reinterpret_cast<const uint8_t*>(queries.data()),
        fp32 ? reinterpret_cast<const float*>(queries.data()) : nullptr,
        fp32, candidates.data(), batch, width, stride, topk,
        actual.data(), actual_d.data(), threads, false, prefetch, nullptr, counts);
    if (actual != expected || actual_d != expected_d)
        throw std::runtime_error("shared rerank differs from scalar sort/unique reference");
    std::vector<int> order(rows), scan_candidates = candidates;
    for (int r = 0; r < rows; ++r) order[r] = rows - 1 - r;
    for (auto& id : scan_candidates)
        id = id >= 0 && id < rows ? rows - 1 - id : -1;
    ivf_rerank_host(storage,
        fp32 ? nullptr : reinterpret_cast<const uint8_t*>(queries.data()),
        fp32 ? reinterpret_cast<const float*>(queries.data()) : nullptr,
        fp32, scan_candidates.data(), batch, width, stride, topk,
        actual.data(), actual_d.data(), threads, false, prefetch, nullptr, counts, order.data());
    if (actual != expected || actual_d != expected_d)
        throw std::runtime_error("mapped candidate rerank changed IDs or distances");
    std::vector<T> sorted_base(base.size());
    for (int r = 0; r < rows; ++r)
        std::copy_n(base.data() + size_t(order[r]) * dim, dim,
                    sorted_base.data() + size_t(r) * dim);
    if constexpr (fp32) storage.fp32 = sorted_base.data();
    else storage.u8 = sorted_base.data();
    storage.source_rows = order.data();
    ivf_rerank_host(storage,
        fp32 ? nullptr : reinterpret_cast<const uint8_t*>(queries.data()),
        fp32 ? reinterpret_cast<const float*>(queries.data()) : nullptr,
        fp32, scan_candidates.data(), batch, width, stride, topk,
        actual.data(), actual_d.data(), threads, false, prefetch, nullptr, counts);
    if (actual != expected || actual_d != expected_d)
        throw std::runtime_error("materialized scalar rerank changed ties or IDs");
    int invalid[batch] = {0, width + 1, 0, 0};
    bool rejected = false;
    try {
        ivf_rerank_host(storage,
            fp32 ? nullptr : reinterpret_cast<const uint8_t*>(queries.data()),
            fp32 ? reinterpret_cast<const float*>(queries.data()) : nullptr,
            fp32, candidates.data(), batch, width, stride, topk,
            actual.data(), actual_d.data(), threads, false, prefetch, nullptr, invalid);
    } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("invalid per-query candidate count accepted");
}

// A compact scan may address only three rows of a much larger exact store.
void check_subset() {
    const uint8_t base[] = {90, 80, 70, 60, 50, 40, 30, 20};
    const uint8_t query[] = {19};
    const int order[] = {7, 2, 5};
    const int candidates[] = {1, -1, 0, 2, 0};
    int ids[3]; float distances[3];
    paraselect::search::IVFHostStorageView host{nullptr, base, 8, 1};
    ivf_rerank_host(host, query, nullptr, false, candidates, 1, 5, 5, 3,
        ids, distances, 2, false, 4, nullptr, nullptr, order, 3);
    if (ids[0] != 7 || ids[1] != 5 || ids[2] != 2 ||
        distances[0] != 1 || distances[1] != 441 || distances[2] != 2601)
        throw std::runtime_error("subset rerank used scan rows as exact rows");
    const int invalid[] = {3}; // In the exact store, but outside the mapping.
    bool rejected = false;
    try {
        ivf_rerank_host(host, query, nullptr, false, invalid, 1, 1, 1, 1,
            ids, distances, 2, false, 4, nullptr, nullptr, order, 3);
    } catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("out-of-subset candidate accepted");
}

int main() {
    check_subset();
    for (int dim : {100, 128})
        for (int width : {128, 256, 512, 1024, 2048})
            for (int threads : {1, 3})
                for (int prefetch : {0, 4}) {
                    check<uint8_t>(dim, width, threads, prefetch);
                    check<float>(dim, width, threads, prefetch);
                }
    std::cout << "PASS: U8/FP32, dimensions 100/128, rerank 128..2048, "
                 "variable counts, padding, duplicates, ties, persistent reuse\n";
}
