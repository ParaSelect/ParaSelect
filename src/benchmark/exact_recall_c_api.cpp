#include "benchmark/exact_recall_evaluator.h"

#include <unordered_map>

extern "C" float paraselect_exact_threshold_recall_l2_selected(
    const float* queries,
    const float* selected_base,
    const long long* selected_base_ids,
    int selected_n,
    const long long* result_ids,
    const long long* gt_ids,
    int n_query,
    int dim,
    int k) {
    if (!queries || !selected_base || !selected_base_ids || !result_ids || !gt_ids ||
        selected_n <= 0 || n_query <= 0 || dim <= 0 || k <= 0) {
        return -1.0f;
    }

    std::unordered_map<long long, int> base_offsets;
    base_offsets.reserve((size_t)selected_n * 2);
    for (int i = 0; i < selected_n; ++i) {
        base_offsets[(long long)selected_base_ids[i]] = i;
    }

    long long hits = 0;
    const float eps = 1e-3f;
    for (int q = 0; q < n_query; ++q) {
        const float* qv = queries + (size_t)q * (size_t)dim;
        const auto gt_it = base_offsets.find((long long)gt_ids[(size_t)q * (size_t)k + (size_t)(k - 1)]);
        if (gt_it == base_offsets.end()) {
            continue;
        }
        const float* gt_bv = selected_base + (size_t)gt_it->second * (size_t)dim;
        const float threshold = paraselect::exact_l2_distance_sq(qv, gt_bv, dim) + eps;

        for (int j = 0; j < k; ++j) {
            const auto it = base_offsets.find((long long)result_ids[(size_t)q * (size_t)k + (size_t)j]);
            if (it == base_offsets.end()) {
                continue;
            }
            const float* bv = selected_base + (size_t)it->second * (size_t)dim;
            const float d = paraselect::exact_l2_distance_sq(qv, bv, dim);
            if (d <= threshold) {
                ++hits;
            }
        }
    }

    return static_cast<float>(hits) / static_cast<float>((size_t)n_query * (size_t)k);
}
