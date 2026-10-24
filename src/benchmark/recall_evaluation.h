#pragma once

#include "utils/pch.h"
#include "benchmark/recall_method.h"

#include <string>

namespace paraselect::benchmark {

struct RecallEvaluationConfig {
    std::string dataset_name;
    int distance_mode = 0;
    int recall_method = IVF_RECALL_IDX_INTERSECTION;
};

struct RecallEvaluationResult {
    bool available = false;
    float recall_at_1 = -1.0f;
    float recall_at_5 = -1.0f;
    float recall_at_10 = -1.0f;
    std::string method;
};

RecallEvaluationResult evaluate_recall(
    const RecallEvaluationConfig& config,
    const idx_t* indices,
    const float* distances,
    int n_query,
    int k_result);

} // namespace paraselect::benchmark
