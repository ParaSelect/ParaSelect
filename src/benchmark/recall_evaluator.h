#pragma once

#include "benchmark/exact_recall_evaluator.h"
#include "benchmark/recall_evaluation.h"

namespace paraselect {
namespace benchmark {

float compute_idx_intersection_recall(
    const idx_t* gt,
    int n_gt,
    const idx_t* result,
    int n_query,
    int k_result,
    int recall_k);

ExactRecallResult compute_idx_intersection_recalls(
    const idx_t* gt,
    int gt_k,
    const idx_t* result,
    int n_query,
    int k_result);

bool compute_disk_adc_threshold_recall(
    const RecallEvaluationConfig& config,
    const float* result_distances,
    const idx_t* gt_indices,
    int n_query,
    int k_result,
    int gt_k,
    ExactRecallResult* out);

bool compute_disk_threshold_recall(
    const RecallEvaluationConfig& config,
    const idx_t* result_indices,
    const idx_t* gt_indices,
    int n_query,
    int k_result,
    int gt_k,
    ExactRecallResult* out);

bool compute_disk_exact_recall(
    const RecallEvaluationConfig& config,
    const idx_t* result_indices,
    const idx_t* gt_indices,
    int n_query,
    int k_result,
    int gt_k,
    ExactRecallResult* out);

}  // namespace benchmark
}  // namespace paraselect
