#pragma once
#include "search/kernels/coarse/stage/coarse_distance_histogram.cuh"
#include "search/kernels/coarse/coarse.h"
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
namespace paraselect::coarse {
// Fixed capacity prepared once. No allocations or host decisions during search.
struct CoarseHistogramTopKWorkspace {
    int capacity = 0, nlist = 0, k = 0;
    int sort_capacity = 0, candidate_stride = 0;
    bool reset_status = true;
    bool estimate_threshold = false;
    CoarseDistanceHistogram histogram;
    float *candidates = nullptr, *sorted_candidates = nullptr;
    int *candidate_ids = nullptr, *sorted_candidate_ids = nullptr;
    int *candidate_counts = nullptr, *candidate_starts = nullptr, *candidate_ends = nullptr;
    int* selection_complete = nullptr;
    int* retry_queries = nullptr;
    int* retry_count = nullptr;
    void* sort_temp = nullptr;
    size_t sort_temp_bytes = 0;
};
// Zero selects the shared score + histogram/sort scratch budget. A positive
// override is a score-only budget for controlled profiling.
inline constexpr size_t coarse_histogram_auto_score_budget = 0;
inline constexpr size_t coarse_histogram_workspace_limit_bytes = 2ULL << 30;
void coarse_histogram_topk_prepare(CoarseHistogramTopKWorkspace&, int batch, int nlist, int k,
    size_t score_budget_bytes = coarse_histogram_auto_score_budget);
size_t coarse_histogram_topk_workspace_bytes(int batch, int nlist, int k = -1,
    size_t score_budget_bytes = coarse_histogram_auto_score_budget);
// Internal scratch tiling only: external batch_size and query_tile do not change.
int coarse_histogram_score_rows(int batch, int nlist,
    size_t score_budget_bytes = coarse_histogram_auto_score_budget, int k = 128);
inline CoarseHistogramTopKWorkspace coarse_histogram_topk_slice(
    const CoarseHistogramTopKWorkspace& source,int offset,int rows) {
    auto w=source; w.capacity=rows; w.reset_status=false;
    w.histogram.counts+=size_t(offset)*CoarseDistanceHistogram::bins;
    w.histogram.minimum+=offset;w.histogram.maximum+=offset;
    w.histogram.lower+=offset;w.histogram.upper+=offset;
    w.histogram.selected_counts+=offset;
    w.candidate_counts+=offset;w.selection_complete+=offset;
    return w;
}
void coarse_histogram_topk_release(CoarseHistogramTopKWorkspace&) noexcept;
// Reads GEMM scores without modification; norms are L2 norms (not squared norms).
// Sorted exact top-k with arbitrary tie membership. Finite scores required;
// histogram.status reports nonfinite input (1) or a count invariant failure (2).
void coarse_histogram_topk_l2(const float* scores, const float* qnorm, const float* cnorm,
    int batch, int nlist, int k, float* values, int64_t* ids,
    CoarseHistogramTopKWorkspace&, cudaStream_t stream,
    CoarseHistogramTiming* timing = nullptr);
}
