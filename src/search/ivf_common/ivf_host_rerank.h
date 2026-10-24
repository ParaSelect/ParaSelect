#ifndef PARASELECT_SEARCH_IVF_COMMON_HOST_RERANK_H
#define PARASELECT_SEARCH_IVF_COMMON_HOST_RERANK_H

#include "search/ivf_common/ivf_rerank_stats.h"
#include "search/ivf_common/ivf_host_storage.h"

void ivf_rerank_host(
    const paraselect::search::IVFHostStorageView& storage,
    const uint8_t* queries_u8,
    const float* queries_fp32,
    bool use_float,
    const int* candidates,
    int batch_n,
    int candidate_count,
    int candidate_stride,
    int topk,
    int* output_ids,
    float* output_distances,
    int max_threads,
    bool pin_threads,
    int prefetch_ahead,
    paraselect::search::IVFRerankStats* stats,
    const int* candidate_counts = nullptr,
    // Optional scan-row -> storage-row permutation; outputs use storage rows.
    const int* candidate_order = nullptr,
    // Zero preserves the full-store permutation contract.
    idx_t candidate_order_count = 0);

void ivf_prewarm_host_rerank(
    int max_threads, int batch_n, bool pin_threads);

#endif  // PARASELECT_SEARCH_IVF_COMMON_HOST_RERANK_H
