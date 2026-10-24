#pragma once
#include "search/kernels/fine/gpu_fine/pq_adc/pq_adc_gpu.cuh"

namespace paraselect::search {
// Exact top-512 over seed probes. Scratch holds only 512 keys per query/list,
// never every scanned row. Distance and row ID jointly determine the order.
void pq_query_lut_seed_top512(
    const PQQueryLutView& view, const idx_t* probes,
    const uint8_t* codes, const long long* offsets, const int* counts,
    int queries, int probe_count, int seed_probes, int pq_m,
    bool four_accumulators, uint64_t* scratch, uint64_t* merge_scratch, size_t scratch_keys,
    int* output_ids, float* output_distances, int* status,
    unsigned long long* scanned_rows, cudaStream_t stream);

// Runtime n_rerank; output uses a packed query-major stride of n_rerank.
// Prepared scratch must fit one query (seed_probes * n_rerank keys); larger
// batches reuse it in stream-ordered query chunks without additional allocation.
void pq_query_lut_seed_topk(
    const PQQueryLutView& view, const idx_t* probes,
    const uint8_t* codes, const long long* offsets, const int* counts,
    int queries, int probe_count, int seed_probes, int pq_m, int n_rerank,
    bool four_accumulators, uint64_t* scratch, uint64_t* merge_scratch, size_t scratch_keys,
    int* output_ids, float* output_distances, int* status,
    unsigned long long* scanned_rows, cudaStream_t stream);

// Prepared mean is a scheduling hint only; unknown shapes retain one list/CTA.
inline int pq_seed_histogram_list_group(int queries, int seeds, float mean_rows) {
    if (queries <= 0 || seeds <= 0 || !(mean_rows > 0.f)) return 1;
    int group = 1;
    if (mean_rows <= 32.f) group = 32;
    else if (mean_rows <= 256.f && queries >= 1024) group = 8;
    while (group > 1 && size_t(queries) * ((seeds + group - 1) / group) < 256)
        group = group == 32 ? 8 : 1;
    return group;
}

// Collect raw distances from the selected seed lists. This is deliberately
// separate from top-k seed output because top-k discards the histogram tail.
void pq_query_lut_seed_histogram(
    const PQQueryLutView& view, const idx_t* probes,
    const uint8_t* codes, const long long* offsets, const int* counts,
    int queries, int probe_count, int seed_probes, int nlist, int pq_m,
    bool four_accumulators, unsigned int* histogram, int histogram_bins,
    float log_min, float log_step, cudaStream_t stream,
    float mean_list_rows = 0.f);
}

namespace paraselect::search {
// Borrowed scratch for a one-pass seed histogram. No probes or coarse state.
struct PQSeedHistogramWorkspace {
    unsigned int* counts = nullptr;
    float* max_distance = nullptr;
    float* thresholds = nullptr;
    int* status = nullptr;
    int bins = 256;
    float log_min = -8.0f;
    float log_step = 0.125f;
};

// Sample distinct, evenly spaced row positions. The sample must contain at
// least rank rows; the returned bucket upper edge safely admits the full
// range's ADC top-rank. Seed and scan use the same score helper.
void pq_query_lut_seed_histogram_rows(
    const PQQueryLutView& view, const uint8_t* codes,
    const int* rows, const int* groups, int row_count, int sample_count,
    int group_count, int queries, int pq_m, int rank,
    const PQSeedHistogramWorkspace& workspace, cudaStream_t stream);
} // namespace paraselect::search
