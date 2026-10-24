#ifndef PARASELECT_PQ_QUERY_LUT_SCAN_CUH
#define PARASELECT_PQ_QUERY_LUT_SCAN_CUH

#include "search/ivf_common/ivf_probe_schedule.cuh"
#include "search/ivf_common/ivf_search_types.h"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_adc_gpu.cuh"

namespace paraselect::search {

struct PQAdcThresholdCandidatePoolView {
    int capacity = 0;
    int* indices = nullptr;
    float* distances = nullptr;
    int* queries = nullptr;
    int* query_counts = nullptr;
    int* total_count = nullptr;
    int* status = nullptr;
    unsigned long long* accepted_counts = nullptr;
};


// Query-owned staging storage for bounded threshold scans.  A query owns a
// fixed slab [q * query_stride, (q + 1) * query_stride); producers never
// reserve from a batch-global counter.  The reducer can therefore tighten a
// query threshold between scan rounds without racing unrelated queries.
struct PQAdcQueryBufferView {
    int query_stride = 0;
    int capacity = 0;
    int* indices = nullptr;
    float* distances = nullptr;
    int* counts = nullptr;
    float* thresholds = nullptr;
    int* overflow = nullptr;
};

void pq_adc_query_lut_pair_scan(
    const PQQueryLutView& pq_query_lut,
    const idx_t* top_probes,
    const uint8_t* codes,
    const long long* list_offsets,
    const int* list_counts,
    const float* thresholds,
    int query_count,
    int list_count,
    int probe_count,
    int pq_m,
    const PQAdcThresholdCandidatePoolView& output,
    cudaStream_t stream,
    PQAdcLutReadMode lut_read_mode = PQAdcLutReadMode::Shared,
    PQAdcSharedLutLayout shared_lut_layout = PQAdcSharedLutLayout::Padded,
    int probes_per_cta = 1);

void pq_adc_query_lut_segmented_scan(
    const PQQueryLutView& pq_query_lut,
    const IVFProbeScheduleView& schedule,
    const IVFProbeTileScheduleView* tile_schedule,
    const uint8_t* codes,
    const long long* list_offsets,
    const int* list_counts,
    const float* thresholds,
    int query_count,
    int probe_count,
    int pq_m,
    PQAdcListScanRowMode row_mode,
    PQAdcLutReadMode lut_read_mode,
    unsigned long long* physical_code_rows,
    const PQAdcThresholdCandidatePoolView& output,
    cudaStream_t stream,
    PQAdcSharedLutLayout shared_lut_layout = PQAdcSharedLutLayout::Padded,
    int probe_begin = 0,
    int probe_end = -1);

}  // namespace paraselect::search

#endif  // PARASELECT_PQ_QUERY_LUT_SCAN_CUH
