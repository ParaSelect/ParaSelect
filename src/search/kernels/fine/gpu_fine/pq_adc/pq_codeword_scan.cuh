#pragma once

#include "search/kernels/fine/gpu_fine/pq_adc/pq_query_lut_scan.cuh"

namespace paraselect::search {
// Experimental M16/D128 dispatcher; includes quantization and scheduling costs.
void pq_codeword_scan_u8_hybrid(
    const float* queries, const float* codebook, const PQQueryLutView& lut,
    const IVFProbeScheduleView& schedule, const uint8_t* codes,
    const long long* offsets, const int* counts, const float* thresholds,
    int query_count, int probe_count, const PQAdcThresholdCandidatePoolView& output,
    cudaStream_t stream, int probe_begin, int probe_end,
    unsigned long long* physical_code_rows = nullptr);
// Static M16/D96-or-D128 hybrid: list fanout >24 uses compact FP16 TC groups of
// at most 64 queries; all other entries preserve their original LUT grouping.
// Scheduling and temporary storage are timed in scan.
void pq_codeword_scan_hybrid(
    const float* queries, const float* codebook, const PQQueryLutView& lut,
    const IVFProbeScheduleView& schedule,
    const uint8_t* codes, const long long* offsets, const int* counts,
    const float* thresholds, int query_count, int probe_count,
    const PQAdcThresholdCandidatePoolView& output, cudaStream_t stream,
    int probe_begin=0, int probe_end=-1, unsigned long long* physical_code_rows=nullptr, int raw_dim=128);
}  // namespace paraselect::search
