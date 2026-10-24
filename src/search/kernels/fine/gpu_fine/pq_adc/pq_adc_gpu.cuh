#ifndef PARASELECT_PQ_ADC_GPU_CUH
#define PARASELECT_PQ_ADC_GPU_CUH

#include "search/kernels/fine/fine_search_strategy.h"
#include "search/ivf_common/ivf_search_types.h"
#include "search/ivf_common/quantized/res_query_lut_bias.h"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_adc_workspace.cuh"

#include <stdexcept>
#include "search/kernels/fine/gpu_fine/pq_adc/pq_query_lut.cuh"

namespace paraselect::search {

struct IVFProbeScheduleView;
struct IVFProbeTileScheduleView;

struct PQQueryLutView {
    const float* query_codeword_dot_lut = nullptr;
    const float* query_centroid_distances = nullptr;
    QueryLutBias query_lut_bias;
    const uint8_t* query_codeword_dot_lut_u8 = nullptr;
    const float* query_lut_scale = nullptr;
    const float* query_lut_lo_sum = nullptr;
    const float* query_lut_inflation = nullptr;
    SignResidualDeviceView sign_codec;
    PQQueryLutKind kind = PQQueryLutKind::ResidualDot;

    bool enabled() const noexcept {
        if (sign_codec.codes) return sign_codec.queries && sign_codec.centroids && sign_codec.scales;
        const bool has_fp32 = query_codeword_dot_lut != nullptr;
        const bool has_u8 = query_codeword_dot_lut_u8 && query_lut_scale &&
            query_lut_lo_sum;
        if (kind == PQQueryLutKind::PlainL2) return has_fp32 || has_u8;
        const bool has_query_lut_bias_fp32 = query_lut_bias.fp32 != nullptr;
        const bool has_query_lut_bias_u16 = query_lut_bias.u16 != nullptr &&
            query_lut_bias.scale > 0.0f;
        return (has_fp32 || has_u8) && query_centroid_distances &&
            (has_query_lut_bias_fp32 != has_query_lut_bias_u16);
    }

    bool quantized() const noexcept {
        return query_codeword_dot_lut_u8 != nullptr;
    }

    bool quantized_query_lut_bias() const noexcept {
        return query_lut_bias.u16 != nullptr;
    }
};

class PQAdcInsufficientSeedCandidates : public std::runtime_error {
public:
    PQAdcInsufficientSeedCandidates()
        : std::runtime_error(
              "pq ADC threshold seed probes produced fewer candidates than global_k") {}
};

}  // namespace paraselect::search

void pq_adc_search_gpu(
    paraselect::search::PQAdcDistanceMode distance_mode,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    const paraselect::search::ResidualPQGpuSearchSpec& spec,
    int* d_probe_candidate_idx,
    float* d_probe_candidate_dist,
    int* d_global_candidate_idx,
    float* d_global_candidate_dist,
    cudaStream_t stream,
    ResidualPQGpuDebugTiming* timing = nullptr,
    paraselect::search::ResidualPQGroupedWorkspace* grouped_workspace = nullptr);

void pq_adc_threshold_search_gpu(
    paraselect::search::PQAdcDistanceMode distance_mode,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    const paraselect::search::ResidualPQGpuSearchSpec& spec,
    const paraselect::search::IVFProbeScheduleView* probe_schedule,
    const paraselect::search::IVFProbeTileScheduleView* probe_tile_schedule,
    cudaStream_t stream,
    paraselect::search::PQAdcThresholdWorkspace* threshold_workspace,
    paraselect::search::ResidualPQGroupedWorkspace* grouped_workspace,
    ResidualPQGpuDebugTiming* timing = nullptr);

void pq_query_lut_threshold_search_gpu(
    const paraselect::search::PQQueryLutView& lut,
    const idx_t* d_top_nprobe, const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets, const int* d_cluster_counts,
    int n_query, int n_total_clusters, int n_probes, int pq_m,
    const paraselect::search::PQAdcThresholdParams& params,
    const paraselect::search::IVFProbeScheduleView* probe_schedule,
    const paraselect::search::IVFProbeTileScheduleView* probe_tile_schedule,
    cudaStream_t stream, paraselect::search::PQAdcThresholdWorkspace* threshold_workspace,
    ResidualPQGpuDebugTiming* timing = nullptr,
    const float* raw_queries = nullptr, const float* raw_codebook = nullptr, int raw_dim = 0);

#endif  // PARASELECT_PQ_ADC_GPU_CUH
