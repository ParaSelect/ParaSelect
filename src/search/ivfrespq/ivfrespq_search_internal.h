#ifndef PARASELECT_IVFRESPQ_SEARCH_INTERNAL_H
#define PARASELECT_IVFRESPQ_SEARCH_INTERNAL_H

#include "quantization/pca/pca_storage.h"

#include "search/ivfrespq/ivfrespq_search.cuh"

using namespace paraselect::search;

struct IVFResPQState {
    const uint8_t* codes = nullptr;
    const paraselect::cpu_fine::CpuPqCodeStorageView* code_storage = nullptr;
    const float* codebook = nullptr;
    int M = 0;
    int K = 0;
};

struct IVFResTurboQuantState {
    const std::uint32_t* codes = nullptr;
    const float* scales = nullptr;
};

struct IVFResSearchState {
    ResidualCodecKind residual_codec = ResidualCodecKind::PQ;
    PQAdcDistanceMode pq_distance_mode = PQAdcDistanceMode::Residual;
    const float* h_queries_fp32 = nullptr;
    const uint8_t* h_queries_u8 = nullptr;
    const std::uint16_t* h_queries_fp16 = nullptr;
    IVFListLayoutView lists;
    paraselect::quantization::PCAStorageView pca_storage;
    IVFExactStorageView exact_storage;
    const float* h_cluster_centers = nullptr;
    IVFResPQState pq;
    QueryLutBias query_lut_bias;
    IVFResTurboQuantState turboquant;
    const float* h_residual_rotation = nullptr;

    int n_query = 0;
    int n_dim = 0;
    int n_probes = 0;
    int k = 0;
    int query_batch_size = 0;
    int query_tile = 0;
    int distance_mode = 0;
    bool fp16_coarse = false;
    int pq_M = 0;
    int pq_K = 0;
    int pq_rerank_n = 0;
    int shortlist_count = 0;
    IVFRefineBackend pca_backend = IVFRefineBackend::None;
    int pq_use_float_rerank = 0;
    bool return_adc_distances = false;
    IVFResPQFineBackend fine_backend = IVFResPQFineBackend::ResidualPQGpu;
    IVFResPQGpuConfig gpu;
    IVFResPQCpuConfig cpu;
    const char* resident_combo_name = nullptr;
    bool cpu_batch_pipeline_overlap = false;
    paraselect::search::PQAdcSearchStrategy adc_strategy =
        paraselect::search::PQAdcSearchStrategy::Dense;
    paraselect::search::ResidualPQAdcMode residual_adc_mode =
        paraselect::search::ResidualPQAdcMode::ProbeResidualLut;
    bool query_lut_codebook_fp16 = false;
    paraselect::search::PQAdcLutQuantizationMode query_lut_quantization =
        paraselect::search::PQAdcLutQuantizationMode::FP32;
    paraselect::search::PQAdcThresholdParams adc_threshold;
    paraselect::search::IVFFileBlockSelection file_block_selection;
    IVFFileReadConfig file_read;
    const int* host_rows = nullptr;

    bool candidate_only = false;
    int* h_candidate_local_ids_out = nullptr;
    float* h_candidate_adc_distances_out = nullptr;
    int candidate_capacity = 0;

    idx_t* h_topk_index_out = nullptr;
    float* h_topk_dist_out = nullptr;
    int* h_coarse_cluster_ids_out = nullptr;

};

#endif  // PARASELECT_IVFRESPQ_SEARCH_INTERNAL_H
