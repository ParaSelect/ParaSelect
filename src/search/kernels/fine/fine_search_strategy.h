#ifndef PARASELECT_FINE_SEARCH_STRATEGY_H
#define PARASELECT_FINE_SEARCH_STRATEGY_H

#include "search/ivf_common/quantized/residual_codec_view.h"

namespace paraselect {
namespace search {

enum class ResidualPQGpuImpl {
    Dense = 0,
    ProbeChunk = 1,
    VectorChunk = 2,
    GroupedVectorChunk = 3,
};

enum class ResidualPQGpuScanMode {
    SharedLut = 0,
    DirectCodebook = 1,
};

enum class PQAdcDistanceMode {
    Plain = 0,
    Residual = 1,
};

enum class PQAdcSearchStrategy {
    Dense = 0,
    Threshold = 1,
};

enum class ResidualPQAdcMode {
    ProbeResidualLut = 0,
    QueryLut = 1,
};

enum class PQAdcThresholdScanLayout {
    PairCentric = 0,
    ListCentric = 1,
};

enum class PQAdcListScanWorkMode {
    Entry = 0,
    FixedTile = 1,
};

enum class PQAdcListScanRowMode {
    Serial = 0,
    Interleaved = 1,
    QueryRowTile4 = 2,
};

// Controls where the list-centric query LUT is read during ADC.  Shared is
// the default and keeps the existing CTA-local staging path; Global is a
// semantic A/B path that reads the same online-built LUT directly.
enum class PQAdcLutReadMode {
    Shared = 0,
    Global = 1,
};

// Query-LUT representation used by residual-PQ ADC.  FP32 is the historical
// path; U8_AFFINE stores each (query, subspace) row as uint8 plus scale/offset.
enum class PQAdcLutQuantizationMode {
    FP32 = 0,
    U8Affine = 1,
};

// Layout of the CTA-local LUT used by the shared read path. Padded preserves
// the historical 257-float row stride; Compact uses the exact 256 codewords.
enum class PQAdcSharedLutLayout {
    Padded = 0,
    Compact = 1,
};

// The consumer is the innermost code-tile scoring operator.  It is orthogonal
// to query-LUT construction/quantization and to probe scheduling.
enum class PQAdcCodeTileConsumer {
    LutScalar = 0,
    Fp16Hybrid = 2,
    U8Hybrid = 3, // Global-affine U8 TC / LUT dispatcher.
};

// Per-query threshold policies. Each policy emits one scalar cutoff from the
// sorted seed row, leaving the scan kernel independent of policy details.
enum class PQThresholdPolicy {
    FixedMultiplier = 0,
    HistogramMargin = 1,
    Bucket = 2,
    DistanceHistogram = 3,
};

struct PQAdcThresholdParams {
    int seed_probes = 10;
    int candidate_capacity = 2048;
    // Number of candidates retained by the GPU threshold cap-select. This is
    // independent from the CPU exact rerank count in IVFResPQSearchOptions.
    int candidate_k = 512;
    // Exact query-LUT seed width, resolved from the CPU rerank budget.
    int n_rerank = 512;
    int seed_rank = 512;
    int reference_rank = 10;
    int histogram_bins = 256;
    float seed_mean_list_rows = 0.f;
    PQAdcThresholdScanLayout scan_layout = PQAdcThresholdScanLayout::PairCentric;
    PQAdcListScanWorkMode list_work_mode = PQAdcListScanWorkMode::Entry;
    PQAdcListScanRowMode list_row_mode = PQAdcListScanRowMode::Serial;
    PQAdcLutReadMode lut_read_mode = PQAdcLutReadMode::Shared;
    PQAdcSharedLutLayout shared_lut_layout = PQAdcSharedLutLayout::Padded;
    PQAdcCodeTileConsumer code_tile_consumer = PQAdcCodeTileConsumer::LutScalar;
    // Diagnostics are resolved by the caller at configuration time. Keeping
    // them in the prepared spec avoids environment lookups in every search.
    bool audit = false;
    bool trace_seed = false;
    PQThresholdPolicy policy = PQThresholdPolicy::FixedMultiplier;
    float multiplier = 1.0f;
    float margin = 0.25f;
    float bucket_edge = 1.5f;
    float bucket_alpha_low = 1.15f;
    float bucket_alpha_high = 1.35f;
    float bucket_edge0 = 0.58f;
    float bucket_edge1 = 0.83f;
    float bucket_alpha0 = 1.05f;
    float bucket_alpha1 = 1.15f;
    float bucket_alpha2 = 1.30f;
    float ledger_multiplier = 1.09304f;
    float ledger_bin_width = 160.0f;
    float ledger_margin = 50.0f;
    float histogram_edge1 = 1.25f;
    float histogram_edge2 = 1.50f;
    float histogram_edge3 = 2.00f;
    float histogram_alpha0 = 1.10f;
    float histogram_alpha1 = 1.20f;
    float histogram_alpha2 = 1.35f;
    float histogram_alpha3 = 1.50f;
};

enum class FineSearchStrategy {
    Flat,
    PQ_CPU,
    PQ_GPU,
    ResidualPQ,
};

inline const char* fine_search_strategy_name(FineSearchStrategy strategy) {
    switch (strategy) {
        case FineSearchStrategy::Flat: return "flat";
        case FineSearchStrategy::PQ_CPU: return "pq_cpu";
        case FineSearchStrategy::PQ_GPU: return "pq_gpu";
        case FineSearchStrategy::ResidualPQ: return "residual_pq";
    }
    return "unknown";
}

struct FineSearchSpec {
    FineSearchStrategy strategy = FineSearchStrategy::Flat;
    int topk = 0;
    int pq_m = 0;
    int pq_k = 0;
    int rerank_n = 0;
};

struct ResidualPQGpuSearchSpec {
    int topk = 0;
    int local_k = 0;
    int thread_k = 0;
    int global_k = 0;
    ResidualPQGpuImpl impl = ResidualPQGpuImpl::Dense;
    ResidualPQGpuScanMode scan_mode = ResidualPQGpuScanMode::SharedLut;
    int pq_m = 0;
    int pq_k = 0;
    int pq_d_sub = 0;
    int pq_d_sub_padded = 0;
    int pq_dim_padded = 0;
    int probes_per_chunk = 1;
    int vector_chunk_size = 0;
    int max_vector_chunks_per_probe = 1;
    int group_chunks = 1;
    int fine_query_tile = 0;
    int worker_blocks = 512;
    ResidualPQAdcMode adc_mode = ResidualPQAdcMode::ProbeResidualLut;
    PQAdcSearchStrategy search_strategy = PQAdcSearchStrategy::Dense;
    PQAdcThresholdParams threshold;
    SignResidualDeviceView sign_codec;
};

}  // namespace search
}  // namespace paraselect

#endif  // PARASELECT_FINE_SEARCH_STRATEGY_H
