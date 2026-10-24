#ifndef PARASELECT_CUDA_IVFRESPQ_SEARCH_CUH
#define PARASELECT_CUDA_IVFRESPQ_SEARCH_CUH

#include "quantization/pca/pca_storage.h"
#include "search/ivf_common/ivf_search_types.h"
#include "search/ivf_common/ivf_storage.h"
#include "search/ivf_common/quantized/res_query_lut_bias.h"
#include "search/ivf_common/file/ivf_file_block_selection.h"
#include "search/kernels/fine/fine_search_strategy.h"
#include "search/ivfrespq/residual_assets_view.h"
#include <memory>

namespace paraselect::search {

enum class IVFResPQFineBackend {
    ResidualPQGpu,
    GpuLutCpuScan,
    CpuLut,
};

enum class IVFResPQCpuScheduler {
    QueryBalanced,
    ClusterBased,
};

struct IVFResPQGpuConfig {
    const char* combo = "auto";
    int thread_topk = -1;
    int local_topk = -1;
    int candidate_count = -1;
    int query_tile = 512;
    int worker_blocks = 512;
    int group_chunks = 0;
};

struct IVFResPQCpuConfig {
    IVFResPQCpuScheduler scheduler = IVFResPQCpuScheduler::QueryBalanced;
    int lut_tile = 1;
    int rerank_threads = 0;
    bool pin_rerank_threads = false;
    int rerank_prefetch = 8;
};

struct IVFResPQStorageView {
    IVFPQCodeStorageView pq;
    paraselect::quantization::PCAStorageView pca;
    ResidualAssetsView assets;
    // Optional PQ-row -> shared CPU-store row mapping (lists.n_vectors entries).
    // Exact store rows are returned as result IDs. Borrowed for session lifetime.
    const int* host_rows = nullptr;
};

struct IVFResPQIndexView {
    IVFCoarseIndexView coarse;
    IVFListLayoutView lists;
    IVFResPQStorageView storage;
};

struct IVFResPQSearchOptions {
    int query_tile = 0; // zero: full query batch; positive: explicit coarse query cap
    ResidualCodecKind residual_codec = ResidualCodecKind::PQ;
    PQAdcDistanceMode pq_distance_mode = PQAdcDistanceMode::Residual;
    int n_rerank = 0;
    int shortlist_count = 0;
    IVFRefineBackend pca_backend = IVFRefineBackend::None;
    bool float_rerank = false;
    bool return_adc_distances = false;
    IVFResPQFineBackend fine_backend = IVFResPQFineBackend::ResidualPQGpu;
    IVFResPQGpuConfig gpu;
    IVFResPQCpuConfig cpu;
    const char* resident_combo = "full";
    bool overlap_cpu_pipeline = false;
    paraselect::search::PQAdcSearchStrategy adc_strategy =
        paraselect::search::PQAdcSearchStrategy::Dense;
    paraselect::search::ResidualPQAdcMode adc_mode =
        paraselect::search::ResidualPQAdcMode::ProbeResidualLut;
    bool query_lut_codebook_fp16 = false;
    paraselect::search::PQAdcLutQuantizationMode query_lut_quantization =
        paraselect::search::PQAdcLutQuantizationMode::FP32;
    paraselect::search::PQAdcThresholdParams adc_threshold;
    paraselect::search::IVFFileBlockSelection file_block_selection;
    IVFFileReadConfig file_read;
};

struct IVFResPQSearchRequest {
    IVFResPQIndexView index;
    IVFQueryBatchView queries;
    IVFSearchParams params;
    IVFSearchOutputView output;
    IVFResPQSearchOptions options;
};

struct IVFResPQCandidateOutputView {
    int* local_ids = nullptr;
    float* adc_distances = nullptr;
    int capacity = 0;
};

struct IVFResPQCandidateIndexView {
    IVFCoarseIndexView coarse;
    IVFListLayoutView lists;
    IVFPQCodeStorageView pq;
    ResidualAssetsView assets;
};

struct IVFResPQCandidateRequest {
    IVFResPQCandidateIndexView index;
    IVFQueryBatchView queries;
    IVFSearchParams params;
    IVFResPQSearchOptions options;
    IVFResPQCandidateOutputView output;
};

void ivfrespq_search(
    const IVFResPQSearchRequest& request,
    IVFSearchStats* stats = nullptr);

// Fixed request and borrowed host assets must outlive this session. Calls are
// synchronous; a session is not shared by concurrent search/update callers.
class IVFResPQSession {
public:
    explicit IVFResPQSession(const IVFResPQSearchRequest& request);
    // Prepared candidate generation for callers owning a separate exact store.
    explicit IVFResPQSession(const IVFResPQCandidateRequest& request);
    int candidate_count() const;
    ~IVFResPQSession();
    IVFResPQSession(const IVFResPQSession&) = delete;
    IVFResPQSession& operator=(const IVFResPQSession&) = delete;
    void search(IVFSearchStats* stats = nullptr);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

int ivfrespq_collect_candidates(const IVFResPQCandidateRequest& request);

}  // namespace paraselect::search

#endif  // PARASELECT_CUDA_IVFRESPQ_SEARCH_CUH
