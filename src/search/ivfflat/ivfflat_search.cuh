#ifndef PARASELECT_CUDA_IVFFLAT_SEARCH_CUH
#define PARASELECT_CUDA_IVFFLAT_SEARCH_CUH

#include "utils/pch.h"
#include "search/ivf_common/ivf_search_types.h"

#include <memory>

namespace paraselect::search {

struct IVFFlatStorageView {
    const float* rowmajor_fp32 = nullptr;
    const uint8_t* rowmajor_u8 = nullptr;
    const float* norms = nullptr;
    const float* aosoa_fp32 = nullptr;
    const long long* aosoa_offsets = nullptr;
    int aosoa_tile = 16;

    const float* resident_fp32 = nullptr;
    const uint8_t* resident_u8 = nullptr;
    const float* resident_norms = nullptr;
    const idx_t* resident_ids = nullptr;

};

struct IVFFlatIndexView {
    IVFCoarseIndexView coarse;
    IVFListLayoutView lists;
    IVFFlatStorageView storage;
};

struct IVFFlatSearchOptions {
    IVFFlatFineKind fine_kind = IVFFlatFineKind::Cpu;
    IVFFlatThresholdParams threshold;
    const char* cpu_fine_combo = "default";
    const char* resident_combo = "chunked";
    int resident_chunk_blocks = 0;
    bool overlap_cpu_pipeline = false;
};

struct IVFFlatSearchRequest {
    IVFFlatIndexView index;
    IVFQueryBatchView queries;
    IVFSearchParams params;
    IVFSearchOutputView output;
    IVFFlatSearchOptions options;
};

void ivfflat_search(
    const IVFFlatSearchRequest& request,
    IVFSearchStats* stats = nullptr);

}  // namespace paraselect::search

#endif  // PARASELECT_CUDA_IVFFLAT_SEARCH_CUH
