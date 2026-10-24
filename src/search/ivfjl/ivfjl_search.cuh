#ifndef PARASELECT_CUDA_IVFJL_SEARCH_CUH
#define PARASELECT_CUDA_IVFJL_SEARCH_CUH

#include "utils/pch.h"
#include "search/ivf_common/ivf_search_types.h"

namespace paraselect::search {

struct IVFJLStorageView {
    const float* projected_rowmajor = nullptr;
    const float* projected_norms = nullptr;
};

struct IVFJLIndexView {
    IVFCoarseIndexView coarse;
    IVFListLayoutView lists;
    IVFJLStorageView storage;
};

struct IVFJLSearchOptions {
    uint64_t seed = 12345ULL;
    const char* cpu_fine_combo = nullptr;
    bool overlap_cpu_pipeline = false;
};

struct IVFJLSearchRequest {
    IVFJLIndexView index;
    IVFQueryBatchView queries;
    IVFSearchParams params;
    IVFSearchOutputView output;
    IVFJLSearchOptions options;
};

void ivfjl_search(
    const IVFJLSearchRequest& request,
    IVFSearchStats* stats = nullptr);

}  // namespace paraselect::search

#endif  // PARASELECT_CUDA_IVFJL_SEARCH_CUH
