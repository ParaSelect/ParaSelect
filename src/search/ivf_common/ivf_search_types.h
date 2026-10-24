#ifndef PARASELECT_SEARCH_IVF_COMMON_IVF_SEARCH_TYPES_H
#define PARASELECT_SEARCH_IVF_COMMON_IVF_SEARCH_TYPES_H

#include "search/ivf_common/ivf_list_layout.h"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace paraselect::search {

struct IVFSearchBatchTiming {
    int batch_id = 0;
    int start_query = 0;
    int n_queries = 0;
    double time_ms = 0.0;
};

struct IVFSearchStageTiming {
    std::string name;
    double time_ms = 0.0;
};

struct IVFSearchStats {
    double total_time_ms = 0.0;
    double avg_batch_time_ms = 0.0;
    double query_preprocess_ms = 0.0;
    double coarse_search_ms = 0.0;
    double probe_metadata_ms = 0.0;
    double cpu_fine_ms = 0.0;
    double recall_at_1 = -1.0;
    double recall_at_5 = -1.0;
    double recall_at_10 = -1.0;
    std::vector<IVFSearchBatchTiming> batch_timings;
    std::vector<IVFSearchStageTiming> stage_timings;
    std::unordered_map<std::string, double> numeric_fields;
    std::unordered_map<std::string, std::string> string_fields;
};

struct IVFCoarseIndexView {
    const float* centroids = nullptr;
    int nlist = 0;
    int dim = 0;
};

struct IVFQueryBatchView {
    const float* fp32 = nullptr;
    const uint8_t* u8 = nullptr;
    int count = 0;
    int dim = 0;
    // Optional host FP16 view used by the coarse Tensor Core path.  The
    // storage is laid out row-major and contains the same values as fp32.
    const std::uint16_t* fp16 = nullptr;
};

enum class IVFExactRerankDType {
    Disabled,
    U8,
    FP32,
};

struct IVFSearchParams {
    int nprobe = 0;
    int topk = 0;
    int batch_size = 0;
    int distance = 0;
    bool fp16_coarse = false;
};

struct IVFSearchOutputView {
    idx_t* indices = nullptr;
    float* distances = nullptr;
    int* coarse_ids = nullptr;
};


enum class IVFFlatFineKind {
    Cpu,
    Resident,
};

struct IVFFlatThresholdParams {
    int seed_probes = 10;
    int candidate_capacity = 8192;
};

inline void validate_ivf_index_shape(
    const IVFCoarseIndexView& coarse, const IVFListLayoutView& lists) {
    validate_ivf_list_layout(lists);
    if (!coarse.centroids || coarse.nlist != lists.nlist || coarse.nlist <= 0 || coarse.dim <= 0) {
        throw std::invalid_argument("invalid IVF coarse index view");
    }
}

}  // namespace paraselect::search

#endif  // PARASELECT_SEARCH_IVF_COMMON_IVF_SEARCH_TYPES_H
