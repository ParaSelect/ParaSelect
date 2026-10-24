#ifndef PARASELECT_SEARCH_IVF_COMMON_IVF_PROBE_SCHEDULE_CUH
#define PARASELECT_SEARCH_IVF_COMMON_IVF_PROBE_SCHEDULE_CUH

#include "search/ivf_common/ivf_probe_view.h"

#include <cuda_runtime.h>

#include <memory>

namespace paraselect::search {

enum class IVFProbeScheduleStrategy {
    Auto,
    Dense,
    Sparse,
};

// Grouped query-list work consumed by resident IVF scan kernels.
struct IVFProbeScheduleView {
    const int* clusters = nullptr;
    const int* starts = nullptr;
    const int* counts = nullptr;
    const int* queries = nullptr;
    const int* probe_ranks = nullptr;
    int entry_count = 0;
    // Whole-list fanout before entry splitting; -1 for manually constructed views.
    int max_list_queries = -1;
};

struct IVFProbeScheduleResult {
    IVFProbeScheduleView view;
    IVFProbeScheduleStrategy strategy = IVFProbeScheduleStrategy::Dense;
    unsigned long long scanned_vectors = 0;
    int reference_count = 0;
};

// Fixed-row work over an existing grouped query-list schedule. Work item w
// belongs to the entry whose prefix interval contains w; the scan kernel can
// recover the entry and tile without materializing two arrays per work item.
struct IVFProbeTileScheduleView {
    const int* entry_tile_offsets = nullptr;
    int entry_count = 0;
    int work_count = 0;
    int tile_rows = 0;
};

struct IVFProbeTileScheduleResult {
    IVFProbeTileScheduleView view;
};

IVFProbeScheduleStrategy choose_ivf_probe_schedule_strategy(
    IVFProbeScheduleStrategy requested,
    size_t probe_reference_count,
    int nlist);

const char* ivf_probe_schedule_strategy_name(IVFProbeScheduleStrategy strategy);

class IVFProbeScheduleBuilder {
public:
    IVFProbeScheduleBuilder(
        int nlist,
        int max_probe_references,
        const int* device_list_counts,
        IVFProbeScheduleStrategy strategy = IVFProbeScheduleStrategy::Auto);
    ~IVFProbeScheduleBuilder();

    IVFProbeScheduleBuilder(const IVFProbeScheduleBuilder&) = delete;
    IVFProbeScheduleBuilder& operator=(const IVFProbeScheduleBuilder&) = delete;

    IVFProbeScheduleResult build(
        const IVFDeviceProbeView& probes,
        int probe_limit,
        cudaStream_t stream,
        bool allow_padding = false);
    IVFProbeTileScheduleResult build_tiles(
        const IVFProbeScheduleView& schedule,
        int tile_rows,
        cudaStream_t stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace paraselect::search

#endif  // PARASELECT_SEARCH_IVF_COMMON_IVF_PROBE_SCHEDULE_CUH
