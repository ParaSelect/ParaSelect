#ifndef PARASELECT_SEARCH_IVF_COMMON_IVF_PCA_RESIDENT_H
#define PARASELECT_SEARCH_IVF_COMMON_IVF_PCA_RESIDENT_H

#include "quantization/pca/pca_storage.h"

#include "search/ivf_common/ivf_list_layout.h"
#include "search/ivf_common/ivf_probe_schedule.cuh"
#include "search/ivf_common/ivf_probe_view.h"
#include "search/ivf_common/ivf_storage.h"

#include <cuda_runtime.h>

#include <memory>

namespace paraselect::search {

enum class IVFPCAResidentMode {
    Full,
    Threshold,
};

enum class IVFPCAScanKind {
    FP32,
    I8Dp4a,
};

enum class IVFPCADp4aKernelVersion {
    V1 = 1,
    V2 = 2,
    V3 = 3,
};

struct IVFPCAResidentConfig {
    int max_batch = 0;
    int max_nprobe = 0;
    int output_count = 0;
    IVFPCAResidentMode mode = IVFPCAResidentMode::Threshold;
    IVFPCAScanKind scan = IVFPCAScanKind::I8Dp4a;
    IVFPCADp4aKernelVersion dp4a_kernel = IVFPCADp4aKernelVersion::V2;
    IVFProbeScheduleStrategy probe_schedule = IVFProbeScheduleStrategy::Auto;
    int threshold_seed_probes = 8;
    int threshold_candidate_capacity = 8192;
    int prepare_chunk_vectors = 1 << 20;
};

struct IVFPCAResidentStats {
    double metadata_ms = 0.0;
    double query_transform_ms = 0.0;
    double scan_ms = 0.0;
    double reduce_ms = 0.0;
    double avg_scanned_vectors = 0.0;
    double avg_reduce_input_candidates = 0.0;
    IVFProbeScheduleStrategy probe_schedule = IVFProbeScheduleStrategy::Dense;
};

// Owns the query-time GPU representation. The persisted PCA cache remains
// row-major; construction converts it once to the resident interleaved layout.
class IVFPCAResidentSearch {
public:
    IVFPCAResidentSearch(
        const paraselect::quantization::PCAStorageView& storage,
        const IVFListLayoutView& lists,
        const IVFPCAResidentConfig& config);
    ~IVFPCAResidentSearch();
    IVFPCAResidentSearch(const IVFPCAResidentSearch&) = delete;
    IVFPCAResidentSearch& operator=(const IVFPCAResidentSearch&) = delete;

    void search(
        const float* host_queries,
        const idx_t* host_top_probes,
        int batch_n,
        int nprobe,
        int* host_output_compact_ids,
        IVFPCAResidentStats* stats,
        cudaStream_t stream);

    void search_device_probes(
        const float* host_queries,
        const IVFDeviceProbeView& probes,
        int* host_output_compact_ids,
        IVFPCAResidentStats* stats,
        cudaStream_t stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace paraselect::search

#endif  // PARASELECT_SEARCH_IVF_COMMON_IVF_PCA_RESIDENT_H
