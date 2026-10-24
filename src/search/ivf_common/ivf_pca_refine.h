#ifndef PARASELECT_SEARCH_IVF_COMMON_IVF_PCA_REFINE_H
#define PARASELECT_SEARCH_IVF_COMMON_IVF_PCA_REFINE_H

#include "quantization/pca/pca_storage.h"

#include "search/ivf_common/ivf_list_layout.h"
#include <cuda_runtime_api.h>

#include <memory>

namespace paraselect::search {

struct IVFPCARefineStats {
    double query_transform_ms = 0.0;
    double select_ms = 0.0;
    double distance_ms = 0.0;
    double reduce_ms = 0.0;
    double avg_input_candidates = 0.0;
    int threads = 0;
};

void ivf_pca_refine_cpu(
    const paraselect::quantization::PCAStorageView& storage,
    const float* queries,
    const int* candidates,
    int batch_n,
    int candidate_stride,
    int output_count,
    int max_threads,
    int* output_candidates,
    IVFPCARefineStats* stats);

void ivf_pca_refine_cpu_prewarm(int max_threads, int batch_n);

class IVFPCAGpuRefiner {
public:
    IVFPCAGpuRefiner(
        const paraselect::quantization::PCAStorageView& storage,
        const int8_t* device_vectors,
        int max_batch,
        int max_candidates,
        int max_output);
    ~IVFPCAGpuRefiner();
    IVFPCAGpuRefiner(const IVFPCAGpuRefiner&) = delete;
    IVFPCAGpuRefiner& operator=(const IVFPCAGpuRefiner&) = delete;

    void run(
        const float* device_queries,
        const int* device_candidates,
        int batch_n,
        int candidate_stride,
        int candidate_count,
        int output_count,
        int* device_output,
        IVFPCARefineStats* stats,
        cudaStream_t stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace paraselect::search

#endif  // PARASELECT_SEARCH_IVF_COMMON_IVF_PCA_REFINE_H
