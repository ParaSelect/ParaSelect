#ifndef PARASELECT_SEARCH_IVFFLAT_IVFFLAT_INDEX_H
#define PARASELECT_SEARCH_IVFFLAT_IVFFLAT_INDEX_H

#include "clustering/kmeans/kmeans_policy.cuh"
#include "dataset/common/dataset_source.h"
#include "search/ivf_common/ivf_search_types.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

namespace paraselect::search {

enum class IVFFlatResidentDType {
    Auto,
    FP32,
    U8,
};

struct IVFFlatBuildConfig {
    explicit IVFFlatBuildConfig(paraselect::dataset::DatasetSource dataset_source)
        : source(std::move(dataset_source)) {}

    paraselect::dataset::DatasetSource source;
    size_t n_vectors = 0;
    int nlist = 0;
    KMeansSpec kmeans;
    IVFFlatFineKind fine_kind = IVFFlatFineKind::Cpu;
    IVFFlatThresholdParams threshold;
    std::string cpu_fine_combo = "default";
    std::string resident_combo = "chunked";
    int resident_chunk_blocks = 0;
    IVFFlatResidentDType resident_dtype = IVFFlatResidentDType::Auto;
    bool numa_first_touch = false;
    bool overlap_cpu_pipeline = false;
};

class IVFFlatIndex {
public:
    static IVFFlatIndex build(IVFFlatBuildConfig config);

    ~IVFFlatIndex();
    IVFFlatIndex(IVFFlatIndex&&) noexcept;
    IVFFlatIndex& operator=(IVFFlatIndex&&) noexcept;
    IVFFlatIndex(const IVFFlatIndex&) = delete;
    IVFFlatIndex& operator=(const IVFFlatIndex&) = delete;

    void search(
        const IVFQueryBatchView& queries,
        const IVFSearchParams& params,
        const IVFSearchOutputView& output,
        IVFSearchStats* stats = nullptr);

    size_t size() const;
    int dim() const;
    int nlist() const;
    double build_time_ms() const;
    double kmeans_time_ms() const;

private:
    struct Impl;
    explicit IVFFlatIndex(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace paraselect::search

#endif  // PARASELECT_SEARCH_IVFFLAT_IVFFLAT_INDEX_H
