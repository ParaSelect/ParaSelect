#ifndef PARASELECT_SEARCH_IVFRESPQ_FILE_IVFRESPQ_FILE_INDEX_H
#define PARASELECT_SEARCH_IVFRESPQ_FILE_IVFRESPQ_FILE_INDEX_H

#include "clustering/kmeans/kmeans_policy.cuh"
#include "dataset/common/dataset_source.h"
#include "search/ivfrespq/ivfrespq_search.cuh"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

namespace paraselect::search {

struct IVFResPQFileBuildConfig {
    explicit IVFResPQFileBuildConfig(
        paraselect::dataset::DatasetSource dataset_source)
        : source(std::move(dataset_source)) {}

    paraselect::dataset::DatasetSource source;
    size_t n_vectors = 0;
    int nlist = 0;
    KMeansSpec kmeans;
    int pq_m = 16;
    int pq_k = 256;
    int pq_max_gpus = 8;
    int pq_train_n = 50000000;
    int pq_encode_chunk = 1 << 20;
    int pca_dim = 0;
    int pca_train_n = 100000;
    int pca_encode_chunk = 1 << 18;
    std::string gpu_combo = "auto";
    std::string temporary_directory;
    IVFResPQSearchOptions options;
};

class IVFResPQFileIndex {
public:
    static IVFResPQFileIndex build(IVFResPQFileBuildConfig config);

    ~IVFResPQFileIndex();
    IVFResPQFileIndex(IVFResPQFileIndex&&) noexcept;
    IVFResPQFileIndex& operator=(IVFResPQFileIndex&&) noexcept;
    IVFResPQFileIndex(const IVFResPQFileIndex&) = delete;
    IVFResPQFileIndex& operator=(const IVFResPQFileIndex&) = delete;

    void search(
        const IVFQueryBatchView& queries,
        const IVFSearchParams& params,
        const IVFSearchOutputView& output,
        IVFSearchStats* stats = nullptr);

    size_t size() const;
    int dim() const;
    int nlist() const;
    int pq_m() const;
    int pq_k() const;
    double build_time_ms() const;
    double exact_storage_build_time_ms() const;

private:
    struct Impl;
    explicit IVFResPQFileIndex(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace paraselect::search

#endif  // PARASELECT_SEARCH_IVFRESPQ_FILE_IVFRESPQ_FILE_INDEX_H
