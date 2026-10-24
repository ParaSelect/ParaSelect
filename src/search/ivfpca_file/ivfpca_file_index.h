#ifndef PARASELECT_SEARCH_IVFPCA_FILE_IVFPCA_FILE_INDEX_H
#define PARASELECT_SEARCH_IVFPCA_FILE_IVFPCA_FILE_INDEX_H

#include "clustering/kmeans/kmeans_policy.cuh"
#include "dataset/common/dataset_source.h"
#include "search/ivf_common/ivf_pca_resident.h"
#include "search/ivf_common/ivf_search_types.h"
#include "search/ivf_common/ivf_storage.h"
#include "search/ivf_common/file/ivf_file_block_selection.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

namespace paraselect::search {

struct IVFPCAFileBuildConfig {
    explicit IVFPCAFileBuildConfig(paraselect::dataset::DatasetSource dataset_source)
        : source(std::move(dataset_source)) {}

    paraselect::dataset::DatasetSource source;
    size_t n_vectors = 0;
    int nlist = 0;
    KMeansSpec kmeans;
    int pca_dim = 64;
    int pca_train_n = 100000;
    int pca_encode_chunk = 1 << 18;
    paraselect::search::IVFPCAResidentConfig scan;
    IVFFileReadConfig file_read;
    paraselect::search::IVFFileBlockSelection file_block_selection;
    std::string temporary_directory;
    int distance = 0;
};

struct IVFPCAFileSearchStats {
    double wall_ms = 0.0;
    double coarse_ms = 0.0;
    double pca_metadata_ms = 0.0;
    double pca_query_transform_ms = 0.0;
    double pca_scan_ms = 0.0;
    double pca_reduce_ms = 0.0;
    double file_plan_ms = 0.0;
    double file_read_ms = 0.0;
    double exact_distance_ms = 0.0;
    double file_topk_ms = 0.0;
    double avg_pca_candidates = 0.0;
    double avg_pca_reduce_input = 0.0;
    long long unique_blocks = 0;
    long long bytes_read = 0;
    long long cache_hits = 0;
    long long cache_misses = 0;
    long long blocks_considered = 0;
    long long blocks_skipped = 0;
    long long distance_evaluations = 0;
    int batches = 0;
    int sparse_schedule_batches = 0;
    int dense_schedule_batches = 0;
};

class IVFPCAFileIndex {
public:
    static IVFPCAFileIndex build(IVFPCAFileBuildConfig config);

    ~IVFPCAFileIndex();
    IVFPCAFileIndex(IVFPCAFileIndex&&) noexcept;
    IVFPCAFileIndex& operator=(IVFPCAFileIndex&&) noexcept;
    IVFPCAFileIndex(const IVFPCAFileIndex&) = delete;
    IVFPCAFileIndex& operator=(const IVFPCAFileIndex&) = delete;

    void search(
        const IVFQueryBatchView& queries,
        const IVFSearchParams& params,
        const IVFSearchOutputView& output,
        IVFPCAFileSearchStats* stats = nullptr);

    size_t size() const;
    int dim() const;
    int nlist() const;
    double build_time_ms() const;

private:
    struct Impl;
    explicit IVFPCAFileIndex(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace paraselect::search

#endif  // PARASELECT_SEARCH_IVFPCA_FILE_IVFPCA_FILE_INDEX_H
