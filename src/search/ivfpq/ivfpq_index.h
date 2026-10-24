#ifndef PARASELECT_SEARCH_IVFPQ_IVFPQ_INDEX_H
#define PARASELECT_SEARCH_IVFPQ_IVFPQ_INDEX_H

#include "dataset/common/dataset_source.h"
#include "search/ivf_common/ivf_search_types.h"
#include "search/kernels/fine/fine_search_strategy.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

namespace paraselect::search {

enum class IVFPQTransformKind {
    Plain,
    OPQ,
};

enum class IVFPQFineStorage {
    Host,
    Resident,
};

enum class IVFPQExactRerank {
    Disabled,
    Flat,
};

struct IVFPQBuildConfig {
    explicit IVFPQBuildConfig(paraselect::dataset::DatasetSource dataset_source)
        : source(std::move(dataset_source)) {}

    paraselect::dataset::DatasetSource source;
    size_t n_vectors = 0;
    int nlist = 0;
    KMeansSpec kmeans;
    int pq_m = 16;
    int pq_k = 256;
    int pq_iters = 12;
    int pq_train_n = 50000000;
    IVFPQTransformKind transform = IVFPQTransformKind::Plain;
    int opq_iters = 0;
    int opq_train_n = 0;
    unsigned seed = 12345;
    size_t prepare_chunk_rows = 1u << 20;
    IVFPQFineStorage fine_storage = IVFPQFineStorage::Host;
    std::string resident_combo = "auto";
    IVFPQExactRerank exact_rerank = IVFPQExactRerank::Disabled;
    int n_rerank = 0;
    int cpu_rerank_threads = 0;
    bool pin_rerank_threads = false;
    bool overlap_cpu_pipeline = true;
    PQAdcLutQuantizationMode query_lut_quantization = PQAdcLutQuantizationMode::U8Affine;
    int rerank_prefetch = 4;
    // Sort retained IDs for CPU locality; does not change candidate selection.
    bool order_candidate_ids = false;
    paraselect::search::PQAdcSearchStrategy adc_strategy =
        paraselect::search::PQAdcSearchStrategy::Dense;
    paraselect::search::PQAdcThresholdParams threshold;
};

class IVFPQIndex {
public:
    static IVFPQIndex build(IVFPQBuildConfig config);

    ~IVFPQIndex();
    IVFPQIndex(IVFPQIndex&&) noexcept;
    IVFPQIndex& operator=(IVFPQIndex&&) noexcept;
    IVFPQIndex(const IVFPQIndex&) = delete;
    IVFPQIndex& operator=(const IVFPQIndex&) = delete;

    void search(
        const IVFQueryBatchView& queries,
        const IVFSearchParams& params,
        const IVFSearchOutputView& output,
        IVFSearchStats* stats = nullptr);

    size_t size() const;
    int dim() const;
    int nlist() const;
    double build_time_ms() const;

private:
    struct Impl;
    explicit IVFPQIndex(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace paraselect::search

#endif  // PARASELECT_SEARCH_IVFPQ_IVFPQ_INDEX_H
