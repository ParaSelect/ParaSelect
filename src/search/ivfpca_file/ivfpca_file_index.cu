#include "quantization/pca/pca_storage.h"
#include "search/ivfpca_file/ivfpca_file_index.h"

#include "search/ivf_common/ivf_coarse_ctx.h"
#include "search/ivf_common/file/ivf_file_rerank.h"
#include "search/ivfpca_file/ivfpca_file_build.h"
#include "search/ivf_common/file/ivf_file_reader.h"
#include "search/kernels/coarse/coarse_policy.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace paraselect::search;

namespace {

using Clock = std::chrono::steady_clock;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void validate_index(
    const IVFPCAFileBuildConfig& config,
    const IVFCoarseIndexView& coarse,
    const IVFListLayoutView& lists,
    const paraselect::quantization::PCAStorageView& pca,
    const IVFFileStorageView& exact) {
    validate_ivf_index_shape(coarse, lists);
    if (!pca.enabled() || !pca.projection || !pca.bias || !pca.scales ||
        pca.count != lists.n_vectors || pca.source_dim != coarse.dim ||
        pca.dim != config.pca_dim) {
        throw std::invalid_argument("invalid IVFPCAFile PCA storage");
    }
    if (exact.fd < 0 || exact.count != lists.n_vectors ||
        exact.dim != coarse.dim ||
        exact.dtype == IVFExactRerankDType::Disabled) {
        throw std::invalid_argument("invalid IVFPCAFile exact storage");
    }
    if (config.distance != static_cast<int>(L2_DISTANCE)) {
        throw std::invalid_argument("IVFPCAFile supports L2 only");
    }
}

}  // namespace

namespace paraselect::search {

struct IVFPCAFileIndex::Impl {
    explicit Impl(IVFPCAFileBuildConfig requested) : config(std::move(requested)) {
        const auto build_start = Clock::now();
        artifacts = paraselect::ivfpca_file::build_artifacts(config);

        coarse_view = {
            artifacts.clusters.centroids.data(), config.nlist, config.source.spec().dim};
        list_view = {
            artifacts.clusters.cluster_offsets.data(),
            artifacts.clusters.reordered_indices.data(),
            config.nlist,
            static_cast<idx_t>(config.n_vectors)};
        pca_view = {
            artifacts.pca.vectors.data(),
            artifacts.pca.projection.data(),
            artifacts.pca.bias.data(),
            artifacts.pca.scales.data(),
            static_cast<idx_t>(artifacts.pca.request.n_vectors),
            artifacts.pca.request.source_dim,
            artifacts.pca.request.dim};
        exact_view = artifacts.exact.view();
        validate_index(config, coarse_view, list_view, pca_view, exact_view);

        paraselect::search::IVFCoarseConfig coarse_config;
        coarse_config.dim = coarse_view.dim;
        coarse_config.nlist = coarse_view.nlist;
        coarse_config.max_batch = config.scan.max_batch;
        coarse_config.max_nprobe = config.scan.max_nprobe;
        coarse_config.distance = config.distance;
        coarse_config.select_l2_topk_per_request = true;
        coarse.init(coarse_config, coarse_view.centroids);
        pca = std::make_unique<paraselect::search::IVFPCAResidentSearch>(
            pca_view, list_view, config.scan);
        reader = paraselect::search::make_ivf_file_reader(
            exact_view, config.file_read);
        shortlist.resize(
            static_cast<size_t>(config.scan.max_batch) * config.scan.output_count);
        local_topk.resize(
            static_cast<size_t>(config.scan.max_batch) * config.scan.output_count);
        check_cuda(cudaStreamCreate(&stream), "create IVFPCAFile stream");
        build_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - build_start).count();
    }

    ~Impl() {
        if (stream) cudaStreamDestroy(stream);
    }

    IVFPCAFileBuildConfig config;
    paraselect::ivfpca_file::BuildArtifacts artifacts;
    IVFCoarseIndexView coarse_view;
    IVFListLayoutView list_view;
    paraselect::quantization::PCAStorageView pca_view;
    IVFFileStorageView exact_view;
    paraselect::search::IVFCoarseCtx coarse;
    std::unique_ptr<paraselect::search::IVFPCAResidentSearch> pca;
    std::shared_ptr<paraselect::search::IVFFileReader> reader;
    std::vector<int> shortlist;
    std::vector<int> local_topk;
    cudaStream_t stream = nullptr;
    double build_ms = 0.0;
};

IVFPCAFileIndex IVFPCAFileIndex::build(IVFPCAFileBuildConfig config) {
    return IVFPCAFileIndex(std::make_unique<Impl>(std::move(config)));
}

IVFPCAFileIndex::IVFPCAFileIndex(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

IVFPCAFileIndex::~IVFPCAFileIndex() = default;
IVFPCAFileIndex::IVFPCAFileIndex(IVFPCAFileIndex&&) noexcept = default;
IVFPCAFileIndex& IVFPCAFileIndex::operator=(IVFPCAFileIndex&&) noexcept = default;

void IVFPCAFileIndex::search(
    const IVFQueryBatchView& queries,
    const IVFSearchParams& params,
    const IVFSearchOutputView& output,
    IVFPCAFileSearchStats* stats) {
    if (!queries.fp32 || queries.count <= 0 ||
        queries.dim != impl_->coarse_view.dim || params.batch_size <= 0 ||
        params.batch_size > impl_->config.scan.max_batch || params.nprobe <= 0 ||
        params.nprobe > impl_->config.scan.max_nprobe || params.topk <= 0 ||
        params.topk > impl_->config.scan.output_count ||
        params.distance != impl_->config.distance || !output.indices ||
        !output.distances) {
        throw std::invalid_argument("invalid IVFPCAFile request");
    }
    if (output.coarse_ids) {
        throw std::invalid_argument("IVFPCAFile does not materialize coarse IDs");
    }
    if (impl_->exact_view.dtype == IVFExactRerankDType::U8 && !queries.u8) {
        throw std::invalid_argument("U8 IVFPCAFile requires U8 queries");
    }
    if (impl_->exact_view.dtype != IVFExactRerankDType::U8 &&
        impl_->exact_view.dtype != IVFExactRerankDType::FP32) {
        throw std::invalid_argument("unknown IVFPCAFile exact dtype");
    }

    IVFPCAFileSearchStats result;
    double scanned_sum = 0.0;
    double reduce_input_sum = 0.0;
    const auto wall_begin = Clock::now();
    for (int begin = 0; begin < queries.count; begin += params.batch_size) {
        const int batch_n = std::min(params.batch_size, queries.count - begin);
        const float* batch_fp32 =
            queries.fp32 + static_cast<size_t>(begin) * queries.dim;
        const uint8_t* batch_u8 = queries.u8
            ? queries.u8 + static_cast<size_t>(begin) * queries.dim
            : nullptr;

        paraselect::search::IVFCoarseTiming coarse_stats;
        const auto probes = impl_->coarse.search_device(
            batch_fp32, batch_n, params.nprobe, impl_->stream, &coarse_stats);
        result.coarse_ms += coarse_stats.total_ms;

        paraselect::search::IVFPCAResidentStats pca_stats;
        impl_->pca->search_device_probes(
            batch_fp32,
            probes,
            impl_->shortlist.data(),
            &pca_stats,
            impl_->stream);
        result.pca_metadata_ms += pca_stats.metadata_ms;
        result.pca_query_transform_ms += pca_stats.query_transform_ms;
        result.pca_scan_ms += pca_stats.scan_ms;
        result.pca_reduce_ms += pca_stats.reduce_ms;
        scanned_sum += pca_stats.avg_scanned_vectors * batch_n;
        reduce_input_sum += pca_stats.avg_reduce_input_candidates * batch_n;
        if (pca_stats.probe_schedule ==
            paraselect::search::IVFProbeScheduleStrategy::Sparse) {
            ++result.sparse_schedule_batches;
        } else {
            ++result.dense_schedule_batches;
        }

        paraselect::search::IVFRerankStats rerank_stats;
        paraselect::search::ivf_file_rerank(
            *impl_->reader,
            impl_->exact_view,
            batch_u8,
            batch_fp32,
            impl_->shortlist.data(),
            batch_n,
            impl_->config.scan.output_count,
            params.topk,
            impl_->local_topk.data(),
            output.distances + static_cast<size_t>(begin) * params.topk,
            impl_->config.file_block_selection,
            &rerank_stats);
        result.file_plan_ms += rerank_stats.plan_ms;
        result.file_read_ms += rerank_stats.read_ms;
        result.exact_distance_ms += rerank_stats.distance_ms;
        result.file_topk_ms += rerank_stats.topk_ms;
        result.unique_blocks += rerank_stats.unique_blocks;
        result.bytes_read += rerank_stats.bytes_read;
        result.cache_hits += rerank_stats.cache_hits;
        result.cache_misses += rerank_stats.cache_misses;
        result.blocks_considered += rerank_stats.blocks_considered;
        result.blocks_skipped += rerank_stats.blocks_skipped;
        result.distance_evaluations += rerank_stats.distance_evaluations;

        for (int row = 0; row < batch_n; ++row) {
            for (int rank = 0; rank < params.topk; ++rank) {
                const int local = impl_->local_topk[
                    static_cast<size_t>(row) * params.topk + rank];
                output.indices[
                    static_cast<size_t>(begin + row) * params.topk + rank] =
                    local >= 0 ? impl_->list_view.reordered_ids[local] : -1;
            }
        }
        ++result.batches;
    }
    result.wall_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - wall_begin).count();
    result.avg_pca_candidates = scanned_sum / queries.count;
    result.avg_pca_reduce_input = reduce_input_sum / queries.count;
    if (stats) *stats = result;
}

size_t IVFPCAFileIndex::size() const { return impl_->config.n_vectors; }
int IVFPCAFileIndex::dim() const { return impl_->config.source.spec().dim; }
int IVFPCAFileIndex::nlist() const { return impl_->config.nlist; }
double IVFPCAFileIndex::build_time_ms() const { return impl_->build_ms; }

}  // namespace paraselect::search
