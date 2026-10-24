#include "search/ivfrespq_file/ivfrespq_file_index.h"

#include "dataset/common/dataset_spec.h"
#include "search/ivfrespq_file/ivfrespq_file_build.h"

#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using Clock = std::chrono::steady_clock;

}  // namespace

namespace paraselect::search {

struct IVFResPQFileIndex::Impl {
    explicit Impl(IVFResPQFileBuildConfig requested) : config(std::move(requested)) {
        const auto build_start = Clock::now();
        artifacts = paraselect::ivfrespq_file::build_artifacts(config);

        index.coarse = {
            artifacts.caches.clusters.centroids.data(),
            config.nlist,
            config.source.spec().dim};
        index.lists = {
            artifacts.caches.clusters.cluster_offsets.data(),
            artifacts.caches.clusters.reordered_indices.data(),
            config.nlist,
            static_cast<idx_t>(config.n_vectors)};
        index.storage.pq.codes = artifacts.caches.pq.codes.data();
        index.storage.pq.codebook = artifacts.caches.pq.codebook.data();
        index.storage.pq.subvectors = config.pq_m;
        index.storage.pq.codebook_size = config.pq_k;
        if (artifacts.pca.hit) {
            index.storage.pca = {
                artifacts.pca.vectors.data(),
                artifacts.pca.projection.data(),
                artifacts.pca.bias.data(),
                artifacts.pca.scales.data(),
                static_cast<idx_t>(config.n_vectors),
                config.source.spec().dim,
                config.pca_dim};
        }
        index.storage.assets.exact = IVFExactStorageView::from_file(artifacts.exact.view());

        validate_ivf_index_shape(index.coarse, index.lists);
        if (!index.storage.pq.codes || !index.storage.pq.codebook ||
            index.storage.pq.subvectors <= 0 ||
            index.storage.pq.codebook_size <= 0) {
            throw std::runtime_error("IVFResPQFile cache produced invalid storage");
        }
        if (config.gpu_combo.empty()) {
            throw std::invalid_argument("IVFResPQFile GPU combo is empty");
        }
        options = config.options;
        options.gpu.combo = config.gpu_combo.c_str();
        build_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - build_start).count();
    }

    IVFResPQFileBuildConfig config;
    paraselect::ivfrespq_file::BuildArtifacts artifacts;
    IVFResPQIndexView index;
    IVFResPQSearchOptions options;
    double build_ms = 0.0;
};

IVFResPQFileIndex IVFResPQFileIndex::build(IVFResPQFileBuildConfig config) {
    return IVFResPQFileIndex(std::make_unique<Impl>(std::move(config)));
}

IVFResPQFileIndex::IVFResPQFileIndex(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

IVFResPQFileIndex::~IVFResPQFileIndex() = default;
IVFResPQFileIndex::IVFResPQFileIndex(IVFResPQFileIndex&&) noexcept = default;
IVFResPQFileIndex& IVFResPQFileIndex::operator=(IVFResPQFileIndex&&) noexcept =
    default;

void IVFResPQFileIndex::search(
    const IVFQueryBatchView& queries,
    const IVFSearchParams& params,
    const IVFSearchOutputView& output,
    IVFSearchStats* stats) {
    if (queries.dim != impl_->index.coarse.dim) {
        throw std::invalid_argument("IVFResPQFile query and index dimensions differ");
    }
    ivfrespq_search(
        {impl_->index, queries, params, output, impl_->options},
        stats);
}

size_t IVFResPQFileIndex::size() const { return impl_->config.n_vectors; }
int IVFResPQFileIndex::dim() const { return impl_->config.source.spec().dim; }
int IVFResPQFileIndex::nlist() const { return impl_->config.nlist; }
int IVFResPQFileIndex::pq_m() const { return impl_->config.pq_m; }
int IVFResPQFileIndex::pq_k() const { return impl_->config.pq_k; }
double IVFResPQFileIndex::build_time_ms() const { return impl_->build_ms; }
double IVFResPQFileIndex::exact_storage_build_time_ms() const {
    return impl_->artifacts.exact_build_ms;
}

}  // namespace paraselect::search
