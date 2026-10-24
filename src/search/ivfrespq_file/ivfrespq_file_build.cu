#include "search/ivfrespq_file/ivfrespq_file_build.h"

#include <chrono>
#include <limits>
#include <stdexcept>

namespace paraselect::ivfrespq_file {
using namespace paraselect::search;

BuildArtifacts build_artifacts(const IVFResPQFileBuildConfig& config) {
    const auto base = config.source.base_info();
    if (config.n_vectors == 0 || config.n_vectors > base.n ||
        config.n_vectors > static_cast<size_t>(std::numeric_limits<idx_t>::max()) ||
        config.nlist <= 0 || config.pq_m <= 0 || config.pq_k <= 0 ||
        config.pq_max_gpus <= 0 || config.pq_train_n <= 0 ||
        config.pq_encode_chunk <= 0 ||
        config.source.spec().dim % config.pq_m != 0) {
        throw std::invalid_argument("invalid IVFResPQFile build configuration");
    }
    const bool use_pca = config.options.pca_backend != IVFRefineBackend::None;
    if (use_pca &&
        (config.pca_dim <= 0 || config.pca_dim > config.source.spec().dim ||
         config.pca_train_n <= 0 || config.pca_encode_chunk <= 0 ||
         config.options.shortlist_count <= 0)) {
        throw std::invalid_argument("invalid IVFResPQFile PCA configuration");
    }
    if (!use_pca &&
        (config.pca_dim != 0 || config.options.shortlist_count != 0)) {
        throw std::invalid_argument("IVFResPQFile PCA storage is not enabled");
    }

    ivf_cache::ResidualPQParams pq;
    pq.m = config.pq_m;
    pq.k = config.pq_k;
    pq.max_gpus = config.pq_max_gpus;
    pq.train_n = config.pq_train_n;
    pq.encode_chunk = config.pq_encode_chunk;

    BuildArtifacts out;
    out.caches = ivf_cache::ensure_respq_mapped(
        config.source, config.n_vectors, config.nlist, config.kmeans, pq);
    if (use_pca) {
        ivf_cache::PCAParams pca;
        pca.dim = config.pca_dim;
        pca.train_n = config.pca_train_n;
        pca.encode_chunk = config.pca_encode_chunk;
        out.pca = ivf_cache::ensure_pca_mapped(
            config.source, out.caches.clusters, pca);
    }

    dataset::TemporaryExactStoreOptions exact;
    exact.directory = config.temporary_directory;
    const auto exact_start = std::chrono::steady_clock::now();
    out.exact = dataset::TemporaryExactStore::build(
        config.source,
        out.caches.clusters.reordered_indices.data(),
        static_cast<idx_t>(config.n_vectors),
        config.source.spec().dim,
        exact);
    out.exact_build_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - exact_start).count();
    return out;
}

}  // namespace paraselect::ivfrespq_file
