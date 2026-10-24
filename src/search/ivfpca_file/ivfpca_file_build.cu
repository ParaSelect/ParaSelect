#include "search/ivfpca_file/ivfpca_file_build.h"

#include "index/ivf_cache.h"

#include <limits>
#include <stdexcept>

namespace paraselect::ivfpca_file {
using namespace paraselect::search;

BuildArtifacts build_artifacts(const IVFPCAFileBuildConfig& config) {
    const auto base = config.source.base_info();
    if (config.n_vectors == 0 || config.n_vectors > base.n ||
        config.n_vectors > static_cast<size_t>(std::numeric_limits<idx_t>::max()) ||
        config.nlist <= 0 || config.pca_dim <= 0 ||
        config.pca_dim > config.source.spec().dim || config.pca_train_n <= 0 ||
        config.pca_encode_chunk <= 0) {
        throw std::invalid_argument("invalid IVFPCAFile build configuration");
    }

    BuildArtifacts out;
    out.clusters = ivf_cache::ensure_kmeans_mapped(
        config.source, config.n_vectors, config.nlist, config.kmeans);

    ivf_cache::PCAParams pca;
    pca.dim = config.pca_dim;
    pca.train_n = config.pca_train_n;
    pca.encode_chunk = config.pca_encode_chunk;
    out.pca = ivf_cache::ensure_pca_mapped(config.source, out.clusters, pca);

    dataset::TemporaryExactStoreOptions exact;
    exact.directory = config.temporary_directory;
    out.exact = dataset::TemporaryExactStore::build(
        config.source,
        out.clusters.reordered_indices.data(),
        static_cast<idx_t>(config.n_vectors),
        config.source.spec().dim,
        exact);
    return out;
}

}  // namespace paraselect::ivfpca_file
