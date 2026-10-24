#ifndef PARASELECT_SEARCH_IVFPCA_FILE_IVFPCA_FILE_BUILD_H
#define PARASELECT_SEARCH_IVFPCA_FILE_IVFPCA_FILE_BUILD_H

#include "dataset/cache/cluster_cache.h"
#include "dataset/cache/pca_cache.h"
#include "dataset/temporary_exact_store.h"
#include "search/ivfpca_file/ivfpca_file_index.h"

namespace paraselect::ivfpca_file {

struct BuildArtifacts {
    dataset::ClusterCache::MappedResult clusters;
    dataset::PCACacheMappedResult pca;
    dataset::TemporaryExactStore exact;
};

BuildArtifacts build_artifacts(const search::IVFPCAFileBuildConfig& config);

}  // namespace paraselect::ivfpca_file

#endif  // PARASELECT_SEARCH_IVFPCA_FILE_IVFPCA_FILE_BUILD_H
