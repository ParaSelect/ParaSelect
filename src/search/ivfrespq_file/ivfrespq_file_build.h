#ifndef PARASELECT_SEARCH_IVFRESPQ_FILE_IVFRESPQ_FILE_BUILD_H
#define PARASELECT_SEARCH_IVFRESPQ_FILE_IVFRESPQ_FILE_BUILD_H

#include "dataset/cache/pca_cache.h"
#include "dataset/temporary_exact_store.h"
#include "index/ivf_cache.h"
#include "search/ivfrespq_file/ivfrespq_file_index.h"

namespace paraselect::ivfrespq_file {

struct BuildArtifacts {
    ivf_cache::MappedPQCaches caches;
    dataset::PCACacheMappedResult pca;
    dataset::TemporaryExactStore exact;
    double exact_build_ms = 0.0;
};

BuildArtifacts build_artifacts(const search::IVFResPQFileBuildConfig& config);

}  // namespace paraselect::ivfrespq_file

#endif  // PARASELECT_SEARCH_IVFRESPQ_FILE_IVFRESPQ_FILE_BUILD_H
