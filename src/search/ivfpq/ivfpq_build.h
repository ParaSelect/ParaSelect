#ifndef PARASELECT_SEARCH_IVFPQ_IVFPQ_BUILD_H
#define PARASELECT_SEARCH_IVFPQ_IVFPQ_BUILD_H

#include "index/ivf_cache.h"
#include "search/ivfpq/ivfpq_index.h"
#include "transform/opq_transform.cuh"

#include <cstdint>
#include <memory>
#include <vector>

namespace paraselect::ivfpq {

struct BuildArtifacts {
    paraselect::ivf_cache::PQCaches caches;
    std::unique_ptr<paraselect::transform::OPQTransform> transform;
    std::vector<float> transformed_centroids;
    std::vector<uint8_t> rerank_u8;
    std::vector<float> rerank_fp32;
};

BuildArtifacts build_artifacts(const search::IVFPQBuildConfig& config);

}  // namespace paraselect::ivfpq

#endif  // PARASELECT_SEARCH_IVFPQ_IVFPQ_BUILD_H
