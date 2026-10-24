#ifndef PARASELECT_SEARCH_IVFFLAT_IVFFLAT_BUILD_H
#define PARASELECT_SEARCH_IVFFLAT_IVFFLAT_BUILD_H

#include "dataset/dataset.cuh"
#include "search/ivfflat/ivfflat_index.h"

namespace paraselect::ivfflat {

struct BuildTiming {
    double kmeans_ms = 0.0;
};

BuildTiming build_storage(
    const search::IVFFlatBuildConfig& config, ClusterDataset* dataset);

}  // namespace paraselect::ivfflat

#endif  // PARASELECT_SEARCH_IVFFLAT_IVFFLAT_BUILD_H
