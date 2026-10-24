#ifndef PARASELECT_INDEX_IVF_CACHE_KMEANS_H
#define PARASELECT_INDEX_IVF_CACHE_KMEANS_H

#include "dataset/cache/cluster_cache.h"
#include "dataset/common/dataset_source.h"

#include <cstddef>
#include <string>

namespace paraselect::ivf_cache::detail {

bool valid_cluster_cache(
    const dataset::ClusterCache::CacheResult& cache,
    size_t n,
    int dim,
    int nlist);

}  // namespace paraselect::ivf_cache::detail

#endif
