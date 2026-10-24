#ifndef PARASELECT_INDEX_IVF_CACHE_H
#define PARASELECT_INDEX_IVF_CACHE_H

#include "dataset/cache/cluster_cache.h"
#include "dataset/common/dataset_source.h"
#include "dataset/cache/pca_cache.h"
#include "dataset/cache/residual_pq_cache.h"
#include "dataset/cache/turboquant_cache.h"

#include <cstdint>
#include <vector>

namespace paraselect::ivf_cache {

struct PlainPQParams {
    int m = 16;
    int k = 256;
    int iters = 12;
    int train_n = 50000000;
    int encode_chunk = 1 << 20;
};

struct ResidualPQParams {
    int m = 16;
    int k = 256;
    int max_gpus = 8;
    int train_n = 50000000;
    int encode_chunk = 1 << 20;
    bool ensure_row_correction = false;
};

// Build wall times, in seconds. encode_s includes H2D, GPU encoding and D2H;
// it must not be reported as pure kernel time. Cache hits are explicit.
struct ResidualPQBuildTiming {
    bool cache_hit=false;
    double lookup_s=0, layout_s=0, train_prepare_s=0, train_s=0;
    double encode_prepare_s=0, residual_io_s=0, encode_s=0, code_reorder_s=0;
    double commit_s=0, total_s=0;
    double row_correction_build_s=0, row_correction_quantize_s=0,
           row_correction_commit_s=0;
};

struct PCAParams {
    int dim = 64;
    int train_n = 100000;
    int encode_chunk = 1 << 18;
};

struct PQCaches {
    dataset::ClusterCache::CacheResult clusters;
    dataset::PQCacheResult pq;
};

struct MappedPQCaches {
    dataset::ClusterCache::MappedResult clusters;
    dataset::PQCacheMappedResult pq;
};

struct TurboQuantParams {
    int rotation_seed = 17;
    size_t encode_chunk = 1 << 19;
};

struct MappedTurboQuantCaches {
    dataset::ClusterCache::MappedResult clusters;
    dataset::TurboQuantCacheMappedResult tq;
};

MappedTurboQuantCaches ensure_turboquant_mapped(
    const dataset::DatasetSource& source,
    size_t n_vectors,
    int nlist,
    const KMeansSpec& kmeans,
    const TurboQuantParams& params);

dataset::ClusterCache::CacheResult ensure_kmeans(
    const dataset::DatasetSource& source,
    size_t n_vectors,
    int nlist,
    const KMeansSpec& spec);

dataset::ClusterCache::MappedResult ensure_kmeans_mapped(
    const dataset::DatasetSource& source,
    size_t n_vectors,
    int nlist,
    const KMeansSpec& spec);

PQCaches ensure_pq(
    const dataset::DatasetSource& source,
    size_t n_vectors,
    int nlist,
    const KMeansSpec& kmeans,
    const PlainPQParams& pq);

dataset::PQCacheResult ensure_pq(
    const dataset::DatasetSource& source,
    const dataset::ClusterCache::CacheResult& clusters,
    const KMeansSpec& kmeans,
    const PlainPQParams& pq);

PQCaches ensure_opq(
    const dataset::DatasetSource& source,
    size_t n_vectors,
    int nlist,
    const KMeansSpec& kmeans,
    const PlainPQParams& pq,
    const std::vector<float>& rotation,
    int opq_iters,
    int opq_train_n,
    unsigned seed);

PQCaches ensure_respq(
    const dataset::DatasetSource& source,
    size_t n_vectors,
    int nlist,
    const KMeansSpec& kmeans,
    const ResidualPQParams& pq);

MappedPQCaches ensure_pq_mapped(
    const dataset::DatasetSource& source,
    size_t n_vectors, int nlist,
    const KMeansSpec& kmeans, const PlainPQParams& pq);

MappedPQCaches ensure_respq_mapped(
    const dataset::DatasetSource& source,
    size_t n_vectors,
    int nlist,
    const KMeansSpec& kmeans,
    const ResidualPQParams& pq,
    ResidualPQBuildTiming* timing=nullptr);

dataset::PCACacheMappedResult ensure_pca_mapped(
    const dataset::DatasetSource& source,
    const dataset::ClusterCache::MappedResult& clusters,
    const PCAParams& params);

}  // namespace paraselect::ivf_cache

#endif
