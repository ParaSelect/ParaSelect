#ifndef PARASELECT_CUDA_DATASET_CACHE_CLUSTER_CACHE_H
#define PARASELECT_CUDA_DATASET_CACHE_CLUSTER_CACHE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "dataset/common/mapped_array.h"

#include "utils/types.h"

namespace paraselect {
namespace dataset {

struct ClusterCacheSpec {
    int n_vectors = 0;
    int n_lists = 0;
    int dim = 0;
    std::string build_recipe_json;
    std::string layout_id;  // Optional explicit manifest selector.
};

class ClusterCache {
public:
    struct CacheResult {
        ClusterCacheSpec request;
        bool hit = false;
        std::string entry_dir;
        std::string layout_id;
        std::vector<float> centroids;
        std::vector<idx_t> reordered_indices;
        std::vector<int> cluster_counts;
        std::vector<idx_t> cluster_offsets;
    };

    struct MappedResult {
        ClusterCacheSpec request;
        bool hit = false;
        std::string entry_dir;
        std::string layout_id;
        MappedArray<float> centroids;
        MappedArray<idx_t> reordered_indices;
        MappedArray<int> cluster_counts;
        MappedArray<idx_t> cluster_offsets;
    };

    explicit ClusterCache(const std::string& cache_dir = "");

    CacheResult try_load(
        const std::string& dataset_name,
        const ClusterCacheSpec& spec);

    MappedResult try_map(
        const std::string& dataset_name,
        const ClusterCacheSpec& spec);

    std::string publish_staged(
        const std::string& entry_dir,
        const std::string& dataset_name,
        const ClusterCacheSpec& spec);

    void save(
        const std::string& dataset_name,
        const ClusterCacheSpec& spec,
        const float* centroids,
        const idx_t* reordered_indices,
        const int* cluster_counts,
        const idx_t* cluster_offsets,
        int n_clusters);

    std::string get_cache_dir() const { return cache_dir_; }

private:
    std::string cache_dir_;

    std::string read_manifest_layout_id(const std::string& entry_dir) const;
    std::string compute_layout_id(const std::string& entry_dir) const;
};

} // namespace dataset
} // namespace paraselect

#endif
