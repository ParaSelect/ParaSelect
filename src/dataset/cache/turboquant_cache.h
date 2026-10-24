#ifndef PARASELECT_CUDA_DATASET_CACHE_TURBOQUANT_CACHE_H
#define PARASELECT_CUDA_DATASET_CACHE_TURBOQUANT_CACHE_H

#include <cstdint>
#include <string>

#include "dataset/common/mapped_array.h"

namespace paraselect {
namespace dataset {

// Identity of a cached TurboQuant 1-bit sign encoding. Any field that changes
// the encoded bytes (dimension, words per code, rotation seed, builder
// revision) is a manifest field so incompatible entries cannot be mapped.
struct TurboQuantCacheRequest {
    int n_vectors = 0;
    int dim = 0;
    int n_lists = 0;
    int rotation_seed = 0;
    std::string cluster_layout_id;
    std::string builder = "chunk-residual-l1-v1";
    std::string codes_layout = "cluster-reordered-[N,words]-u32";

    int words_per_code() const { return dim / 32; }
};

struct TurboQuantCacheMappedResult {
    TurboQuantCacheRequest request;
    bool hit = false;
    std::string entry_dir;
    MappedArray<uint32_t> codes;   // [n_vectors, words_per_code]
    MappedArray<float> scales;     // [n_vectors, 1]
};

class TurboQuantCache {
public:
    TurboQuantCache(std::string cache_dir, std::string dataset_name);

    TurboQuantCacheMappedResult try_map(
        const TurboQuantCacheRequest& spec) const;

    void publish_staged(const std::string& entry_dir,
                                          const TurboQuantCacheRequest& spec) const;

    std::string get_cache_dir() const { return cache_dir_; }

private:
    std::string cache_dir_;
    std::string dataset_name_;
};

}  // namespace dataset
}  // namespace paraselect

#endif  // PARASELECT_CUDA_DATASET_CACHE_TURBOQUANT_CACHE_H
