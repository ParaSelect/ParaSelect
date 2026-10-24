#ifndef PARASELECT_CUDA_DATASET_CACHE_RESIDUAL_PQ_CACHE_H
#define PARASELECT_CUDA_DATASET_CACHE_RESIDUAL_PQ_CACHE_H

#include <cstdint>
#include <string>
#include <vector>
#include "dataset/common/mapped_array.h"

namespace paraselect {
namespace dataset {

struct PQCacheRequest {
    int n_vectors = 0;
    int dim = 0;
    int n_lists = 0;
    std::string cluster_layout_id;
    int pq_m = 0;
    int pq_k = 0;
    int dsub() const { return pq_m > 0 ? dim / pq_m : 0; }
    int pq_iters = 0;
    std::string trainer = "gpu-ooc";
    std::string training_sampling = "full";
    int training_rows = 0;  // Zero means all n_vectors for full training.
    int opq_iters = 0;
    int opq_train_rows = 0;
    unsigned opq_seed = 0;
    std::string pq_type = "residual";
    std::string codes_layout = "cluster-reordered-[N,M]";
};

struct PQCacheResult {
    PQCacheRequest request;
    bool hit = false;            // full hit: codebook + codes
    bool codebook_hit = false;   // codebook exists even if codes are missing
    std::string entry_dir;
    std::vector<float> codebook;   // [M, K, d_sub]
    std::vector<uint8_t> codes;    // [N, M], layout described by spec.codes_layout
};

struct PQCacheMappedResult {
    PQCacheRequest request;
    bool hit = false;
    std::string entry_dir;
    MappedArray<float> codebook;
    MappedArray<uint8_t> codes;
};

class ResidualPQCache {
public:
    ResidualPQCache(std::string cache_dir, std::string dataset_name);

    PQCacheResult try_load(
        const PQCacheRequest& spec) const;

    PQCacheMappedResult try_map(
        const PQCacheRequest& spec) const;

    void publish_staged(const std::string& entry, const PQCacheRequest& spec) const;

    void save(
        const PQCacheRequest& spec,
        const float* codebook,
        const uint8_t* codes) const;

    std::string get_cache_dir() const { return cache_dir_; }
private:
    std::string cache_dir_;
    std::string dataset_name_;

    bool load_codebook(
        const std::string& path,
        const PQCacheRequest& spec,
        std::vector<float>& out) const;
    bool load_codes(
        const std::string& path,
        const PQCacheRequest& spec,
        std::vector<uint8_t>& out) const;
    void write_meta(
        const std::string& path,
        const std::string& dataset_name,
        const PQCacheRequest& spec) const;
};


}  // namespace dataset
}  // namespace paraselect

#endif  // PARASELECT_CUDA_DATASET_CACHE_RESIDUAL_PQ_CACHE_H
