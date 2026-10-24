#ifndef PARASELECT_DATASET_CACHE_PCA_CACHE_H
#define PARASELECT_DATASET_CACHE_PCA_CACHE_H

#include "dataset/common/mapped_array.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace paraselect::dataset {

struct PCACacheSpec {
    int n_vectors = 0;
    int source_dim = 0;
    int dim = 0;
    int train_n = 0;
    std::string cluster_layout_id;
    std::string quantization = "symmetric-per-dimension-int8-v1";
};

struct PCACacheMappedResult {
    PCACacheSpec request;
    bool hit = false;
    std::string entry_dir;
    MappedArray<int8_t> vectors;
    MappedArray<float> projection;
    MappedArray<float> bias;
    MappedArray<float> scales;
};

class PCACacheWriter {
public:
    ~PCACacheWriter();
    PCACacheWriter(const PCACacheWriter&) = delete;
    PCACacheWriter& operator=(const PCACacheWriter&) = delete;
    PCACacheWriter(PCACacheWriter&&) noexcept;
    PCACacheWriter& operator=(PCACacheWriter&&) noexcept;

    int8_t* vectors();
    void commit(
        const std::vector<float>& projection,
        const std::vector<float>& bias,
        const std::vector<float>& scales);

private:
    friend class PCACache;
    struct Impl;
    explicit PCACacheWriter(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class PCACache {
public:
    PCACache(std::string directory, std::string dataset_name);

    PCACacheMappedResult try_map(
        const PCACacheSpec& spec) const;

    PCACacheWriter create(
        const PCACacheSpec& spec) const;

private:

    std::string directory_;
    std::string dataset_name_;
};

}  // namespace paraselect::dataset

#endif  // PARASELECT_DATASET_CACHE_PCA_CACHE_H
