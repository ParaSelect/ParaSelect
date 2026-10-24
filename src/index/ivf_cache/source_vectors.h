#ifndef PARASELECT_INDEX_IVF_CACHE_SOURCE_VECTORS_H
#define PARASELECT_INDEX_IVF_CACHE_SOURCE_VECTORS_H

#include "dataset/common/dataset_source.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace paraselect::ivf_cache::detail {

constexpr int kIoChunk = 1 << 20;

const float* as_fp32(dataset::VectorView vectors, std::vector<float>& converted);
float* load_fp32_prefix(const dataset::DatasetSource& source, size_t n);
uint8_t* load_u8_prefix(const dataset::DatasetSource& source, size_t n);

}  // namespace paraselect::ivf_cache::detail

#endif
