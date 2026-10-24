#include "index/ivf_cache/source_vectors.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <vector>

namespace paraselect::ivf_cache::detail {

const float* as_fp32(dataset::VectorView vectors, std::vector<float>& converted) {
    if (vectors.dtype == dataset::VectorDType::F32) return vectors.fp32();
    converted.resize(vectors.n * static_cast<size_t>(vectors.dim));
    std::transform(vectors.u8(), vectors.u8() + converted.size(), converted.begin(),
                   [](uint8_t value) { return static_cast<float>(value); });
    return converted.data();
}

float* load_fp32_prefix(const dataset::DatasetSource& source, size_t n) {
    const auto info = source.base_info();
    if (n == 0 || n > info.n || n > static_cast<size_t>(INT_MAX)) {
        throw std::invalid_argument("invalid KMeans source prefix size");
    }
    const size_t values = n * static_cast<size_t>(source.spec().dim);
    auto* output = static_cast<float*>(std::aligned_alloc(64, ((values * sizeof(float) + 63) / 64) * 64));
    if (!output) throw std::bad_alloc();
    try {
        if (source.spec().dtype == dataset::VectorDType::F32) {
            if (source.read_base_range(0, n, output) != n) {
                throw std::runtime_error("short read while loading KMeans input");
            }
        } else {
            std::vector<uint8_t> chunk;
            for (size_t begin = 0; begin < n; begin += kIoChunk) {
                const size_t count = std::min<size_t>(kIoChunk, n - begin);
                chunk.resize(count * static_cast<size_t>(source.spec().dim));
                if (source.read_base_range(begin, count, chunk.data()) != count) {
                    throw std::runtime_error("short read while loading KMeans input");
                }
                std::transform(chunk.begin(), chunk.end(),
                               output + begin * static_cast<size_t>(source.spec().dim),
                               [](uint8_t value) { return static_cast<float>(value); });
            }
        }
        return output;
    } catch (...) {
        std::free(output);
        throw;
    }
}

uint8_t* load_u8_prefix(const dataset::DatasetSource& source, size_t n) {
    const auto info = source.base_info();
    if (source.spec().dtype != dataset::VectorDType::U8 || n == 0 || n > info.n ||
        n > static_cast<size_t>(INT_MAX)) {
        throw std::invalid_argument("CAGRA Lloyd requires a valid U8 source prefix");
    }
    const size_t values = n * static_cast<size_t>(source.spec().dim);
    auto* output = static_cast<uint8_t*>(
        std::aligned_alloc(64, ((values + 63) / 64) * 64));
    if (!output) throw std::bad_alloc();
    if (source.read_base_range(0, n, output) != n) {
        std::free(output);
        throw std::runtime_error("short read while loading CAGRA KMeans input");
    }
    return output;
}

}  // namespace paraselect::ivf_cache::detail
