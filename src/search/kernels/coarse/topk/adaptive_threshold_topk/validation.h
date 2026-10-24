#pragma once

#include <cstdint>
#include <stdexcept>

namespace paraselect::coarse::detail {

// Validate before any in-place score conversion or kernel launch. TMA consumes
// 256-element row strides and a 16-byte-aligned input allocation/subview.
inline void validate_topk_input(
    const float* scores, int batch, int length, int stride, int k,
    const float* values, const int64_t* indices) {
    if (batch <= 0 || k < 8 || k > 16384 || k % 8 || length < k ||
        length >= (1 << 23) || stride < length || stride % 256 ||
        !scores || !values || !indices ||
        (reinterpret_cast<std::uintptr_t>(scores) & 15u)) {
        throw std::invalid_argument("Adaptive Threshold TopK: unsupported shape or input alignment");
    }
}

inline void validate_l2_input(
    const float* scores, const float* query_norm, const float* center_norm,
    int batch, int length, int k, const float* values, const int64_t* indices) {
    validate_topk_input(scores, batch, length, length, k, values, indices);
    if (!query_norm || !center_norm) {
        throw std::invalid_argument("Adaptive Threshold TopK: missing L2 norms");
    }
}

}  // namespace paraselect::coarse::detail
