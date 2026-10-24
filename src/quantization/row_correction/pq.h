#pragma once

#include <cstddef>
#include <cstdint>

namespace paraselect::row_correction {

// Match the FP32 norm/dot producer used by the fixed-model U16 range bound.
inline float pq_value(const float* centroid, const float* codebook,
                      const std::uint8_t* code, int m, int k, int dsub) {
    float norm = 0, dot = 0;
    for (int sub = 0; sub < m; ++sub) {
        const float* word = codebook + (std::size_t(sub)*k + code[sub])*dsub;
        for (int j = 0; j < dsub; ++j) {
            norm += word[j]*word[j];
            dot += centroid[sub*dsub+j]*word[j];
        }
    }
    return norm + 2.0f*dot;
}

} // namespace paraselect::row_correction
