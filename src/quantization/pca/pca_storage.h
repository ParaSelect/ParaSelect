#pragma once

#include "utils/types.h"
#include <cstdint>

namespace paraselect::quantization {

// Borrowed row-major PCA/int8 representation; row ordering belongs to the index.
struct PCAStorageView {
    const int8_t* vectors = nullptr;
    const float* projection = nullptr;
    const float* bias = nullptr;
    const float* scales = nullptr;
    idx_t count = 0;
    int source_dim = 0;
    int dim = 0;

    bool enabled() const noexcept { return vectors != nullptr; }
    bool empty() const noexcept {
        return !vectors && !projection && !bias && !scales &&
               count == 0 && source_dim == 0 && dim == 0;
    }
};


}  // namespace paraselect::quantization
