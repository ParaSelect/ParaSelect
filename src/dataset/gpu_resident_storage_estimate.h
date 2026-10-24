#ifndef PARASELECT_DATASET_GPU_RESIDENT_STORAGE_ESTIMATE_H
#define PARASELECT_DATASET_GPU_RESIDENT_STORAGE_ESTIMATE_H

#include "utils/pch.h"

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace paraselect::dataset {

struct GpuResidentStorageEstimate {
    size_t vector_bytes = 0;
    size_t norm_bytes = 0;
    size_t id_bytes = 0;

    size_t total_bytes() const {
        if (vector_bytes > std::numeric_limits<size_t>::max() - norm_bytes ||
            vector_bytes + norm_bytes > std::numeric_limits<size_t>::max() - id_bytes) {
            throw std::overflow_error("GPU resident storage byte count overflow");
        }
        return vector_bytes + norm_bytes + id_bytes;
    }
};

inline size_t checked_storage_product(size_t lhs, size_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::overflow_error("GPU resident storage byte count overflow");
    }
    return lhs * rhs;
}

inline GpuResidentStorageEstimate estimate_gpu_resident_storage(
    size_t vector_rows,
    size_t id_rows,
    int dim,
    size_t vector_element_bytes) {
    if (dim <= 0 || vector_element_bytes == 0) {
        throw std::invalid_argument(
            "GPU resident storage requires positive dimension and element size");
    }
    const size_t vector_elements =
        checked_storage_product(vector_rows, static_cast<size_t>(dim));
    return {
        checked_storage_product(vector_elements, vector_element_bytes),
        checked_storage_product(vector_rows, sizeof(float)),
        checked_storage_product(id_rows, sizeof(idx_t)),
    };
}

} // namespace paraselect::dataset

#endif // PARASELECT_DATASET_GPU_RESIDENT_STORAGE_ESTIMATE_H
