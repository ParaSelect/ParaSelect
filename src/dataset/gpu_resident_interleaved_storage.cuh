#ifndef PARASELECT_GPU_RESIDENT_INTERLEAVED_STORAGE_CUH
#define PARASELECT_GPU_RESIDENT_INTERLEAVED_STORAGE_CUH

#include "utils/pch.h"
#include <cuda_runtime.h>
#include <cstddef>

namespace paraselect::dataset {

struct GpuResidentInterleavedStorage {
    float* d_vectors = nullptr;
    uint8_t* d_vectors_u8 = nullptr;
    float* d_norms = nullptr;
    idx_t* d_reordered_indices = nullptr;
    size_t bytes = 0;
    int n_vectors = 0;
    int n_padded = 0;
    int dim = 0;
    int nlist = 0;

    bool uses_u8() const { return d_vectors_u8 != nullptr; }

    bool is_valid() const {
        const bool has_one_vector_storage = (d_vectors != nullptr) != (d_vectors_u8 != nullptr);
        return has_one_vector_storage && d_norms && d_reordered_indices &&
               n_vectors > 0 && n_padded >= n_vectors && dim > 0 && nlist > 0;
    }

    void release() {
        if (d_vectors) cudaFree(d_vectors);
        if (d_vectors_u8) cudaFree(d_vectors_u8);
        if (d_norms) cudaFree(d_norms);
        if (d_reordered_indices) cudaFree(d_reordered_indices);
        d_vectors = nullptr;
        d_vectors_u8 = nullptr;
        d_norms = nullptr;
        d_reordered_indices = nullptr;
        bytes = 0;
        n_vectors = 0;
        n_padded = 0;
        dim = 0;
        nlist = 0;
    }
};

}  // namespace paraselect::dataset

#endif  // PARASELECT_GPU_RESIDENT_INTERLEAVED_STORAGE_CUH
