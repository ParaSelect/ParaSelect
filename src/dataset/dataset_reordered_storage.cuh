#ifndef PARASELECT_DATASET_REORDERED_STORAGE_CUH
#define PARASELECT_DATASET_REORDERED_STORAGE_CUH

#include "dataset/ownership_helper.h"

namespace paraselect {
namespace dataset {

struct ReorderedRowMajorStorage {
    float* fp32 = nullptr;      // row-major [n_total_vectors, vector_dim]
    uint8_t* u8 = nullptr;      // row-major uint8 [n_total_vectors, vector_dim]
    paraselect::ownership::BufferOwner fp32_owner = paraselect::ownership::BufferOwner::None;
    paraselect::ownership::BufferOwner u8_owner = paraselect::ownership::BufferOwner::None;
};

}  // namespace dataset
}  // namespace paraselect

#endif  // PARASELECT_DATASET_REORDERED_STORAGE_CUH
