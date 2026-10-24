#ifndef PARASELECT_DATASET_COMMON_VECTOR_DTYPE_H
#define PARASELECT_DATASET_COMMON_VECTOR_DTYPE_H
#include <cstddef>
namespace paraselect::dataset {
enum class VectorDType { U8, F32 };
inline constexpr size_t vector_dtype_size(VectorDType dtype) { return dtype == VectorDType::U8 ? 1 : sizeof(float); }
inline constexpr const char* vector_dtype_name(VectorDType dtype) { return dtype == VectorDType::U8 ? "u8" : "fp32"; }
}
#endif
