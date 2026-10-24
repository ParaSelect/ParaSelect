#ifndef PARASELECT_DATASET_COMMON_VECTOR_BUFFER_H
#define PARASELECT_DATASET_COMMON_VECTOR_BUFFER_H
#include "vector_dtype.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>
namespace paraselect::dataset {
struct VectorView {
    VectorDType dtype = VectorDType::F32; const void* data = nullptr; size_t n = 0; int dim = 0;
    size_t bytes() const { return n * static_cast<size_t>(dim) * vector_dtype_size(dtype); }
    const uint8_t* u8() const { return dtype == VectorDType::U8 ? static_cast<const uint8_t*>(data) : nullptr; }
    const float* fp32() const { return dtype == VectorDType::F32 ? static_cast<const float*>(data) : nullptr; }
};
inline void convert_u8_to_fp32(VectorView view, std::vector<float>* output) {
    if (view.dtype != VectorDType::U8) return;
    output->resize(view.n * static_cast<size_t>(view.dim));
    std::transform(view.u8(), view.u8() + output->size(), output->begin(),
                   [](uint8_t value) { return static_cast<float>(value); });
}

class VectorBuffer {
public:
    using Deleter = void (*)(void*);
    VectorBuffer() = default;
    VectorBuffer(VectorDType dtype, void* data, size_t n, int dim, Deleter deleter) : dtype_(dtype), data_(data), n_(n), dim_(dim), deleter_(deleter) { if ((n > 0 && !data) || dim <= 0) throw std::invalid_argument("invalid VectorBuffer shape"); }
    ~VectorBuffer() { reset(); }
    VectorBuffer(const VectorBuffer&) = delete; VectorBuffer& operator=(const VectorBuffer&) = delete;
    VectorBuffer(VectorBuffer&& other) noexcept { move_from(std::move(other)); }
    VectorBuffer& operator=(VectorBuffer&& other) noexcept { if (this != &other) { reset(); move_from(std::move(other)); } return *this; }
    VectorView view() const { return {dtype_, data_, n_, dim_}; } bool empty() const { return n_ == 0; }
    void reset() { if (data_ && deleter_) deleter_(data_); data_ = nullptr; n_ = 0; dim_ = 0; deleter_ = nullptr; }
private:
    void move_from(VectorBuffer&& o) { dtype_=o.dtype_; data_=o.data_; n_=o.n_; dim_=o.dim_; deleter_=o.deleter_; o.data_=nullptr; o.n_=0; o.dim_=0; o.deleter_=nullptr; }
    VectorDType dtype_=VectorDType::F32; void* data_=nullptr; size_t n_=0; int dim_=0; Deleter deleter_=nullptr;
};
}
#endif
