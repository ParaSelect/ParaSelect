#ifndef PARASELECT_CUDA_TRANSFORM_OPQ_TRANSFORM_CUH
#define PARASELECT_CUDA_TRANSFORM_OPQ_TRANSFORM_CUH

#include "utils/pch.h"

#include <cstddef>
#include <vector>

namespace paraselect::transform {

struct OPQTransformSpec {
    int dim = 0;
};

class OPQTransform {
public:
    explicit OPQTransform(const OPQTransformSpec& spec);
    explicit OPQTransform(int dim);
    ~OPQTransform();

    OPQTransform(const OPQTransform&) = delete;
    OPQTransform& operator=(const OPQTransform&) = delete;

    void set_matrix(const std::vector<float>& matrix_rowmajor_out_in);
    void set_identity();
    void prepare();
    void release();

    void apply_device(const float* d_input, float* d_output, size_t n, cudaStream_t stream) const;
    void apply_host_to_host(const float* h_input, float* h_output, size_t n, cudaStream_t stream) const;

    int dim() const { return spec_.dim; }
    const std::vector<float>& host_matrix() const { return h_matrix_; }
    const float* device_matrix() const { return d_matrix_; }
    bool prepared() const { return prepared_; }

private:
    OPQTransformSpec spec_;
    std::vector<float> h_matrix_;
    float* d_matrix_ = nullptr;
    cublasHandle_t cublas_handle_ = nullptr;
    bool prepared_ = false;

    void validate_spec() const;
};

void opq_apply_device_gemm(
    cublasHandle_t handle,
    const float* d_input,
    const float* d_matrix_rowmajor_out_in,
    float* d_output,
    size_t n,
    int dim,
    cudaStream_t stream);

}  // namespace paraselect::transform

#endif
