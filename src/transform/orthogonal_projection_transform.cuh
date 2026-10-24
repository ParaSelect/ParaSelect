#ifndef PARASELECT_CUDA_TRANSFORM_ORTHOGONAL_PROJECTION_TRANSFORM_CUH
#define PARASELECT_CUDA_TRANSFORM_ORTHOGONAL_PROJECTION_TRANSFORM_CUH

#include "utils/pch.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace paraselect::transform {

struct OrthogonalProjectionTransformSpec {
    int d_in = 0;
    int d_out = 0;
    uint64_t seed = 12345;
};

class OrthogonalProjectionTransform {
public:
    explicit OrthogonalProjectionTransform(const OrthogonalProjectionTransformSpec& spec);
    ~OrthogonalProjectionTransform();

    OrthogonalProjectionTransform(const OrthogonalProjectionTransform&) = delete;
    OrthogonalProjectionTransform& operator=(const OrthogonalProjectionTransform&) = delete;

    void prepare();
    void release();

    void apply_device(const float* d_input, float* d_output, size_t n, cudaStream_t stream) const;
    void apply_host_to_host(const float* h_input, float* h_output, size_t n, cudaStream_t stream) const;

    int d_in() const { return spec_.d_in; }
    int d_out() const { return spec_.d_out; }
    uint64_t seed() const { return spec_.seed; }
    const OrthogonalProjectionTransformSpec& spec() const { return spec_; }
    const std::vector<float>& host_matrix() const { return h_matrix_; }
    const float* device_matrix() const { return d_matrix_; }
    bool prepared() const { return prepared_; }

private:
    OrthogonalProjectionTransformSpec spec_;
    std::vector<float> h_matrix_;
    float* d_matrix_ = nullptr;
    cublasHandle_t cublas_handle_ = nullptr;
    bool prepared_ = false;

    void validate_spec() const;
    void generate_matrix();
};

void orthogonal_projection_apply_device_gemm(
    cublasHandle_t handle,
    const float* d_input,
    const float* d_matrix_rowmajor,
    float* d_output,
    size_t n,
    int d_in,
    int d_out,
    cudaStream_t stream);

}  // namespace paraselect::transform

#endif  // PARASELECT_CUDA_TRANSFORM_ORTHOGONAL_PROJECTION_TRANSFORM_CUH
