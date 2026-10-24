#include "transform/orthogonal_projection_transform.cuh"

#include <cmath>
#include <random>
#include <stdexcept>

namespace paraselect::transform {

OrthogonalProjectionTransform::OrthogonalProjectionTransform(const OrthogonalProjectionTransformSpec& spec) : spec_(spec) {
    validate_spec();
}

OrthogonalProjectionTransform::~OrthogonalProjectionTransform() {
    release();
}

void OrthogonalProjectionTransform::validate_spec() const {
    if (spec_.d_in <= 0) throw std::invalid_argument("OrthogonalProjectionTransform: d_in must be positive");
    if (spec_.d_out <= 0) throw std::invalid_argument("OrthogonalProjectionTransform: d_out must be positive");
    if (spec_.d_out > spec_.d_in) {
        throw std::invalid_argument("OrthogonalProjectionTransform: d_out must not exceed d_in for orthogonal projection");
    }
}

void OrthogonalProjectionTransform::generate_matrix() {
    const size_t elems = (size_t)spec_.d_out * (size_t)spec_.d_in;
    h_matrix_.resize(elems);
    std::mt19937_64 rng(spec_.seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (size_t i = 0; i < elems; ++i) {
        h_matrix_[i] = normal(rng);
    }

    for (int r = 0; r < spec_.d_out; ++r) {
        float* row = h_matrix_.data() + (size_t)r * spec_.d_in;
        for (int prev = 0; prev < r; ++prev) {
            const float* basis = h_matrix_.data() + (size_t)prev * spec_.d_in;
            double dot = 0.0;
            for (int c = 0; c < spec_.d_in; ++c) dot += (double)row[c] * (double)basis[c];
            for (int c = 0; c < spec_.d_in; ++c) row[c] -= (float)dot * basis[c];
        }

        double norm2 = 0.0;
        for (int c = 0; c < spec_.d_in; ++c) norm2 += (double)row[c] * (double)row[c];
        if (norm2 <= 0.0) {
            throw std::runtime_error("OrthogonalProjectionTransform: failed to generate full-rank projection row");
        }
        const float inv_norm = 1.0f / std::sqrt((float)norm2);
        for (int c = 0; c < spec_.d_in; ++c) row[c] *= inv_norm;
    }

    const float scale = std::sqrt((float)spec_.d_in / (float)spec_.d_out);
    for (float& v : h_matrix_) v *= scale;
}

void OrthogonalProjectionTransform::prepare() {
    if (prepared_) return;
    validate_spec();
    generate_matrix();
    const size_t bytes = h_matrix_.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_matrix_, bytes));
    CUDA_CHECK(cudaMemcpy(d_matrix_, h_matrix_.data(), bytes, cudaMemcpyHostToDevice));
    if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
        CUDA_CHECK(cudaFree(d_matrix_));
        d_matrix_ = nullptr;
        throw std::runtime_error("OrthogonalProjectionTransform: cublasCreate failed");
    }
    prepared_ = true;
}

void OrthogonalProjectionTransform::release() {
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
        cublas_handle_ = nullptr;
    }
    if (d_matrix_) {
        cudaFree(d_matrix_);
        d_matrix_ = nullptr;
    }
    prepared_ = false;
}

void OrthogonalProjectionTransform::apply_device(
    const float* d_input,
    float* d_output,
    size_t n,
    cudaStream_t stream) const
{
    if (!prepared_ || !d_matrix_ || !cublas_handle_) {
        throw std::runtime_error("OrthogonalProjectionTransform: prepare() must be called before apply_device");
    }
    orthogonal_projection_apply_device_gemm(
        cublas_handle_, d_input, d_matrix_, d_output, n, spec_.d_in, spec_.d_out, stream);
}

void OrthogonalProjectionTransform::apply_host_to_host(
    const float* h_input,
    float* h_output,
    size_t n,
    cudaStream_t stream) const
{
    if (!prepared_) throw std::runtime_error("OrthogonalProjectionTransform: prepare() must be called before apply_host_to_host");
    if (n == 0) return;
    float* d_input = nullptr;
    float* d_output = nullptr;
    const size_t in_bytes = n * (size_t)spec_.d_in * sizeof(float);
    const size_t out_bytes = n * (size_t)spec_.d_out * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input, in_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, out_bytes));
    try {
        CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, in_bytes, cudaMemcpyHostToDevice, stream));
        apply_device(d_input, d_output, n, stream);
        CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, out_bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (...) {
        cudaFree(d_input);
        cudaFree(d_output);
        throw;
    }
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

void orthogonal_projection_apply_device_gemm(
    cublasHandle_t handle,
    const float* d_input,
    const float* d_matrix_rowmajor,
    float* d_output,
    size_t n,
    int d_in,
    int d_out,
    cudaStream_t stream)
{
    if (n == 0) return;
    if (!handle || !d_input || !d_matrix_rowmajor || !d_output) {
        throw std::invalid_argument("orthogonal_projection_apply_device_gemm: null pointer");
    }
    if (d_in <= 0 || d_out <= 0) {
        throw std::invalid_argument("orthogonal_projection_apply_device_gemm: invalid dimensions");
    }
    if (n > (size_t)std::numeric_limits<int>::max()) {
        throw std::invalid_argument("orthogonal_projection_apply_device_gemm: n exceeds cublas int range");
    }

    // Same row-major trick as IVFFlat coarse:
    // row-major A[d_out, d_in] is interpreted as column-major A_col[d_in, d_out].
    // row-major X[n, d_in] is interpreted as column-major X_col[d_in, n].
    // cublas computes C_col[d_out, n] = A_col^T * X_col, whose memory is row-major Y[n, d_out].
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSetStream(handle, stream);
    const cublasStatus_t st = cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        d_out,
        (int)n,
        d_in,
        &alpha,
        d_matrix_rowmajor,
        d_in,
        d_input,
        d_in,
        &beta,
        d_output,
        d_out);
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("orthogonal_projection_apply_device_gemm: cublasSgemm failed");
    }
    CUDA_LAUNCH_CHECK("orthogonal_projection_apply_device_gemm cublasSgemm");
}

}  // namespace paraselect::transform
