/**
 * LinearQuantizerFP16:fp32 <-> fp16 (cast +  scale/zero_point)
 */
#include "quantization/linear/linear_fp16.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstring>

namespace ivf {

namespace {

__global__ void encode_fp32_to_fp16_kernel(
    const float* __restrict__ d_data,
    int n, int dim,
    float scale, float zero_point,
    half* __restrict__ d_codes
) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)n * dim;
    if (i >= total) return;
    float v = (d_data[i] - zero_point) * scale;
    d_codes[i] = __float2half_rn(v);
}

__global__ void decode_fp16_to_fp32_kernel(
    const half* __restrict__ d_codes,
    int n, int dim,
    float scale, float zero_point,
    float* __restrict__ d_data
) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)n * dim;
    if (i >= total) return;
    float v = __half2float(d_codes[i]);
    d_data[i] = v / scale + zero_point;
}

} // namespace

LinearQuantizerFP16GPU::LinearQuantizerFP16GPU(float scale, float zero_point)
    : scale_(scale), zero_point_(zero_point) {}

void LinearQuantizerFP16GPU::train(const float* /*h_data*/, int /*n*/, int /*dim*/) {
    // fp16  cast ; per-tensor scale  min/max  scale/zero_point
}

void LinearQuantizerFP16GPU::encode(const float* d_data, int n, int dim, void* d_codes) {
    encode_async(d_data, n, dim, d_codes, nullptr);
}

void LinearQuantizerFP16GPU::encode_async(
    const float* d_data,
    int n,
    int dim,
    void* d_codes,
    cudaStream_t stream) const {
    size_t total = (size_t)n * dim;
    if (total == 0) return;
    dim3 blk(256);
    dim3 grid((unsigned int)((total + 255) / 256));
    encode_fp32_to_fp16_kernel<<<grid, blk, 0, stream>>>(
        d_data, n, dim, scale_, zero_point_, (half*)d_codes);
}

void LinearQuantizerFP16GPU::decode(const void* d_codes, int n, int dim, float* d_data) {
    decode_async(d_codes, n, dim, d_data, nullptr);
}

void LinearQuantizerFP16GPU::decode_async(
    const void* d_codes,
    int n,
    int dim,
    float* d_data,
    cudaStream_t stream) const {
    size_t total = (size_t)n * dim;
    if (total == 0) return;
    dim3 blk(256);
    dim3 grid((unsigned int)((total + 255) / 256));
    decode_fp16_to_fp32_kernel<<<grid, blk, 0, stream>>>(
        (const half*)d_codes, n, dim, scale_, zero_point_, d_data);
}

int LinearQuantizerFP16GPU::code_size_bytes(int n, int dim) const {
    return (int)((size_t)n * dim * sizeof(half));
}

} // namespace ivf
