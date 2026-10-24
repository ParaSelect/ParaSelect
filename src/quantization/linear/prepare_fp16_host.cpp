/**
 * Host  fp32→fp16  device, fp16 ( host ).
 */
#include "quantization/linear/prepare_fp16_host.h"
#include "quantization/linear/linear_fp16.cuh"
#include <vector>
#include <cstring>

namespace ivf {

void convert_fp32_to_fp16_upload(
    const float* h_fp32,
    int n,
    int dim,
    __half** d_fp16_out
) {
    if (!h_fp32 || !d_fp16_out || n <= 0 || dim <= 0) {
        if (d_fp16_out) *d_fp16_out = nullptr;
        return;
    }
    LinearQuantizerFP16CPU quant(1.f, 0.f);
    const size_t code_bytes = quant.code_size_bytes(n, dim);
    std::vector<uint8_t> h_fp16(code_bytes);
    quant.encode(h_fp32, n, dim, h_fp16.data());
    __half* d_fp16 = nullptr;
    if (cudaMalloc(&d_fp16, code_bytes) != cudaSuccess) {
        *d_fp16_out = nullptr;
        return;
    }
    cudaMemcpy(d_fp16, h_fp16.data(), code_bytes, cudaMemcpyHostToDevice);
    *d_fp16_out = reinterpret_cast<__half*>(d_fp16);
}

void convert_fp32_to_fp16_inplace(
    const float* h_fp32,
    int n,
    int dim,
    __half* d_fp16_buf
) {
    if (!h_fp32 || !d_fp16_buf || n <= 0 || dim <= 0) return;
    LinearQuantizerFP16CPU quant(1.f, 0.f);
    const size_t code_bytes = quant.code_size_bytes(n, dim);
    std::vector<uint8_t> h_fp16(code_bytes);
    quant.encode(h_fp32, n, dim, h_fp16.data());
    cudaMemcpy(d_fp16_buf, h_fp16.data(), code_bytes, cudaMemcpyHostToDevice);
}

} // namespace ivf
