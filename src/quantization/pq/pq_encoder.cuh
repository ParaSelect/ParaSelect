#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace paraselect::quantization::pq {
// Fixed-capacity host->GPU->host encoder. The caller supplies residuals for
// ResPQ or original vectors for plain PQ. Codebook and storage persist across
// chunks; encode never allocates or changes capacity.
class PQEncoderGPU {
public:
    PQEncoderGPU(const float* codebook, int dim, int m, int k, size_t max_rows);
    ~PQEncoderGPU();
    PQEncoderGPU(const PQEncoderGPU&)=delete;
    PQEncoderGPU& operator=(const PQEncoderGPU&)=delete;
    void encode(const float* data, int rows, uint8_t* codes);
private:
    void release() noexcept;
    int dim_, m_, k_, subdim_;
    size_t max_rows_;
    bool warp_;
    float* data_=nullptr;
    float* codebook_=nullptr;
    uint8_t* codes_=nullptr;
    cudaStream_t stream_=nullptr;
};
}
