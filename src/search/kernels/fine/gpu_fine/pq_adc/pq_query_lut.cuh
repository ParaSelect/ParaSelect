#ifndef PARASELECT_PQ_QUERY_LUT_CUH
#define PARASELECT_PQ_QUERY_LUT_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace paraselect::search {

enum class PQQueryLutKind { ResidualDot, PlainL2 };

inline constexpr int kPqQueryLutGlobalStride = 256;

#ifdef __CUDACC__
// Query scales are immutable during a scan. The caller supplies a 16-byte
// aligned address (cudaMalloc base + q*M floats, M divisible by four).
// Explicit vector load prevents scalarization into four memory instructions.
__device__ __forceinline__ float4 load_query_lut_scale4(const float* scales) {
    float4 values;
    asm("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
        : "=f"(values.x), "=f"(values.y), "=f"(values.z), "=f"(values.w)
        : "l"(scales));
    return values;
}
#endif

}  // namespace paraselect::search

void pq_build_query_lut_gpu(
    const float* d_queries,
    const float* d_codebook,
    int n_query,
    int pq_m,
    int pq_d_sub_padded,
    float* d_query_codeword_dot_lut,
    cudaStream_t stream,
    paraselect::search::PQQueryLutKind kind = paraselect::search::PQQueryLutKind::ResidualDot);

void pq_build_query_lut_gpu(
    const float* d_queries,
    const __half* d_codebook,
    int n_query,
    int pq_m,
    int pq_d_sub_padded,
    float* d_query_codeword_dot_lut,
    cudaStream_t stream,
    paraselect::search::PQQueryLutKind kind = paraselect::search::PQQueryLutKind::ResidualDot);

void pq_quantize_query_lut_u8_gpu(
    const float* d_query_codeword_dot_lut,
    int n_query,
    int pq_m,
    uint8_t* d_query_codeword_dot_lut_u8,
    float* d_query_lut_scale,
    float* d_query_lut_lo_sum,
    float* d_query_lut_inflation,
    cudaStream_t stream,
    paraselect::search::PQQueryLutKind kind = paraselect::search::PQQueryLutKind::ResidualDot);

#endif  // PARASELECT_PQ_QUERY_LUT_CUH
