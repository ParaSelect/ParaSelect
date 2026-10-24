#pragma once
#include "search/kernels/fine/gpu_fine/pq_adc/pq_adc_gpu.cuh"

namespace paraselect::search {
// Shared accumulation order: seed thresholds and exhaustive admission must
// compare the same ADC score, including row-correction decoding.
template<bool CorrectionU16, bool PlainL2 = false>
__device__ __forceinline__ float pq_query_lut_fp32_score(
    const PQQueryLutView& view, const uint8_t* codes, int query, int group,
    int centroid_stride, int row, int m, int lut_stride = 256) {
    float dot = 0.0f;
    for (int j = 0; j < m; ++j)
        dot += view.query_codeword_dot_lut[(size_t(query) * m + j) * lut_stride +
                                         codes[size_t(row) * m + j]];
    if constexpr (PlainL2) return dot;
    const float correction = CorrectionU16
        ? fmaf(float(view.query_lut_bias.u16[row]), view.query_lut_bias.scale,
               view.query_lut_bias.offset)
        : view.query_lut_bias.fp32[row];
    return view.query_centroid_distances[size_t(query) * centroid_stride + group]
           + correction - 2.0f * dot;
}
template<int M, bool U8, bool QUERY_LUT_BIAS_U16, bool FourAcc, bool PlainL2 = false>
__device__ __forceinline__ float pq_query_lut_seed_score(
    PQQueryLutView v, const uint8_t* codes, int q, int probe, int nprobe, int row) {
    if constexpr (!U8 && !FourAcc)
        return pq_query_lut_fp32_score<QUERY_LUT_BIAS_U16, PlainL2>(v, codes, q, probe, nprobe, row, M);
    const uint8_t* code = codes + size_t(row) * M;
    float a = U8 ? v.query_lut_lo_sum[q] : 0.f;
    float b = 0.f, c = 0.f, d = 0.f;
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const size_t slot = (size_t(q) * M + m) * kPqQueryLutGlobalStride + code[m];
        float term;
        if constexpr (U8) term = v.query_lut_scale[q * M + m] * float(v.query_codeword_dot_lut_u8[slot]);
        else term = v.query_codeword_dot_lut[slot];
        if constexpr (FourAcc) {
            if ((m & 3) == 0) a += term;
            else if ((m & 3) == 1) b += term;
            else if ((m & 3) == 2) c += term;
            else d += term;
        } else a += term;
    }
    const float dot = FourAcc ? (a + b) + (c + d) : a;
    if constexpr (PlainL2) return dot;
    const float query_lut_bias = QUERY_LUT_BIAS_U16 ? fmaf(float(v.query_lut_bias.u16[row]),
        v.query_lut_bias.scale, v.query_lut_bias.offset) : v.query_lut_bias.fp32[row];
    return v.query_centroid_distances[size_t(q) * nprobe + probe] + query_lut_bias - 2.f * dot;
}

} // namespace paraselect::search
