#ifndef PARASELECT_CPU_DISTANCE_FP32_L2_AVX2_H
#define PARASELECT_CPU_DISTANCE_FP32_L2_AVX2_H

#include <immintrin.h>

namespace paraselect {
namespace cpu_distance {

inline float hsum_ps_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

inline void fp32_prefetch_avx2(const float* p, int dim) {
    const char* cp = reinterpret_cast<const char*>(p);
    for (int b = 0; b < dim * static_cast<int>(sizeof(float)); b += 64) {
        _mm_prefetch(cp + b, _MM_HINT_T0);
    }
}

inline float fp32_l2_avx2(const float* q, const float* v, int dim) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= dim; i += 32) {
        __m256 q0 = _mm256_loadu_ps(q + i + 0);
        __m256 v0 = _mm256_loadu_ps(v + i + 0);
        __m256 d0 = _mm256_sub_ps(q0, v0);
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        __m256 q1 = _mm256_loadu_ps(q + i + 8);
        __m256 v1 = _mm256_loadu_ps(v + i + 8);
        __m256 d1 = _mm256_sub_ps(q1, v1);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        __m256 q2 = _mm256_loadu_ps(q + i + 16);
        __m256 v2 = _mm256_loadu_ps(v + i + 16);
        __m256 d2 = _mm256_sub_ps(q2, v2);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);
        __m256 q3 = _mm256_loadu_ps(q + i + 24);
        __m256 v3 = _mm256_loadu_ps(v + i + 24);
        __m256 d3 = _mm256_sub_ps(q3, v3);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
    }
    for (; i + 8 <= dim; i += 8) {
        __m256 qq = _mm256_loadu_ps(q + i);
        __m256 vv = _mm256_loadu_ps(v + i);
        __m256 dd = _mm256_sub_ps(qq, vv);
        acc0 = _mm256_fmadd_ps(dd, dd, acc0);
    }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    float out = hsum_ps_avx2(acc);
    for (; i < dim; ++i) {
        float d = q[i] - v[i];
        out += d * d;
    }
    return out;
}

inline void fp32_l2_q4_avx2(
    const float* q0,
    const float* q1,
    const float* q2,
    const float* q3,
    const float* v,
    int dim,
    float& out0,
    float& out1,
    float& out2,
    float& out3) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 vv = _mm256_loadu_ps(v + i);
        __m256 qq0 = _mm256_loadu_ps(q0 + i);
        __m256 qq1 = _mm256_loadu_ps(q1 + i);
        __m256 qq2 = _mm256_loadu_ps(q2 + i);
        __m256 qq3 = _mm256_loadu_ps(q3 + i);
        __m256 d0 = _mm256_sub_ps(qq0, vv);
        __m256 d1 = _mm256_sub_ps(qq1, vv);
        __m256 d2 = _mm256_sub_ps(qq2, vv);
        __m256 d3 = _mm256_sub_ps(qq3, vv);
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
    }
    out0 = hsum_ps_avx2(acc0);
    out1 = hsum_ps_avx2(acc1);
    out2 = hsum_ps_avx2(acc2);
    out3 = hsum_ps_avx2(acc3);
    for (; i < dim; ++i) {
        float a0 = q0[i] - v[i]; out0 += a0 * a0;
        float a1 = q1[i] - v[i]; out1 += a1 * a1;
        float a2 = q2[i] - v[i]; out2 += a2 * a2;
        float a3 = q3[i] - v[i]; out3 += a3 * a3;
    }
}

}  // namespace cpu_distance
}  // namespace paraselect

#endif  // PARASELECT_CPU_DISTANCE_FP32_L2_AVX2_H
