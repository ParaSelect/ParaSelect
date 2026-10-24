#ifndef PARASELECT_CPU_DISTANCE_FP32_L2_AVX512_H
#define PARASELECT_CPU_DISTANCE_FP32_L2_AVX512_H

#include <immintrin.h>

namespace paraselect {
namespace cpu_distance {

inline void fp32_prefetch_avx512(const float* p, int dim) {
    const char* cp = reinterpret_cast<const char*>(p);
    for (int b = 0; b < dim * static_cast<int>(sizeof(float)); b += 64) {
        _mm_prefetch(cp + b, _MM_HINT_T0);
    }
}

inline float fp32_l2_avx512(const float* q, const float* v, int dim) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 64 <= dim; i += 64) {
        __m512 q0 = _mm512_loadu_ps(q + i + 0);
        __m512 v0 = _mm512_loadu_ps(v + i + 0);
        __m512 d0 = _mm512_sub_ps(q0, v0);
        acc0 = _mm512_fmadd_ps(d0, d0, acc0);
        __m512 q1 = _mm512_loadu_ps(q + i + 16);
        __m512 v1 = _mm512_loadu_ps(v + i + 16);
        __m512 d1 = _mm512_sub_ps(q1, v1);
        acc1 = _mm512_fmadd_ps(d1, d1, acc1);
        __m512 q2 = _mm512_loadu_ps(q + i + 32);
        __m512 v2 = _mm512_loadu_ps(v + i + 32);
        __m512 d2 = _mm512_sub_ps(q2, v2);
        acc2 = _mm512_fmadd_ps(d2, d2, acc2);
        __m512 q3 = _mm512_loadu_ps(q + i + 48);
        __m512 v3 = _mm512_loadu_ps(v + i + 48);
        __m512 d3 = _mm512_sub_ps(q3, v3);
        acc3 = _mm512_fmadd_ps(d3, d3, acc3);
    }
    for (; i + 16 <= dim; i += 16) {
        __m512 qq = _mm512_loadu_ps(q + i);
        __m512 vv = _mm512_loadu_ps(v + i);
        __m512 dd = _mm512_sub_ps(qq, vv);
        acc0 = _mm512_fmadd_ps(dd, dd, acc0);
    }
    __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    float out = _mm512_reduce_add_ps(acc);
    for (; i < dim; ++i) {
        float d = q[i] - v[i];
        out += d * d;
    }
    return out;
}

inline void fp32_l2_q4_avx512(
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
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 vv = _mm512_loadu_ps(v + i);
        __m512 qq0 = _mm512_loadu_ps(q0 + i);
        __m512 qq1 = _mm512_loadu_ps(q1 + i);
        __m512 qq2 = _mm512_loadu_ps(q2 + i);
        __m512 qq3 = _mm512_loadu_ps(q3 + i);
        __m512 d0 = _mm512_sub_ps(qq0, vv);
        __m512 d1 = _mm512_sub_ps(qq1, vv);
        __m512 d2 = _mm512_sub_ps(qq2, vv);
        __m512 d3 = _mm512_sub_ps(qq3, vv);
        acc0 = _mm512_fmadd_ps(d0, d0, acc0);
        acc1 = _mm512_fmadd_ps(d1, d1, acc1);
        acc2 = _mm512_fmadd_ps(d2, d2, acc2);
        acc3 = _mm512_fmadd_ps(d3, d3, acc3);
    }
    out0 = _mm512_reduce_add_ps(acc0);
    out1 = _mm512_reduce_add_ps(acc1);
    out2 = _mm512_reduce_add_ps(acc2);
    out3 = _mm512_reduce_add_ps(acc3);
    for (; i < dim; ++i) {
        float a0 = q0[i] - v[i]; out0 += a0 * a0;
        float a1 = q1[i] - v[i]; out1 += a1 * a1;
        float a2 = q2[i] - v[i]; out2 += a2 * a2;
        float a3 = q3[i] - v[i]; out3 += a3 * a3;
    }
}

}  // namespace cpu_distance
}  // namespace paraselect

#endif  // PARASELECT_CPU_DISTANCE_FP32_L2_AVX512_H
