#ifndef PARASELECT_CPU_DISTANCE_U8_L2_AVX512_H
#define PARASELECT_CPU_DISTANCE_U8_L2_AVX512_H

#include <cstdint>
#include <immintrin.h>

namespace paraselect {
namespace cpu_distance {

inline void u8_prefetch_avx512(const uint8_t* p, int dim) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(p, 0, 1);
    // A SIFT row is 128 B = two cache lines; prefetch both so the
    // distance loop never stalls on the second line.
    if (dim > 64) __builtin_prefetch(p + 64, 0, 1);
#else
    (void)p;
    (void)dim;
#endif
}

inline int32_t u8_l2_avx512(const uint8_t* q, const uint8_t* v, int dim) {
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    int i = 0;
    for (; i + 64 <= dim; i += 64) {
        __m512i da = _mm512_sub_epi16(
            _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + i + 0))),
            _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(v + i + 0))));
        __m512i db = _mm512_sub_epi16(
            _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + i + 32))),
            _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(v + i + 32))));
        acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(da, da));
        acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(db, db));
    }
    for (; i + 32 <= dim; i += 32) {
        __m512i dd = _mm512_sub_epi16(
            _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + i))),
            _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(v + i))));
        acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(dd, dd));
    }
    int32_t out = _mm512_reduce_add_epi32(_mm512_add_epi32(acc0, acc1));
    for (; i < dim; ++i) {
        int d = static_cast<int>(q[i]) - static_cast<int>(v[i]);
        out += d * d;
    }
    return out;
}

inline void u8_l2_q4_avx512(
    const uint8_t* q0,
    const uint8_t* q1,
    const uint8_t* q2,
    const uint8_t* q3,
    const uint8_t* v,
    int dim,
    int32_t& out0,
    int32_t& out1,
    int32_t& out2,
    int32_t& out3) {
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    __m512i acc2 = _mm512_setzero_si512();
    __m512i acc3 = _mm512_setzero_si512();
    int i = 0;
    for (; i + 32 <= dim; i += 32) {
        __m256i b8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(v + i));
        __m512i bi = _mm512_cvtepu8_epi16(b8);
        __m512i q0i = _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q0 + i)));
        __m512i q1i = _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q1 + i)));
        __m512i q2i = _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q2 + i)));
        __m512i q3i = _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q3 + i)));
        __m512i d0 = _mm512_sub_epi16(q0i, bi);
        __m512i d1 = _mm512_sub_epi16(q1i, bi);
        __m512i d2 = _mm512_sub_epi16(q2i, bi);
        __m512i d3 = _mm512_sub_epi16(q3i, bi);
        acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(d0, d0));
        acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(d1, d1));
        acc2 = _mm512_add_epi32(acc2, _mm512_madd_epi16(d2, d2));
        acc3 = _mm512_add_epi32(acc3, _mm512_madd_epi16(d3, d3));
    }
    out0 = _mm512_reduce_add_epi32(acc0);
    out1 = _mm512_reduce_add_epi32(acc1);
    out2 = _mm512_reduce_add_epi32(acc2);
    out3 = _mm512_reduce_add_epi32(acc3);
    for (; i < dim; ++i) {
        int s0 = static_cast<int>(q0[i]) - static_cast<int>(v[i]); out0 += s0 * s0;
        int s1 = static_cast<int>(q1[i]) - static_cast<int>(v[i]); out1 += s1 * s1;
        int s2 = static_cast<int>(q2[i]) - static_cast<int>(v[i]); out2 += s2 * s2;
        int s3 = static_cast<int>(q3[i]) - static_cast<int>(v[i]); out3 += s3 * s3;
    }
}

}  // namespace cpu_distance
}  // namespace paraselect

#endif  // PARASELECT_CPU_DISTANCE_U8_L2_AVX512_H
