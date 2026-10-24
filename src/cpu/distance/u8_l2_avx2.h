#ifndef PARASELECT_CPU_DISTANCE_U8_L2_AVX2_H
#define PARASELECT_CPU_DISTANCE_U8_L2_AVX2_H

#include <cstdint>
#include <immintrin.h>

namespace paraselect {
namespace cpu_distance {

inline int32_t hsum_epi32_avx2(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

inline void u8_prefetch_avx2(const uint8_t* p, int dim) {
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

inline int32_t u8_l2_avx2(const uint8_t* q, const uint8_t* v, int dim) {
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    int i = 0;
    for (; i + 32 <= dim; i += 32) {
        __m256i d0 = _mm256_sub_epi16(
            _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q + i + 0))),
            _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(v + i + 0))));
        __m256i d1 = _mm256_sub_epi16(
            _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q + i + 16))),
            _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(v + i + 16))));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(d0, d0));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(d1, d1));
    }
    for (; i + 16 <= dim; i += 16) {
        __m256i d = _mm256_sub_epi16(
            _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q + i))),
            _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(v + i))));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(d, d));
    }
    int32_t out = hsum_epi32_avx2(_mm256_add_epi32(acc0, acc1));
    for (; i < dim; ++i) {
        int d = static_cast<int>(q[i]) - static_cast<int>(v[i]);
        out += d * d;
    }
    return out;
}

inline void u8_l2_q4_avx2(
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
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m128i b8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(v + i));
        __m256i bi = _mm256_cvtepu8_epi16(b8);
        __m256i q0i = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q0 + i)));
        __m256i q1i = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q1 + i)));
        __m256i q2i = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q2 + i)));
        __m256i q3i = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q3 + i)));
        __m256i d0 = _mm256_sub_epi16(q0i, bi);
        __m256i d1 = _mm256_sub_epi16(q1i, bi);
        __m256i d2 = _mm256_sub_epi16(q2i, bi);
        __m256i d3 = _mm256_sub_epi16(q3i, bi);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(d0, d0));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(d1, d1));
        acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(d2, d2));
        acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(d3, d3));
    }
    out0 = hsum_epi32_avx2(acc0);
    out1 = hsum_epi32_avx2(acc1);
    out2 = hsum_epi32_avx2(acc2);
    out3 = hsum_epi32_avx2(acc3);
    for (; i < dim; ++i) {
        int s0 = static_cast<int>(q0[i]) - static_cast<int>(v[i]); out0 += s0 * s0;
        int s1 = static_cast<int>(q1[i]) - static_cast<int>(v[i]); out1 += s1 * s1;
        int s2 = static_cast<int>(q2[i]) - static_cast<int>(v[i]); out2 += s2 * s2;
        int s3 = static_cast<int>(q3[i]) - static_cast<int>(v[i]); out3 += s3 * s3;
    }
}

}  // namespace cpu_distance
}  // namespace paraselect

#endif  // PARASELECT_CPU_DISTANCE_U8_L2_AVX2_H
