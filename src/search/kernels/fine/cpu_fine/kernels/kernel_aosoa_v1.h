#pragma once

#include <cstddef>

#if defined(__AVX512F__) && defined(__FMA__)
#include <immintrin.h>
#define PARASELECT_AOSOA_V1_HAS_AVX512 1
#else
#define PARASELECT_AOSOA_V1_HAS_AVX512 0
#endif

#if defined(__GNUC__) || defined(__clang__)
#define PARASELECT_AOSOA_V1_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define PARASELECT_AOSOA_V1_ALWAYS_INLINE inline
#endif

namespace paraselect {
namespace cpu_fine {

struct Fp32AosoaTile16KernelV1 {
    static constexpr int kTile = 16;
    static constexpr int kPrefetchDimAhead = 8;

    static PARASELECT_AOSOA_V1_ALWAYS_INLINE void prefetch_tile_dim(
        const float* tile_base,
        int dim_idx)
    {
        if (dim_idx >= 0) {
            __builtin_prefetch(tile_base + (size_t)dim_idx * kTile, 0, 1);
        }
    }

    static PARASELECT_AOSOA_V1_ALWAYS_INLINE void distance_q4_tile(
        const float* q0,
        const float* q1,
        const float* q2,
        const float* q3,
        const float* tile_base,
        int dim,
        float* d0,
        float* d1,
        float* d2,
        float* d3)
    {
#if PARASELECT_AOSOA_V1_HAS_AVX512
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        for (int d = 0; d < dim; ++d) {
            const int pf_d = d + kPrefetchDimAhead;
            if (pf_d < dim) prefetch_tile_dim(tile_base, pf_d);

            const __m512 bv = _mm512_loadu_ps(tile_base + (size_t)d * kTile);
            const __m512 qv0 = _mm512_set1_ps(q0[d]);
            const __m512 qv1 = _mm512_set1_ps(q1[d]);
            const __m512 qv2 = _mm512_set1_ps(q2[d]);
            const __m512 qv3 = _mm512_set1_ps(q3[d]);
            const __m512 diff0 = _mm512_sub_ps(qv0, bv);
            const __m512 diff1 = _mm512_sub_ps(qv1, bv);
            const __m512 diff2 = _mm512_sub_ps(qv2, bv);
            const __m512 diff3 = _mm512_sub_ps(qv3, bv);
            acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);
            acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);
            acc2 = _mm512_fmadd_ps(diff2, diff2, acc2);
            acc3 = _mm512_fmadd_ps(diff3, diff3, acc3);
        }
        _mm512_storeu_ps(d0, acc0);
        _mm512_storeu_ps(d1, acc1);
        _mm512_storeu_ps(d2, acc2);
        _mm512_storeu_ps(d3, acc3);
#else
        for (int lane = 0; lane < kTile; ++lane) {
            d0[lane] = 0.0f;
            d1[lane] = 0.0f;
            d2[lane] = 0.0f;
            d3[lane] = 0.0f;
        }
        for (int d = 0; d < dim; ++d) {
            const int pf_d = d + kPrefetchDimAhead;
            if (pf_d < dim) prefetch_tile_dim(tile_base, pf_d);

            const float* base_lane = tile_base + (size_t)d * kTile;
            const float qd0 = q0[d];
            const float qd1 = q1[d];
            const float qd2 = q2[d];
            const float qd3 = q3[d];
            for (int lane = 0; lane < kTile; ++lane) {
                const float bv = base_lane[lane];
                const float diff0 = qd0 - bv;
                const float diff1 = qd1 - bv;
                const float diff2 = qd2 - bv;
                const float diff3 = qd3 - bv;
                d0[lane] += diff0 * diff0;
                d1[lane] += diff1 * diff1;
                d2[lane] += diff2 * diff2;
                d3[lane] += diff3 * diff3;
            }
        }
#endif
    }

    static PARASELECT_AOSOA_V1_ALWAYS_INLINE void distance_q1_tile(
        const float* q,
        const float* tile_base,
        int dim,
        float* d0)
    {
#if PARASELECT_AOSOA_V1_HAS_AVX512
        __m512 acc = _mm512_setzero_ps();
        for (int d = 0; d < dim; ++d) {
            const int pf_d = d + kPrefetchDimAhead;
            if (pf_d < dim) prefetch_tile_dim(tile_base, pf_d);

            const __m512 qv = _mm512_set1_ps(q[d]);
            const __m512 bv = _mm512_loadu_ps(tile_base + (size_t)d * kTile);
            const __m512 diff = _mm512_sub_ps(qv, bv);
            acc = _mm512_fmadd_ps(diff, diff, acc);
        }
        _mm512_storeu_ps(d0, acc);
#else
        for (int lane = 0; lane < kTile; ++lane) d0[lane] = 0.0f;
        for (int d = 0; d < dim; ++d) {
            const int pf_d = d + kPrefetchDimAhead;
            if (pf_d < dim) prefetch_tile_dim(tile_base, pf_d);

            const float qd = q[d];
            const float* base_lane = tile_base + (size_t)d * kTile;
            for (int lane = 0; lane < kTile; ++lane) {
                const float diff = qd - base_lane[lane];
                d0[lane] += diff * diff;
            }
        }
#endif
    }
};

}  // namespace cpu_fine
}  // namespace paraselect

#undef PARASELECT_AOSOA_V1_ALWAYS_INLINE
