/**
 * Residual PQ + rerank fine kernel.
 *
 * (per query, per probed cluster):
 *   1) q_resid = q_fp32 - centroids[cid]                             (128  sub)
 *   2) build LUT[M][K] = ||q_resid_seg_m - codebook[m][k]||²           (M*K  8-D L2)
 *   3) scan cluster:
 *        for v in cluster:
 *            d_approx = Σ_m LUT[m][ pq_codes[v][m] ]
 *            push_stage1_topN(q, d_approx, v)
 *
 * Stage 2 rerank(per query):
 *   1) consume top-N candidates in rerank input order
 *   2) for each candidate idx: d_exact = L2_u8(q_u8, base_u8[idx])
 *      or L2_fp32(q_fp32, base_fp32[idx]) when float rerank is enabled.
 *   3) push_topk(q, d_exact, idx)
 *
 * :per-query ( query ).
 *
 *  - v2 (tile + cluster-major fan-out) :
 *    " cluster  query "  pq_codes .
 *   nlist=32768,TILE_SIZE=16 ,tile-level  k = TILE_SIZE *
 *   nprobe / nlist = 16 * nprobe / 32768  < 1(np=512  0.25),
 *    amortization ,fan-out  fine
 *   +60%. cluster-major  batch-level  +  heap
 *   ,. v1-opt  PQ kernel .
 */

#include "search/ivfrespq/stage/ivfrespq_cpu_rerank.h"
#include "search/ivf_common/ivf_host_rerank.h"
#include "cpu/distance/distance_dispatch.h"
#include "search/kernels/fine/cpu_fine/cpu_fine_common.h"
#include "search/ivfrespq/stage/ivfrespq_cluster_scheduler.h"
#include "search/ivfrespq/stage/ivfrespq_query_scheduler.h"
#include "search/ivfrespq/stage/ivfrespq_pq_storage.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#define IVFT_PQ_HAS_AVX512BW 1
#else
#define IVFT_PQ_HAS_AVX512BW 0
#endif

namespace paraselect {
namespace cpu_fine {

namespace {

/* ============================================================
 * :high-resolution wall clock in ms
 * ============================================================ */
inline double pq_now_ms() {
    using clk = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(
        clk::now().time_since_epoch()).count();
}

struct PQScanBreakdown {
    double adc_lookup_ms = 0.0;
    double heap_push_ms = 0.0;
    long long heap_pushes = 0;
};

/* ============================================================
 * LUT :
 *    q_resid (1 × dim float), codebook (M × K × d_sub float)
 *    LUT (M × K float)  (K  ≤ 256  pq_codes  uint8)
 *
 *  m:LUT[m][k] = Σ_{i=0..d_sub-1} (q_resid[m*d_sub+i] - codebook[m][k][i])²
 *
 * SIMD (AVX-512,d_sub=8):
 *    2  code( 8 float = 32 bytes), __m256  subtract,
 *    horizontal add  2  scalar.
 *    d_sub=4/8 (+).
 * ============================================================ */
static inline void build_lut(
    const float* __restrict__ q_resid,   /* [dim] */
    const float* __restrict__ codebook,  /* [M, K, d_sub] */
    int M, int K, int d_sub,
    float* __restrict__ LUT              /* [M, K] */
) {
#if IVFT_PQ_HAS_AVX512BW
    if (d_sub == 8) {
        for (int m = 0; m < M; ++m) {
            __m256 q_seg = _mm256_loadu_ps(q_resid + m * 8);
            const float* cb_m = codebook + (size_t)m * K * 8;
            float* lut_m = LUT + (size_t)m * K;
            for (int k = 0; k < K; ++k) {
                __m256 cb = _mm256_loadu_ps(cb_m + k * 8);
                __m256 d = _mm256_sub_ps(q_seg, cb);
                __m256 sq = _mm256_mul_ps(d, d);
                __m128 lo = _mm256_castps256_ps128(sq);
                __m128 hi = _mm256_extractf128_ps(sq, 1);
                __m128 sum = _mm_add_ps(lo, hi);
                sum = _mm_hadd_ps(sum, sum);
                sum = _mm_hadd_ps(sum, sum);
                lut_m[k] = _mm_cvtss_f32(sum);
            }
        }
        return;
    }
#endif
    /*  fallback */
    for (int m = 0; m < M; ++m) {
        const float* q_seg = q_resid + m * d_sub;
        const float* cb_m = codebook + (size_t)m * K * d_sub;
        float* lut_m = LUT + (size_t)m * K;
        for (int k = 0; k < K; ++k) {
            const float* cb = cb_m + k * d_sub;
            float s = 0.0f;
            for (int i = 0; i < d_sub; ++i) {
                float d = q_seg[i] - cb[i];
                s += d * d;
            }
            lut_m[k] = s;
        }
    }
}

static inline void build_lut_tile_M16_K256_D8(
    const float* const* __restrict__ q_resids,
    int valid,
    const float* __restrict__ codebook,
    float* __restrict__ lut_tile)
{
#if IVFT_PQ_HAS_AVX512BW
    for (int m = 0; m < 16; ++m) {
        const __m256 q0 = _mm256_loadu_ps(q_resids[0] + m * 8);
        const __m256 q1 = valid > 1 ? _mm256_loadu_ps(q_resids[1] + m * 8) : _mm256_setzero_ps();
        const __m256 q2 = valid > 2 ? _mm256_loadu_ps(q_resids[2] + m * 8) : _mm256_setzero_ps();
        const __m256 q3 = valid > 3 ? _mm256_loadu_ps(q_resids[3] + m * 8) : _mm256_setzero_ps();
        const float* cb_m = codebook + (size_t)m * 256 * 8;
        float* lut0 = lut_tile + ((size_t)0 * 16 + (size_t)m) * 256;
        float* lut1 = lut_tile + ((size_t)1 * 16 + (size_t)m) * 256;
        float* lut2 = lut_tile + ((size_t)2 * 16 + (size_t)m) * 256;
        float* lut3 = lut_tile + ((size_t)3 * 16 + (size_t)m) * 256;
        for (int k = 0; k < 256; ++k) {
            __m256 cb = _mm256_loadu_ps(cb_m + k * 8);
            __m256 d0 = _mm256_sub_ps(q0, cb);
            __m256 sq0 = _mm256_mul_ps(d0, d0);
            __m128 lo0 = _mm256_castps256_ps128(sq0);
            __m128 hi0 = _mm256_extractf128_ps(sq0, 1);
            __m128 sum0 = _mm_add_ps(lo0, hi0);
            sum0 = _mm_hadd_ps(sum0, sum0);
            sum0 = _mm_hadd_ps(sum0, sum0);
            lut0[k] = _mm_cvtss_f32(sum0);

            if (valid > 1) {
                __m256 d1 = _mm256_sub_ps(q1, cb);
                __m256 sq1 = _mm256_mul_ps(d1, d1);
                __m128 lo1 = _mm256_castps256_ps128(sq1);
                __m128 hi1 = _mm256_extractf128_ps(sq1, 1);
                __m128 sum1 = _mm_add_ps(lo1, hi1);
                sum1 = _mm_hadd_ps(sum1, sum1);
                sum1 = _mm_hadd_ps(sum1, sum1);
                lut1[k] = _mm_cvtss_f32(sum1);
            }
            if (valid > 2) {
                __m256 d2 = _mm256_sub_ps(q2, cb);
                __m256 sq2 = _mm256_mul_ps(d2, d2);
                __m128 lo2 = _mm256_castps256_ps128(sq2);
                __m128 hi2 = _mm256_extractf128_ps(sq2, 1);
                __m128 sum2 = _mm_add_ps(lo2, hi2);
                sum2 = _mm_hadd_ps(sum2, sum2);
                sum2 = _mm_hadd_ps(sum2, sum2);
                lut2[k] = _mm_cvtss_f32(sum2);
            }
            if (valid > 3) {
                __m256 d3 = _mm256_sub_ps(q3, cb);
                __m256 sq3 = _mm256_mul_ps(d3, d3);
                __m128 lo3 = _mm256_castps256_ps128(sq3);
                __m128 hi3 = _mm256_extractf128_ps(sq3, 1);
                __m128 sum3 = _mm_add_ps(lo3, hi3);
                sum3 = _mm_hadd_ps(sum3, sum3);
                sum3 = _mm_hadd_ps(sum3, sum3);
                lut3[k] = _mm_cvtss_f32(sum3);
            }
        }
    }
#else
    for (int t = 0; t < valid; ++t) {
        build_lut(q_resids[t], codebook, 16, 256, 8, lut_tile + (size_t)t * 16 * 256);
    }
#endif
}

/* ============================================================
 * PQ :
 *   dist = Σ_{m=0..M-1} LUT[m][ pq[m] ]
 *
 *  M=16,K≤256:16  + 15 ;LUT 16 KB  L1.
 * SIMD : vpgatherdd  16  float( gather  CPU ),
 *           ,.
 * ============================================================ */
static inline float pq_distance(
    const float* __restrict__ LUT,  /* [M, K] */
    const uint8_t* __restrict__ pq, /* [M] */
    int M, int K
) {
    /* ( unroll/schedule; gather ) */
    float s = 0.0f;
    const float* lut_m = LUT;
    for (int m = 0; m < M; ++m) {
        s += lut_m[pq[m]];
        lut_m += K;
    }
    return s;
}

/* ============================================================
 * :M=16, K=256(SIFT )
 * : 16  load +  reduction,
 *  16  4, 5-10%.
 * ============================================================ */
static inline float pq_distance_M16_K256(
    const float* __restrict__ LUT,   /* 16 * 256 floats */
    const uint8_t* __restrict__ pq   /* 16 bytes */
) {
    float s0 = LUT[ 0 * 256 + pq[ 0]] + LUT[ 1 * 256 + pq[ 1]];
    float s1 = LUT[ 2 * 256 + pq[ 2]] + LUT[ 3 * 256 + pq[ 3]];
    float s2 = LUT[ 4 * 256 + pq[ 4]] + LUT[ 5 * 256 + pq[ 5]];
    float s3 = LUT[ 6 * 256 + pq[ 6]] + LUT[ 7 * 256 + pq[ 7]];
    float s4 = LUT[ 8 * 256 + pq[ 8]] + LUT[ 9 * 256 + pq[ 9]];
    float s5 = LUT[10 * 256 + pq[10]] + LUT[11 * 256 + pq[11]];
    float s6 = LUT[12 * 256 + pq[12]] + LUT[13 * 256 + pq[13]];
    float s7 = LUT[14 * 256 + pq[14]] + LUT[15 * 256 + pq[15]];
    float t0 = (s0 + s1) + (s2 + s3);
    float t1 = (s4 + s5) + (s6 + s7);
    return t0 + t1;
}

/* ============================================================
 * :M=8, K=256(BW-,pq_codes  8 B/vec)
 * : BW-bound  bs +  nprobe ,pq_M=8  Stage 1 BW ;
 *  rerank_n  recall.
 *  3(8 → 4 → 2 → 1).
 * ============================================================ */
static inline float pq_distance_M8_K256(
    const float* __restrict__ LUT,   /* 8 * 256 floats */
    const uint8_t* __restrict__ pq   /* 8 bytes */
) {
    float s0 = LUT[0 * 256 + pq[0]] + LUT[1 * 256 + pq[1]];
    float s1 = LUT[2 * 256 + pq[2]] + LUT[3 * 256 + pq[3]];
    float s2 = LUT[4 * 256 + pq[4]] + LUT[5 * 256 + pq[5]];
    float s3 = LUT[6 * 256 + pq[6]] + LUT[7 * 256 + pq[7]];
    return (s0 + s1) + (s2 + s3);
}

static inline void pq_distance_aosoa_M16_K256_T16_scalar(
    const float* __restrict__ LUT,
    const uint8_t* __restrict__ block,
    float* __restrict__ out_dist)
{
    for (int lane = 0; lane < 16; ++lane) {
        float d = 0.0f;
        for (int m = 0; m < 16; ++m) {
            const uint8_t code = block[(size_t)m * 16 + lane];
            d += LUT[(size_t)m * 256 + code];
        }
        out_dist[lane] = d;
    }
}

#if IVFT_PQ_HAS_AVX512BW
static inline __m512 pq_distance_aosoa_M16_K256_T16_gather(
    const float* __restrict__ LUT,
    const uint8_t* __restrict__ block)
{
    __m512 acc = _mm512_setzero_ps();
    for (int m = 0; m < 16; ++m) {
        const __m128i codes8 = _mm_loadu_si128((const __m128i*)(block + (size_t)m * 16));
        __m512i idx = _mm512_cvtepu8_epi32(codes8);
        idx = _mm512_add_epi32(idx, _mm512_set1_epi32(m * 256));
        const __m512 vals = _mm512_i32gather_ps(idx, LUT, 4);
        acc = _mm512_add_ps(acc, vals);
    }
    return acc;
}
#endif

static inline long long scan_pq_cluster_kernel(
    const uint8_t* __restrict__ pq_base,
    long long base_off,
    int count,
    const float* __restrict__ lut,
    int pq_M,
    int pq_K,
    TopKHeap& heap,
    PQScanBreakdown* breakdown = nullptr
) {
    constexpr float FLOAT_INF = std::numeric_limits<float>::max();
    float top_thresh = (heap.size >= heap.k) ? heap.dists[0] : FLOAT_INF;
    const double scan_t0 = breakdown ? pq_now_ms() : 0.0;
    double heap_ms = 0.0;

    if (pq_M == 16 && pq_K == 256) {
        for (int vi = 0; vi < count; ++vi) {
            const uint8_t* pq = pq_base + (size_t)vi * 16;
            float d_approx = pq_distance_M16_K256(lut, pq);
            if (d_approx >= top_thresh) continue;
            const double heap_t0 = breakdown ? pq_now_ms() : 0.0;
            heap.push(d_approx, (int)(base_off + vi));
            if (breakdown) {
                heap_ms += pq_now_ms() - heap_t0;
                ++breakdown->heap_pushes;
            }
            top_thresh = (heap.size >= heap.k) ? heap.dists[0] : FLOAT_INF;
        }
    } else if (pq_M == 8 && pq_K == 256) {
        for (int vi = 0; vi < count; ++vi) {
            const uint8_t* pq = pq_base + (size_t)vi * 8;
            float d_approx = pq_distance_M8_K256(lut, pq);
            if (d_approx >= top_thresh) continue;
            const double heap_t0 = breakdown ? pq_now_ms() : 0.0;
            heap.push(d_approx, (int)(base_off + vi));
            if (breakdown) {
                heap_ms += pq_now_ms() - heap_t0;
                ++breakdown->heap_pushes;
            }
            top_thresh = (heap.size >= heap.k) ? heap.dists[0] : FLOAT_INF;
        }
    } else {
        for (int vi = 0; vi < count; ++vi) {
            const uint8_t* pq = pq_base + (size_t)vi * pq_M;
            float d_approx = pq_distance(lut, pq, pq_M, pq_K);
            if (d_approx >= top_thresh) continue;
            const double heap_t0 = breakdown ? pq_now_ms() : 0.0;
            heap.push(d_approx, (int)(base_off + vi));
            if (breakdown) {
                heap_ms += pq_now_ms() - heap_t0;
                ++breakdown->heap_pushes;
            }
            top_thresh = (heap.size >= heap.k) ? heap.dists[0] : FLOAT_INF;
        }
    }
    if (breakdown) {
        breakdown->heap_push_ms += heap_ms;
        breakdown->adc_lookup_ms += std::max(0.0, pq_now_ms() - scan_t0 - heap_ms);
    }
    return (long long)count * pq_M;
}

static inline long long scan_pq_cluster_aosoa_kernel(
    const uint8_t* __restrict__ pq_cluster_aosoa,
    long long base_off,
    int count,
    const float* __restrict__ lut,
    int pq_M,
    int pq_K,
    int tile,
    TopKHeap& heap,
    PQScanBreakdown* breakdown = nullptr
) {
    constexpr float FLOAT_INF = std::numeric_limits<float>::max();
    float top_thresh = (heap.size >= heap.k) ? heap.dists[0] : FLOAT_INF;
    const int groups = (count + tile - 1) / tile;
    const double scan_t0 = breakdown ? pq_now_ms() : 0.0;
    double heap_ms = 0.0;

    if (pq_M == 16 && pq_K == 256 && tile == 16) {
        for (int g = 0; g < groups; ++g) {
            const int lane_begin = g * 16;
            const int valid = std::min(16, count - lane_begin);
            const uint8_t* block = pq_cluster_aosoa + (size_t)g * 16 * 16;
            alignas(64) float ds[16];
#if IVFT_PQ_HAS_AVX512BW
            const __m512 dvec = pq_distance_aosoa_M16_K256_T16_gather(lut, block);
            __mmask16 candidate_mask = _mm512_cmp_ps_mask(
                dvec, _mm512_set1_ps(top_thresh), _CMP_LT_OQ);
            if (valid < 16) candidate_mask &= (__mmask16)((1u << valid) - 1u);
            if (candidate_mask == 0) continue;
            _mm512_storeu_ps(ds, dvec);
#else
            pq_distance_aosoa_M16_K256_T16_scalar(lut, block, ds);
#endif
            for (int lane = 0; lane < valid; ++lane) {
#if IVFT_PQ_HAS_AVX512BW
                if (((candidate_mask >> lane) & 1u) == 0) continue;
#endif
                const float d = ds[lane];
                if (d >= top_thresh) continue;
                const double heap_t0 = breakdown ? pq_now_ms() : 0.0;
                heap.push(d, (int)(base_off + lane_begin + lane));
                if (breakdown) {
                    heap_ms += pq_now_ms() - heap_t0;
                    ++breakdown->heap_pushes;
                }
                top_thresh = (heap.size >= heap.k) ? heap.dists[0] : FLOAT_INF;
            }
        }
        if (breakdown) {
            breakdown->heap_push_ms += heap_ms;
            breakdown->adc_lookup_ms += std::max(0.0, pq_now_ms() - scan_t0 - heap_ms);
        }
        return (long long)count * pq_M;
    }

    for (int g = 0; g < groups; ++g) {
        const int lane_begin = g * tile;
        const int valid = std::min(tile, count - lane_begin);
        const uint8_t* block = pq_cluster_aosoa + (size_t)g * (size_t)pq_M * (size_t)tile;
        for (int lane = 0; lane < valid; ++lane) {
            float d = 0.0f;
            for (int m = 0; m < pq_M; ++m) {
                const uint8_t code = block[(size_t)m * tile + lane];
                d += lut[(size_t)m * pq_K + code];
            }
            if (d >= top_thresh) continue;
            const double heap_t0 = breakdown ? pq_now_ms() : 0.0;
            heap.push(d, (int)(base_off + lane_begin + lane));
            if (breakdown) {
                heap_ms += pq_now_ms() - heap_t0;
                ++breakdown->heap_pushes;
            }
            top_thresh = (heap.size >= heap.k) ? heap.dists[0] : FLOAT_INF;
        }
    }
    if (breakdown) {
        breakdown->heap_push_ms += heap_ms;
        breakdown->adc_lookup_ms += std::max(0.0, pq_now_ms() - scan_t0 - heap_ms);
    }
    return (long long)count * pq_M;
}

static inline void build_query_residual_kernel(
    const float* __restrict__ q_fp32,
    const float* __restrict__ centroid,
    int dim,
    float* __restrict__ q_resid
) {
#if IVFT_PQ_HAS_AVX512BW
    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 a = _mm512_loadu_ps(q_fp32 + i);
        __m512 b = _mm512_loadu_ps(centroid + i);
        _mm512_storeu_ps(q_resid + i, _mm512_sub_ps(a, b));
    }
    for (; i < dim; ++i) q_resid[i] = q_fp32[i] - centroid[i];
#else
    for (int i = 0; i < dim; ++i) q_resid[i] = q_fp32[i] - centroid[i];
#endif
}

struct PQResidOps {
    using ScanBreakdown = PQScanBreakdown;

    static inline double now_ms() {
        return pq_now_ms();
    }

    static inline void build_query_residual(
        const float* q_fp32,
        const float* centroid,
        int dim,
        float* q_resid)
    {
        build_query_residual_kernel(q_fp32, centroid, dim, q_resid);
    }

    static inline void build_lut(
        const float* q_resid,
        const float* codebook,
        int M,
        int K,
        int d_sub,
        float* lut)
    {
        ::paraselect::cpu_fine::build_lut(q_resid, codebook, M, K, d_sub, lut);
    }

    static inline void build_lut_tile(
        const float* const* q_resids,
        int valid,
        const float* codebook,
        int M,
        int K,
        int d_sub,
        float* lut_tile)
    {
        if (M == 16 && K == 256 && d_sub == 8 && valid > 0 && valid <= 4) {
            build_lut_tile_M16_K256_D8(q_resids, valid, codebook, lut_tile);
            return;
        }
        for (int t = 0; t < valid; ++t) {
            build_lut(q_resids[t], codebook, M, K, d_sub, lut_tile + (size_t)t * M * K);
        }
    }

    static inline long long scan_pq_cluster(
        const uint8_t* pq_base,
        long long base_off,
        int count,
        const float* lut,
        int pq_M,
        int pq_K,
        TopKHeap& heap,
        ScanBreakdown* breakdown = nullptr)
    {
        return scan_pq_cluster_kernel(pq_base, base_off, count, lut, pq_M, pq_K, heap, breakdown);
    }

    static inline long long scan_pq_cluster_aosoa(
        const uint8_t* pq_cluster_aosoa,
        long long base_off,
        int count,
        const float* lut,
        int pq_M,
        int pq_K,
        int tile,
        TopKHeap& heap,
        ScanBreakdown* breakdown = nullptr)
    {
        return scan_pq_cluster_aosoa_kernel(
            pq_cluster_aosoa, base_off, count, lut, pq_M, pq_K, tile, heap, breakdown);
    }
};

}  // namespace

/* ============================================================
 *  kernel
 * ============================================================ */
long long cpu_fine_kernel_pq_resid_cpu_lut(
    const uint8_t* h_base_u8,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const CpuPqCodeStorageView& pq_storage,
    const float* h_codebook,
    const float* h_centroids,
    const uint8_t* h_query_u8,
    const float* h_query_fp32,
    const float* h_precomputed_lut,
    const float* h_base_fp32,
    int use_float_rerank,
    const idx_t* h_coarse_cluster_ids,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    int topk,
    int rerank_n,
    int pq_M,
    int pq_K,
    int num_threads,
    int* h_topk_local_idx,
    float* h_topk_dist,
    double* out_stage1_ms,
    double* out_stage2_ms
) {
    PQResidCpuLUTTiming timing;
    const long long total_fma = cpu_fine_kernel_pq_resid_cpu_lut_timed(
        h_base_u8,
        h_cluster_offsets,
        h_cluster_counts,
        pq_storage,
        h_codebook,
        h_centroids,
        h_query_u8,
        h_query_fp32,
        h_precomputed_lut,
        h_base_fp32,
        use_float_rerank,
        h_coarse_cluster_ids,
        n_query,
        dim,
        n_total_clusters,
        n_probes,
        topk,
        rerank_n,
        pq_M,
        pq_K,
        num_threads,
        h_topk_local_idx,
        h_topk_dist,
        paraselect::search::IVFResPQCpuScheduler::QueryBalanced,
        1,
        &timing);
    if (out_stage1_ms) *out_stage1_ms = timing.workspace_ms + timing.adc_lut_scan_ms;
    if (out_stage2_ms) *out_stage2_ms = timing.rerank_ms;
    return total_fma;
}

long long cpu_fine_kernel_pq_resid_cpu_lut_timed(
    const uint8_t* h_base_u8,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const CpuPqCodeStorageView& pq_storage,
    const float* h_codebook,
    const float* h_centroids,
    const uint8_t* h_query_u8,
    const float* h_query_fp32,
    const float* h_precomputed_lut,
    const float* h_base_fp32,
    int use_float_rerank,
    const idx_t* h_coarse_cluster_ids,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    int topk,
    int rerank_n,
    int pq_M,
    int pq_K,
    int num_threads,
    int* h_topk_local_idx,
    float* h_topk_dist,
    paraselect::search::IVFResPQCpuScheduler scheduler,
    int lut_tile,
    PQResidCpuLUTTiming* timing
) {
    if (num_threads <= 0) num_threads = omp_get_max_threads();
    if (rerank_n < topk) rerank_n = topk;
    constexpr int kDefaultQueryBalancedPQStage1K = 512;
    rerank_n = std::max(rerank_n, kDefaultQueryBalancedPQStage1K);
    if (pq_K > 256) pq_K = 256;  /* pq_codes  u8, K>256 */
    const int d_sub = dim / pq_M;

    const double workspace_t0 = pq_now_ms();
    std::vector<float> s1_dists((size_t)n_query * (size_t)rerank_n);
    std::vector<int>   s1_idxs ((size_t)n_query * (size_t)rerank_n);
    std::vector<int>   s1_sizes((size_t)n_query, 0);
    const double workspace_t1 = pq_now_ms();
    if (timing) timing->workspace_ms = workspace_t1 - workspace_t0;

    const double s1_t0 = pq_now_ms();
    long long adc_lookups = 0;
    long long scanned_codes = 0;
    long long heap_pushes = 0;
    double residual_ms = 0.0;
    double lut_build_ms = 0.0;
    double adc_lookup_ms = 0.0;
    double heap_push_ms = 0.0;
    const bool profile_breakdown = []() {
        const char* env = std::getenv("PARASELECT_PQ_BREAKDOWN");
        return env && env[0] && std::string(env) != "0";
    }();
    if (scheduler == paraselect::search::IVFResPQCpuScheduler::ClusterBased) {
        constexpr int kClusterBasedPQLocalK = 256;
        const cluster_based_scheduler_pq_detail::ClusterBasedPQScheduleStats s1_stats =
            cluster_based_scheduler_pq_detail::run_cluster_based_scheduler_pq<PQResidOps>(
                h_cluster_offsets,
                h_cluster_counts,
                pq_storage.row_major,
                h_codebook,
                h_centroids,
                h_query_fp32,
                h_coarse_cluster_ids,
                n_query,
                dim,
                n_total_clusters,
                n_probes,
                rerank_n,
                std::min(kClusterBasedPQLocalK, rerank_n),
                pq_M,
                pq_K,
                d_sub,
                num_threads,
                s1_dists.data(),
                s1_idxs.data(),
                s1_sizes.data());
        adc_lookups = s1_stats.adc_lookups;
        scanned_codes = s1_stats.scanned_codes;
    } else {
        const query_balanced_scheduler_pq_detail::QueryBalancedPQScheduleStats s1_stats =
            query_balanced_scheduler_pq_detail::run_query_balanced_scheduler_pq<PQResidOps>(
            h_cluster_offsets,
            h_cluster_counts,
            pq_storage,
            h_codebook,
            h_centroids,
            h_query_fp32,
            h_coarse_cluster_ids,
            n_query,
            dim,
            n_total_clusters,
            n_probes,
            rerank_n,
            pq_M,
            pq_K,
            d_sub,
            h_precomputed_lut,
            num_threads,
            s1_dists.data(),
            s1_idxs.data(),
            s1_sizes.data(),
                profile_breakdown,
                lut_tile);
        adc_lookups = s1_stats.adc_lookups;
        scanned_codes = s1_stats.scanned_codes;
        heap_pushes = s1_stats.heap_pushes;
        residual_ms = s1_stats.residual_ms;
        lut_build_ms = s1_stats.lut_build_ms;
        adc_lookup_ms = s1_stats.adc_lookup_ms;
        heap_push_ms = s1_stats.heap_push_ms;
    }
    const double s1_t1 = pq_now_ms();
    if (timing) {
        timing->adc_lut_scan_ms = s1_t1 - s1_t0;
        timing->residual_ms = residual_ms;
        timing->lut_build_ms = lut_build_ms;
        timing->adc_lookup_ms = adc_lookup_ms;
        timing->heap_push_ms = heap_push_ms;
        timing->adc_lookups = adc_lookups;
        timing->scanned_codes = scanned_codes;
        timing->heap_pushes = heap_pushes;
    }

    const double s2_t0 = pq_now_ms();
    const paraselect::search::IVFHostStorageView storage{
        h_base_fp32, h_base_u8,
        static_cast<idx_t>(h_cluster_offsets[n_total_clusters]), dim};
    ivf_rerank_host(storage, h_query_u8, h_query_fp32,
        use_float_rerank != 0 && h_base_fp32 != nullptr,
        s1_idxs.data(), n_query, rerank_n, rerank_n, topk,
        h_topk_local_idx, h_topk_dist, num_threads, false, 4, nullptr,
        s1_sizes.data());

    const double s2_t1 = pq_now_ms();
    if (timing) timing->rerank_ms = s2_t1 - s2_t0;

    return adc_lookups;
}

}  // namespace cpu_fine
}  // namespace paraselect
