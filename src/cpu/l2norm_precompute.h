/**
 * CPU L2
 *
 * : CPU  L2 , AVX2 SIMD
 *  CPU (/), AVX/FPU
 */

#ifndef PARASELECT_CPU_L2NORM_PRECOMPUTE_H
#define PARASELECT_CPU_L2NORM_PRECOMPUTE_H

#include <cstddef>
#include <cstdlib>
#include <cstring>

// OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

// AVX2 (x86_64 )
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define PARASELECT_CPU_HAS_AVX2
#endif

// Linux
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace paraselect {
namespace cpu {

/**
 *  CPU ()
 */
inline int get_physical_cores() {
    unsigned int logical_cores = 0;

    #ifdef _OPENMP
    logical_cores = omp_get_max_threads();
    #else
    //  sysconf
    #ifdef _SC_NPROCESSORS_ONLN
    logical_cores = sysconf(_SC_NPROCESSORS_ONLN);
    #endif
    #endif

    if (logical_cores == 0) {
        logical_cores = 4; //
    }

    // : 2  1 ()
    return logical_cores > 1 ? logical_cores / 2 : 1;
}

/**
 *  CPU
 *
 * @param physical_core_id  ID(0, 1, 2, ...)
 * @param smt_threads_per_core ( 2)
 */
inline void pin_to_physical_core(int physical_core_id, int smt_threads_per_core = 2) {
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    //
    // : 0  0, 1; 1  2, 3
    int logical_core_id = physical_core_id * smt_threads_per_core;
    CPU_SET(logical_core_id, &cpuset);

    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

/**
 *  AVX2  L2
 */
#ifdef PARASELECT_CPU_HAS_AVX2
__attribute__((target("avx2,fma")))
inline float norm2_avx2(const float* vec, int dim) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;

    //  8  float
    for (; i <= dim - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(vec + i);
        sum = _mm256_fmadd_ps(v, v, sum);
    }

    //
    __m128 vlow = _mm256_castps256_ps128(sum);
    __m128 vhigh = _mm256_extractf128_ps(sum, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 vshuf = _mm_movehl_ps(vlow, vlow);
    vlow = _mm_add_ps(vlow, vshuf);
    __m128 vshuf2 = _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(1, 1, 1, 1));
    vlow = _mm_add_ss(vlow, vshuf2);
    float result = _mm_cvtss_f32(vlow);

    //
    for (; i < dim; ++i) {
        result += vec[i] * vec[i];
    }

    return result;
}
#endif

/**
 *  L2
 */
inline float norm2_scalar(const float* vec, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum += vec[i] * vec[i];
    }
    return sum;
}

/**
 *  L2
 *
 * @param vectors  [n_vectors, dim],row-major
 * @param n_vectors
 * @param dim
 * @param out_norms  [n_vectors],
 * @param bind_cores ( true)
 */
inline void precompute_l2_norms(
    const float* vectors,
    int n_vectors,
    int dim,
    float* out_norms,
    bool bind_cores = true
) {
    if (!vectors || !out_norms || n_vectors <= 0 || dim <= 0) {
        return;
    }

    #if defined(__linux__)
    cpu_set_t original_affinity;
    const bool restore_affinity = bind_cores &&
        pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &original_affinity) == 0;
    #else
    const bool restore_affinity = false;
    #endif

    #ifdef _OPENMP
    #pragma omp parallel
    {
        // ,
        if (bind_cores) {
            int thread_id = omp_get_thread_num();
            pin_to_physical_core(thread_id);
        }

        #pragma omp for schedule(dynamic, 1024)
        for (int i = 0; i < n_vectors; ++i) {
            const float* vec = vectors + (size_t)i * dim;

            #ifdef PARASELECT_CPU_HAS_AVX2
            out_norms[i] = norm2_avx2(vec, dim);
            #else
            out_norms[i] = norm2_scalar(vec, dim);
            #endif
        }
    }
    #if defined(__linux__)
    if (restore_affinity) {
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &original_affinity);
    }
    #endif
    #else
    //  OpenMP ,
    for (int i = 0; i < n_vectors; ++i) {
        const float* vec = vectors + (size_t)i * dim;

        #ifdef PARASELECT_CPU_HAS_AVX2
        out_norms[i] = norm2_avx2(vec, dim);
        #else
        out_norms[i] = norm2_scalar(vec, dim);
        #endif
    }
    #endif
}

/**
 *  L2
 *
 * @param vectors  [n_vectors, dim]
 * @param n_vectors
 * @param dim
 * @param bind_cores
 * @return , free()
 */
inline float* allocate_and_precompute_l2_norms(
    const float* vectors,
    int n_vectors,
    int dim,
    bool bind_cores = true
) {
    if (!vectors || n_vectors <= 0 || dim <= 0) {
        return nullptr;
    }

    float* norms = (float*)std::malloc((size_t)n_vectors * sizeof(float));
    if (!norms) {
        return nullptr;
    }

    precompute_l2_norms(vectors, n_vectors, dim, norms, bind_cores);
    return norms;
}

} // namespace cpu
} // namespace paraselect

#endif // PARASELECT_CPU_L2NORM_PRECOMPUTE_H
