#ifndef TEST_UTILS_H
#define TEST_UTILS_H
#include <string>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "src/utils/pch.h"
#include "output_macros.cuh"
#include "params_macros.cuh"
#include "metrics_collector.cuh"
#include "test_context.cuh"
#include "vector_nd.cuh"
#include "compare_utils.cuh"
#include "cpu/utils/cpu_distance.h"

/**
 * row-major [n_vec][n_dim]: L2 .
 *  test_gemv_interleaved  static compute_norms , cpu_distance::compute_l2_norm.
 */
inline void compute_norms(const float* data, float* norms, int n_vec, int n_dim) {
    for (int i = 0; i < n_vec; ++i) {
        norms[i] = compute_l2_norm(
            data + static_cast<size_t>(i) * static_cast<size_t>(n_dim),
            n_dim);
    }
}

/**
 * @brief ,().
 * @param TESTNAME ,.
 * @param ... .
 */
#define MEASURE_MS(TESTNAME, ...) \
    do { \
        auto __measure_start = std::chrono::high_resolution_clock::now(); \
        __VA_ARGS__; \
        auto __measure_end = std::chrono::high_resolution_clock::now(); \
        auto __measure_dur = std::chrono::duration_cast<std::chrono::microseconds>(__measure_end - __measure_start); \
        COUT_VAL((TESTNAME), ((double)__measure_dur.count()/1000.0) , "ms"); \
    } while(0)

/**
 * @brief ,().
 * @param TESTNAME ,.
 * @param TIMESPEND_VAR ( double, long long ).
 * @param ... .
 */
#define MEASURE_MS_AND_SAVE(TESTNAME, TIMESPEND_VAR, ...) \
    do { \
        auto __measure_start = std::chrono::high_resolution_clock::now(); \
        __VA_ARGS__; \
        auto __measure_end = std::chrono::high_resolution_clock::now(); \
        auto __measure_dur = std::chrono::duration_cast<std::chrono::microseconds>(__measure_end - __measure_start); \
        (TIMESPEND_VAR) = __measure_dur.count() / 1000.0; \
    } while(0);


inline bool check_pass(std::string test_names, bool pass){
    COUT_ENDL( (test_names), (pass) ? "✅ PASS" : "❌ FAIL");
    return pass;
}

bool compare_float(float a, float b, float epsilon = 1e-5f);
bool compare_float_relative(float a, float b, float epsilon = 1e-5f);

//
void* generate_cluster_query_data(int* query_cluster_group, int n_query, int k, int batch_size);
void get_cluster_vector(int cluster_id, float** cluster_vector, int* vector_num);

#endif
