/**
 * gemv interleaved ( + L2 )
 *  mode  launch_gemm_interleaved_cos  launch_gemm_interleaved_l2
 */

#include "src/utils/pch.h"
#include "src/search/kernels/fine/gpu_fine/gemv_interleaved.cuh"
#include "src/layout/ivf_interleaved_layout.cuh"
#include "common/test_utils.cuh"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>

#define EPSILON 1e-3f

/// CPU : out[q][v] = sum_d query[q][d] * vecs[v][d]
static void cpu_inner_product(
    const float* query,
    const float* vecs,
    float* out,
    int n_query,
    int n_vec,
    int n_dim) {
    for (int q = 0; q < n_query; ++q) {
        for (int v = 0; v < n_vec; ++v) {
            float dot = 0.0f;
            for (int d = 0; d < n_dim; ++d) {
                dot += query[(size_t)q * n_dim + d] * vecs[(size_t)v * n_dim + d];
            }
            out[(size_t)q * n_vec + v] = dot;
        }
    }
}

/// CPU : out[q][v] = 1 - cos_sim
static void cpu_cosine_distance(
    const float* query,
    const float* vecs,
    const float* query_norm,
    const float* vec_norm,
    float* out,
    int n_query,
    int n_vec,
    int n_dim) {
    for (int q = 0; q < n_query; ++q) {
        for (int v = 0; v < n_vec; ++v) {
            float dot = 0.0f;
            for (int d = 0; d < n_dim; ++d) {
                dot += query[(size_t)q * n_dim + d] * vecs[(size_t)v * n_dim + d];
            }
            float qn = query_norm[q];
            float vn = vec_norm[v];
            if (qn >= 1e-6f && vn >= 1e-6f) {
                out[(size_t)q * n_vec + v] = 1.0f - dot / (qn * vn);
            } else {
                out[(size_t)q * n_vec + v] = 1.0f;
            }
        }
    }
}

/// CPU :L2  out[q][v] = ||q||^2 + ||v||^2 - 2*q·v
static void cpu_l2_squared(
    const float* query,
    const float* vecs,
    const float* query_norm,
    const float* vec_norm,
    float* out,
    int n_query,
    int n_vec,
    int n_dim) {
    for (int q = 0; q < n_query; ++q) {
        for (int v = 0; v < n_vec; ++v) {
            float dot = 0.0f;
            for (int d = 0; d < n_dim; ++d) {
                dot += query[(size_t)q * n_dim + d] * vecs[(size_t)v * n_dim + d];
            }
            float qn = query_norm[q];
            float vn = vec_norm[v];
            out[(size_t)q * n_vec + v] = qn * qn + vn * vn - 2.0f * dot;
        }
    }
}

/// ,use_cosine: true=(launch_gemm_interleaved_cos),false=L2 (launch_gemm_interleaved_l2)
/// n_vec , 32  padding  0,kernel  vec_idx >= n_vec  mask.
static std::vector<double> run_case(int n_query, int n_vec, int n_dim, bool use_cosine) {
    float* h_query = (float*)std::malloc((size_t)n_query * n_dim * sizeof(float));
    float* h_vecs = (float*)std::malloc((size_t)n_vec * n_dim * sizeof(float));
    float* h_interleaved = (float*)std::malloc(paraselect::interleaved_storage_bytes(n_vec, n_dim));
    float* h_query_norm = (float*)std::malloc((size_t)n_query * sizeof(float));
    float* h_vec_norm = (float*)std::malloc((size_t)n_vec * sizeof(float));
    float* h_cpu_out = (float*)std::malloc((size_t)n_query * n_vec * sizeof(float));
    float* h_gpu_out = (float*)std::malloc((size_t)n_query * n_vec * sizeof(float));

    std::srand(12345);
    for (size_t i = 0; i < (size_t)n_query * n_dim; ++i)
        h_query[i] = (float)std::rand() / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < (size_t)n_vec * n_dim; ++i)
        h_vecs[i] = (float)std::rand() / RAND_MAX * 2.0f - 1.0f;

    /* interleaved  32 , 0(pack_rowmajor_to_interleaved  memset) */
    paraselect::pack_rowmajor_to_interleaved(h_vecs, h_interleaved, n_vec, n_dim);
    compute_norms(h_query, h_query_norm, n_query, n_dim);
    compute_norms(h_vecs, h_vec_norm, n_vec, n_dim);

    float *d_query = nullptr, *d_interleaved = nullptr, *d_out = nullptr;
    float *d_query_norm = nullptr, *d_vec_norm = nullptr;
    cudaMalloc(&d_query, (size_t)n_query * n_dim * sizeof(float));
    cudaMalloc(&d_interleaved, paraselect::interleaved_storage_bytes(n_vec, n_dim));
    cudaMalloc(&d_out, (size_t)n_query * n_vec * sizeof(float));
    cudaMalloc(&d_query_norm, (size_t)n_query * sizeof(float));
    cudaMalloc(&d_vec_norm, (size_t)n_vec * sizeof(float));

    cudaMemcpy(d_query, h_query, (size_t)n_query * n_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_interleaved, h_interleaved, paraselect::interleaved_storage_bytes(n_vec, n_dim), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_norm, h_query_norm, (size_t)n_query * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec_norm, h_vec_norm, (size_t)n_vec * sizeof(float), cudaMemcpyHostToDevice);

    double cpu_ms = 0.0;
    if (use_cosine) {
        MEASURE_MS_AND_SAVE("cpu", cpu_ms,
            cpu_cosine_distance(h_query, h_vecs, h_query_norm, h_vec_norm, h_cpu_out, n_query, n_vec, n_dim);
        );
    } else {
        MEASURE_MS_AND_SAVE("cpu", cpu_ms,
            cpu_l2_squared(h_query, h_vecs, h_query_norm, h_vec_norm, h_cpu_out, n_query, n_vec, n_dim);
        );
    }

    double gpu_ms = 0.0;
    if (use_cosine) {
        MEASURE_MS_AND_SAVE("gpu", gpu_ms,
            launch_gemm_interleaved_cos(d_query, d_interleaved,
                d_query_norm, d_vec_norm,
                n_query, n_vec, n_dim, d_out, 0);
            cudaDeviceSynchronize();
        );
    } else {
        MEASURE_MS_AND_SAVE("gpu", gpu_ms,
            launch_gemm_interleaved_l2(d_query, d_interleaved,
                d_query_norm, d_vec_norm,
                n_query, n_vec, n_dim, d_out, 0);
            cudaDeviceSynchronize();
        );
    }

    cudaMemcpy(h_gpu_out, d_out, (size_t)n_query * n_vec * sizeof(float), cudaMemcpyDeviceToHost);

    bool pass = compare_1D(h_gpu_out, h_cpu_out, n_query * n_vec, EPSILON);
    double pass_rate = pass ? 1.0 : 0.0;
    double speedup = (gpu_ms > 1e-6) ? (cpu_ms / gpu_ms) : 0.0;

    cudaFree(d_query);
    cudaFree(d_interleaved);
    cudaFree(d_out);
    cudaFree(d_query_norm);
    cudaFree(d_vec_norm);
    std::free(h_query);
    std::free(h_vecs);
    std::free(h_interleaved);
    std::free(h_query_norm);
    std::free(h_vec_norm);
    std::free(h_cpu_out);
    std::free(h_gpu_out);

    return {pass_rate, cpu_ms, gpu_ms, speedup};
}

int main(int argc, char** argv) {
    MetricsCollector metrics;
    metrics.set_columns("pass_rate", "n_query", "n_vec", "n_dim", "mode",
                        "cpu_ms", "gpu_ms", "speedup");
    metrics.set_num_repeats(1);

    cudaSetDevice(0);

    COUT_ENDL(" gemv_interleaved (cos / L2)...");

    /*  32  n_vec (kernel  vec_idx >= n_vec  mask,interleaved  32 ) */
    // const int non_multiple_n_vec[] = {33};
    const int non_multiple_n_vec[] = {31, 33, 50, 100, 1000};
    for (int n_vec : non_multiple_n_vec) {
        for (int use_cosine : {0}) {
            auto r = run_case(4, n_vec, 64, use_cosine != 0);
            metrics.add_row(
                r[0],
                1.0,
                (double)n_vec,
                64.0,
                (double)use_cosine,
                r[1],
                r[2],
                r[3]
            );
        }
    }

    // PARAM_3D(n_query, (64, 256),
    //          n_vec, (1024, 4096),
    //          n_dim, (64, 128))
    // {
    //     for (int use_cosine : {0, 1}) {
    //         auto r = run_case(n_query, n_vec, n_dim, use_cosine != 0);
    //         metrics.add_row(
    //             r[0],
    //             (double)n_query,
    //             (double)n_vec,
    //             (double)n_dim,
    //             (double)use_cosine,
    //             r[1],
    //             r[2],
    //             r[3]
    //         );
    //     }
    // }

    cudaDeviceSynchronize();

    metrics.print_table();
    metrics.export_csv("gemv_interleaved_metrics.csv");

    bool all_pass = true;
    for (size_t i = 0; i < metrics.size(); ++i) {
        auto row = metrics.get_row(i);
        if (row.size() > 0 && row[0] < 1.0) all_pass = false;
    }
    COUT_ENDL(all_pass ? "all_test_passed ✅" : "some tests failed ❌");
    return all_pass ? 0 : 1;
}
