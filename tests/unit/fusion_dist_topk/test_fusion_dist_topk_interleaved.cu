/**
 * test_fusion_cos_topk_interleaved
 *
 *  test_fusion_cos_topk: interleaved  topk  CPU .
 * - GPU:launch_gemm_interleaved_cos ( test_gemv_interleaved ), CPU  topk
 * - CPU:cpu_cos_distance_topk(row-major )
 */

#include <stdlib.h>
#include <limits>
#include <string>
#include <algorithm>
#include <vector>

#include "src/search/kernels/fine/gpu_fine/gemv_interleaved.cuh"
#include "src/layout/ivf_interleaved_layout.cuh"
#include "src/utils/pch.h"
#include "common/test_utils.cuh"
#include "cpu/cpu_utils.h"

#define EPSILON 1e-2
#define DIV_EPSILON 1e-4

///  row-major  L2 ()
static void compute_norms_from_pointers(
    const float** vectors,
    float* norms,
    int n_vec,
    int n_dim) {
    for (int i = 0; i < n_vec; ++i) {
        float sum = 0.0f;
        for (int d = 0; d < n_dim; ++d) {
            float v = vectors[i][d];
            sum += v * v;
        }
        norms[i] = std::sqrt(sum);
    }
}

/**
 *  - interleaved  cosine topk
 *  {pass_rate, n_query, n_batch, n_dim, k, gpu_ms, cpu_ms, speedup, memory_mb}
 */
std::vector<double> test_cos_topk_interleaved(
    int n_query,
    int n_batch,
    int n_dim,
    int k,
    const float** h_query_vectors,
    const float** h_data_vectors) {
    bool pass = true;
    double memory_mb = (double)(n_query * n_dim + n_batch * n_dim + n_query * n_batch) * sizeof(float) / (double)(1024 * 1024);

    if (!QUIET) {
        COUT_VAL(":", n_query, " ×", n_batch, ", :", n_dim, ", top", k, "(interleaved)");
    }

    // n_batch  32(interleaved )
    int n_batch_aligned = ((n_batch + 31) / 32) * 32;

    idx_t** topk_index_cpu = malloc_vector_list<idx_t>(n_query, k);
    idx_t** topk_index_gpu = malloc_vector_list<idx_t>(n_query, k);
    float** topk_dist_cpu = malloc_vector_list<float>(n_query, k);
    float** topk_dist_gpu = malloc_vector_list<float>(n_query, k);

    for (int i = 0; i < n_query; ++i) {
        for (int j = 0; j < k; ++j) {
            topk_dist_cpu[i][j] = -std::numeric_limits<float>::max();
            topk_dist_gpu[i][j] = -std::numeric_limits<float>::max();
        }
    }

    //  interleaved
    float* h_interleaved = (float*)std::malloc(paraselect::interleaved_storage_bytes(n_batch_aligned, n_dim));
    float* h_query_norm = (float*)std::malloc((size_t)n_query * sizeof(float));
    float* h_data_norm = (float*)std::malloc((size_t)n_batch_aligned * sizeof(float));

    //  data  row-major  pack; n_batch_aligned
    float* h_data_row = (float*)std::malloc((size_t)n_batch_aligned * n_dim * sizeof(float));
    std::memset(h_data_row, 0, (size_t)n_batch_aligned * n_dim * sizeof(float));
    for (int i = 0; i < n_batch; ++i) {
        for (int d = 0; d < n_dim; ++d) {
            h_data_row[(size_t)i * n_dim + d] = h_data_vectors[i][d];
        }
    }
    paraselect::pack_rowmajor_to_interleaved(h_data_row, h_interleaved, n_batch_aligned, n_dim);

    compute_norms_from_pointers(h_query_vectors, h_query_norm, n_query, n_dim);
    compute_norms_from_pointers(h_data_vectors, h_data_norm, n_batch, n_dim);
    for (int i = n_batch; i < n_batch_aligned; ++i) {
        h_data_norm[i] = 1e-6f;  // padding
    }

    // GPU
    float *d_query = nullptr, *d_interleaved = nullptr, *d_out = nullptr;
    float *d_query_norm = nullptr, *d_data_norm = nullptr;

    cudaMalloc(&d_query, (size_t)n_query * n_dim * sizeof(float));
    cudaMalloc(&d_interleaved, paraselect::interleaved_storage_bytes(n_batch_aligned, n_dim));
    cudaMalloc(&d_out, (size_t)n_query * n_batch_aligned * sizeof(float));
    cudaMalloc(&d_query_norm, (size_t)n_query * sizeof(float));
    cudaMalloc(&d_data_norm, (size_t)n_batch_aligned * sizeof(float));

    for (int q = 0; q < n_query; ++q) {
        cudaMemcpy(d_query + (size_t)q * n_dim, h_query_vectors[q], (size_t)n_dim * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_interleaved, h_interleaved, paraselect::interleaved_storage_bytes(n_batch_aligned, n_dim), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_norm, h_query_norm, (size_t)n_query * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_norm, h_data_norm, (size_t)n_batch_aligned * sizeof(float), cudaMemcpyHostToDevice);

    float* h_gpu_dist = (float*)std::malloc((size_t)n_query * n_batch_aligned * sizeof(float));

    double gpu_duration_ms = 0, cpu_duration_ms = 0;

    /* GPU: launch_gemm_interleaved_cos ->  */
    MEASURE_MS_AND_SAVE("gpu interleaved cos:", gpu_duration_ms,
        launch_gemm_interleaved_cos(
            d_query, d_interleaved,
            d_query_norm, d_data_norm,
            n_query, n_batch_aligned, n_dim,
            d_out, 0);
        cudaDeviceSynchronize();
    );

    cudaMemcpy(h_gpu_dist, d_out, (size_t)n_query * n_batch_aligned * sizeof(float), cudaMemcpyDeviceToHost);

    /*  GPU  topk( n_batch ) */
    for (int q = 0; q < n_query; ++q) {
        std::vector<std::pair<float, int>> pairs(n_batch);
        for (int j = 0; j < n_batch; ++j) {
            pairs[j] = {h_gpu_dist[(size_t)q * n_batch_aligned + j], j};
        }
        int topk_count = std::min(k, n_batch);
        std::partial_sort(pairs.begin(), pairs.begin() + topk_count, pairs.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                if (a.first != b.first) return a.first < b.first;
                return a.second < b.second;
            });
        for (int i = 0; i < topk_count; ++i) {
            topk_index_gpu[q][i] = pairs[i].second;
            topk_dist_gpu[q][i] = pairs[i].first;
        }
    }

    /* CPU  */
    MEASURE_MS_AND_SAVE("cpu cos topk:", cpu_duration_ms,
        cpu_cos_distance_topk(
            h_query_vectors, h_data_vectors,
            topk_index_cpu, topk_dist_cpu,
            n_query, n_batch, n_dim, k);
    );

    double speedup = (gpu_duration_ms > 1e-9) ? (cpu_duration_ms / gpu_duration_ms) : 0.0;

    // :( compare_set_2D,)
    pass &= compare_set_2D<float>(topk_dist_gpu, topk_dist_cpu, n_query, k, EPSILON);

    //
    cudaFree(d_query);
    cudaFree(d_interleaved);
    cudaFree(d_out);
    cudaFree(d_query_norm);
    cudaFree(d_data_norm);
    std::free(h_interleaved);
    std::free(h_query_norm);
    std::free(h_data_norm);
    std::free(h_data_row);
    std::free(h_gpu_dist);
    free_vector_list((void**)topk_index_gpu);
    free_vector_list((void**)topk_index_cpu);
    free_vector_list((void**)topk_dist_cpu);
    free_vector_list((void**)topk_dist_gpu);

    return {
        pass ? 1.0 : 0.0,
        (double)n_query, (double)n_batch, (double)n_dim, (double)k,
        gpu_duration_ms, cpu_duration_ms, speedup, memory_mb
    };
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    srand(time(0));

    bool all_pass = true;
    MetricsCollector metrics;
    metrics.set_columns("pass_rate", "n_query", "n_batch", "n_dim", "k", "avg_gpu_ms", "avg_cpu_ms", "avg_speedup", "memory_mb");

    cudaSetDevice(0);

    COUT_ENDL(": interleaved  Topk (launch_gemm_interleaved_cos + CPU topk )");

    // warmup
    const float** q_warm = generate_vector_list<const float>(64, 128);
    const float** d_warm = generate_vector_list<const float>(256, 128);
    test_cos_topk_interleaved(64, 256, 128, 16, q_warm, d_warm);
    free_vector_list((void**)q_warm);
    free_vector_list((void**)d_warm);

    const float** cached_h_query = nullptr;
    const float** cached_h_data = nullptr;
    int cached_n_query = -1;
    int cached_n_batch = -1;
    int cached_n_dim = -1;

    // ( test_fusion_cos_topk )
    PARAM_2D(n_probes, (1, 5, 10, 20, 40),
             n_batch, (1024))
    {
        int n_query = 10000;
        int n_dim = 128;
        int k = n_probes;

        bool need_regen_query = (cached_n_query != n_query || cached_n_dim != n_dim);
        bool need_regen_data = (cached_n_batch != n_batch || cached_n_dim != n_dim);

        if (need_regen_query) {
            if (cached_h_query) free_vector_list((void**)cached_h_query);
            cached_h_query = generate_vector_list<const float>(n_query, n_dim);
            cached_n_query = n_query;
            cached_n_dim = n_dim;
        }
        if (need_regen_data) {
            if (cached_h_data) free_vector_list((void**)cached_h_data);
            cached_h_data = generate_vector_list<const float>(n_batch, n_dim);
            cached_n_batch = n_batch;
            cached_n_dim = n_dim;
        }

        auto avg_result = metrics.add_row_averaged([&]() -> std::vector<double> {
            auto r = test_cos_topk_interleaved(
                n_query, n_batch, n_dim, k,
                cached_h_query, cached_h_data);
            all_pass &= (r[0] == 1.0);
            return r;
        });
    }

    if (cached_h_query) free_vector_list((void**)cached_h_query);
    if (cached_h_data) free_vector_list((void**)cached_h_data);

    metrics.print_table();
    metrics.export_csv("fusion_cos_topk_interleaved_metrics.csv");

    COUT_ENDL("\n:", all_pass ? "✅ PASS" : "❌ FAIL");
    return all_pass ? 0 : 1;
}
