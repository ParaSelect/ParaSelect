#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "src/utils/pch.h"
#include "src/clustering/kmeans/kmeans.cuh"
#include "cpu/utils/cpu_array_utils.h"
#include "common/test_utils.cuh"
#include "cpu/cpu_utils.h"
#include "src/utils/utils.cuh"
// ============================================================
// Algorithm Version Enum
// ============================================================
enum KMeansAlgorithm {
    KMEANS_LLOYD = 0,      //  Lloyd
    KMEANS_MINIBATCH = 1   // Minibatch
};

struct KMeansCompareRun {
    bool ok = false;
    float objective = 0.0f;
    KMeansLloydStats stats;
    std::vector<float> centroids;
    std::vector<idx_t> assign;
};

// ============================================================
// Utilities (CPU)
// ============================================================
// cpu_kmeans_lloyd  ../common/cpu_kmeans.h

static double wall_ms_now() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static bool parse_int_arg(const char* text, int* out) {
    if (!text || !out) return false;
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (!end || *end != '\0' || value <= 0 || value > INT_MAX) {
        return false;
    }
    *out = static_cast<int>(value);
    return true;
}

static KMeansCompareRun run_lloyd_compare_path(
    const KMeansCase& cfg_in,
    const float* h_data,
    const float* h_init_centroids)
{
    KMeansCompareRun out;
    KMeansCase cfg = cfg_in;
    cfg.collect_timing = true;

    const int n = cfg.n;
    const int dim = cfg.dim;
    const int k = cfg.k;

    float* d_centroids = nullptr;
    idx_t* d_assign = nullptr;
    cudaMalloc(&d_centroids, sizeof(float) * (size_t)k * dim);
    cudaMalloc(&d_assign, sizeof(idx_t) * (size_t)n);
    cudaMemcpy(d_centroids, h_init_centroids, sizeof(float) * (size_t)k * dim, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS;

    float objective = 0.0f;
    const double outer_start = wall_ms_now();
    gpu_kmeans_lloyd(cfg, h_data, d_assign, d_centroids, &objective);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;
    const double outer_ms = wall_ms_now() - outer_start;

    out.ok = true;
    out.objective = objective;
    out.stats = get_last_kmeans_lloyd_stats();
    if (!out.stats.timing_valid) {
        out.stats.total_ms = outer_ms;
    }
    out.centroids.resize((size_t)k * dim);
    out.assign.resize((size_t)n);
    cudaMemcpy(out.centroids.data(), d_centroids, sizeof(float) * (size_t)k * dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(out.assign.data(), d_assign, sizeof(idx_t) * (size_t)n, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERRORS;

    cudaFree(d_assign);
    cudaFree(d_centroids);
    return out;
}

static int run_compare_flash_mode(int argc, char** argv) {
    int n = 1 << 20;
    int k = 4096;
    int dim = 96;
    int iters = 4;
    int seed = 1234;
    if (argc >= 3 && !parse_int_arg(argv[2], &n)) {
        fprintf(stderr, "usage: %s --compare-flash [n] [k] [dim] [iters] [seed]\n", argv[0]);
        return 2;
    }
    if (argc >= 4 && !parse_int_arg(argv[3], &k)) {
        fprintf(stderr, "usage: %s --compare-flash [n] [k] [dim] [iters] [seed]\n", argv[0]);
        return 2;
    }
    if (argc >= 5 && !parse_int_arg(argv[4], &dim)) {
        fprintf(stderr, "usage: %s --compare-flash [n] [k] [dim] [iters] [seed]\n", argv[0]);
        return 2;
    }
    if (argc >= 6 && !parse_int_arg(argv[5], &iters)) {
        fprintf(stderr, "usage: %s --compare-flash [n] [k] [dim] [iters] [seed]\n", argv[0]);
        return 2;
    }
    if (argc >= 7 && !parse_int_arg(argv[6], &seed)) {
        fprintf(stderr, "usage: %s --compare-flash [n] [k] [dim] [iters] [seed]\n", argv[0]);
        return 2;
    }

    float* h_data = nullptr;
    float* h_init_centroids = nullptr;
    cudaMallocHost(&h_data, sizeof(float) * (size_t)n * dim);
    cudaMallocHost(&h_init_centroids, sizeof(float) * (size_t)k * dim);
    CHECK_CUDA_ERRORS;
    constexpr int kCompareInitThreads = 16;
    init_array_multithreaded(h_data, (size_t)n * dim, seed, -1.0f, 1.0f, kCompareInitThreads);

    KMeansCase cfg;
    cfg.n = n;
    cfg.dim = dim;
    cfg.k = k;
    cfg.iters = iters;
    cfg.minibatch_iters = 20;
    cfg.seed = seed;
    cfg.dist = L2_DISTANCE;
    cfg.precision = KMeansComputePrecision::Tf32;
    cfg.init_policy = KMEANS_INIT_FORCE_RANDOM;
    cfg.collect_timing = true;

    init_centroids_by_sampling(cfg, h_data, h_init_centroids);

    KMeansCase baseline_cfg = cfg;
    baseline_cfg.use_flash_assign = false;
    fprintf(stderr,
            "[paraselect:kmeans_flash_compare:phase] baseline_start n=%d k=%d dim=%d iters=%d seed=%d init_threads=%d\n",
            n, k, dim, iters, seed, kCompareInitThreads);
    auto baseline = run_lloyd_compare_path(baseline_cfg, h_data, h_init_centroids);

    KMeansCase flash_cfg = cfg;
    flash_cfg.use_flash_assign = true;
    fprintf(stderr,
            "[paraselect:kmeans_flash_compare:phase] flash_start n=%d k=%d dim=%d iters=%d seed=%d init_threads=%d\n",
            n, k, dim, iters, seed, kCompareInitThreads);
    auto flash = run_lloyd_compare_path(flash_cfg, h_data, h_init_centroids);

    double centroid_rmse = 0.0;
    if (baseline.centroids.size() == flash.centroids.size() && !baseline.centroids.empty()) {
        double sum = 0.0;
        for (size_t i = 0; i < baseline.centroids.size(); ++i) {
            const double diff = (double)baseline.centroids[i] - (double)flash.centroids[i];
            sum += diff * diff;
        }
        centroid_rmse = std::sqrt(sum / (double)baseline.centroids.size());
    }

    int assignment_mismatches = 0;
    const size_t assign_n = std::min(baseline.assign.size(), flash.assign.size());
    for (size_t i = 0; i < assign_n; ++i) {
        assignment_mismatches += (baseline.assign[i] != flash.assign[i]) ? 1 : 0;
    }
    const double mismatch_rate = assign_n > 0 ? (double)assignment_mismatches / (double)assign_n : 1.0;
    const double rel_obj_diff =
        std::abs((double)baseline.objective - (double)flash.objective) /
        std::max(1.0, std::abs((double)baseline.objective));
    const double speedup =
        flash.stats.total_ms > 1e-9 ? baseline.stats.total_ms / flash.stats.total_ms : 0.0;

    const bool pass =
        baseline.ok &&
        flash.ok &&
        std::isfinite(baseline.objective) &&
        std::isfinite(flash.objective) &&
        baseline.objective > 0.0f &&
        flash.objective > 0.0f &&
        rel_obj_diff < 0.20;

    std::cout << "[paraselect:kmeans_flash_compare:summary]"
              << " n=" << n
              << " k=" << k
              << " dim=" << dim
              << " iters=" << iters
              << " seed=" << seed
              << " init_threads=" << kCompareInitThreads
              << " baseline_total_ms=" << baseline.stats.total_ms
              << " flash_total_ms=" << flash.stats.total_ms
              << " speedup=" << speedup
              << " baseline_batch_ms=" << baseline.stats.batch_ms
              << " flash_batch_ms=" << flash.stats.batch_ms
              << " baseline_final_assign_ms=" << baseline.stats.final_assign_ms
              << " flash_final_assign_ms=" << flash.stats.final_assign_ms
              << " baseline_obj=" << baseline.objective
              << " flash_obj=" << flash.objective
              << " rel_obj_diff=" << rel_obj_diff
              << " centroid_rmse=" << centroid_rmse
              << " assignment_mismatches=" << assignment_mismatches
              << " mismatch_rate=" << mismatch_rate
              << " pass=" << (pass ? 1 : 0)
              << std::endl;

    cudaFreeHost(h_init_centroids);
    cudaFreeHost(h_data);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;
    return pass ? 0 : 1;
}

// ============================================================
// Test Case Runner
// ============================================================
static std::vector<double> run_case(const KMeansCase& cfg, float* h_pool_data, KMeansAlgorithm algo) {
    const int n = cfg.n, dim = cfg.dim, k = cfg.k;


    // ()
    float* h_data = h_pool_data;

    // :CPUGPU
    //  cfg.seed ( pinned memory)
    float* h_init_centroids = nullptr;
    cudaMallocHost(&h_init_centroids, sizeof(float) * (size_t)k * dim);
    init_centroids_by_sampling(cfg, h_data, h_init_centroids);

    // CPU reference - GPU
    std::vector<int> h_assign_cpu(n);
    std::vector<float> h_centroids_cpu((size_t)k * dim);
    std::memcpy(h_centroids_cpu.data(), h_init_centroids, sizeof(float) * (size_t)k * dim);
    float cpu_obj = 0.0f;

    double cpu_ms = 0.0;
    MEASURE_MS_AND_SAVE("cpu_kmeans:", cpu_ms,
        // cpu_kmeans_lloyd(cfg,
        //                  h_data,
        //                  h_assign_cpu.data(),
        //                  h_centroids_cpu.data(),
        //                  &cpu_obj);
    );

    // GPU buffers - CPU
    float* d_centroids = nullptr;
    cudaMalloc(&d_centroids, sizeof(float) * (size_t)k * dim);
    cudaMemcpyAsync(d_centroids, h_init_centroids, sizeof(float) * (size_t)k * dim,
                   cudaMemcpyHostToDevice);

    //  in-place reorder
    idx_t* h_indices = (idx_t*)std::aligned_alloc(64, (size_t)n * sizeof(idx_t));
    if (!h_indices) {
        fprintf(stderr, "[test] ERROR: Failed to alloc h_indices\n");
        cudaFree(d_centroids);
        cudaFreeHost(h_init_centroids);
        return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    for (int i = 0; i < n; ++i) {
        h_indices[i] = i;
    }

    //
    size_t data_size = (size_t)n * (size_t)dim;
    size_t data_bytes = data_size * sizeof(float);
    float* h_data_original = (float*)std::aligned_alloc(64, data_bytes);
    if (!h_data_original) {
        fprintf(stderr, "[test] ERROR: Failed to alloc h_data_original\n");
        std::free(h_indices);
        cudaFree(d_centroids);
        cudaFreeHost(h_init_centroids);
        return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    std::memcpy(h_data_original, h_data, data_bytes);

    //  ivf_kmeans_inplace :K-means + permutation + in-place
    float gpu_obj = 0.0f;
    ClusterInfo h_cluster_info;
    bool use_minibatch = (algo == KMEANS_MINIBATCH);
    const int BATCH_SIZE = 1 << 20;  // 1M per batch

    double gpu_ms = 0.0;
    double reorder_ms = 0.0;
    bool reorder_success = false;

    MEASURE_MS_AND_SAVE("ivf_kmeans_inplace:", gpu_ms,
        reorder_success = ivf_kmeans_inplace(cfg, h_data, d_centroids,
                                              &h_cluster_info, use_minibatch, 0, BATCH_SIZE, &gpu_obj,
                                              h_indices);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    );

    // Copy GPU centroids back for verification
    std::vector<float> h_centroids_gpu((size_t)k * dim);
    cudaMemcpy(h_centroids_gpu.data(), d_centroids, sizeof(float) * (size_t)k * dim, cudaMemcpyDeviceToHost);

    // cluster
    bool ok_info = true;
    if (h_cluster_info.offsets && h_cluster_info.counts && h_cluster_info.k > 0) {
        long long total = 0;
        for (int c = 0; c < h_cluster_info.k; ++c) {
            int cnt = h_cluster_info.counts[c];
            if (cnt < 0 || cnt > n || h_cluster_info.offsets[c] != total) {
                ok_info = false;
                break;
            }
            total += (long long)cnt;
        }
        if (total != (long long)n) ok_info = false;
    } else {
        ok_info = false;
    }

    // (bijection)
    // :h_indices  0  n-1 ,
    bool ok_bijection = true;
    if (h_indices) {
        //
        std::vector<bool> seen(n, false);
        for (int i = 0; i < n; ++i) {
            idx_t idx = h_indices[i];
            //
            if (idx < 0 || idx >= n) {
                ok_bijection = false;
                if (!QUIET) {
                    fprintf(stderr, "[test] ERROR: Invalid index at position %d: %d (should be in [0, %d))\n",
                           i, idx, n);
                }
                break;
            }
            // ()
            if (seen[idx]) {
                ok_bijection = false;
                if (!QUIET) {
                    fprintf(stderr, "[test] ERROR: Duplicate index %d found at position %d\n", idx, i);
                }
                break;
            }
            seen[idx] = true;
        }
        // ()
        if (ok_bijection) {
            for (int i = 0; i < n; ++i) {
                if (!seen[i]) {
                    ok_bijection = false;
                    if (!QUIET) {
                        fprintf(stderr, "[test] ERROR: Index %d is missing from indices\n", i);
                    }
                    break;
                }
            }
        }
    } else {
        ok_bijection = false;
    }

    reorder_success = reorder_success && ok_info && ok_bijection;

    if (QUIET) {
        if (reorder_success) {
            fprintf(stderr, "[test] IVF K-means In-place completed successfully. "
                   "Sample: data[0..3] first dim0: %f %f %f %f\n",
                   h_data[0], h_data[(size_t)dim],
                   h_data[(size_t)2 * dim], h_data[(size_t)3 * dim]);
            fprintf(stderr, "[test] Index bijection check: PASSED (all indices 0-%d present and unique)\n", n-1);
        } else {
            fprintf(stderr, "[test] IVF K-means In-place FAILED! (cluster_info=%d, bijection=%d)\n",
                   ok_info ? 1 : 0, ok_bijection ? 1 : 0);
        }
    }

    // Cleanup
    std::free(h_indices);
    std::free(h_data_original);
    free_cluster_info(&h_cluster_info, false);
    cudaFree(d_centroids);
    cudaFreeHost(h_init_centroids);

    CHECK_CUDA_ERRORS;

    // verify: compare objective (relative) + centroid L2 error
    // :FP16 + atomic ,
    double rel_obj_err = std::abs((double)gpu_obj - (double)cpu_obj) / std::max(1.0, (double)cpu_obj);

    double centroid_rmse = 0.0;
    {
        float s = l2_distance_squared(h_centroids_cpu.data(), h_centroids_gpu.data(), h_centroids_cpu.size());
        centroid_rmse = std::sqrt(s / std::max<size_t>(1, h_centroids_cpu.size()));
    }

    double obj_tol = 1e-4;
    bool pass = reorder_success;

    double pass_rate = reorder_success ? 1.0 : 0.0;
    double speedup = (gpu_ms > 1e-6) ? (cpu_ms / gpu_ms) : 0.0;
    reorder_ms = gpu_ms;  // ivf_kmeans

    if (!QUIET) {
        COUT_ENDL("----- KMeans Verify -----");
        COUT_VAL("cpu_obj=", cpu_obj, " gpu_obj=", gpu_obj, " rel_obj_err=", rel_obj_err);
        COUT_VAL("centroid_rmse=", centroid_rmse, " reorder_success=", (reorder_success ? 1 : 0), " pass=", (pass ? 1 : 0));
        COUT_ENDL("-------------------------");
    }

    return {pass_rate, gpu_ms, cpu_ms, speedup, (double)rel_obj_err, (double)centroid_rmse, (double)gpu_obj, (double)cpu_obj, reorder_ms};
}

// ============================================================
// Main (Metrics table like your template)
// ============================================================
// , CUDA  std::function
#ifdef __CUDACC__
__host__
#endif
int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--compare-flash") == 0) {
        return run_compare_flash_mode(argc, argv);
    }

    MetricsCollector metrics;
    metrics.set_columns("pass_rate", "n", "k", "dim", "iters", "dist", "algo",
                        "gpu_ms", "cpu_ms", "speedup", "rel_obj_err", "centroid_rmse",
                        "gpu_obj", "cpu_obj");
    metrics.set_num_repeats(1);

    // PARAM_3D(n, (10000, 20000),
    //  dim, (4,10,96,128,200),
    PARAM_1D(dim, (96)){
        // ,pinned memory
        // :max(n) * dim
        //  n = 100000000, 100000000 * dim  floats
        const size_t MAX_N = 100000000;
        // const size_t MAX_N = 500000000;
        const size_t POOL_SIZE = MAX_N * dim;
        float* h_pool_data = nullptr;
        cudaMallocHost(&h_pool_data, sizeof(float) * POOL_SIZE);
        init_array_multithreaded(h_pool_data, POOL_SIZE, 1234, -1.0f, 1.0f);
        if (!QUIET) {
            COUT_ENDL("========================================");
            COUT_VAL("Created pinned memory pool: ", POOL_SIZE, " floats (",
                    (double)POOL_SIZE * sizeof(float) / (1024.0 * 1024.0 * 1024.0), " GB)");
            COUT_ENDL("========================================");
        }

        // PARAM_3D(n, (20000, 1000000, 10000000, 100000000, 500000000),
        // PARAM_3D(n, (10000, 20000, 50000, 100000),
        // PARAM_3D(n, (100000000),
        PARAM_3D(n, (10000000),
                dist, (L2_DISTANCE),
                // algo, (KMEANS_MINIBATCH))
                algo, (KMEANS_LLOYD))
    {

        // K = N^(2/3)(, 8)
        // int k = std::max(8, (int)std::round(std::pow((double)0.1 * n, 2.0 / 3.0)));
        // int k = 131072;
        int k = 8196;
            if ((size_t)n * dim > POOL_SIZE) {
                fprintf(stderr, "Error: n=%d * dim=%d = %zu exceeds pool_size=%zu\n",
                        n, dim, (size_t)n * dim, POOL_SIZE);
                std::abort();
            }
        KMeansCase cfg;
        cfg.n = n;
        cfg.dim = dim;
        cfg.k = k;
        cfg.iters = 8;
        cfg.minibatch_iters = 20;
        cfg.seed = 1234;
        cfg.dist = dist;

        if (!QUIET) {
            COUT_ENDL("========================================");
            COUT_VAL("KMeans Test: n=", cfg.n,
                    " k=", cfg.k,
                    " dim=", cfg.dim,
                    " iters=", cfg.iters,
                        " dist=", (cfg.dist == L2_DISTANCE ? "L2" : "COSINE"),
                        " algo=", (algo == KMEANS_LLOYD ? "LLOYD" : "MINIBATCH")
                    );
            COUT_ENDL("========================================");
        }

        auto row = metrics.add_row_averaged([&]() -> std::vector<double> {
                // run_case(),
                auto r = run_case(cfg, h_pool_data, algo);
            return {
                r[0],                         // pass_rate
                (double)cfg.n,
                (double)cfg.k,
                (double)cfg.dim,
                (double)cfg.iters,
                (double)cfg.dist,
                    (double)algo,                 // algorithm version
                r[1],                         // gpu_ms
                r[2],                         // cpu_ms
                r[3],                         // speedup
                r[4],                         // rel_obj_err
                r[5],                         // centroid_rmse
                r[6],                         // gpu_obj
                r[7],                         // cpu_obj
            };
        });


        }
        // pinned memory()
        cudaFreeHost(h_pool_data);
    }

    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;

    metrics.print_table();
    metrics.export_csv("kmeans_cluster_metrics.csv");
    COUT_ENDL("KMeans tests completed successfully!");
    return 0;
}
