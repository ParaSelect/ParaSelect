#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "src/utils/pch.h"
#include "src/clustering/hierarchical/hierarchical.cuh"
#include "cpu/utils/cpu_array_utils.h"
#include "common/test_utils.cuh"
#include "src/utils/utils.cuh"

// ============================================================
// Test Case Runner
// ============================================================
static std::vector<double> run_case(int n, int dim, int M, int split_k, float* h_pool_data, DistanceType dist) {
    ClusteringConfig cfg;
    cfg.n_clusters   = M;
    cfg.max_iters    = 8;
    cfg.distance_mode = dist;
    cfg.seed         = 1234;
    cfg.use_minibatch = false;
    cfg.batch_size   = 1 << 20;
    cfg.device_id    = 0;

    double fit_ms = 0.0;
    bool pass = false;
    int k_final = 0;
    long long total_count = 0;
    bool ok_bijection = false;
    bool ok_offsets   = false;

    BasicHierarchicalClustering model(split_k);

    MEASURE_MS_AND_SAVE("basic hierarchical fit:", fit_ms,
        model.fit(h_pool_data, n, dim, cfg);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    );

    ClusteringResult result = model.get_result();
    ClusterInfo* ci = result.cluster_info;

    //  cluster_info
    if (ci && ci->k > 0 && ci->offsets && ci->counts) {
        k_final = ci->k;

        //  offsets  + counts  == n
        ok_offsets = true;
        long long running = 0;
        for (int c = 0; c < k_final; ++c) {
            if (ci->counts[c] < 0 || ci->offsets[c] != running) {
                ok_offsets = false;
                break;
            }
            running += ci->counts[c];
        }
        total_count = running;
        if (total_count != (long long)n) ok_offsets = false;

        //  reordered_indices
        ok_bijection = true;
        if (result.reordered_indices) {
            std::vector<bool> seen(n, false);
            for (int i = 0; i < n; ++i) {
                int idx = result.reordered_indices[i];
                if (idx < 0 || idx >= n || seen[idx]) {
                    ok_bijection = false;
                    if (!QUIET) {
                        fprintf(stderr, "[test] ERROR: reordered_indices[%d]=%d invalid\n", i, idx);
                    }
                    break;
                }
                seen[idx] = true;
            }
        } else {
            ok_bijection = false;
        }
    }

    // k_final >= M()
    bool ok_k = (k_final >= M);

    pass = ok_offsets && ok_bijection && ok_k;

    if (!QUIET) {
        COUT_ENDL("----- Basic Hierarchical Verify -----");
        COUT_VAL("n=", n, " M=", M, " split_k=", split_k,
                 " k_final=", k_final, " total_count=", (long long)total_count);
        COUT_VAL("ok_offsets=", (int)ok_offsets,
                 " ok_bijection=", (int)ok_bijection,
                 " ok_k(>=M)=", (int)ok_k,
                 " pass=", (int)pass);
        COUT_ENDL("-------------------------------------");
    }

    return {pass ? 1.0 : 0.0, fit_ms, (double)k_final, (double)total_count};
}

// ============================================================
// Main
// ============================================================
#ifdef __CUDACC__
__host__
#endif
int main(int argc, char** argv) {
    MetricsCollector metrics;
    metrics.set_columns("pass_rate", "n", "M", "dim", "split_k", "dist",
                        "fit_ms", "k_final", "total_count");
    metrics.set_num_repeats(1);

    PARAM_1D(dim, (128)) {
        const size_t MAX_N = 10000000;
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

        PARAM_3D(n,     (10000000),
                 dist,  (L2_DISTANCE),
                 split_k, (10))   //  split_k
        {
            // M = N^(2/3), 8
            // int M = std::max(8, (int)std::round(std::pow((double)0.1 * n, 2.0 / 3.0)));
            int M = 10000;

            if ((size_t)n * dim > POOL_SIZE) {
                fprintf(stderr, "Error: n=%d * dim=%d exceeds pool_size=%zu\n",
                        n, dim, POOL_SIZE);
                std::abort();
            }

            if (!QUIET) {
                COUT_ENDL("========================================");
                COUT_VAL("Basic Hierarchical Test: n=", n, " M=", M, " dim=", dim,
                         " split_k=", split_k,
                         " dist=", (dist == L2_DISTANCE ? "L2" : "COSINE"));
                COUT_ENDL("========================================");
            }

            auto row = metrics.add_row_averaged([&]() -> std::vector<double> {
                auto r = run_case(n, dim, M, split_k, h_pool_data, dist);
                return {
                    r[0],           // pass_rate
                    (double)n,
                    (double)M,
                    (double)dim,
                    (double)split_k,
                    (double)dist,
                    r[1],           // fit_ms
                    r[2],           // k_final
                    r[3],           // total_count
                };
            });
        }

        cudaFreeHost(h_pool_data);
    }

    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;

    metrics.print_table();
    metrics.export_csv("basic_hierarchical_cluster_metrics.csv");
    COUT_ENDL("Basic hierarchical clustering tests completed!");
    return 0;
}
