/**
 * KMeans
 *
 * : SIFT, DEEP, TEXT
 * : 1M, 10M, 100M
 *
 * : ./test_kmeans_stats <dataset> <size> [n_clusters] [kmeans_iters]
 *   dataset: sift | deep | text
 *   size: 1m | 10m | 100m
 *
 * :
 *   ./test_kmeans_stats sift 1m
 *   ./test_kmeans_stats deep 10m 1024 10
 *   ./test_kmeans_stats text 100m 10240 5
 */

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <memory>

#include "src/utils/pch.h"
#include "src/dataset/dataset.cuh"
#include "src/dataset/common/dataset_source.h"
#include "src/clustering/kmeans/kmeans_stats.cuh"
#include "common/test_utils.cuh"

using namespace paraselect;
using namespace paraselect::dataset;

// ============================================================
//
// ============================================================
struct StatsInput { DatasetSpec spec; };

static StatsInput make_stats_input(const std::string& dataset, const std::string& size) {
    return {describe_dataset(dataset, size)};
}

// ============================================================
// (, pinned memory)
// ============================================================
struct LoadedData {
    VectorBuffer native;
    std::vector<float> converted;
    float* h_base = nullptr;
    size_t n_total = 0;
    std::string dataset_name;
    int dim = 0;

};

static LoadedData load_dataset(const StatsInput& cfg) {
    LoadedData data;
    data.dataset_name = dataset_key_name(cfg.spec);
    data.dim = cfg.spec.dim;
    std::printf("[Step 1] Loading %s...\n", display_dataset_name(cfg.spec).c_str());
    auto t0 = std::chrono::high_resolution_clock::now();
    data.native = open_dataset(cfg.spec.name, cfg.spec.size_label).load_base();
    const auto view = data.native.view();
    data.n_total = view.n;
    if (view.dtype == VectorDType::F32) {
        data.h_base = const_cast<float*>(view.fp32());
    } else {
        data.converted.resize(view.n * static_cast<size_t>(view.dim));
        for (size_t i = 0; i < data.converted.size(); ++i) {
            data.converted[i] = static_cast<float>(view.u8()[i]);
        }
        data.h_base = data.converted.data();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("  Loaded %zu vectors (dim=%d) in %.2f ms\n\n",
                data.n_total, data.dim, load_ms);

    return data;
}

// ============================================================
//
// ============================================================
static void export_residuals_per_cluster(
    const float* h_data,
    const float* h_centroids,
    const int* h_assign,
    int n,
    int dim,
    int k,
    DistanceType dist_type,
    const std::string& filepath)
{
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        std::cerr << "[ERROR] Failed to open " << filepath << std::endl;
        return;
    }

    ofs << "cluster_id,residual\n";

    for (int i = 0; i < n; ++i) {
        int cid = h_assign[i];
        if (cid < 0 || cid >= k) continue;
        const float* vec = h_data + (size_t)i * dim;
        const float* cent = h_centroids + (size_t)cid * dim;

        float dist = 0.0f;
        if (dist_type == L2_DISTANCE) {
            float sum = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float diff = vec[d] - cent[d];
                sum += diff * diff;
            }
            dist = sum;
        } else {
            float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
            for (int d = 0; d < dim; ++d) {
                dot += vec[d] * cent[d];
                norm_a += vec[d] * vec[d];
                norm_b += cent[d] * cent[d];
            }
            float denom = std::sqrt(norm_a * norm_b);
            dist = (denom < 1e-8f) ? 1.0f : (1.0f - dot / denom);
        }
        ofs << cid << "," << dist << "\n";
    }

    ofs.close();
    std::printf("[Stats] Per-vector residuals exported to %s\n", filepath.c_str());
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("Usage: %s <dataset> <size> [n_clusters] [kmeans_iters]\n", argv[0]);
        std::printf("  dataset: sift | deep | text\n");
        std::printf("  size: 1m | 10m | 100m\n");
        std::printf("  n_clusters: default 10240\n");
        std::printf("  kmeans_iters: default 10\n");
        return 1;
    }

    cudaSetDevice(0);
    (void)cudaGetLastError();

    std::string dataset_name = argv[1];
    std::string size_str = argv[2];
    int n_clusters = (argc > 3) ? std::stoi(argv[3]) : 10240;
    int kmeans_iters = (argc > 4) ? std::stoi(argv[4]) : 10;
    DistanceType dist = L2_DISTANCE;

    //
    StatsInput cfg = make_stats_input(dataset_name, size_str);

    std::printf("========================================\n");
    std::printf("KMeans Stats Test on %s\n", display_dataset_name(cfg.spec).c_str());
    std::printf("  n_vectors=%zu  dim=%d  n_clusters=%d  iters=%d\n",
                cfg.spec.n_vectors, cfg.spec.dim, n_clusters, kmeans_iters);
    std::printf("========================================\n\n");

    // ====================================================================
    // 1.
    // ====================================================================
    LoadedData data = load_dataset(cfg);

    // ====================================================================
    // 2.  K-means
    // ====================================================================
    std::printf("[Step 2] Running K-means clustering...\n");
    auto t0 = std::chrono::high_resolution_clock::now();

    ClusterDataset dataset;
    float kmeans_objective = 0.0f;

    dataset.init_with_kmeans(
        (int)data.n_total,
        data.dim,
        n_clusters,
        &kmeans_objective,
        data.h_base,
        kmeans_iters,
        false,           // use_minibatch
        dist,
        1234,            // seed
        (1 << 20),       // batch_size
        0,               // device_id
        paraselect::ownership::BufferOwner::Borrowed,
        1.0f,            // train_sample_ratio
        0                // train_max_samples
    );

    auto t1 = std::chrono::high_resolution_clock::now();
    double kmeans_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("  K-means completed: objective=%.4f  time=%.2f ms\n", kmeans_objective, kmeans_ms);
    std::printf("  Actual clusters: %d\n\n", dataset.cluster_info.k);

    // ====================================================================
    // 3.
    // ====================================================================
    std::printf("[Step 3] Computing residual statistics...\n");
    t0 = std::chrono::high_resolution_clock::now();

    int* h_assign = new int[data.n_total];
    {
        for (int i = 0; i < (int)data.n_total; ++i) {
            const float* vec = data.h_base + (size_t)i * data.dim;
            float best_dist = 1e30f;
            int best_cid = 0;
            for (int c = 0; c < dataset.cluster_info.k; ++c) {
                const float* cent = dataset.centroids + (size_t)c * data.dim;
                float dist_val = 0.0f;
                for (int d = 0; d < data.dim; ++d) {
                    float diff = vec[d] - cent[d];
                    dist_val += diff * diff;
                }
                if (dist_val < best_dist) {
                    best_dist = dist_val;
                    best_cid = c;
                }
            }
            h_assign[i] = best_cid;
        }
    }

    KMeansStatsResult stats_result;
    compute_kmeans_residual_stats(
        data.h_base,
        dataset.centroids,
        h_assign,
        (int)data.n_total,
        data.dim,
        dataset.cluster_info.k,
        dist,
        &stats_result,
        50   // num_bins
    );

    t1 = std::chrono::high_resolution_clock::now();
    double stats_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("  Stats computed in %.2f ms\n\n", stats_ms);

    // ====================================================================
    // 4.
    // ====================================================================
    print_kmeans_stats(stats_result);

    std::string out_dir = "tests/unit/stats/output";
    system(("mkdir -p " + out_dir).c_str());

    std::string csv_path = out_dir + "/kmeans_stats_" + cfg.spec.name + cfg.spec.size_label + ".csv";
    export_kmeans_stats_to_csv(stats_result, csv_path);

    std::string residual_csv_path = out_dir + "/kmeans_residuals_" + cfg.spec.name + cfg.spec.size_label + ".csv";
    export_residuals_per_cluster(
        data.h_base, dataset.centroids, h_assign,
        (int)data.n_total, data.dim, dataset.cluster_info.k, dist,
        residual_csv_path);

    // ====================================================================
    // 5.
    // ====================================================================
    delete[] h_assign;
    dataset.release();

    std::printf("\nDone! CSV files:\n");
    std::printf("  - %s  ()\n", csv_path.c_str());
    std::printf("  - %s  (, Python )\n", residual_csv_path.c_str());

    return 0;
}
