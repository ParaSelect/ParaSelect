/**
 * @file test_gemv_interleaved_page.cu
 *
 * @brief `launch_gemv_page_l2` ( Interleaved ).
 */

#include "src/utils/pch.h"
#include "src/search/kernels/fine/gpu_fine/gemv_page_kernel.cuh"
#include "cpu/utils/cpu_array_utils.h"
#include "common/test_utils.cuh"
#include "common/params_macros.cuh"
#include "common/vector_nd.cuh"
#include "cpu/search/cpu_fine.h"
#include "cpu/search/cpu_fine_page.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>

#define EPSILON 1e-3f

static constexpr int kWarpSortCap = 256;
static constexpr int QPB = 8; // QueriesPerBlock

/* ============================================================================
 * CPU  ()
 * ============================================================================ */
static void cpu_ground_truth(
    int n_query, int n_dim, int k, int n_probes,
    const float* h_query,
    const float* h_vectors_rm, //  row-major
    const std::vector<int>& h_pvoc,
    const std::vector<int>& h_pvc,
    const std::vector<int>& phys_pages, // Mapping:  ->
    const std::vector<std::vector<int>>& query_to_clusters,
    std::vector<float>& gt_dist,
    std::vector<int>& gt_index
) {
    // TODO:  CPU  L2 TopK
}

/* ============================================================================
 *  Case
 * ============================================================================ */
static std::vector<double> run_case(
    int n_query, int n_cluster, int n_dim, int k, int n_probes) {

    std::mt19937 rng(42);

    std::vector<int> h_pvc(n_cluster);
    std::vector<int> h_pvoc(n_cluster);
    int total_vecs = 0;
    int total_pages = 0;

    // 1.  Cluster
    for (int c = 0; c < n_cluster; ++c) {
        h_pvc[c] = 100 + rng() % 5000; //
        h_pvoc[c] = total_vecs;
        total_vecs += h_pvc[c];
        total_pages += (h_pvc[c] + 255) / 256;
    }

    // 2.  ()
    std::vector<int> phys_pages(total_pages);
    for (int i = 0; i < total_pages; ++i) { phys_pages[i] = i; }
    std::shuffle(phys_pages.begin(), phys_pages.end(), rng);

    // 3. Query
    std::vector<std::vector<int>> query_to_clusters(n_query, std::vector<int>(n_probes));
    for (int q = 0; q < n_query; ++q) {
        std::vector<int> c_idx(n_cluster);
        for (int c = 0; c < n_cluster; ++c) c_idx[c] = c;
        std::shuffle(c_idx.begin(), c_idx.end(), rng);
        for (int p = 0; p < n_probes; ++p) query_to_clusters[q][p] = c_idx[p];
    }

    // 4.  Entry
    std::vector<int> h_cluster_page_table, h_ecid, h_epid, h_eqs, h_eqc, h_eq, h_epi;
    int q_offset = 0;
    int page_offset_global = 0;

    for (int c = 0; c < n_cluster; ++c) {
        std::vector<std::pair<int, int>> probing_queries;
        for (int q = 0; q < n_query; ++q) {
            for (int p = 0; p < n_probes; ++p) {
                if (query_to_clusters[q][p] == c) probing_queries.push_back({q, p});
            }
        }

        int num_pages_for_c = (h_pvc[c] + 255) / 256;
        if (probing_queries.empty()) {
            page_offset_global += num_pages_for_c;
            continue;
        }

        // QPB
        for (size_t i = 0; i < probing_queries.size(); i += QPB) {
            int chunk_size = std::min<int>(QPB, probing_queries.size() - i);
            for (int p = 0; p < num_pages_for_c; ++p) {
                h_cluster_page_table.push_back(phys_pages[page_offset_global + p]);
                h_ecid.push_back(c);
                h_epid.push_back(p); //  Cluster
                h_eqs.push_back(q_offset);
                h_eqc.push_back(chunk_size);
                for (int j = 0; j < chunk_size; ++j) {
                    h_eq.push_back(probing_queries[i + j].first);
                    h_epi.push_back(probing_queries[i + j].second);
                }
                q_offset += chunk_size;
            }
        }
        page_offset_global += num_pages_for_c;
    }
    int n_entry = h_cluster_page_table.size();

    COUT_ENDL("Setup: Queries=", n_query, ", Entries=", n_entry, ", Layout=Interleaved");

    // 5. Host  Layout
    std::vector<float> h_query(n_query * n_dim);
    std::vector<float> h_query_norm(n_query);
    init_array_multithreaded<float>(h_query.data(), h_query.size(), 123u, -1.0f, 1.0f, 0);
    init_array_multithreaded<float>(h_query_norm.data(), h_query_norm.size(), 789u, 0.5f, 2.0f, 0);

    //  Row-major
    std::vector<float> h_vectors_rm(total_pages * 256 * n_dim);
    init_array_multithreaded<float>(h_vectors_rm.data(), h_vectors_rm.size(), 456u, -1.0f, 1.0f, 0);

    //  Interleaved
    std::vector<float> h_vectors_interleaved(total_pages * 256 * n_dim, 0.0f);
    for (int p = 0; p < total_pages; ++p) {
        paraselect::pack_rowmajor_to_interleaved(
            h_vectors_rm.data() + p * 256 * n_dim,
            h_vectors_interleaved.data() + p * 256 * n_dim,
            256, n_dim
        );
    }

    // 6.  Device
    float *d_query, *d_vectors, *d_qnorm;
    int *d_pvoc, *d_pvc, *d_cpt, *d_ecid, *d_epid, *d_eqs, *d_eqc, *d_eq, *d_epi;
    float *d_p_dist, *d_pr_dist, *d_f_dist;
    int *d_p_idx, *d_pr_idx, *d_mutex, *d_f_idx;

    cudaMalloc(&d_query, h_query.size() * sizeof(float));
    cudaMalloc(&d_vectors, h_vectors_interleaved.size() * sizeof(float));
    cudaMalloc(&d_qnorm, h_query_norm.size() * sizeof(float));

    cudaMalloc(&d_pvoc, h_pvoc.size() * sizeof(int));
    cudaMalloc(&d_pvc, h_pvc.size() * sizeof(int));
    cudaMalloc(&d_cpt, h_cluster_page_table.size() * sizeof(int));
    cudaMalloc(&d_ecid, h_ecid.size() * sizeof(int));
    cudaMalloc(&d_epid, h_epid.size() * sizeof(int));
    cudaMalloc(&d_eqs, h_eqs.size() * sizeof(int));
    cudaMalloc(&d_eqc, h_eqc.size() * sizeof(int));
    cudaMalloc(&d_eq, h_eq.size() * sizeof(int));
    cudaMalloc(&d_epi, h_epi.size() * sizeof(int));

    cudaMalloc(&d_p_dist, (size_t)n_entry * QPB * k * sizeof(float));
    cudaMalloc(&d_p_idx, (size_t)n_entry * QPB * k * sizeof(int));
    cudaMalloc(&d_pr_dist, (size_t)n_query * n_probes * k * sizeof(float));
    cudaMalloc(&d_pr_idx, (size_t)n_query * n_probes * k * sizeof(int));
    cudaMalloc(&d_mutex, (size_t)n_query * n_probes * sizeof(int));
    cudaMalloc(&d_f_dist, (size_t)n_query * k * sizeof(float));
    cudaMalloc(&d_f_idx, (size_t)n_query * k * sizeof(int));

    cudaMemcpy(d_query, h_query.data(), h_query.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vectors, h_vectors_interleaved.data(), h_vectors_interleaved.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qnorm, h_query_norm.data(), h_query_norm.size() * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_pvoc, h_pvoc.data(), h_pvoc.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pvc, h_pvc.data(), h_pvc.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cpt, h_cluster_page_table.data(), h_cluster_page_table.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ecid, h_ecid.data(), h_ecid.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_epid, h_epid.data(), h_epid.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eqs, h_eqs.data(), h_eqs.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eqc, h_eqc.data(), h_eqc.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eq, h_eq.data(), h_eq.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_epi, h_epi.data(), h_epi.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaStream_t st;
    cudaStreamCreate(&st);

    COUT_ENDL("Kernel launch...");
    double gpu_ms = 0.0;

    MEASURE_MS_AND_SAVE("gpu", gpu_ms,
        paraselect::launch_gemv_page_l2(
            n_dim, n_entry, n_probes, k,
            d_query, d_vectors, d_cpt,
            d_pvoc, d_pvc, d_ecid, d_epid, d_eqs, d_eqc, d_eq, d_epi, d_qnorm,
            n_query,
            d_p_dist, d_p_idx, d_pr_dist, d_pr_idx, d_mutex, d_f_dist, d_f_idx, st);
        cudaStreamSynchronize(st);
    );

    COUT_ENDL("gpu_ms:", gpu_ms);

    //  GPU
    std::vector<float> gpu_dist(n_query * k);
    std::vector<int> gpu_index(n_query * k);
    cudaMemcpy(gpu_dist.data(), d_f_dist, gpu_dist.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_index.data(), d_f_idx, gpu_index.size() * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<float> gt_dist(n_query * k);
    std::vector<int> gt_index(n_query * k);
    double cpu_ms = 0.0;
    MEASURE_MS_AND_SAVE("cpu", cpu_ms,
        cpu_ground_truth(n_query, n_dim, k, n_probes, h_query.data(), h_query_norm.data(), h_vectors_rm.data(),
                     h_pvoc, h_pvc, phys_pages, query_to_clusters, gt_dist, gt_index);
    );

    //
    float** cpu_dist_rows = malloc_vector_2D<float>(n_query, k);
    float** gpu_dist_rows = malloc_vector_2D<float>(n_query, k);
    int** cpu_idx_rows = malloc_vector_2D<int>(n_query, k);
    int** gpu_idx_rows = malloc_vector_2D<int>(n_query, k);

    for (int q = 0; q < n_query; ++q) {
        for (int i = 0; i < k; ++i) {
            cpu_dist_rows[q][i] = gt_dist[q * k + i];
            gpu_dist_rows[q][i] = gpu_dist[q * k + i];
            cpu_idx_rows[q][i] = gt_index[q * k + i];
            gpu_idx_rows[q][i] = gpu_index[q * k + i];
        }
    }

    auto mismatches = compare_set_2D_with_mismatches<float>(
        cpu_dist_rows, gpu_dist_rows,
        cpu_idx_rows, gpu_idx_rows,
        n_query, k, EPSILON);

    int num_mismatches = mismatches.size();
    double pass_rate = (num_mismatches == 0) ? 1.0 : 0.0;
    double speedup = (gpu_ms > 1e-6) ? (cpu_ms / gpu_ms) : 0.0;

    free_vector<float>(cpu_dist_rows);
    free_vector<float>(gpu_dist_rows);
    free_vector<int>(cpu_idx_rows);
    free_vector<int>(gpu_idx_rows);

    cudaFree(d_query); cudaFree(d_vectors); cudaFree(d_qnorm);
    cudaFree(d_pvoc); cudaFree(d_pvc); cudaFree(d_cpt);
    cudaFree(d_ecid); cudaFree(d_epid); cudaFree(d_eqs); cudaFree(d_eqc); cudaFree(d_eq); cudaFree(d_epi);
    cudaFree(d_p_dist); cudaFree(d_p_idx); cudaFree(d_pr_dist); cudaFree(d_pr_idx); cudaFree(d_mutex);
    cudaFree(d_f_dist); cudaFree(d_f_idx);
    cudaStreamDestroy(st);

    return {pass_rate, (double)num_mismatches, cpu_ms, gpu_ms, speedup};
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    MetricsCollector metrics;
    metrics.set_columns("pass_rate", "mismatches", "n_query", "n_cluster", "n_dim", "k", "n_probes", "cpu_ms", "gpu_ms", "speedup");
    metrics.set_num_repeats(1);

    cudaSetDevice(0);
    COUT_ENDL("test_gemv_interleaved_page (L2) - Interleaved Layout  ...");

    // :

    // :
    PARAM_3D(n_query, (16, 64, 256),
             n_cluster, (10, 20, 100),
            //  n_cluster, (10, 20),
             n_dim, (64, 128, 256)){
            //  n_dim, (64)){
            PARAM_2D(k, (10, 50),
             n_probes, (2, 10))
        {
            auto r = run_case(n_query, n_cluster, n_dim, k, n_probes);
            metrics.add_row(r[0], r[1], (double)n_query, (double)n_cluster, (double)n_dim,
                            (double)k, (double)n_probes, r[2], r[3], r[4]);
        }
    }
    cudaDeviceSynchronize();
    metrics.print_table();
    metrics.export_csv("gemv_interleaved_page_metrics.csv");

    bool all_pass = true;
    for (size_t i = 0; i < metrics.size(); ++i) {
        auto row = metrics.get_row(i);
        if (row.size() > 0 && row[0] < 1.0) all_pass = false;
    }
    COUT_ENDL(all_pass ? "all_test_passed" : "some tests failed");
    return all_pass ? 0 : 1;
}
