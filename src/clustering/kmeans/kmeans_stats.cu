#include "clustering/kmeans/kmeans_stats.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>
#include <thread>
#include <mutex>
#include <atomic>

// ---------------------------------------------------------------------------
// CPU :
// ---------------------------------------------------------------------------
static inline float compute_distance(
    const float* vec,
    const float* centroid,
    int dim,
    DistanceType dist_type)
{
    if (dist_type == L2_DISTANCE) {
        float sum = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = vec[d] - centroid[d];
            sum += diff * diff;
        }
        return sum;
    } else {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (int d = 0; d < dim; ++d) {
            dot += vec[d] * centroid[d];
            norm_a += vec[d] * vec[d];
            norm_b += centroid[d] * centroid[d];
        }
        float denom = std::sqrt(norm_a * norm_b);
        if (denom < 1e-8f) return 1.0f;
        return 1.0f - dot / denom;
    }
}

// ---------------------------------------------------------------------------
// :
// ---------------------------------------------------------------------------
static void collect_residuals_mt(
    const float* h_data,
    const float* h_centroids,
    const int* h_assign,
    int n, int dim, int k,
    DistanceType dist_type,
    std::vector<std::vector<double>>& cluster_residuals,
    std::vector<double>& all_residuals)
{
    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 4;

    //
    std::vector<std::vector<std::vector<double>>> local_cluster_residuals(n_threads,
        std::vector<std::vector<double>>(k));
    std::vector<std::vector<double>> local_all_residuals(n_threads);

    std::vector<std::thread> threads;
    int chunk = (n + (int)n_threads - 1) / (int)n_threads;

    for (unsigned int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            int start = (int)t * chunk;
            if (start >= n) return;
            int end = std::min(start + chunk, n);
            auto& local_c = local_cluster_residuals[t];
            auto& local_a = local_all_residuals[t];
            local_a.reserve(std::max(0, end - start));

            for (int i = start; i < end; ++i) {
                int cid = h_assign[i];
                if (cid < 0 || cid >= k) continue;
                const float* vec = h_data + (size_t)i * dim;
                const float* cent = h_centroids + (size_t)cid * dim;
                float dist = compute_distance(vec, cent, dim, dist_type);
                local_c[cid].push_back(dist);
                local_a.push_back(dist);
            }
        });
    }

    for (auto& th : threads) th.join();

    //
    all_residuals.reserve(n);
    for (unsigned int t = 0; t < n_threads; ++t) {
        all_residuals.insert(all_residuals.end(),
                             local_all_residuals[t].begin(),
                             local_all_residuals[t].end());
        for (int c = 0; c < k; ++c) {
            cluster_residuals[c].insert(cluster_residuals[c].end(),
                                        local_cluster_residuals[t][c].begin(),
                                        local_cluster_residuals[t][c].end());
        }
    }
}

// ---------------------------------------------------------------------------
// :( nth_element,)
// ---------------------------------------------------------------------------
static double compute_quantile(std::vector<double>& data, double q) {
    if (data.empty()) return 0.0;
    size_t idx = (size_t)(q * (data.size() - 1));
    std::nth_element(data.begin(), data.begin() + idx, data.end());
    return data[idx];
}

// ---------------------------------------------------------------------------
// :()
// ---------------------------------------------------------------------------
static void compute_cluster_stats_mt(
    const std::vector<std::vector<double>>& cluster_residuals,
    int k,
    std::vector<ClusterResidualStats>& per_cluster,
    double& global_sum,
    double& global_min,
    double& global_max,
    int& empty_count,
    int min_cluster_size = 32)
{
    global_sum = 0.0;
    global_min = 1e300;
    global_max = -1e300;
    empty_count = 0;

    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 4;

    std::vector<std::thread> threads;
    std::mutex mtx;
    int chunk = (k + (int)n_threads - 1) / (int)n_threads;

    for (unsigned int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            int start = (int)t * chunk;
            int end = std::min(start + chunk, k);
            double local_sum = 0.0;
            double local_min = 1e300;
            double local_max = -1e300;
            int local_empty = 0;

            for (int cid = start; cid < end; ++cid) {
                auto& stats = per_cluster[cid];
                stats.cluster_id = cid;
                stats.count = (int)cluster_residuals[cid].size();

                if (stats.count == 0) {
                    local_empty++;
                    stats.mean = 0.0;
                    stats.variance = 0.0;
                    stats.min_val = 0.0;
                    stats.max_val = 0.0;
                    stats.mean_radius = 0.0;
                    stats.std_radius = 0.0;
                    stats.q95_radius = 0.0;
                    stats.q99_radius = 0.0;
                    stats.valid = false;
                    continue;
                }

                // ---- () ----
                double sum = 0.0;
                double min_v = 1e300;
                double max_v = -1e300;
                for (double v : cluster_residuals[cid]) {
                    sum += v;
                    if (v < min_v) min_v = v;
                    if (v > max_v) max_v = v;
                }
                stats.mean = sum / stats.count;
                stats.min_val = min_v;
                stats.max_val = max_v;

                double var_sum = 0.0;
                for (double v : cluster_residuals[cid]) {
                    double diff = v - stats.mean;
                    var_sum += diff * diff;
                }
                stats.variance = var_sum / stats.count;

                // ---- ( = sqrt()) ----
                std::vector<double> radii;
                radii.reserve(stats.count);
                double radius_sum = 0.0;
                double radius_sq_sum = 0.0;
                for (double v : cluster_residuals[cid]) {
                    double r = std::sqrt(std::max(v, 0.0));
                    radii.push_back(r);
                    radius_sum += r;
                    radius_sq_sum += r * r;
                }
                stats.mean_radius = radius_sum / stats.count;
                double radius_var = (radius_sq_sum / stats.count) - (stats.mean_radius * stats.mean_radius);
                stats.std_radius = std::sqrt(std::max(radius_var, 0.0));

                // ( nth_element)
                std::vector<double> radii_copy = radii;
                stats.q95_radius = compute_quantile(radii_copy, 0.95);
                radii_copy = radii;
                stats.q99_radius = compute_quantile(radii_copy, 0.99);

                //
                stats.valid = (stats.count >= min_cluster_size);

                local_sum += sum;
                if (min_v < local_min) local_min = min_v;
                if (max_v > local_max) local_max = max_v;
            }

            std::lock_guard<std::mutex> lock(mtx);
            global_sum += local_sum;
            if (local_min < global_min) global_min = local_min;
            if (local_max > global_max) global_max = local_max;
            empty_count += local_empty;
        });
    }

    for (auto& th : threads) th.join();
}

// ---------------------------------------------------------------------------
// : +
// ---------------------------------------------------------------------------
static void compute_global_var_and_histogram_mt(
    const std::vector<double>& all_residuals,
    double global_mean,
    int num_bins,
    double global_min,
    double global_max,
    double& global_variance,
    std::vector<int>& counts,
    std::vector<double>& bin_edges)
{
    int n = (int)all_residuals.size();
    if (n == 0 || num_bins <= 0) return;

    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 4;

    //
    std::vector<double> local_var_sums(n_threads, 0.0);
    int chunk = (n + (int)n_threads - 1) / (int)n_threads;
    std::vector<std::thread> threads;

    for (unsigned int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            int start = (int)t * chunk;
            int end = std::min(start + chunk, n);
            double sum = 0.0;
            for (int i = start; i < end; ++i) {
                double diff = all_residuals[i] - global_mean;
                sum += diff * diff;
            }
            local_var_sums[t] = sum;
        });
    }
    for (auto& th : threads) th.join();

    double total_var_sum = 0.0;
    for (double s : local_var_sums) total_var_sum += s;
    global_variance = total_var_sum / n;

    //
    double range = global_max - global_min;
    if (range < 1e-12) range = 1.0;
    double bin_width = range / num_bins;

    for (int b = 0; b <= num_bins; ++b) {
        bin_edges[b] = global_min + b * bin_width;
    }

    std::fill(counts.begin(), counts.end(), 0);
    std::vector<std::vector<int>> local_counts(n_threads, std::vector<int>(num_bins, 0));
    threads.clear();

    for (unsigned int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            int start = (int)t * chunk;
            int end = std::min(start + chunk, n);
            for (int i = start; i < end; ++i) {
                int bin = (int)((all_residuals[i] - global_min) / bin_width);
                if (bin < 0) bin = 0;
                if (bin >= num_bins) bin = num_bins - 1;
                local_counts[t][bin]++;
            }
        });
    }
    for (auto& th : threads) th.join();

    for (unsigned int t = 0; t < n_threads; ++t) {
        for (int b = 0; b < num_bins; ++b) {
            counts[b] += local_counts[t][b];
        }
    }
}

// ---------------------------------------------------------------------------
// :()
// ---------------------------------------------------------------------------
void compute_kmeans_residual_stats(
    const float* h_data,
    const float* h_centroids,
    const int* h_assign,
    int n,
    int dim,
    int k,
    DistanceType dist_type,
    KMeansStatsResult* result,
    int num_bins)
{
    if (!result) return;

    result->n_vectors = n;
    result->dim = dim;
    result->k = k;
    result->per_cluster.resize(k);
    result->histogram.num_bins = num_bins;
    result->histogram.counts.assign(num_bins, 0);
    result->histogram.bin_edges.resize(num_bins + 1);

    // :
    std::vector<std::vector<double>> cluster_residuals(k);
    std::vector<double> all_residuals;
    collect_residuals_mt(h_data, h_centroids, h_assign,
                         n, dim, k, dist_type,
                         cluster_residuals, all_residuals);

    // :
    double global_sum = 0.0;
    double global_min = 1e300;
    double global_max = -1e300;
    int empty_count = 0;

    compute_cluster_stats_mt(cluster_residuals, k,
                             result->per_cluster,
                             global_sum, global_min, global_max, empty_count,
                             32);

    // :
    result->global.mean = n > 0 ? global_sum / n : 0.0;
    result->global.min_val = global_min;
    result->global.max_val = global_max;
    result->global.empty_clusters = empty_count;

    // : +
    result->histogram.min_val = global_min;
    result->histogram.bin_width = (global_max - global_min) / num_bins;

    compute_global_var_and_histogram_mt(
        all_residuals, result->global.mean, num_bins,
        global_min, global_max,
        result->global.variance,
        result->histogram.counts,
        result->histogram.bin_edges);
}

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
void print_kmeans_stats(const KMeansStatsResult& result)
{
    std::printf("\n========== KMeans Residual Stats ==========\n");
    std::printf("n_vectors=%d  dim=%d  k=%d\n", result.n_vectors, result.dim, result.k);
    std::printf("\n--- Global ---\n");
    std::printf("  mean=%.6f  variance=%.6f  min=%.6f  max=%.6f\n",
                result.global.mean, result.global.variance,
                result.global.min_val, result.global.max_val);
    std::printf("  empty_clusters=%d\n", result.global.empty_clusters);

    std::printf("\n--- Per Cluster (first 10) ---\n");
    int print_n = std::min((int)result.per_cluster.size(), 10);
    for (int i = 0; i < print_n; ++i) {
        const auto& c = result.per_cluster[i];
        std::printf("  cid=%d  count=%d  mean=%.6f  var=%.6f  min=%.6f  max=%.6f  "
                    "mean_r=%.6f  std_r=%.6f  q95_r=%.6f  q99_r=%.6f  valid=%d\n",
                    c.cluster_id, c.count, c.mean, c.variance, c.min_val, c.max_val,
                    c.mean_radius, c.std_radius, c.q95_radius, c.q99_radius,
                    c.valid ? 1 : 0);
    }
    if ((int)result.per_cluster.size() > print_n) {
        std::printf("  ... (%d more clusters)\n", (int)result.per_cluster.size() - print_n);
    }

    std::printf("\n--- Histogram ---\n");
    std::printf("  bins=%d  width=%.6f\n", result.histogram.num_bins, result.histogram.bin_width);
    int print_bins = std::min(result.histogram.num_bins, 20);
    for (int i = 0; i < print_bins; ++i) {
        std::printf("  [%.4f, %.4f): %d\n",
                    result.histogram.bin_edges[i],
                    result.histogram.bin_edges[i + 1],
                    result.histogram.counts[i]);
    }
    if (result.histogram.num_bins > print_bins) {
        std::printf("  ... (%d more bins)\n", result.histogram.num_bins - print_bins);
    }
    std::printf("============================================\n\n");
}

// ---------------------------------------------------------------------------
//  CSV
// ---------------------------------------------------------------------------
void export_kmeans_stats_to_csv(
    const KMeansStatsResult& result,
    const std::string& filepath)
{
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        std::cerr << "[ERROR] Failed to open " << filepath << std::endl;
        return;
    }

    ofs << "n_vectors," << result.n_vectors << "\n";
    ofs << "dim," << result.dim << "\n";
    ofs << "k," << result.k << "\n";
    ofs << "global_mean," << result.global.mean << "\n";
    ofs << "global_variance," << result.global.variance << "\n";
    ofs << "global_min," << result.global.min_val << "\n";
    ofs << "global_max," << result.global.max_val << "\n";
    ofs << "empty_clusters," << result.global.empty_clusters << "\n";
    ofs << "\n";

    ofs << "cluster_id,count,mean,variance,min,max,mean_radius,std_radius,q95_radius,q99_radius,valid\n";
    for (const auto& c : result.per_cluster) {
        ofs << c.cluster_id << ","
            << c.count << ","
            << c.mean << ","
            << c.variance << ","
            << c.min_val << ","
            << c.max_val << ","
            << c.mean_radius << ","
            << c.std_radius << ","
            << c.q95_radius << ","
            << c.q99_radius << ","
            << (c.valid ? 1 : 0) << "\n";
    }

    ofs << "\n";
    ofs << "bin_start,bin_end,count\n";
    for (int i = 0; i < result.histogram.num_bins; ++i) {
        ofs << result.histogram.bin_edges[i] << ","
            << result.histogram.bin_edges[i + 1] << ","
            << result.histogram.counts[i] << "\n";
    }

    ofs.close();
    std::printf("[Stats] Exported to %s\n", filepath.c_str());
}
