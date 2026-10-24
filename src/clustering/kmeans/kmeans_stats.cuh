#ifndef KMEANS_STATS_CUH
#define KMEANS_STATS_CUH

#include "utils/pch.h"
#include <vector>
#include <cmath>

// ============================================================
// KMeans (, IVF )
// ============================================================

/**
 *
 */
struct ClusterResidualStats {
    int cluster_id;       //  ID
    int count;            //
    double mean;          //
    double variance;      //
    double min_val;       //
    double max_val;       //
    // ---- (,) ----
    double mean_radius;   //
    double std_radius;    //
    double q95_radius;    // 95
    double q99_radius;    // 99
    bool valid;           // (count >= min_cluster_size)
};

/**
 *
 */
struct GlobalResidualStats {
    double mean;          //
    double variance;      //
    double min_val;       //
    double max_val;       //
    int empty_clusters;   //
};

/**
 *
 */
struct ResidualHistogram {
    int num_bins;                    // bin
    double bin_width;                //  bin
    double min_val;                  //
    std::vector<int> counts;         // [num_bins]  bin
    std::vector<double> bin_edges;   // [num_bins+1] bin
};

/**
 *
 */
struct KMeansStatsResult {
    std::vector<ClusterResidualStats> per_cluster;  // [k]
    GlobalResidualStats global;                       //
    ResidualHistogram histogram;                      //
    int n_vectors;
    int dim;
    int k;
};

// ============================================================
//
// ============================================================

/**
 *  KMeans
 *
 * :
 *   - h_data:  [n, dim] row-major(CPU pinned memory)
 *   - h_centroids:  [k, dim](CPU memory)
 *   - h_assign:  [n](CPU memory)
 *   - n, dim, k:
 *   - dist_type:
 *   - num_bins:  bin ( 50)
 *
 * :
 *   - result: (CPU )
 */
void compute_kmeans_residual_stats(
    const float* h_data,        // [n, dim] CPU
    const float* h_centroids,   // [k, dim] CPU
    const int* h_assign,        // [n] CPU
    int n,
    int dim,
    int k,
    DistanceType dist_type,
    KMeansStatsResult* result,
    int num_bins = 50
);

/**
 *
 */
void print_kmeans_stats(const KMeansStatsResult& result);

/**
 *  CSV()
 */
void export_kmeans_stats_to_csv(
    const KMeansStatsResult& result,
    const std::string& filepath
);

#endif // KMEANS_STATS_CUH
