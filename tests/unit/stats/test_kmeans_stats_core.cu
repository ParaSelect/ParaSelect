/**
 * KMeans
 *  compute_kmeans_residual_stats
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "src/utils/pch.h"
#include "src/clustering/kmeans/kmeans_stats.cuh"
#include "common/test_context.cuh"

int main() {
    paraselect::test::TestContext ctx("kmeans_stats_core", paraselect::test::TestKind::Unit);
    cudaSetDevice(0);
    (void)cudaGetLastError();

    ctx.run_case("two_cluster_residual_stats", [&] {
        float h_data[] = {
            0.0f, 0.0f,
            1.0f, 1.0f,
            10.0f, 10.0f,
            11.0f, 11.0f,
        };
        float h_centroids[] = {
            0.5f, 0.5f,
            10.5f, 10.5f,
        };
        int h_assign[] = {0, 0, 1, 1};

        KMeansStatsResult result;
        compute_kmeans_residual_stats(
            h_data, h_centroids, h_assign,
            4, 2, 2,
            L2_DISTANCE, &result, 10);

        ctx.check_near(result.per_cluster[0].mean, 0.5, 1e-6, "cluster 0 mean");
        ctx.check_near(result.per_cluster[0].variance, 0.0, 1e-6, "cluster 0 variance");
        ctx.check_near(result.per_cluster[1].mean, 0.5, 1e-6, "cluster 1 mean");
        ctx.check_near(result.global.mean, 0.5, 1e-6, "global mean");
    });

    return ctx.exit_code();
}
