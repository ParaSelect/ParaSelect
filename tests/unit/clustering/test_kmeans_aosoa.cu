#include "dataset/dataset_aosoa.cuh"
#include "test_context.cuh"
#include "search/kernels/fine/cpu_fine/cpu_fine_layout.h"
#include "search/kernels/fine/cpu_fine/cpu_fine_layout_aosoa.h"
#include "utils/pch.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static float make_value(int row, int dim) {
    return static_cast<float>(row * 100 + dim);
}

static bool expect_eq(float actual, float expected, const char* label) {
    if (std::fabs(actual - expected) <= 1e-6f) return true;
    std::fprintf(stderr, "[FAIL] %s actual=%.6f expected=%.6f\n", label, actual, expected);
    return false;
}

static bool check_slot(
    const float* aosoa,
    long long cluster_base,
    int group,
    int d,
    int vector_dim,
    int tile,
    int lane,
    float expected,
    const char* label)
{
    const long long pos = cluster_base + (long long)group * vector_dim * tile + (long long)d * tile + lane;
    return expect_eq(aosoa[pos], expected, label);
}

int main() {
    paraselect::test::TestContext ctx("kmeans_aosoa_layout", paraselect::test::TestKind::Unit);
    ctx.add_tag("cpu_fine");
    ctx.add_tag("clustering");
    ctx.add_tag("aosoa");

    constexpr int n = 6;
    constexpr int dim = 3;
    constexpr int n_clusters = 3;
    constexpr int tile = 4;
    ctx.add_param("n", n);
    ctx.add_param("dim", dim);
    ctx.add_param("n_clusters", n_clusters);
    ctx.add_param("tile", tile);

    std::vector<float> base((size_t)n * dim);
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < dim; ++d) {
            base[(size_t)i * dim + d] = make_value(i, d);
        }
    }

    const std::vector<idx_t> reordered_indices = {
        2, 0, 5,  // cluster 0
        1, 4      // cluster 2; cluster 1 is empty
    };
    const idx_t cluster_offsets[n_clusters + 1] = {0, 3, 3, 5};
    const int cluster_counts[n_clusters] = {3, 0, 2};

    paraselect::dataset::DatasetAosoaStorage storage;
    paraselect::dataset::build_direct_aosoa_from_reordered_indices(
        &storage,
        base.data(),
        reordered_indices.data(),
        cluster_offsets,
        cluster_counts,
        n_clusters,
        dim,
        tile,
        1);

    ctx.run_case("offsets", [&] {
        const long long* aosoa_offsets = storage.offsets;
        const long long total = static_cast<long long>(storage.size(n_clusters));
        ctx.check(storage.is_valid(), "AoSoA storage is valid");
        ctx.check_eq(total, 24LL, "AoSoA padded storage size");
        ctx.check_eq(aosoa_offsets[0], 0LL, "cluster 0 offset");
        ctx.check_eq(aosoa_offsets[1], 12LL, "cluster 1 offset keeps empty cluster");
        ctx.check_eq(aosoa_offsets[2], 12LL, "cluster 2 offset");
        ctx.check_eq(aosoa_offsets[3], 24LL, "end offset");
    });

    ctx.run_case("slot_values", [&] {
        PARASELECT_REQUIRE(ctx, storage.is_valid());
        const long long* aosoa_offsets = storage.offsets;
        const float* aosoa = storage.data;
        ctx.check(check_slot(aosoa, aosoa_offsets[0], 0, 0, dim, tile, 0, make_value(2, 0), "c0 d0 lane0"), "c0 d0 lane0");
        ctx.check(check_slot(aosoa, aosoa_offsets[0], 0, 0, dim, tile, 1, make_value(0, 0), "c0 d0 lane1"), "c0 d0 lane1");
        ctx.check(check_slot(aosoa, aosoa_offsets[0], 0, 0, dim, tile, 2, make_value(5, 0), "c0 d0 lane2"), "c0 d0 lane2");
        ctx.check(check_slot(aosoa, aosoa_offsets[0], 0, 0, dim, tile, 3, 0.0f, "c0 d0 padding"), "c0 d0 padding");
        ctx.check(check_slot(aosoa, aosoa_offsets[0], 0, 1, dim, tile, 0, make_value(2, 1), "c0 d1 lane0"), "c0 d1 lane0");
        ctx.check(check_slot(aosoa, aosoa_offsets[0], 0, 2, dim, tile, 2, make_value(5, 2), "c0 d2 lane2"), "c0 d2 lane2");

        ctx.check(check_slot(aosoa, aosoa_offsets[2], 0, 0, dim, tile, 0, make_value(1, 0), "c2 d0 lane0"), "c2 d0 lane0");
        ctx.check(check_slot(aosoa, aosoa_offsets[2], 0, 0, dim, tile, 1, make_value(4, 0), "c2 d0 lane1"), "c2 d0 lane1");
        ctx.check(check_slot(aosoa, aosoa_offsets[2], 0, 1, dim, tile, 0, make_value(1, 1), "c2 d1 lane0"), "c2 d1 lane0");
        ctx.check(check_slot(aosoa, aosoa_offsets[2], 0, 2, dim, tile, 1, make_value(4, 2), "c2 d2 lane1"), "c2 d2 lane1");
        ctx.check(check_slot(aosoa, aosoa_offsets[2], 0, 2, dim, tile, 2, 0.0f, "c2 d2 padding2"), "c2 d2 padding2");
        ctx.check(check_slot(aosoa, aosoa_offsets[2], 0, 2, dim, tile, 3, 0.0f, "c2 d2 padding3"), "c2 d2 padding3");
    });

    storage.release();
    return ctx.exit_code();
}
