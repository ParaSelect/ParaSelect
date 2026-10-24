#include "quantization/pq/residual_pq_trainer.cuh"
#include "common/test_utils.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using paraselect::quantization::pq::ResidualPQCase;
using paraselect::quantization::pq::ResidualPQInput;
using paraselect::quantization::pq::ResidualPQTrainerCPU;

bool all_finite(const std::vector<float>& values) {
    for (float v : values) {
        if (!std::isfinite(v)) return false;
    }
    return true;
}

bool codes_in_range(const std::vector<uint8_t>& codes, int k) {
    for (uint8_t c : codes) {
        if ((int)c >= k) return false;
    }
    return true;
}

float encoded_residual_sse(
    const float* vec,
    const float* centroid,
    const float* codebook,
    const uint8_t* code,
    int dim,
    int m,
    int k,
    int d_sub)
{
    (void)k;
    float sse = 0.0f;
    for (int sub = 0; sub < m; ++sub) {
        const int base = sub * d_sub;
        const float* cb = codebook +
            ((size_t)sub * (size_t)k + (size_t)code[sub]) * (size_t)d_sub;
        for (int d = 0; d < d_sub; ++d) {
            const float residual = vec[base + d] - centroid[base + d];
            const float diff = residual - cb[d];
            sse += diff * diff;
        }
    }
    (void)dim;
    return sse;
}

void run_basic_fit_encode(paraselect::test::TestContext& ctx) {
    const int n = 12;
    const int dim = 4;
    const int nlist = 2;
    const int m = 2;
    const int k = 4;
    const int d_sub = dim / m;

    std::vector<float> data = {
        0.0f, 0.1f, 1.0f, 1.1f,
        0.2f, 0.0f, 1.1f, 0.9f,
        -0.1f, 0.2f, 0.9f, 1.2f,
        0.3f, -0.1f, 1.2f, 1.0f,
        -0.2f, -0.2f, 0.8f, 0.8f,
        0.1f, 0.3f, 1.0f, 1.3f,
        10.0f, 9.9f, -1.0f, -1.1f,
        10.2f, 10.0f, -0.9f, -1.2f,
        9.8f, 10.1f, -1.2f, -1.0f,
        10.1f, 9.7f, -0.8f, -1.3f,
        9.9f, 10.3f, -1.1f, -0.9f,
        10.3f, 9.8f, -1.3f, -1.1f,
    };
    std::vector<float> centroids = {
        0.0f, 0.0f, 1.0f, 1.0f,
        10.0f, 10.0f, -1.0f, -1.0f,
    };
    std::vector<idx_t> offsets = {0, 6, 12};
    std::vector<int> counts = {6, 6};

    ResidualPQCase cfg;
    cfg.n = n;
    cfg.dim = dim;
    cfg.nlist = nlist;
    cfg.m = m;
    cfg.k = k;
    cfg.iters = 3;
    cfg.num_threads = 2;

    ResidualPQInput input;
    input.data = data.data();
    input.coarse_centroids = centroids.data();
    input.cluster_offsets = offsets.data();
    input.cluster_counts = counts.data();

    ResidualPQTrainerCPU trainer(cfg);
    const auto result = trainer.fit_encode(input);

    ctx.check_eq(result.m, m, "ResidualPQResult.m");
    ctx.check_eq(result.k, k, "ResidualPQResult.k");
    ctx.check_eq(result.d_sub, d_sub, "ResidualPQResult.d_sub");
    ctx.check_eq((long long)result.codebook.size(), (long long)m * k * d_sub, "codebook size");
    ctx.check_eq((long long)result.codes.size(), (long long)n * m, "codes size");
    ctx.check(all_finite(result.codebook), "codebook values should be finite");
    ctx.check(codes_in_range(result.codes, k), "codes should be in [0, k)");
    ctx.check(result.fit_ms >= 0.0, "fit_ms should be non-negative");
    ctx.check(result.encode_ms >= 0.0, "encode_ms should be non-negative");
    ctx.check(result.total_ms >= 0.0, "total_ms should be non-negative");

    float total_sse = 0.0f;
    for (int cid = 0; cid < nlist; ++cid) {
        for (int j = 0; j < counts[cid]; ++j) {
            const int ridx = (int)offsets[cid] + j;
            total_sse += encoded_residual_sse(
                data.data() + (size_t)ridx * dim,
                centroids.data() + (size_t)cid * dim,
                result.codebook.data(),
                result.codes.data() + (size_t)ridx * m,
                dim,
                m,
                k,
                d_sub);
        }
    }
    ctx.check(std::isfinite(total_sse), "encoded residual SSE should be finite");
    ctx.check(total_sse < 10.0f, "encoded residual SSE should be bounded for the synthetic case");
}

void run_split_fit_then_encode(paraselect::test::TestContext& ctx) {
    const int n = 8;
    const int dim = 4;
    const int nlist = 2;
    const int m = 2;
    const int k = 4;
    const int d_sub = dim / m;

    std::vector<float> data = {
        1.0f, 1.0f, 0.0f, 0.0f,
        1.1f, 0.9f, 0.1f, -0.1f,
        0.8f, 1.2f, -0.2f, 0.2f,
        1.2f, 0.8f, 0.2f, -0.2f,
        -1.0f, -1.0f, 2.0f, 2.0f,
        -0.9f, -1.1f, 1.9f, 2.1f,
        -1.2f, -0.8f, 2.2f, 1.8f,
        -0.8f, -1.2f, 1.8f, 2.2f,
    };
    std::vector<float> centroids = {
        1.0f, 1.0f, 0.0f, 0.0f,
        -1.0f, -1.0f, 2.0f, 2.0f,
    };
    std::vector<idx_t> offsets = {0, 4, 8};
    std::vector<int> counts = {4, 4};

    ResidualPQCase cfg;
    cfg.n = n;
    cfg.dim = dim;
    cfg.nlist = nlist;
    cfg.m = m;
    cfg.k = k;
    cfg.iters = 2;
    cfg.num_threads = 1;

    ResidualPQInput input;
    input.data = data.data();
    input.coarse_centroids = centroids.data();
    input.cluster_offsets = offsets.data();
    input.cluster_counts = counts.data();

    ResidualPQTrainerCPU trainer(cfg);
    std::vector<float> codebook((size_t)m * k * d_sub);
    std::vector<uint8_t> codes((size_t)n * m);
    trainer.fit(input, codebook.data());
    trainer.encode(input, codebook.data(), codes.data());

    ctx.check(all_finite(codebook), "split fit codebook should be finite");
    ctx.check(codes_in_range(codes, k), "split encode codes should be in [0, k)");
}

void run_validation_case(paraselect::test::TestContext& ctx) {
    ResidualPQCase cfg;
    cfg.n = 4;
    cfg.dim = 5;
    cfg.nlist = 1;
    cfg.m = 2;
    cfg.k = 4;
    cfg.iters = 1;
    cfg.num_threads = 1;

    std::vector<float> data((size_t)cfg.n * cfg.dim, 0.0f);
    std::vector<float> centroids((size_t)cfg.nlist * cfg.dim, 0.0f);
    std::vector<idx_t> offsets = {0, 4};
    std::vector<int> counts = {4};

    ResidualPQInput input;
    input.data = data.data();
    input.coarse_centroids = centroids.data();
    input.cluster_offsets = offsets.data();
    input.cluster_counts = counts.data();

    bool threw = false;
    try {
        ResidualPQTrainerCPU trainer(cfg);
        (void)trainer.fit_encode(input);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ctx.check(threw, "invalid dim % m should throw");
}

}  // namespace

int main() {
    paraselect::test::TestContext ctx(
        "residual_pq_trainer",
        paraselect::test::TestKind::Unit);

    ctx.run_case("basic_fit_encode", [&] {
        run_basic_fit_encode(ctx);
    });
    ctx.run_case("split_fit_then_encode", [&] {
        run_split_fit_then_encode(ctx);
    });
    ctx.run_case("validation", [&] {
        run_validation_case(ctx);
    });

    return ctx.exit_code();
}
