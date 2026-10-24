/**
 * SIFT dataset reader integration test.
 * Paths come from environment variables; the row limit keeps this a correctness test.
 */

#include "dataset/common/dataset_source.h"
#include "common/test_context.cuh"
#include "utils/check_cuda_error.cuh"
#include <cstdio>
#include <chrono>

using namespace paraselect::dataset;

int main() {
    paraselect::test::TestContext ctx("sift_dataset_io", paraselect::test::TestKind::Integration);
    ctx.add_tag("dataset");
    ctx.add_tag("io");
    ctx.add_tag("sift");

    const auto source = open_dataset("sift", "1M");

    const size_t N_TRAIN = 10000;
    const size_t N_QUERY = 10000;
    ctx.add_param("n_train", static_cast<long long>(N_TRAIN));
    ctx.add_param("n_query", static_cast<long long>(N_QUERY));

    // 1.
    VectorBuffer train;
    DatasetInfo train_info;
    VectorBuffer query;
    DatasetInfo query_info;
    GroundTruth gt;

    ctx.run_case("load_train", [&] {
        auto start = std::chrono::high_resolution_clock::now();
        train = source.load_base(N_TRAIN);
        train_info = {train.view().n, train.view().dim, train.view().bytes()};
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        ctx.check_eq(train_info.n, N_TRAIN, "train vector count");
        ctx.check_eq(train_info.dim, 128, "SIFT dimension");
        ctx.check(!train.empty(), "train pinned buffer allocated");
        ctx.add_metric("train_load_seconds", elapsed.count());
        ctx.add_metric("train_load_mbps", (train_info.data_size / (1024.0 * 1024.0)) / elapsed.count());
    });

    ctx.run_case("load_query", [&] {
        auto start = std::chrono::high_resolution_clock::now();
        query = source.load_queries(N_QUERY);
        query_info = {query.view().n, query.view().dim, query.view().bytes()};
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        ctx.check_eq(query_info.n, N_QUERY, "query vector count");
        ctx.check_eq(query_info.dim, 128, "query dimension");
        ctx.check(!query.empty(), "query pinned buffer allocated");
        ctx.add_metric("query_load_seconds", elapsed.count());
    });

    ctx.run_case("load_ground_truth", [&] {
        try {
            gt = source.load_groundtruth();
            ctx.check(gt.n > 0, "ground truth query count");
            ctx.check(gt.k > 0, "ground truth neighbor count");
            ctx.add_metric("ground_truth_queries", static_cast<double>(gt.n));
            ctx.add_metric("ground_truth_dim", static_cast<double>(gt.k));
        } catch (const std::exception& e) {
            ctx.skip(std::string("ground truth optional: ") + e.what());
        }
    });

    ctx.run_case("h2d_transfer", [&] {
        PARASELECT_REQUIRE(ctx, !train.empty());
        float* d_train = nullptr;
        CUDA_CHECK(cudaMalloc(&d_train, train_info.data_size));

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        auto start = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpyAsync(d_train, train.view().data, train_info.data_size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        ctx.add_metric("h2d_seconds", elapsed.count());
        ctx.add_metric("h2d_gbps", (train_info.data_size / (1024.0 * 1024.0 * 1024.0)) / elapsed.count());
        ctx.check(elapsed.count() > 0.0, "H2D transfer completed");

        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(d_train));
    });

    return ctx.exit_code();
}
