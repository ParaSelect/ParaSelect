#include "clustering/kmeans/kmeans_cagra_train.cuh"
#include "clustering/kmeans/kmeans_final_assign.cuh"
#include "common/test_context.cuh"
#include "utils/check_cuda_error.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

int main() {
    paraselect::test::TestContext ctx(
        "kmeans_cagra_fp32", paraselect::test::TestKind::Unit);

    constexpr int n = 4096;
    constexpr int dim = 32;
    constexpr int k = 64;
    std::vector<float> data((size_t)n * dim);
    for (int row = 0; row < n; ++row) {
        const int cluster = row % k;
        for (int d = 0; d < dim; ++d) {
            data[(size_t)row * dim + d] =
                (float)(cluster * 4 + d) + (float)(row / k % 7) * 0.01f;
        }
    }

    KMeansCagraTrainConfig cfg;
    cfg.n = n;
    cfg.dim = dim;
    cfg.k = k;
    cfg.iters = 2;
    cfg.params.train_batch = 1024;
    cfg.params.final_batch = 1024;
    cfg.params.graph_degree = 16;
    cfg.params.intermediate_graph_degree = 32;
    cfg.params.build_algo = KMeansCagraBuildAlgo::NnDescent;
    cfg.params.index_scale = 1.0f;

    ctx.run_case("train_and_assign", [&] {
        std::vector<float> centroids;
        if (!ctx.require(
                kmeans_train_centroids_cagra_fp32(cfg, data.data(), &centroids),
                "FP32 CAGRA Lloyd training failed")) {
            return;
        }
        ctx.check_eq(
            centroids.size(), (size_t)k * dim,
            "FP32 CAGRA centroid shape mismatch");
        for (float value : centroids) {
            ctx.check(std::isfinite(value), "FP32 CAGRA produced a non-finite centroid");
        }

        float* d_centroids = nullptr;
        float* d_data = nullptr;
        idx_t* d_assign = nullptr;
        CUDA_CHECK(cudaMalloc(&d_centroids, centroids.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_data, data.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_assign, (size_t)n * sizeof(idx_t)));
        CUDA_CHECK(cudaMemcpy(
            d_centroids, centroids.data(), centroids.size() * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_data, data.data(), data.size() * sizeof(float),
            cudaMemcpyHostToDevice));

        GpuCentroidAssignmentConfig assign_cfg;
        assign_cfg.strategy = KMeansFinalAssignStrategy::Cagra;
        assign_cfg.combo = GpuCentroidAssignmentCombo::CagraTopkSelect;
        assign_cfg.batch_size = 1024;
        assign_cfg.cagra_graph_degree = 16;
        assign_cfg.cagra_intermediate_graph_degree = 32;
        assign_cfg.cagra_build_algo = KMeansCagraBuildAlgo::NnDescent;
        assign_cfg.cagra_index_scale = 1.0f;
        GpuCentroidAssignmentCagraWorkspace workspace;
        KMeansCase shape{};
        shape.n = n;
        shape.dim = dim;
        shape.k = k;
        shape.dist = L2_DISTANCE;
        ctx.check(!assign_vectors_to_centroids_gpu_cagra_with_workspace(
            assign_cfg, shape, data.data(), d_centroids, workspace, d_assign),
            "Unprepared CAGRA workspace must reject search");
        if (!ctx.require(gpu_centroid_assignment_cagra_workspace_build(
                workspace, assign_cfg, d_centroids, k, dim, assign_cfg.batch_size),
                "Explicit CAGRA build failed")) {
            cudaFree(d_assign); cudaFree(d_data); cudaFree(d_centroids);
            return;
        }
        if (!ctx.require(
                assign_vectors_to_centroids_gpu_cagra_with_workspace(
                    assign_cfg, shape, data.data(), d_centroids, workspace, d_assign),
                "FP32 host CAGRA assignment failed")) {
            gpu_centroid_assignment_cagra_workspace_release(workspace);
            cudaFree(d_assign);
            cudaFree(d_data);
            cudaFree(d_centroids);
            return;
        }
        std::vector<idx_t> host_assignments(n);
        CUDA_CHECK(cudaMemcpy(
            host_assignments.data(), d_assign,
            host_assignments.size() * sizeof(idx_t),
            cudaMemcpyDeviceToHost));
        for (idx_t cluster : host_assignments) {
            ctx.check(cluster >= 0 && cluster < k, "FP32 CAGRA assignment is out of range");
        }

        if (!ctx.require(
                assign_device_vectors_to_centroids_gpu_cagra_with_workspace(
                    assign_cfg, shape, d_data, d_centroids, workspace, d_assign),
                "FP32 device CAGRA assignment failed")) {
            gpu_centroid_assignment_cagra_workspace_release(workspace);
            cudaFree(d_assign);
            cudaFree(d_data);
            cudaFree(d_centroids);
            return;
        }
        std::vector<idx_t> device_assignments(n);
        CUDA_CHECK(cudaMemcpy(
            device_assignments.data(), d_assign,
            device_assignments.size() * sizeof(idx_t),
            cudaMemcpyDeviceToHost));
        for (int row = 0; row < n; ++row) {
            ctx.check_eq(
                device_assignments[(size_t)row], host_assignments[(size_t)row],
                "FP32 host/device CAGRA assignment mismatch");
        }
        gpu_centroid_assignment_cagra_workspace_release(workspace);
        cudaFree(d_assign);
        cudaFree(d_data);
        cudaFree(d_centroids);
    });

    return ctx.exit_code();
}
