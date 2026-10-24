#include "search/ivf_common/ivf_coarse_ctx.h"

#include "clustering/kmeans/kmeans_final_assign.cuh"
#include "utils/check_cuda_error.cuh"

#include <algorithm>
#include <stdexcept>

namespace paraselect::search {
namespace {

GpuCentroidAssignmentConfig make_cagra_assignment_config(int batch) {
    GpuCentroidAssignmentConfig cfg{};
    cfg.combo = GpuCentroidAssignmentCombo::CagraTopkSelect;
    cfg.batch_size = batch;
    cfg.cagra_topk_centroids = 8;
    cfg.cagra_itopk_size = 128;
    cfg.cagra_search_width = 4;
    cfg.cagra_max_iterations = 32;
    return cfg;
}

KMeansCase make_assignment_case(const IVFCoarseConfig& cfg) {
    KMeansCase assignment{};
    assignment.dim = cfg.dim;
    assignment.k = cfg.nlist;
    assignment.dist = static_cast<DistanceType>(cfg.distance);
    assignment.precision = KMeansComputePrecision::Tf32;
    return assignment;
}

} // namespace

void IVFCoarseCtx::assign_nearest(
    const float* h_vectors, int n, idx_t* h_assign) const {
    if (!initialized_) throw std::runtime_error("IVFCoarseCtx is not initialized");
    if (!h_vectors || !h_assign || n <= 0 || cfg_.max_batch <= 0) {
        throw std::invalid_argument("IVFCoarseCtx::assign_nearest invalid input");
    }
    // Nearest-centroid assignment is top-1 coarse search: reuse the fused coarse
    // operators instead of materializing every batch-by-centroid distance tile.
    for (int begin = 0; begin < n; begin += cfg_.max_batch) {
        const int rows = std::min(cfg_.max_batch, n - begin);
        search(h_vectors + (size_t)begin * cfg_.dim, rows, 1, h_assign + begin);
    }
}

void IVFCoarseCtx::assign_nearest_cagra(
    const float* h_vectors, int n, idx_t* h_assign) const {
    if (!initialized_) throw std::runtime_error("IVFCoarseCtx is not initialized");
    if (!cfg_.enable_cagra_assignment || !h_vectors || !h_assign || n <= 0)
        throw std::invalid_argument("IVFCoarseCtx::assign_nearest_cagra invalid input");
    const int batch = std::max(1, cfg_.cagra_assignment_batch);
    const auto assign_cfg = make_cagra_assignment_config(batch);
    KMeansCase assignment = make_assignment_case(cfg_);
    for (int begin = 0; begin < n; begin += batch) {
        const int rows = std::min(batch, n - begin);
        assignment.n = rows;
        if (!assign_vectors_to_centroids_gpu_cagra_with_workspace(
                assign_cfg, assignment,
                h_vectors + (size_t)begin * cfg_.dim, d_centroids_,
                cagra_assign_ws_, d_cagra_assign_))
            throw std::runtime_error("CAGRA centroid assignment failed");
        CUDA_CHECK(cudaMemcpy(
            h_assign + begin, d_cagra_assign_, (size_t)rows * sizeof(idx_t),
            cudaMemcpyDeviceToHost));
    }
}

void IVFCoarseCtx::assign_nearest_cagra_u8(
    const uint8_t* h_vectors, int n, idx_t* h_assign) const {
    if (!initialized_) throw std::runtime_error("IVFCoarseCtx is not initialized");
    if (!cfg_.enable_cagra_assignment || !h_vectors || !h_assign || n <= 0)
        throw std::invalid_argument("IVFCoarseCtx::assign_nearest_cagra_u8 invalid input");
    const int batch = std::max(1, cfg_.cagra_assignment_batch);
    const auto assign_cfg = make_cagra_assignment_config(batch);
    KMeansCase assignment = make_assignment_case(cfg_);
    for (int begin = 0; begin < n; begin += batch) {
        const int rows = std::min(batch, n - begin);
        assignment.n = rows;
        if (!assign_u8_vectors_to_centroids_gpu_cagra_with_workspace(
                assign_cfg, assignment,
                h_vectors + (size_t)begin * cfg_.dim, d_centroids_,
                cagra_assign_ws_, d_cagra_assign_))
            throw std::runtime_error("CAGRA U8 centroid assignment failed");
        CUDA_CHECK(cudaMemcpy(
            h_assign + begin, d_cagra_assign_, (size_t)rows * sizeof(idx_t),
            cudaMemcpyDeviceToHost));
    }
}

} // namespace paraselect::search
