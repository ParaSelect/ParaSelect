#include "search/kernels/coarse/coarse_prepare.h"
#include "search/ivf_common/ivf_coarse_ctx.h"

#include "clustering/kmeans/kmeans_final_assign.cuh"
#include "l2norm/l2norm.cuh"
#include "search/kernels/coarse/coarse.h"
#include "search/kernels/coarse/runtime/coarse_runtime.h"
#include "quantization/linear/linear_fp16.cuh"
#include "utils/check_cuda_error.cuh"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <numeric>

namespace paraselect::search {
namespace {

void check_cublas(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed");
    }
}

template <typename T>
void free_device(T*& ptr) {
    if (ptr) cudaFree(ptr);
    ptr = nullptr;
}

} // namespace

IVFCoarseCtx::~IVFCoarseCtx() {
    release();
}

void IVFCoarseCtx::release() {
    if(stream_) CUDA_CHECK(cudaStreamSynchronize(stream_));
    coarse_workspace_.reset();
    free_device(d_centroids_);
    free_device(d_centroid_norms_);
    free_device(d_queries_);
    free_device(d_query_norms_);
    free_device(d_query_fp16_);
    free_device(d_cluster_centers_fp16_);
    free_device(d_inner_);
    free_device(d_top_dist_);
    free_device(d_top_idx_);
    gpu_centroid_assignment_cagra_workspace_release(cagra_assign_ws_);
    free_device(d_cagra_assign_);
    if (cublas_) {
        cublasDestroy(cublas_);
        cublas_ = nullptr;
    }
    if (probes_ready_) {
        cudaEventDestroy(probes_ready_);
        probes_ready_ = nullptr;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    initialized_ = false;
    cfg_ = IVFCoarseConfig{};
}

void IVFCoarseCtx::init(const IVFCoarseConfig& cfg, const float* h_centroids) {
    release();
    if (!h_centroids || cfg.dim <= 0 || cfg.nlist <= 0 || cfg.max_batch <= 0 || cfg.max_nprobe <= 0 || cfg.max_nprobe > cfg.nlist) {
        throw std::invalid_argument("IVFCoarseCtx::init invalid input");
    }
    cfg_ = cfg;
    centers_version_ = 1;
    // Coarse policy is selected only by the formal dispatcher from the actual
    // query geometry. The legacy PARASELECT_COARSE_STRATEGY environment
    // override is intentionally not read by this shared context.
    cfg_.l2_topk_policy = -1;
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUDA_CHECK(cudaEventCreateWithFlags(&probes_ready_, cudaEventDisableTiming));
    check_cublas(cublasCreate(&cublas_), "cublasCreate");
    check_cublas(cublasSetStream(cublas_, stream_), "cublasSetStream");

    const size_t centroid_bytes = (size_t)cfg_.nlist * (size_t)cfg_.dim * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_centroids_, centroid_bytes));
    CUDA_CHECK(cudaMemcpyAsync(d_centroids_, h_centroids, centroid_bytes, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMalloc(&d_centroid_norms_, (size_t)cfg_.nlist * sizeof(float)));
    compute_l2_norm_gpu(d_centroids_, d_centroid_norms_, cfg_.nlist, cfg_.dim, L2NORM_AUTO, stream_);
    if (cfg_.fp16_coarse) {
        CUDA_CHECK(cudaMalloc(&d_cluster_centers_fp16_,
                              (size_t)cfg_.nlist * cfg_.dim * sizeof(__half)));
        ivf::LinearQuantizerFP16GPU fp16;
        fp16.encode_async(d_centroids_, cfg_.nlist, cfg_.dim,
                          d_cluster_centers_fp16_, stream_);
    }

    const size_t vector_rows = (size_t)std::max(cfg_.max_batch, cfg_.max_assign_batch);
    coarse_workspace_=paraselect::coarse::prepare_coarse(
        cfg_.fp16_coarse,cfg_.distance,cfg_.max_batch,cfg_.dim,cfg_.nlist,
        cfg_.max_nprobe);
    const int assign_tile = std::min(cfg_.nlist, std::max(1, cfg_.assign_centroid_tile));
    const size_t assign_inner_elems = (size_t)std::max(0, cfg_.max_assign_batch) * (size_t)assign_tile;
    const size_t inner_elems = assign_inner_elems;
    const size_t query_top_elems = (size_t)cfg_.max_batch * (size_t)cfg_.max_nprobe;
    const size_t top_elems = std::max(query_top_elems, (size_t)std::max(0, cfg_.max_assign_batch));
    CUDA_CHECK(cudaMalloc(&d_queries_, (size_t)vector_rows * (size_t)cfg_.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_query_norms_, vector_rows * sizeof(float)));
    if (cfg_.fp16_coarse) {
        CUDA_CHECK(cudaMalloc(&d_query_fp16_,
                              vector_rows * (size_t)cfg_.dim * sizeof(__half)));
    }
    if(inner_elems) CUDA_CHECK(cudaMalloc(&d_inner_, inner_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_top_dist_, top_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_top_idx_, top_elems * sizeof(idx_t)));
    if (cfg_.enable_cagra_assignment) {
        GpuCentroidAssignmentConfig assign_cfg{};
        assign_cfg.combo = GpuCentroidAssignmentCombo::CagraTopkSelect;
        assign_cfg.batch_size = std::max(1, cfg_.cagra_assignment_batch);
        assign_cfg.cagra_topk_centroids = 8;
        assign_cfg.cagra_itopk_size = 128;
        assign_cfg.cagra_search_width = 4;
        assign_cfg.cagra_max_iterations = 32;
        if (!gpu_centroid_assignment_cagra_workspace_build(
                cagra_assign_ws_, assign_cfg, d_centroids_, cfg_.nlist,
                cfg_.dim, assign_cfg.batch_size))
            throw std::runtime_error("failed to build CAGRA assignment workspace");
        CUDA_CHECK(cudaMalloc(&d_cagra_assign_,
                              (size_t)assign_cfg.batch_size * sizeof(idx_t)));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    initialized_ = true;
}

void IVFCoarseCtx::update_centroids(const float* h_centroids,
                                    const std::vector<int>& clusters) {
    if (!initialized_) throw std::runtime_error("IVFCoarseCtx is not initialized");
    if (!h_centroids)
        throw std::invalid_argument("IVFCoarseCtx centroid input is null");
    if (clusters.empty()) return;
    ++centers_version_;

    std::vector<int> sorted = clusters;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    for (int cluster : sorted) {
        if (cluster < 0 || cluster >= cfg_.nlist)
            throw std::invalid_argument(
                "IVFCoarseCtx centroid index out of range");
    }
    for (size_t i = 0; i < sorted.size();) {
        const int begin = sorted[i];
        int end = begin + 1;
        while (++i < sorted.size() && sorted[i] == end) ++end;
        const size_t rows = (size_t)(end - begin);
        const size_t offset = (size_t)begin * (size_t)cfg_.dim;
        CUDA_CHECK(cudaMemcpyAsync(
            d_centroids_ + offset, h_centroids + offset,
            rows * (size_t)cfg_.dim * sizeof(float), cudaMemcpyHostToDevice,
            stream_));
        compute_l2_norm_gpu(d_centroids_ + offset, d_centroid_norms_ + begin,
                            (int)rows, cfg_.dim, L2NORM_AUTO, stream_);
        if (cfg_.fp16_coarse) {
            ivf::LinearQuantizerFP16GPU fp16;
            fp16.encode_async(d_centroids_ + offset, (int)rows, cfg_.dim,
                              d_cluster_centers_fp16_ + offset, stream_);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    if (cfg_.enable_cagra_assignment) {
        gpu_centroid_assignment_cagra_workspace_release(cagra_assign_ws_);
        GpuCentroidAssignmentConfig assign_cfg{};
        assign_cfg.combo = GpuCentroidAssignmentCombo::CagraTopkSelect;
        assign_cfg.batch_size = std::max(1, cfg_.cagra_assignment_batch);
        assign_cfg.cagra_topk_centroids = 8;
        assign_cfg.cagra_itopk_size = 128;
        assign_cfg.cagra_search_width = 4;
        assign_cfg.cagra_max_iterations = 32;
        if (!gpu_centroid_assignment_cagra_workspace_build(
                cagra_assign_ws_, assign_cfg, d_centroids_, cfg_.nlist,
                cfg_.dim, assign_cfg.batch_size))
            throw std::runtime_error("failed to rebuild CAGRA assignment workspace");
    }
}

void IVFCoarseCtx::enqueue_search(
    const float* h_queries,
    int nq,
    int nprobe,
    IVFCoarseTiming* timing) const {
    if (!initialized_) throw std::runtime_error("IVFCoarseCtx is not initialized");
    if (!h_queries || nq <= 0 || nq > cfg_.max_batch || nprobe <= 0 ||
        nprobe > cfg_.max_nprobe) {
        throw std::invalid_argument("IVFCoarseCtx coarse search invalid input");
    }
    const auto t0 = std::chrono::high_resolution_clock::now();

    const auto h2d_t0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(d_queries_, h_queries, (size_t)nq * (size_t)cfg_.dim * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    if (cfg_.fp16_coarse) {
        ivf::LinearQuantizerFP16GPU fp16;
        fp16.encode_async(d_queries_, nq, cfg_.dim, d_query_fp16_, stream_);
    }
    if (timing) CUDA_CHECK(cudaStreamSynchronize(stream_));
    const auto h2d_t1 = std::chrono::high_resolution_clock::now();

    const auto norm_t0 = std::chrono::high_resolution_clock::now();
    compute_l2_norm_gpu(d_queries_, d_query_norms_, nq, cfg_.dim, L2NORM_AUTO, stream_);
    if (timing) CUDA_CHECK(cudaStreamSynchronize(stream_));
    const auto norm_t1 = std::chrono::high_resolution_clock::now();

    paraselect::coarse::Request args{};
    args.fp16_coarse = cfg_.fp16_coarse;
    args.distance_mode = cfg_.distance;
    args.n_query = nq;
    args.dim = cfg_.dim;
    args.nlist = cfg_.nlist;
    args.nprobe = nprobe;
    args.cluster_tile_size = cfg_.cluster_tile_size;
    args.center_version = centers_version_;

    args.d_query = d_queries_;
    args.d_query_fp16 = d_query_fp16_;
    args.d_query_norm = d_query_norms_;
    args.d_cluster_centers = d_centroids_;
    args.d_cluster_centers_fp16 = d_cluster_centers_fp16_;
    args.d_cluster_centers_norm = d_centroid_norms_;
    args.d_top_nprobe_dist = d_top_dist_;
    args.d_top_nprobe_index = d_top_idx_;
    args.cublas_handle = cublas_;
    args.stream = stream_;

    paraselect::coarse::Timing stage_timing{};
    const auto coarse_t0 = std::chrono::high_resolution_clock::now();
    paraselect::coarse::search(args, *coarse_workspace_, timing ? &stage_timing : nullptr);
    if (timing) CUDA_CHECK(cudaStreamSynchronize(stream_));
    const auto coarse_t1 = std::chrono::high_resolution_clock::now();

    if (timing) {
        const auto t1 = std::chrono::high_resolution_clock::now();
        timing->total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        timing->h2d_ms = std::chrono::duration<double, std::milli>(h2d_t1 - h2d_t0).count();
        timing->query_norm_ms = std::chrono::duration<double, std::milli>(norm_t1 - norm_t0).count();
        timing->coarse_stage_ms = std::chrono::duration<double, std::milli>(coarse_t1 - coarse_t0).count();
        timing->d2h_ms = 0.0;
        timing->gemm_ms = stage_timing.gemm_ms;
        timing->topk_ms = stage_timing.topk_ms;
    }
}

void IVFCoarseCtx::search(
    const float* h_queries,
    int nq,
    int nprobe,
    idx_t* h_top_nprobe,
    IVFCoarseTiming* timing) const {
    if (!h_top_nprobe) {
        throw std::invalid_argument("IVFCoarseCtx::search output is null");
    }
    enqueue_search(h_queries, nq, nprobe, timing);
    const auto d2h_t0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(
        h_top_nprobe, d_top_idx_, (size_t)nq * (size_t)nprobe * sizeof(idx_t),
        cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    const auto d2h_t1 = std::chrono::high_resolution_clock::now();
    if (timing) {
        timing->d2h_ms = std::chrono::duration<double, std::milli>(
            d2h_t1 - d2h_t0).count();
        timing->total_ms += timing->d2h_ms;
    }
}

IVFDeviceProbeView IVFCoarseCtx::search_device(
    const float* h_queries,
    int nq,
    int nprobe,
    cudaStream_t consumer_stream,
    IVFCoarseTiming* timing) const {
    enqueue_search(h_queries, nq, nprobe, timing);
    CUDA_CHECK(cudaEventRecord(probes_ready_, stream_));
    CUDA_CHECK(cudaStreamWaitEvent(consumer_stream, probes_ready_, 0));
    return {d_top_idx_, nq, nprobe};
}

IVFDeviceProbeView IVFCoarseCtx::search_device_into(
    const float* h_queries,
    int nq,
    int nprobe,
    idx_t* device_output,
    cudaStream_t consumer_stream,
    IVFCoarseTiming* timing,
    float* device_distance_output) const {
    if (!device_output) {
        throw std::invalid_argument(
            "IVFCoarseCtx::search_device_into output is null");
    }
    enqueue_search(h_queries, nq, nprobe, timing);
    CUDA_CHECK(cudaMemcpyAsync(
        device_output, d_top_idx_, (size_t)nq * (size_t)nprobe * sizeof(idx_t),
        cudaMemcpyDeviceToDevice, stream_));
    if (device_distance_output) {
        CUDA_CHECK(cudaMemcpyAsync(
            device_distance_output, d_top_dist_,
            (size_t)nq * (size_t)nprobe * sizeof(float),
            cudaMemcpyDeviceToDevice, stream_));
    }
    CUDA_CHECK(cudaEventRecord(probes_ready_, stream_));
    CUDA_CHECK(cudaStreamWaitEvent(consumer_stream, probes_ready_, 0));
    return {device_output, nq, nprobe};
}

} // namespace paraselect::search
