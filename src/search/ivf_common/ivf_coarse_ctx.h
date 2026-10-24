#ifndef PARASELECT_SEARCH_IVF_COMMON_IVF_COARSE_CTX_H
#define PARASELECT_SEARCH_IVF_COMMON_IVF_COARSE_CTX_H

#include "utils/pch.h"
#include "search/ivf_common/ivf_probe_view.h"
#include "search/kernels/coarse/coarse_policy.h"
#include "search/kernels/coarse/runtime/coarse_runtime.h"
#include "clustering/kmeans/kmeans_assignment_workspace.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

namespace paraselect::search {

struct IVFCoarseConfig {
    int dim = 0;
    int nlist = 0;
    int max_batch = 0;
    int max_nprobe = 0;
    int max_assign_batch = 0;
    int assign_centroid_tile = 4096;
    bool enable_cagra_assignment = false;
    int cagra_assignment_batch = 32768;
    int distance = L2_DISTANCE;
    int cluster_tile_size = 0;
    int l2_topk_policy = IVF_COARSE_REDUCER_HISTOGRAM_SORT;
    bool select_l2_topk_per_request = false;
    bool fp16_coarse = false;
};

struct IVFCoarseTiming {
    double total_ms = 0.0;
    double h2d_ms = 0.0;
    double query_norm_ms = 0.0;
    double coarse_stage_ms = 0.0;
    double d2h_ms = 0.0;
    double gemm_ms = 0.0;
    double topk_ms = 0.0;
};

class IVFCoarseCtx {
public:
    IVFCoarseCtx() = default;
    ~IVFCoarseCtx();

    IVFCoarseCtx(const IVFCoarseCtx&) = delete;
    IVFCoarseCtx& operator=(const IVFCoarseCtx&) = delete;

    void init(const IVFCoarseConfig& cfg, const float* h_centroids);
    void update_centroids(const float* h_centroids,
                          const std::vector<int>& clusters);
    void release();

    void search(const float* h_queries, int nq, int nprobe, idx_t* h_top_nprobe,
                IVFCoarseTiming* timing = nullptr) const;
    // The returned view remains valid until the next search on this context.
    // Work submitted to consumer_stream waits for the coarse result automatically.
    IVFDeviceProbeView search_device(
        const float* h_queries,
        int nq,
        int nprobe,
        cudaStream_t consumer_stream,
        IVFCoarseTiming* timing = nullptr) const;
    IVFDeviceProbeView search_device_into(
        const float* h_queries,
        int nq,
        int nprobe,
        idx_t* device_output,
        cudaStream_t consumer_stream,
        IVFCoarseTiming* timing = nullptr,
        float* device_distance_output = nullptr) const;
    void assign_nearest(const float* h_vectors, int n, idx_t* h_assign) const;
    void assign_nearest_cagra(const float* h_vectors, int n,
                              idx_t* h_assign) const;
    void assign_nearest_cagra_u8(const uint8_t* h_vectors, int n,
                                 idx_t* h_assign) const;

    bool initialized() const { return initialized_; }
    int dim() const { return cfg_.dim; }
    int nlist() const { return cfg_.nlist; }

private:
    void enqueue_search(
        const float* h_queries,
        int nq,
        int nprobe,
        IVFCoarseTiming* timing) const;

    IVFCoarseConfig cfg_{};
    bool initialized_ = false;
    mutable cudaStream_t stream_ = nullptr;
    mutable cudaEvent_t probes_ready_ = nullptr;
    mutable cublasHandle_t cublas_ = nullptr;
    float* d_centroids_ = nullptr;
    float* d_centroid_norms_ = nullptr;
    mutable float* d_queries_ = nullptr;
    mutable float* d_query_norms_ = nullptr;
    mutable __half* d_query_fp16_ = nullptr;
    __half* d_cluster_centers_fp16_ = nullptr;
    std::uint64_t centers_version_ = 1;
    mutable float* d_inner_ = nullptr;
    mutable float* d_top_dist_ = nullptr;
    mutable idx_t* d_top_idx_ = nullptr;
    mutable std::unique_ptr<paraselect::coarse::Workspace> coarse_workspace_;
    mutable GpuCentroidAssignmentCagraWorkspace cagra_assign_ws_{};
    mutable idx_t* d_cagra_assign_ = nullptr;
};

} // namespace paraselect::search

#endif // PARASELECT_SEARCH_IVF_COMMON_IVF_COARSE_CTX_H
