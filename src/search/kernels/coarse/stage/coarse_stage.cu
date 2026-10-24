#include "coarse_execution.cuh"
#include "search/kernels/coarse/runtime/coarse_runtime.h"
#include "search/kernels/coarse/coarse_policy.h"
#include "search/kernels/coarse/topk/coarse_histogram_partition.cuh"
#include "search/kernels/coarse/topk/adaptive_threshold_topk/fused_gemm/fused_gemm_topk.cuh"
#ifdef PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK
#include "search/kernels/coarse/topk/adaptive_threshold_topk/adaptive_threshold_topk_fp32.h"
#endif
#include "search/kernels/topk/fusion/fusion_dist_topk.cuh"
#include "coarse_fp16.cuh"
#include "coarse_timing.cuh"
#include "utils/cuda_fill.cuh"
#include <algorithm>
#include <cfloat>
#include <stdexcept>

namespace paraselect::coarse {
namespace {
__global__ void add_cluster_tile_offset_kernel(
    idx_t* idx,
    int n_query,
    int stride,
    int cluster_offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_query * stride;
    if (tid < total && idx[tid] >= 0) {
        idx[tid] += static_cast<idx_t>(cluster_offset);
    }
}

__global__ void merge_tile_topk_kernel(
    float* global_dist,
    idx_t* global_idx,
    const float* tile_dist,
    const idx_t* tile_idx,
    int n_query,
    int global_k,
    int tile_k)
{
    int q = blockIdx.x;
    if (q >= n_query || threadIdx.x != 0) return;

    float* gdist = global_dist + (size_t)q * global_k;
    idx_t* gidx = global_idx + (size_t)q * global_k;
    const float* ldist = tile_dist + (size_t)q * tile_k;
    const idx_t* lidx = tile_idx + (size_t)q * tile_k;

    // Preserve the nearest-first contract used by seed-probe consumers.
    // Normalize and sort the accumulated list as well, so callers need not
    // supply sorted initial candidates. Subsequent tiles take the linear path.
    for (int i = 0; i < global_k; ++i) {
        if (gidx[i] < 0 || !(gdist[i] < FLT_MAX)) {
            gidx[i] = -1;
            gdist[i] = FLT_MAX;
        }
        const float distance = gdist[i];
        const idx_t index = gidx[i];
        int pos = i;
        while (pos > 0 && distance < gdist[pos - 1]) {
            gdist[pos] = gdist[pos - 1];
            gidx[pos] = gidx[pos - 1];
            --pos;
        }
        gdist[pos] = distance;
        gidx[pos] = index;
    }
    int valid = 0;
    while (valid < global_k && gidx[valid] >= 0) ++valid;
    for (int i = 0; i < tile_k; ++i) {
        const float distance = ldist[i];
        const idx_t index = lidx[i];
        if (index < 0 || !(distance < FLT_MAX)) continue;
        if (valid == global_k && !(distance < gdist[global_k - 1])) continue;
        int pos = valid < global_k ? valid++ : global_k - 1;
        while (pos > 0 && distance < gdist[pos - 1]) {
            gdist[pos] = gdist[pos - 1];
            gidx[pos] = gidx[pos - 1];
            --pos;
        }
        gdist[pos] = distance;
        gidx[pos] = index;
    }
}

void run_adaptive_l2_reducer(
    const detail::Execution& a,
    const float* d_cluster_norm,
    int n_clusters,
    int k,
    float* d_out_dist,
    idx_t* d_out_index)
{
#ifdef PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK
    if (a.l2_topk_policy == IVF_COARSE_REDUCER_ADAPTIVE_DIRECT) {
        adaptive_threshold_topk_l2(a.d_inner_product, a.d_query_norm, d_cluster_norm,
            a.n_query, n_clusters, k, d_out_dist, d_out_index, a.stream);
        return;
    }
    if (a.l2_topk_policy == IVF_COARSE_REDUCER_ADAPTIVE_FUSED) {
        adaptive_threshold_topk_fused(a.d_inner_product, a.d_query_norm, d_cluster_norm,
            a.n_query, n_clusters, k, d_out_dist, d_out_index, a.stream);
        return;
    }
#else
    (void)a; (void)d_cluster_norm; (void)n_clusters; (void)k;
    (void)d_out_dist; (void)d_out_index;
#endif
    throw std::runtime_error("adaptive reducer is not compiled or is not an adaptive family");
}

void run_l2_topk_policy(
    const detail::Execution& a,
    const float* d_cluster_norm,
    int n_clusters,
    int k,
    float* d_out_dist,
    idx_t* d_out_index,
    CoarseHistogramTiming* histogram_timing = nullptr)
{
    require_coarse_reducer(a.l2_topk_policy, a.n_query, n_clusters, k, a.fp16_coarse);
    if (coarse_reducer_is_adaptive(a.l2_topk_policy)) {
        run_adaptive_l2_reducer(a, d_cluster_norm, n_clusters, k, d_out_dist, d_out_index);
        return;
    }
    if (coarse_reducer_is_histogram(a.l2_topk_policy)) {
        if(!a.histogram_workspace) throw std::invalid_argument("histogram coarse requires prepared workspace");
        if (a.l2_topk_policy == IVF_COARSE_REDUCER_HISTOGRAM_PARTITION_SMALLK) {
            coarse_histogram_partition_l2(a.d_inner_product,a.d_query_norm,d_cluster_norm,
                a.n_query,n_clusters,k,d_out_dist,d_out_index,*a.histogram_workspace,a.stream);
        } else {
            coarse_histogram_topk_l2(a.d_inner_product,a.d_query_norm,d_cluster_norm,
                a.n_query,n_clusters,k,d_out_dist,d_out_index,*a.histogram_workspace,a.stream,
                histogram_timing);
        }
        return;
    }
    throw std::invalid_argument("unknown coarse L2 reducer");
}


void select_cosine(const detail::Execution& a,const float* norms,int columns,int k,
                   float* values,idx_t* ids) {
    CUDA_CHECK((paraselect::fusion_dist_topk_warpsort::fusion_cos_topk_warpsort<float,idx_t>(
        a.d_query_norm,norms,a.d_inner_product,a.n_query,columns,k,values,ids,true,a.stream,a.cuvs_res)));
}

bool run_fp16_fused(const detail::Execution& a,Timing* timing) {
    if(!a.fp16_coarse || a.distance_mode!=L2_DISTANCE ||
       (!a.fused_gemm && a.l2_topk_policy!=IVF_COARSE_REDUCER_ADAPTIVE_DIRECT))return false;
    detail::StageTimer timer(a.stream,timing!=nullptr);
#ifdef PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK
    if(a.fused_gemm) {
        if(!adaptive_threshold_topk_gemm_l2_fp16(
            a.d_query_fp16,a.d_cluster_centers_fp16,a.d_query_norm,a.d_cluster_centers_norm,
            a.n_query,a.dim,a.nlist,a.nprobe,a.d_inner_product,size_t(a.n_query)*a.nlist,
            a.d_top_nprobe_dist,a.d_top_nprobe_index,a.stream))
            throw std::invalid_argument("prepared FP16 fused GEMM rejected its shape");
        if(timing)++timing->gemm_fused_calls;
        timer.add(timing?&timing->gemm_ms:nullptr);
        return true;
    }
    if(a.l2_topk_policy==IVF_COARSE_REDUCER_ADAPTIVE_DIRECT && adaptive_direct_fp16(
        a.d_query_fp16,a.d_cluster_centers_fp16,a.d_query_norm,a.d_cluster_centers_norm,
        a.n_query,a.dim,a.nlist,a.nprobe,a.d_inner_product,a.nlist,
        a.d_top_nprobe_dist,a.d_top_nprobe_index,a.stream)) {
        timer.add(timing?&timing->topk_ms:nullptr);
        return true;
    }
#endif
    return false;
}

// FP16 uses one centroid span. FP32 may use bounded centroid tiles. Both share
// the same GEMM -> reducer -> optional merge sequence and timing contract.
void run_score_tiles(const detail::Execution& a,int width,Timing* timing) {
    detail::StageTimer timer(a.stream,timing!=nullptr);
    const bool tiled=width<a.nlist;
    const float alpha=1.0f,beta=0.0f;
    if(!a.fp16_coarse && cublasSetStream(a.cublas_handle,a.stream)!=CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("coarse cublasSetStream failed");
    if(tiled) {
        fill_kernel<<<(size_t(a.n_query)*a.nprobe+255)/256,256,0,a.stream>>>(
            a.d_top_nprobe_dist,FLT_MAX,a.n_query*a.nprobe);
        paraselect::cuda_utils::launch_fill_value(a.d_top_nprobe_index,idx_t(-1),a.n_query*a.nprobe,a.stream);
        CUDA_LAUNCH_CHECK("coarse initialize accumulated top-k");
        timer.add(timing?&timing->init_topk_ms:nullptr);
    }
    for(int offset=0;offset<a.nlist;offset+=width) {
        const int columns=std::min(width,a.nlist-offset),k=std::min(a.nprobe,columns);
        float* values=tiled?a.d_tile_top_nprobe_dist:a.d_top_nprobe_dist;
        idx_t* ids=tiled?a.d_tile_top_nprobe_index:a.d_top_nprobe_index;
        if(tiled) {
            fill_kernel<<<(size_t(a.n_query)*k+255)/256,256,0,a.stream>>>(values,FLT_MAX,a.n_query*k);
            paraselect::cuda_utils::launch_fill_value(ids,idx_t(-1),a.n_query*k,a.stream);
            CUDA_LAUNCH_CHECK("coarse initialize tile top-k");
            timer.add(timing?&timing->tile_fill_ms:nullptr);
        }
        if(timing){++timing->materialized_calls;++timing->materialized_query_tiles;}
        if(a.fp16_coarse) {
            coarse_stage_fp16_gemm(a.d_query_fp16,a.d_cluster_centers_fp16,
                a.n_query,a.dim,a.nlist,a.d_inner_product,a.cublas_handle,a.stream);
        } else if(cublasSgemm(a.cublas_handle,CUBLAS_OP_T,CUBLAS_OP_N,
            columns,a.n_query,a.dim,&alpha,a.d_cluster_centers+size_t(offset)*a.dim,
            a.dim,a.d_query,a.dim,&beta,a.d_inner_product,columns)!=CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("coarse cublasSgemm failed");
        }
        timer.add(timing?&timing->gemm_ms:nullptr);
        if(a.distance_mode==COSINE_DISTANCE)
            select_cosine(a,a.d_cluster_centers_norm+offset,columns,k,values,ids);
        else run_l2_topk_policy(a,a.d_cluster_centers_norm+offset,columns,k,values,ids,
                               timing?&timing->histogram:nullptr);
        timer.add(timing?&timing->topk_ms:nullptr);
        if(tiled) {
            add_cluster_tile_offset_kernel<<<(size_t(a.n_query)*k+255)/256,256,0,a.stream>>>(ids,a.n_query,k,offset);
            merge_tile_topk_kernel<<<a.n_query,1,0,a.stream>>>(a.d_top_nprobe_dist,
                a.d_top_nprobe_index,values,ids,a.n_query,a.nprobe,k);
            CUDA_LAUNCH_CHECK("coarse tile merge");
            timer.add(timing?&timing->merge_ms:nullptr);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void detail::run_gemm_score_tiles(const detail::Execution& a,Timing* timing) {
    if(a.n_query<=0 || a.dim<=0 || a.nlist<=0 || a.nprobe<=0 || a.nprobe>a.nlist)
        throw std::invalid_argument("invalid coarse shape");
    if(a.distance_mode!=L2_DISTANCE && a.distance_mode!=COSINE_DISTANCE)
        throw std::invalid_argument("unsupported coarse distance mode");
    if(!a.cublas_handle || !a.d_inner_product || !a.d_top_nprobe_dist || !a.d_top_nprobe_index ||
       !a.d_query_norm || !a.d_cluster_centers_norm ||
       (a.fp16_coarse?(!a.d_query_fp16 || !a.d_cluster_centers_fp16):(!a.d_query || !a.d_cluster_centers)))
        throw std::invalid_argument("missing coarse input/output or workspace");
    const int width=a.fp16_coarse || a.cluster_tile_size<=0?a.nlist:std::min(a.cluster_tile_size,a.nlist);
    if(width<a.nlist && (!a.d_tile_top_nprobe_dist || !a.d_tile_top_nprobe_index))
        throw std::invalid_argument("missing centroid tile workspace");
    if(a.fused_gemm && (!a.fp16_coarse || a.distance_mode!=L2_DISTANCE ||
        !adaptive_threshold_topk_gemm_l2_fp16_supported(a.n_query,a.dim,a.nlist,a.nprobe)))
        throw std::invalid_argument("unsupported prepared FP16 fused GEMM");
    if(!a.fused_gemm && a.distance_mode==L2_DISTANCE) {
        require_coarse_reducer(a.l2_topk_policy,a.n_query,width,std::min(a.nprobe,width),a.fp16_coarse);
        const int tail=a.nlist%width;
        if(tail)require_coarse_reducer(a.l2_topk_policy,a.n_query,tail,std::min(a.nprobe,tail),a.fp16_coarse);
        if(coarse_reducer_needs_histogram(a.l2_topk_policy) && !a.histogram_workspace)
            throw std::invalid_argument("missing histogram workspace");
    }
    if(run_fp16_fused(a,timing))return;
    const bool histogram=a.distance_mode==L2_DISTANCE && a.l2_topk_policy==IVF_COARSE_REDUCER_HISTOGRAM_SORT;
    if(histogram && (a.n_query>a.histogram_workspace->capacity || a.histogram_workspace->sort_capacity<=0))
        throw std::invalid_argument("histogram batch exceeds prepared capacity");
    const int rows=histogram?std::min(a.n_query,a.histogram_workspace->sort_capacity):a.n_query;
    if(rows==a.n_query){run_score_tiles(a,width,timing);return;}
    auto& workspace=*a.histogram_workspace;
    if(workspace.reset_status)CUDA_CHECK(cudaMemsetAsync(workspace.histogram.status,0,sizeof(int),a.stream));
    const int calls=timing?timing->materialized_calls:0;
    for(int offset=0;offset<a.n_query;offset+=rows) {
        auto sub=a;sub.n_query=std::min(rows,a.n_query-offset);
        auto slice=coarse_histogram_topk_slice(workspace,offset,sub.n_query);sub.histogram_workspace=&slice;
        if(a.d_query)sub.d_query=a.d_query+size_t(offset)*a.dim;
        if(a.d_query_fp16)sub.d_query_fp16=a.d_query_fp16+size_t(offset)*a.dim;
        sub.d_query_norm=a.d_query_norm+offset;
        sub.d_top_nprobe_dist=a.d_top_nprobe_dist+size_t(offset)*a.nprobe;
        sub.d_top_nprobe_index=a.d_top_nprobe_index+size_t(offset)*a.nprobe;
        run_score_tiles(sub,width,timing);
    }
    if(timing)timing->materialized_calls=calls+1;
}

} // namespace paraselect::coarse
