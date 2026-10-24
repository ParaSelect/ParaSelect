#include "coarse_runtime.h"
#include "kernels/simd_shape.h"
#include "../stage/coarse_execution.cuh"
#include "../stage/coarse_timing.cuh"
#include "../coarse_prepare.h"
#ifdef PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK
#include "adaptive_run.cuh"
#include "../topk/adaptive_threshold_topk/fused_gemm/fused_gemm_topk.cuh"
#endif
#include "search/kernels/coarse/topk/coarse_histogram_partition.cuh"
#include <mma.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <climits>
#include <cmath>
#include <utility>

namespace paraselect::coarse::current {
__global__ void pad_fp16_rows(const __half* src, __half* dst, int rows,
                              int logical_dim, int storage_dim) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t total=size_t(rows)*storage_dim;
    if(i>=total)return;
    const int d=int(i%storage_dim), r=int(i/storage_dim);
    dst[i]=(d<logical_dim)?src[size_t(r)*logical_dim+d]:__float2half(0.f);
}
__global__ void trim_padded_topk(const float* src_values, const idx_t* src_ids,
                                 float* dst_values, idx_t* dst_ids,
                                 int rows, int padded_k, int logical_k) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(rows) * logical_k;
    if (i >= total) return;
    const int q = int(i / logical_k), r = int(i % logical_k);
    dst_values[i] = src_values[size_t(q) * padded_k + r];
    dst_ids[i] = src_ids[size_t(q) * padded_k + r];
}
template<class T> struct Buffer {
    T* p=nullptr;
    size_t allocation_bytes=0;
    explicit Buffer(size_t n=0) { resize(n); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&)=delete;
    void resize(size_t n) {
        if(n*sizeof(T)<=allocation_bytes)return;
        T* next=nullptr;CUDA_CHECK(cudaMalloc(&next,n*sizeof(T)));
        CUDA_CHECK(cudaFree(p));p=next;allocation_bytes=n*sizeof(T);
    }
};
#ifdef PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK
namespace pc {
#include "kernels/pc_primitives.cuh"
#include "kernels/pc_producer.cuh"
#include "kernels/pc_sort.cuh"
#include "kernels/pc_reuse_sort.cuh"
#include "kernels/pc_count.cuh"

struct Workspace {
    Plan plan;
    Producer producer;
    TopkLaunchCache cache;
    Buffer<float> samples,thresholds,pool,packed_norms,fallback_values,sorted_values;
    Buffer<int> ids,counts,bad,bad_count,sorted_ids,offsets,ends;
    Buffer<__half> packed;
    Buffer<idx_t> fallback_ids;
    Buffer<unsigned char> sort_temp;
    size_t sort_bytes=0,failed_queries=0;
    int* host_bad_count=nullptr;
    int test_mode=0;
    explicit Workspace(Plan p) { configure(p); }
    void configure(Plan p) {
        plan=p;producer=prepare_producer(p);cache={};cache.enabled=true;
        samples.resize(size_t(p.batch)*Samples);thresholds.resize(p.batch);
        const size_t entries=size_t(p.batch)*p.parts*p.capacity;
        pool.resize(entries);ids.resize(entries);counts.resize(size_t(p.batch)*p.parts);
        packed_norms.resize(FallbackBatch);fallback_values.resize(size_t(FallbackBatch)*p.k);
        fallback_ids.resize(size_t(FallbackBatch)*p.k);packed.resize(size_t(FallbackBatch)*128);
        bad.resize(p.batch);bad_count.resize(1);
        if(p.consumer==Consumer::SegmentedSort) {
            sorted_values.resize(entries);sorted_ids.resize(entries);
            offsets.resize(p.batch+1);ends.resize(p.batch);
            cub::DoubleBuffer<float> keys(sorted_values.p,pool.p);
            cub::DoubleBuffer<int> indices(sorted_ids.p,ids.p);
            CUDA_CHECK(sort_admitted_pairs(true,nullptr,sort_bytes,keys,indices,
                entries,p.batch,offsets.p,ends.p,0));
            sort_temp.resize(sort_bytes);
        }
        if(!host_bad_count)CUDA_CHECK(cudaMallocHost(&host_bad_count,sizeof(int)));
    }
    ~Workspace(){cudaFreeHost(host_bad_count);}
    size_t bytes() const {
        return samples.allocation_bytes+thresholds.allocation_bytes+pool.allocation_bytes+
            packed_norms.allocation_bytes+fallback_values.allocation_bytes+sorted_values.allocation_bytes+
            ids.allocation_bytes+counts.allocation_bytes+bad.allocation_bytes+bad_count.allocation_bytes+
            sorted_ids.allocation_bytes+offsets.allocation_bytes+ends.allocation_bytes+
            packed.allocation_bytes+fallback_ids.allocation_bytes+sort_temp.allocation_bytes;
    }
    void consume(const detail::Execution& a) {
        const int stride=plan.parts*plan.capacity;
        switch(plan.consumer) {
        case Consumer::Small:
            select_admitted<128>(a,pool.p,ids.p,stride,cache);break;
        case Consumer::Count512:
            dispatch_counted_512<false>(a,pool.p,ids.p,counts.p,plan.count_cache,true);break;
        case Consumer::Adaptive:
            adaptive_threshold_topk_negative_distance(pool.p,a.n_query,stride,stride,a.nprobe,a.d_top_nprobe_dist,a.d_top_nprobe_index,a.stream);
            finish_ids<<<(size_t(a.n_query)*a.nprobe+255)/256,256,0,a.stream>>>(ids.p,a.d_top_nprobe_index,a.nprobe,a.n_query,stride);break;
        case Consumer::WideSort:
            sort_reusing_outputs(a,pool.p,ids.p,counts.p,plan.parts,plan.capacity,plan.sort_run);break;
        case Consumer::SegmentedSort: {
            segment_offsets<<<(a.n_query+256)/256,256,0,a.stream>>>(offsets.p,a.n_query,stride);
            compact_admitted_for_sort<<<a.n_query,256,0,a.stream>>>(pool.p,ids.p,counts.p,plan.parts,plan.capacity,plan.k,sorted_values.p,sorted_ids.p,ends.p);
            cub::DoubleBuffer<float> keys(sorted_values.p,pool.p);
            cub::DoubleBuffer<int> indices(sorted_ids.p,ids.p);
            CUDA_CHECK(sort_admitted_pairs(true,sort_temp.p,sort_bytes,keys,indices,a.n_query*stride,a.n_query,offsets.p,ends.p,a.stream));
            finish_sorted<<<(size_t(a.n_query)*plan.k+255)/256,256,0,a.stream>>>(keys.Current(),indices.Current(),a.n_query,plan.k,stride,a.d_top_nprobe_dist,a.d_top_nprobe_index);
            break;
        }
        }
    }
    void run(const detail::Execution& a) {
        const int batch=a.n_query,stride=plan.parts*plan.capacity;
        const int rank=int(std::min(double(Samples-1),std::max(3.0,std::ceil(double(plan.k)*Samples*plan.oversample/plan.nlist)-1)));
        produce<true><<<dim3((batch+Rows-1)/Rows,Samples/Cols),Threads,0,a.stream>>>(
            a.d_query_fp16,a.d_cluster_centers_fp16,a.d_query_norm,a.d_cluster_centers_norm,
            batch,a.nlist,plan.parts,plan.capacity,nullptr,samples.p,nullptr,nullptr);
        if(plan.register_producer)sample_threshold<<<batch,128,0,a.stream>>>(samples.p,thresholds.p,rank,test_mode);
        else sample_threshold_histogram<<<batch,128,0,a.stream>>>(samples.p,thresholds.p,rank,test_mode);
        if(plan.consumer==Consumer::Count512) {
            if(plan.splits>1)CUDA_CHECK(cudaMemsetAsync(counts.p,0,size_t(batch)*plan.parts*sizeof(int),a.stream));
        } else initialize_candidate_pool<<<(size_t(batch)*stride+255)/256,256,0,a.stream>>>(pool.p,ids.p,counts.p,size_t(batch)*stride,size_t(batch)*plan.parts);
        producer(a,plan,thresholds.p,pool.p,ids.p,counts.p);
        CUDA_CHECK(cudaMemsetAsync(bad_count.p,0,sizeof(int),a.stream));
        if(plan.register_producer)validate_admission<<<(batch+255)/256,256,0,a.stream>>>(counts.p,batch,plan.parts,plan.capacity,plan.k,bad.p,bad_count.p);
        else validate_admission_warp<<<(batch+3)/4,128,0,a.stream>>>(counts.p,batch,plan.parts,plan.capacity,plan.k,bad.p,bad_count.p);
        CUDA_CHECK(cudaMemcpyAsync(host_bad_count,bad_count.p,sizeof(int),cudaMemcpyDeviceToHost,a.stream));
        CUDA_CHECK(cudaStreamSynchronize(a.stream));
        const int failed=*host_bad_count;
        failed_queries+=failed;
        if(failed<batch)consume(a);
        for(int offset=0;offset<failed;offset+=FallbackBatch) {
            const int rows=std::min(FallbackBatch,failed-offset);
            pack_fallback<<<rows,128,0,a.stream>>>(a.d_query_fp16,a.d_query_norm,bad.p+offset,rows,packed.p,packed_norms.p);
            auto fallback=a;fallback.n_query=rows;fallback.d_query_fp16=packed.p;fallback.d_query_norm=packed_norms.p;
            fallback.d_top_nprobe_dist=fallback_values.p;fallback.d_top_nprobe_index=fallback_ids.p;
            run_adaptive(fallback,Path::GemmAdaptive,1,nullptr,nullptr,nullptr,nullptr);
            scatter_fallback<<<(size_t(rows)*plan.k+255)/256,256,0,a.stream>>>(fallback_values.p,fallback_ids.p,bad.p+offset,rows,plan.k,a.d_top_nprobe_dist,a.d_top_nprobe_index);
        }
    }
};
} // namespace pc
#endif

} // namespace paraselect::coarse::current

namespace paraselect::coarse {
using current::Plan;
using current::Path;
using current::Buffer;
#ifdef PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK
namespace pc = current::pc;
using current::run_adaptive;
#endif
using current::trim_padded_topk;
using current::pad_fp16_rows;

struct Workspace::Impl {
    Plan plan{};
    int request=-1;
    bool asynchronous_only=false,sm120=false;
#ifdef PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK
    std::unique_ptr<pc::Workspace> pc_workspace;
#endif
    Buffer<float> scores,local_values,merge_values,padded_values;
    Buffer<idx_t> local_ids,merge_ids,padded_ids;
    Buffer<__half> padded_query,padded_centers;
    const __half* padded_center_source=nullptr;
    std::uint64_t padded_center_version=0;
    std::unique_ptr<CoarseHistogramTopKWorkspace,void(*)(CoarseHistogramTopKWorkspace*)> histogram{
        new CoarseHistogramTopKWorkspace{}, [](CoarseHistogramTopKWorkspace* w) {
            coarse_histogram_topk_release(*w);delete w;
        }};
    size_t histogram_bytes=0;
    int histogram_reducer=-1;
    explicit Impl() {
        int device;cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDevice(&device));CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
        sm120=prop.major==12 && prop.minor==0;
    }
    void configure(Plan p) {
        validate(p);
        if(!sm120)
            throw std::invalid_argument("optimized coarse runtime requires validated SM120");
        // Allocation failure must not expose an old plan with partially resized scratch.
        plan.batch=0;
        int score_rows=p.batch, width=p.nlist, copies=1;
        size_t local=1,merge=1;
        int reducer=-1;
#ifdef PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK
        if(p.path==Path::ProducerConsumer) {
            if(pc_workspace)pc_workspace->configure(p);
            else pc_workspace=std::make_unique<pc::Workspace>(p);
            score_rows=std::min(p.batch,pc::FallbackBatch);
        } else
#endif
        if(p.path==Path::SimdFused) {
            score_rows=0;local=merge=current::SimdShape(p.batch,p.nlist,p.k).elements;
        } else if(p.path==Path::Histogram || p.path==Path::HistogramPartition) {
            reducer=p.path==Path::Histogram?IVF_COARSE_REDUCER_HISTOGRAM_SORT:IVF_COARSE_REDUCER_HISTOGRAM_PARTITION_SMALLK;
        } else if(p.parts>1)local=size_t(p.batch)*p.parts*p.k;
        if(coarse_reducer_needs_histogram(reducer)) {
            const int k=std::min(width,p.k);
            if(histogram_reducer!=reducer || histogram->capacity<p.batch || histogram->nlist!=width || histogram->k<k) {
                coarse_histogram_topk_release(*histogram);
                prepare_coarse_histogram_workspace(reducer,*histogram,p.batch,width,k);
                histogram_bytes=coarse_histogram_workspace_bytes(reducer,p.batch,width,k);
                histogram_reducer=reducer;
            }
            if(reducer==IVF_COARSE_REDUCER_HISTOGRAM_SORT)score_rows=histogram->sort_capacity;
        }
        scores.resize(std::max(size_t(1),size_t(score_rows)*width*copies));
        if(p.logical_dim<p.dim) {
            padded_query.resize(size_t(p.batch)*p.dim);
            padded_centers.resize(size_t(p.nlist)*p.dim);
            padded_center_source=nullptr;
            padded_center_version=0;
        }
        if(p.logical_k!=p.k) { padded_values.resize(size_t(p.batch)*p.k);padded_ids.resize(size_t(p.batch)*p.k); }
        local_values.resize(local);local_ids.resize(local);
        merge_values.resize(merge);merge_ids.resize(merge);
#ifdef PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK
        if(p.fp16)prepare_adaptive_threshold_topk_gemm_l2_fp16();
#endif
        plan=p;
    }
    void prepare_centers(const Request& request) {
        if (plan.logical_dim >= plan.dim) return;
        // Shape-only construction via prepare_coarse() has no data pointers;
        // defer conversion until the first data-bearing prepare().
        if (!request.d_cluster_centers_fp16) return;
        const bool changed = padded_center_source != request.d_cluster_centers_fp16 ||
            (request.center_version != 0 && padded_center_version != request.center_version);
        if (!changed) return;
        pad_fp16_rows<<<(size_t(plan.nlist)*plan.dim+255)/256,256,0,request.stream>>>(
            request.d_cluster_centers_fp16,padded_centers.p,plan.nlist,
            plan.logical_dim,plan.dim);
        CUDA_CHECK(cudaGetLastError());
        padded_center_source=request.d_cluster_centers_fp16;
        padded_center_version=request.center_version;
    }
    void run(const Request& request,Timing* timing) {
        detail::Execution a(request);
        if(a.fp16_coarse!=plan.fp16 || a.distance_mode!=plan.distance_mode || a.dim!=plan.logical_dim || a.nlist!=plan.nlist || a.nprobe!=plan.logical_k || a.n_query<=0 || a.n_query>plan.batch)
            throw std::invalid_argument("coarse runtime input does not match prepared plan");
        detail::StageTimer timer(a.stream,timing!=nullptr);
        if(plan.logical_dim<plan.dim) {
            if(!a.d_query_fp16 || !a.d_cluster_centers_fp16)
                throw std::invalid_argument("padded FP16 coarse input missing");
            pad_fp16_rows<<<(size_t(a.n_query)*plan.dim+255)/256,256,0,a.stream>>>(
                a.d_query_fp16,padded_query.p,a.n_query,plan.logical_dim,plan.dim);
            a.d_query_fp16=padded_query.p;
            a.d_cluster_centers_fp16=padded_centers.p;
            a.dim=plan.dim;
        }
        if(!a.d_query_norm || !a.d_cluster_centers_norm || !a.d_top_nprobe_dist || !a.d_top_nprobe_index || !a.cublas_handle ||
           (a.fp16_coarse?(!a.d_query_fp16 || !a.d_cluster_centers_fp16):(!a.d_query || !a.d_cluster_centers)))
            throw std::invalid_argument("coarse runtime input/output missing");
        a.d_inner_product=scores.p;
        const int logical_k=plan.logical_k;
        float* const output_values=a.d_top_nprobe_dist;
        idx_t* const output_ids=a.d_top_nprobe_index;
        if (logical_k != plan.k) {
            a.nprobe=plan.k;
            a.d_top_nprobe_dist=padded_values.p;
            a.d_top_nprobe_index=padded_ids.p;
        }
        switch(plan.path) {
#ifdef PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK
        case Path::ProducerConsumer:pc_workspace->run(a);break;
        case Path::Histogram:case Path::HistogramPartition:
            a.fused_gemm=false;a.cluster_tile_size=a.nlist;a.histogram_workspace=histogram.get();
            a.l2_topk_policy=plan.path==Path::Histogram?IVF_COARSE_REDUCER_HISTOGRAM_SORT:IVF_COARSE_REDUCER_HISTOGRAM_PARTITION_SMALLK;
            detail::run_gemm_score_tiles(a);break;
        default:run_adaptive(a,plan.path,plan.parts,local_values.p,merge_values.p,local_ids.p,merge_ids.p);break;
#else
        default:throw std::invalid_argument("optimized coarse not compiled");
#endif
        }
        if (logical_k != plan.k) {
            const size_t total=size_t(a.n_query)*logical_k;
            trim_padded_topk<<<(total+255)/256,256,0,a.stream>>>(
                padded_values.p,padded_ids.p,
                output_values, output_ids,
                a.n_query,plan.k,logical_k);
        }
        CUDA_CHECK(cudaGetLastError());
        timer.add(timing?&timing->topk_ms:nullptr);
    }
    size_t bytes() const {
        return scores.allocation_bytes+local_values.allocation_bytes+local_ids.allocation_bytes+
            merge_values.allocation_bytes+merge_ids.allocation_bytes+histogram_bytes+
            padded_values.allocation_bytes+padded_ids.allocation_bytes+
            padded_query.allocation_bytes+padded_centers.allocation_bytes
#ifdef PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK
            +(pc_workspace?pc_workspace->bytes():0)
#endif
            ;
    }
};
Workspace::Workspace(Plan p):impl_(std::make_unique<Impl>()){impl_->configure(p);}
Workspace::Workspace(const Request& a,int request,bool asynchronous_only):impl_(std::make_unique<Impl>()) {
    impl_->request=request;impl_->asynchronous_only=asynchronous_only;
    prepare(a);
}
void Workspace::prepare(const Request& a) {
    const auto& p=impl_->plan;
    if(!(a.n_query==p.batch && a.nprobe==p.logical_k && a.nlist==p.nlist &&
       a.dim==p.logical_dim && a.fp16_coarse==p.fp16 && a.distance_mode==p.distance_mode))
        impl_->configure(select_coarse_plan(a,impl_->request,impl_->asynchronous_only,impl_->sm120));
    impl_->prepare_centers(a);
}
Workspace::~Workspace()=default;
const Plan& Workspace::plan() const{return impl_->plan;}
size_t Workspace::workspace_bytes() const{return impl_->bytes();}
size_t Workspace::fallback_queries() const{
#ifdef PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK
    return impl_->pc_workspace?impl_->pc_workspace->failed_queries:0;
#else
    return 0;
#endif
}
void Workspace::set_admission_test_mode(int mode) {
#ifdef PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK
    if(mode< -1 || mode>2 || (!impl_->pc_workspace && mode))throw std::invalid_argument("invalid admission test mode");
    if(impl_->pc_workspace)impl_->pc_workspace->test_mode=mode;
#else
    if(mode)throw std::invalid_argument("PC not compiled");
#endif
}
void search(const Request& request,Workspace& workspace,Timing* timing) {
    workspace.prepare(request);
    workspace.impl_->run(request,timing);
}
}
