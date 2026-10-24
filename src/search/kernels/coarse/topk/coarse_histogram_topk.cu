#include "coarse_histogram_topk.cuh"
#include "utils/check_cuda_error.cuh"
#include "search/kernels/coarse/topk/adaptive_threshold_topk/adaptive_threshold_topk_fp32.h"
#include <cub/device/device_segmented_sort.cuh>
#include <climits>
#include <cstdlib>
#include <stdexcept>
#include "search/kernels/topk/fixed_topk.cuh"
namespace paraselect::coarse {
namespace {
template<class F> void timed(cudaStream_t stream, double* out, F&& launch) {
    cudaEvent_t begin=nullptr,end=nullptr;
    CUDA_CHECK(cudaEventCreate(&begin)); CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(begin,stream)); launch();
    CUDA_CHECK(cudaEventRecord(end,stream)); CUDA_CHECK(cudaEventSynchronize(end));
    float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end)); *out += ms;
    cudaEventDestroy(begin); cudaEventDestroy(end);
}
template<class T> void allocate(T*& p,size_t n) {CUDA_CHECK(cudaMalloc(&p,n*sizeof(T)));}
bool adaptive_selector_enabled() {
#ifdef PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK
    const char* value=std::getenv("PARASELECT_HISTOGRAM_SELECTOR");
    return value && std::strcmp(value,"adaptive")==0;
#else
    return false;
#endif
}
size_t sort_workspace_bytes(int batch,int nlist) {
    size_t bytes=0;
    CUDA_CHECK(cub::DeviceSegmentedSort::SortPairs(nullptr,bytes,
        static_cast<const float*>(nullptr),static_cast<float*>(nullptr),
        static_cast<const int*>(nullptr),static_cast<int*>(nullptr),batch*nlist,batch,
        static_cast<const int*>(nullptr),static_cast<const int*>(nullptr)));
    return bytes;
}
__global__ void offsets_kernel(int* starts,int b,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<b)starts[i]=i*n;
}
// Reserve once per warp/tile, rather than waiting for a global atomic
// after every row load. Thread-local rows remain in registers until scatter.
__device__ int reserve(int count,int* global_count) {
    const int lane=threadIdx.x&31;
    int inclusive=count;
    #pragma unroll
    for(int offset=1;offset<32;offset*=2) {
        const int previous=__shfl_up_sync(0xffffffff,inclusive,offset);
        if(lane>=offset)inclusive+=previous;
    }
    const int total=__shfl_sync(0xffffffff,inclusive,31);
    int begin=0;
    if(lane==0 && total)begin=atomicAdd(global_count,total);
    begin=__shfl_sync(0xffffffff,begin,0);
    return begin+inclusive-count;
}
constexpr int compact_rows=16;
template<bool Refine=false,bool Validate=false,bool StoreNegative=false>
__global__ void compact_kernel(const float* scores,const float* query_norm,const float* centroid_norm,
                                int n,CoarseHistogramTopKWorkspace w) {
    const int queries=Refine ? *w.retry_count : gridDim.y;
    for(int row=blockIdx.y;row<queries;row+=gridDim.y) {
    const int q=Refine ? w.retry_queries[row] : row;
    if constexpr(Refine)if(w.selection_complete[q])continue;
    const float upper=w.histogram.upper[q];
    for(int base=blockIdx.x*blockDim.x*compact_rows;base<n;base+=gridDim.x*blockDim.x*compact_rows) {
    const int start=base+threadIdx.x;
    float rows[compact_rows];
    int count=0;
    bool finite=true;
    #pragma unroll
    for(int i=0;i<compact_rows;++i) {
        const int c=start+i*blockDim.x;
        const float d=c<n ? coarse_l2_distance(scores[size_t(q)*n+c],query_norm[q],centroid_norm[c]) : INFINITY;
        rows[i]=d;
        count+=c<n && d<=upper;
        if constexpr(Validate)finite &= c>=n || isfinite(d);
    }
    if constexpr(Validate)if(!finite)atomicExch(w.histogram.status,1);
    int position=reserve(count,w.candidate_counts+q);
    #pragma unroll
    for(int i=0;i<compact_rows;++i) {
        const int c=start+i*blockDim.x;
        if(c<n && rows[i]<=upper) {
            if(position<w.candidate_stride) {
                w.candidates[size_t(q)*w.candidate_stride+position]=StoreNegative ? -rows[i] : rows[i];
                w.candidate_ids[size_t(q)*w.candidate_stride+position]=c;
            }
            ++position;
        }
    }
}
}
}
// Only overflow rows take this path. Four byte histograms locate the exact
// kth float key; retain all strictly smaller values plus the needed ties.
// This is threshold refinement followed by the same segmented sort, not a
// switch to another reducer. The original score matrix remains immutable.
__device__ unsigned ordered_key(float value) {
    if(value==0.f)value=0.f; // canonicalize signed zero for equality admission
    unsigned bits=__float_as_uint(value);
    return bits ^ ((bits&0x80000000u) ? 0xffffffffu : 0x80000000u);
}
template<bool StoreNegative=false>
__global__ void refine_overflow(const float* scores,const float* qn,const float* cn,
                               int n,int k,CoarseHistogramTopKWorkspace w,
                               const int* query_ids,int query_count) {
    if (query_ids && blockIdx.x >= query_count) return;
    int q=query_ids ? query_ids[blockIdx.x] : blockIdx.x;
    if(w.candidate_counts[q]<=w.candidate_stride) {
        if(threadIdx.x==0) {
            const int count=w.candidate_counts[q];
            w.candidate_ends[q]=q*w.candidate_stride+count;
            if(count<k || count!=w.histogram.selected_counts[q])atomicExch(w.histogram.status,2);
        }
        return;
    }
    __shared__ unsigned bins[256],prefix;
    __shared__ int rank,strict_count,less_cursor,equal_cursor;
    if(threadIdx.x==0){prefix=0;rank=k;strict_count=0;less_cursor=0;equal_cursor=0;}
    __syncthreads();
    for(int shift=24;shift>=0;shift-=8) {
        bins[threadIdx.x]=0;__syncthreads();
        for(int c=threadIdx.x;c<n;c+=blockDim.x) {
            float d=coarse_l2_distance(scores[size_t(q)*n+c],qn[q],cn[c]);
            if(!isfinite(d))continue;
            unsigned key=ordered_key(d);
            if(shift==24 || (key>>(shift+8))==(prefix>>(shift+8))) {
                const unsigned bin=(key>>shift)&255;
                const unsigned peers=__match_any_sync(__activemask(),bin);
                if((threadIdx.x&31)==__ffs(peers)-1)
                    atomicAdd(&bins[bin],unsigned(__popc(peers)));
            }
        }
        __syncthreads();
        if(threadIdx.x==0) {
            int before=0,bin=0;
            while(bin<255 && before+int(bins[bin])<rank)before+=bins[bin++];
            prefix|=unsigned(bin)<<shift;rank-=before;strict_count+=before;
        }
        __syncthreads();
    }
    unsigned bits=prefix ^ ((prefix&0x80000000u) ? 0x80000000u : 0xffffffffu);
    float threshold=__uint_as_float(bits);
    for(int c=threadIdx.x;c<n;c+=blockDim.x) {
        float d=coarse_l2_distance(scores[size_t(q)*n+c],qn[q],cn[c]);
        int slot=-1;
        if(d<threshold)slot=atomicAdd(&less_cursor,1);
        else if(d==threshold)slot=strict_count+atomicAdd(&equal_cursor,1);
        if(slot>=0 && slot<k) {
            w.candidates[size_t(q)*w.candidate_stride+slot]=StoreNegative ? -d : d;
            w.candidate_ids[size_t(q)*w.candidate_stride+slot]=c;
        }
    }
    __syncthreads();
    if(threadIdx.x==0) {
        if(less_cursor!=strict_count || strict_count+equal_cursor<k)atomicExch(w.histogram.status,2);
        w.candidate_counts[q]=k;w.histogram.selected_counts[q]=k;w.histogram.upper[q]=threshold;
        w.candidate_ends[q]=q*w.candidate_stride+k;
    }
}
__global__ void collect_overflow(int b,int k,CoarseHistogramTopKWorkspace w) {
    const int q=blockIdx.x*blockDim.x+threadIdx.x;
    if(q>=b)return;
    const int count=w.candidate_counts[q];
    if(count>w.candidate_stride) {
        w.retry_queries[atomicAdd(w.retry_count,1)]=q;
    } else {
        w.candidate_ends[q]=q*w.candidate_stride+count;
        if(count<k || count!=w.histogram.selected_counts[q])
            atomicExch(w.histogram.status,2);
    }
}
__global__ void verify_estimate(int b,int k,CoarseHistogramTopKWorkspace w) {
    __shared__ int retries;
    if(threadIdx.x==0)retries=0;
    __syncthreads();
    for(int q=threadIdx.x;q<b;q+=blockDim.x) {
        const int count=w.candidate_counts[q];
        w.selection_complete[q]=count>=k;
        if(count>=k)w.histogram.selected_counts[q]=count;
        else {
            w.candidate_counts[q]=0;
            w.retry_queries[atomicAdd(&retries,1)]=q;
        }
    }
    __syncthreads();
    if(threadIdx.x==0)*w.retry_count=retries;
}
__global__ void gather_kernel(int b,int n,int k,CoarseHistogramTopKWorkspace w,float* values,int64_t* ids) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(b)*k)return;
    const size_t source=(i/k)*n+i%k;
    values[i]=w.sorted_candidates[source];
    ids[i]=w.sorted_candidate_ids[source];
}
__global__ void initialize_selector_pool(int b,CoarseHistogramTopKWorkspace w) {
    const size_t total=size_t(b)*w.candidate_stride;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total;
        i+=size_t(blockDim.x)*gridDim.x) {
        w.candidates[i]=-INFINITY;
        w.candidate_ids[i]=-1;
    }
}
__global__ void gather_selector_kernel(int b,int k,CoarseHistogramTopKWorkspace w,
                                       const float* values,const int64_t* slots,
                                       float* out_values,int64_t* out_ids) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(b)*k)return;
    const int slot=static_cast<int>(slots[i]);
    out_values[i]=values[i];
    out_ids[i]=(slot>=0 && slot<w.candidate_stride)
        ? static_cast<int64_t>(w.candidate_ids[(i/k)*w.candidate_stride+slot]) : -1;
}
template<int K> __global__ void small_k_stream_kernel(const float* scores,const float* qn,
        const float* cn,int b,int n,CoarseHistogramTopKWorkspace w,float* values,int64_t* ids) {
    const int q=blockIdx.x;
    if(q>=b) return;
    float best[K]; int best_id[K]; int candidate_count=0;
    if(threadIdx.x==0) paraselect::search::topk_detail::init_topk_fixed<K>(best,best_id);
    __syncwarp();
    const float upper=w.histogram.upper[q];
    for(int c=threadIdx.x;c<n;c+=32) {
        const float d=coarse_l2_distance(scores[size_t(q)*n+c],qn[q],cn[c]);
        const unsigned mask=__ballot_sync(0xffffffff,d<=upper);
        if(threadIdx.x==0) candidate_count += __popc(mask);
        unsigned pending=mask;
        while(pending) {
            const int lane=__ffs(pending)-1;
            const float x=__shfl_sync(0xffffffff,d,lane);
            const int id=__shfl_sync(0xffffffff,c,lane);
            if(threadIdx.x==0)
                paraselect::search::topk_detail::insert_topk(x,id,K,best,best_id);
            pending &= pending-1;
        }
    }
    if(threadIdx.x==0) {
        w.candidate_counts[q]=candidate_count;
        w.histogram.selected_counts[q]=candidate_count;
        w.selection_complete[q]=candidate_count>=K;
        for(int i=0;i<K;i++) {
            int p=i;
            for(int j=i+1;j<K;j++) if(best[j]<best[p]) p=j;
            float td=best[i]; best[i]=best[p]; best[p]=td;
            int ti=best_id[i]; best_id[i]=best_id[p]; best_id[p]=ti;
            values[size_t(q)*K+i]=best[i];
            ids[size_t(q)*K+i]=best_id[i];
        }
    }
}

}
bool estimate_threshold_for(int b,int n) {
    return n>=524288 || (n>=131072 && b>=256);
}
int candidate_stride(int b,int n,int k) {
    if(const char* forced=std::getenv("PARASELECT_HISTOGRAM_STRIDE")) {
        const int requested=std::atoi(forced);
        if(requested>=k && requested<=n)return requested;
    }
    const bool estimated=estimate_threshold_for(b,n);
    const size_t padding=estimated ? std::max(size_t(k)/2,size_t(n)/64) : size_t(2048);
    const size_t padded=(size_t(k)+padding+255)/256*256;
    return int(std::min(size_t(n),std::max(size_t(estimated ? 8192:4096),padded)));
}
int coarse_histogram_score_rows(int b,int n,size_t score_budget_bytes,int k) {
    if(b<=0 || b>65535 || n<=0 || k<=0)
        throw std::invalid_argument("invalid histogram scratch shape");
    const int stride=candidate_stride(b,n,std::min(k,n));
    const int capacity=std::min(b,INT_MAX/stride);
    if(score_budget_bytes) {
        if(score_budget_bytes<size_t(n)*sizeof(float))
            throw std::invalid_argument("histogram score budget must fit at least one row");
        return int(std::min(size_t(capacity),score_budget_bytes/(size_t(n)*sizeof(float))));
    }
    const size_t fixed=size_t(b)*(CoarseDistanceHistogram::bins*4+32)+2*sizeof(int);
    // Include CUDA allocation granularity without reserving any PQ code memory.
    const size_t limit=coarse_histogram_workspace_limit_bytes-(16ULL<<20);
    auto bytes=[&](int rows) {
        return fixed+size_t(rows)*(size_t(n)*4+size_t(stride)*16+8)+sort_workspace_bytes(rows,stride);
    };
    int low=0,high=capacity;
    while(low<high) {
        const int mid=low+(high-low+1)/2;
        if(bytes(mid)<=limit)low=mid;else high=mid-1;
    }
    if(!low)throw std::invalid_argument("one histogram row exceeds scratch budget");
    if(low<capacity && low>=64)low=low/64*64;
    return low;
}
size_t coarse_histogram_topk_workspace_bytes(int b,int n,int k,size_t score_budget_bytes) {
    if(k<0)k=n;
    if(b<=0 || b>65535 || n<=0 || k<=0 || k>n)
        throw std::invalid_argument("invalid histogram top-k capacity");
    const int stride=candidate_stride(b,n,k);
    const int rows=std::min(coarse_histogram_score_rows(b,n,score_budget_bytes,k),INT_MAX/stride);
    return size_t(rows)*stride*16 + size_t(b)*(CoarseDistanceHistogram::bins*4+32) +
        size_t(rows)*8 + 2*sizeof(int) +
        sort_workspace_bytes(rows,stride);
}
void coarse_histogram_topk_release(CoarseHistogramTopKWorkspace& w) noexcept {
    cudaFree(w.histogram.counts);cudaFree(w.histogram.minimum);cudaFree(w.histogram.selected_counts);cudaFree(w.histogram.status);
    cudaFree(w.candidates);cudaFree(w.sorted_candidates);
    cudaFree(w.candidate_ids);cudaFree(w.sorted_candidate_ids);
    cudaFree(w.candidate_counts);cudaFree(w.candidate_starts);cudaFree(w.candidate_ends);
    cudaFree(w.selection_complete);cudaFree(w.retry_queries);
    cudaFree(w.sort_temp);
    w={};
}
void coarse_histogram_topk_prepare(CoarseHistogramTopKWorkspace& w,int b,int n,int k,size_t score_budget_bytes) {
    if(w.capacity)throw std::invalid_argument("histogram top-k workspace already prepared");
    if(b<=0||b>65535||n<=0||k<=0||k>n)
        throw std::invalid_argument("invalid histogram top-k capacity");
    const int stride=candidate_stride(b,n,k);
    const int rows=std::min(coarse_histogram_score_rows(b,n,score_budget_bytes,k),INT_MAX/stride);
    w.capacity=b;w.nlist=n;w.k=k;
    w.candidate_stride=stride;
    w.sort_capacity=rows;
    // Small query batches retain the lower-launch exact histogram path.
    w.estimate_threshold=estimate_threshold_for(b,n);
    try {
        allocate(w.histogram.counts,size_t(b)*CoarseDistanceHistogram::bins);
        allocate(w.histogram.minimum,size_t(b)*4);
        w.histogram.maximum=w.histogram.minimum+b;w.histogram.lower=w.histogram.maximum+b;w.histogram.upper=w.histogram.lower+b;
        allocate(w.histogram.selected_counts,b);allocate(w.histogram.status,1);
        allocate(w.candidates,size_t(w.sort_capacity)*w.candidate_stride);allocate(w.sorted_candidates,size_t(w.sort_capacity)*w.candidate_stride);
        allocate(w.candidate_ids,size_t(w.sort_capacity)*w.candidate_stride);allocate(w.sorted_candidate_ids,size_t(w.sort_capacity)*w.candidate_stride);
        allocate(w.candidate_counts,b);allocate(w.candidate_starts,w.sort_capacity);allocate(w.candidate_ends,w.sort_capacity);
        allocate(w.selection_complete,b);
        allocate(w.retry_queries,size_t(b)+1);w.retry_count=w.retry_queries+b;
        CUDA_CHECK(cudaMemsetAsync(w.selection_complete,0,b*sizeof(int),0));
        offsets_kernel<<<(w.sort_capacity+255)/256,256>>>(w.candidate_starts,w.sort_capacity,w.candidate_stride);
        CUDA_LAUNCH_CHECK("histogram top-k offsets");
        CUDA_CHECK(cudaStreamSynchronize(0));
        w.sort_temp_bytes=sort_workspace_bytes(w.sort_capacity,w.candidate_stride);
        CUDA_CHECK(cudaMalloc(&w.sort_temp,w.sort_temp_bytes));
    } catch(...) {coarse_histogram_topk_release(w);throw;}
}
void coarse_histogram_topk_l2(const float* scores,const float* qn,const float* cn,int b,int n,int k,
    float* values,int64_t* ids,CoarseHistogramTopKWorkspace& w,cudaStream_t stream,
    CoarseHistogramTiming* timing) {
    if(b<=0||b>w.capacity||n<=0||n>w.nlist||k<=0||k>w.k||k>n||!scores||!qn||!cn||!values||!ids)
        throw std::invalid_argument("invalid histogram top-k search shape");
    if(w.reset_status)CUDA_CHECK(cudaMemsetAsync(w.histogram.status,0,sizeof(int),stream));
    if(b>w.sort_capacity) {
        for(int off=0;off<b;off+=w.sort_capacity) {
            const int rows=std::min(w.sort_capacity,b-off);
            auto slice=coarse_histogram_topk_slice(w,off,rows);
            coarse_histogram_topk_l2(scores+size_t(off)*n,qn+off,cn,rows,n,k,
                values+size_t(off)*k,ids+size_t(off)*k,slice,stream,timing);
        }
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(w.candidate_counts,0,b*sizeof(int),stream));
    const bool adaptive_selector = adaptive_selector_enabled() &&
        k >= 8 && k <= 2048 && (k % 8) == 0 &&
        w.candidate_stride >= k && (w.candidate_stride % 256) == 0 &&
        w.candidate_stride < (1 << 23);
    if(adaptive_selector) {
        initialize_selector_pool<<<std::min<size_t>(65535,
            (size_t(b)*w.candidate_stride+255)/256),256,0,stream>>>(b,w);
    }
    const dim3 grid((n+256*compact_rows-1)/(256*compact_rows),b);
    if(w.estimate_threshold) {
        auto estimate = [&] { coarse_distance_histogram_estimate_l2(scores,qn,cn,b,n,k,w.histogram,stream); };
        if(timing) timed(stream,&timing->estimate_ms,estimate); else estimate();
        auto compact = [&] {
            if(adaptive_selector)
                compact_kernel<false,true,true><<<grid,256,0,stream>>>(scores,qn,cn,n,w);
            else
                compact_kernel<false,true><<<grid,256,0,stream>>>(scores,qn,cn,n,w);
        };
        if(timing) timed(stream,&timing->compact_ms,compact); else compact();
        auto refine = [&] {
            verify_estimate<<<1,256,0,stream>>>(b,k,w);
            coarse_distance_histogram_select_l2(scores,qn,cn,b,n,k,w.histogram,stream,
                w.selection_complete,{w.retry_queries,w.retry_count});
            const dim3 retry_grid(grid.x,std::min(b,8));
            if(adaptive_selector)
                compact_kernel<true,false,true><<<retry_grid,256,0,stream>>>(scores,qn,cn,n,w);
            else
                compact_kernel<true><<<retry_grid,256,0,stream>>>(scores,qn,cn,n,w);
        };
        if(timing) timed(stream,&timing->refine_ms,refine); else refine();
    } else {
        constexpr bool kEnableSmallKStreaming = false;
        auto compact = [&] {
            coarse_distance_histogram_select_l2(scores,qn,cn,b,n,k,w.histogram,stream);
            if (kEnableSmallKStreaming && b >= 128 && k == 8) small_k_stream_kernel<8><<<b,32,0,stream>>>(scores,qn,cn,b,n,w,values,ids);
            else if (kEnableSmallKStreaming && b >= 128 && k == 16) small_k_stream_kernel<16><<<b,32,0,stream>>>(scores,qn,cn,b,n,w,values,ids);
            else if (kEnableSmallKStreaming && b >= 128 && k == 32) small_k_stream_kernel<32><<<b,32,0,stream>>>(scores,qn,cn,b,n,w,values,ids);
            else if(adaptive_selector)
                compact_kernel<false,false,true><<<grid,256,0,stream>>>(scores,qn,cn,n,w);
            else compact_kernel<false><<<grid,256,0,stream>>>(scores,qn,cn,n,w);
        };
        if(timing) timed(stream,&timing->compact_ms,compact); else compact();
        if (kEnableSmallKStreaming && b >= 128 && (k == 8 || k == 16 || k == 32)) {
            CUDA_LAUNCH_CHECK("histogram small-k stream top-k");
            return;
        }
    }
    const bool sparse_overflow = [] {
        const char* value=std::getenv("PARASELECT_HISTOGRAM_SPARSE_OVERFLOW");
        return value && std::strcmp(value,"1")==0;
    }();
    auto overflow = [&] {
        if(!sparse_overflow) {
            if(adaptive_selector)
                refine_overflow<true><<<b,256,0,stream>>>(scores,qn,cn,n,k,w,nullptr,0);
            else
                refine_overflow<false><<<b,256,0,stream>>>(scores,qn,cn,n,k,w,nullptr,0);
            return;
        }
        CUDA_CHECK(cudaMemsetAsync(w.retry_count,0,sizeof(int),stream));
        collect_overflow<<<(b+255)/256,256,0,stream>>>(b,k,w);
        int overflow_count=0;
        CUDA_CHECK(cudaMemcpyAsync(&overflow_count,w.retry_count,sizeof(int),
                                   cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if(overflow_count>0)
            if(adaptive_selector)
                refine_overflow<true><<<overflow_count,256,0,stream>>>(scores,qn,cn,n,k,w,
                                                                        w.retry_queries,overflow_count);
            else
                refine_overflow<false><<<overflow_count,256,0,stream>>>(scores,qn,cn,n,k,w,
                                                                         w.retry_queries,overflow_count);
    };
    if(timing)timed(stream,&timing->refine_ms,overflow);else overflow();
    auto sort = [&] {
#ifdef PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK
        if(adaptive_selector) {
            adaptive_threshold_topk_negative_distance(w.candidates,b,w.candidate_stride,
                w.candidate_stride,k,values,ids,stream);
        } else
#endif
        {
            CUDA_CHECK(cub::DeviceSegmentedSort::SortPairs(w.sort_temp,w.sort_temp_bytes,
                w.candidates,w.sorted_candidates,w.candidate_ids,w.sorted_candidate_ids,
                b*w.candidate_stride,b,w.candidate_starts,w.candidate_ends,stream));
        }
    };
    if(timing) timed(stream,&timing->sort_ms,sort); else sort();
    auto gather = [&] {
        if(adaptive_selector) {
            gather_selector_kernel<<<(size_t(b)*k+255)/256,256,0,stream>>>(
                b,k,w,values,ids,values,ids);
        } else {
            gather_kernel<<<(size_t(b)*k+255)/256,256,0,stream>>>(b,w.candidate_stride,k,w,values,ids);
        }
    };
    if(timing) timed(stream,&timing->gather_ms,gather); else gather();
    CUDA_LAUNCH_CHECK("histogram exact top-k");
}
}
