#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include "search/kernels/coarse/stage/coarse_distance_histogram.cuh"
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_radix_sort.cuh>
#include "search/kernels/topk/fixed_topk.cuh"

// Experimental reducer implementation. Selection belongs to the preparation layer.
// Each CTA scans a bounded tile; the second launch reduces their exact top-k union.
namespace coarse_small_k {
struct Item { float distance; int id; };
struct Minimum {
    __device__ Item operator()(Item a,Item b) const {
        return a.distance<b.distance || (a.distance==b.distance && a.id<b.id) ? a : b;
    }
};
template<int K,bool Merge,bool L2=false,bool Sparse=true> __global__ void reduce(const float* input,const int* input_ids,
    const float* upper,int n,int tiles,float* output,int* output_ids,const float* qnorm=nullptr,const float* cnorm=nullptr) {
    constexpr int Threads=128;
    using Reduction=cub::BlockReduce<Item,Threads>;
    __shared__ typename Reduction::TempStorage temp;
    __shared__ Item winner;
    const int q=blockIdx.y,tile=blockIdx.x;
    const int width=Merge ? tiles*K : n;
    const int begin=Merge ? 0 : tile*4096;
    const int end=Merge ? width : min(n,begin+4096);
    if constexpr(Sparse) {
    // Sparse tiles avoid per-thread top-k state and K block reductions.
    __shared__ unsigned long long sparse_keys[Threads];
    __shared__ int sparse_count;
    if(threadIdx.x==0)sparse_count=0;
    __syncthreads();
    const int lane=threadIdx.x&31;
    for(int base=begin;base<end;base+=Threads) {
        const int c=base+threadIdx.x;
        float distance=INFINITY;int id=-1;
        if(c<end) {
            distance=input[size_t(q)*width+c];
            if constexpr(L2 && !Merge)distance=paraselect::coarse::coarse_l2_distance(distance,qnorm[q],cnorm[c]);
            id=Merge ? input_ids[size_t(q)*width+c] : c;
        }
        const bool accept=id>=0 && distance<=upper[q];
        const unsigned mask=__ballot_sync(0xffffffff,accept);
        int offset=0;
        if(lane==0 && mask)offset=atomicAdd(&sparse_count,__popc(mask));
        offset=__shfl_sync(0xffffffff,offset,0);
        if(accept) {
            const int slot=offset+__popc(mask&((1u<<lane)-1));
            if(slot<Threads) {
                const unsigned bits=__float_as_uint(distance);
                const unsigned ordered=bits^((bits&0x80000000u)?0xffffffffu:0x80000000u);
                sparse_keys[slot]=(static_cast<unsigned long long>(ordered)<<32)|unsigned(id);
            }
        }
    }
    __syncthreads();
    if(sparse_count<=Threads) {
        unsigned long long key=threadIdx.x<sparse_count?sparse_keys[threadIdx.x]:~0ull;
        __syncthreads();
        #pragma unroll
        for(int width=2;width<=Threads;width*=2) {
            #pragma unroll
            for(int stride=width/2;stride;stride/=2) {
                unsigned long long peer;
                if(stride<32) {
                    peer=__shfl_xor_sync(0xffffffff,key,stride);
                } else {
                    sparse_keys[threadIdx.x]=key;
                    __syncthreads();
                    peer=sparse_keys[threadIdx.x^stride];
                    __syncthreads();
                }
                const bool take_min=((threadIdx.x&width)==0)==((threadIdx.x&stride)==0);
                key=take_min?(key<peer?key:peer):(key>peer?key:peer);
            }
        }
        if(threadIdx.x<K) {
            const size_t out=(size_t(q)*(Merge?1:tiles)+tile)*K+threadIdx.x;
            const unsigned ordered=unsigned(key>>32);
            const unsigned bits=ordered^((ordered&0x80000000u)?0x80000000u:0xffffffffu);
            output[out]=key==~0ull?INFINITY:__uint_as_float(bits);
            output_ids[out]=key==~0ull?-1:int(unsigned(key));
        }
        return;
    }
    if(threadIdx.x==0)output_ids[(size_t(q)*(Merge?1:tiles)+tile)*K]=-2;
    } else {
    // Read the marker once before any output is overwritten.
    __shared__ bool overflow;
    if(threadIdx.x==0)overflow=output_ids[(size_t(q)*(Merge?1:tiles)+tile)*K]==-2;
    __syncthreads();
    if(!overflow)return;
    float best[K];int ids[K];
    paraselect::search::topk_detail::init_topk_fixed<K>(best,ids);
    for(int c=begin+threadIdx.x;c<end;c+=Threads){
        float v=input[size_t(q)*width+c];
        if constexpr(L2 && !Merge) v=paraselect::coarse::coarse_l2_distance(v,qnorm[q],cnorm[c]);
        int id=Merge ? input_ids[size_t(q)*width+c] : c;
        if(id>=0 && v<=upper[q]) {
            // Keep the local list ascending; selection then reads only its head.
            #pragma unroll
            for(int i=0;i<K;++i) {
                if(ids[i]<0 || v<best[i] || (v==best[i] && id<ids[i])) {
                    const float old_distance=best[i];const int old_id=ids[i];
                    best[i]=v;ids[i]=id;v=old_distance;id=old_id;
                }
            }
        }
    }
    int head=0;
    for(int rank=0;rank<K;++rank){
        Item local{INFINITY,2147483647};
        if(head<K && ids[head]>=0)local={best[head],ids[head]};
        Item value=Reduction(temp).Reduce(local,Minimum{});
        if(threadIdx.x==0){winner=value;size_t out=(size_t(q)*(Merge?1:tiles)+tile)*K+rank;output[out]=value.distance;output_ids[out]=value.id==2147483647 ? -1 : value.id;}
        __syncthreads();
        // Most boundary-filtered tiles contain fewer than K candidates.
        // All threads observe the same winner, so this exit is CTA-uniform.
        if(winner.id==2147483647) {
            for(int rest=rank+1+threadIdx.x;rest<K;rest+=Threads) {
                size_t out=(size_t(q)*(Merge?1:tiles)+tile)*K+rest;
                output[out]=INFINITY;output_ids[out]=-1;
            }
            return;
        }
        if(local.id==winner.id)++head;
        __syncthreads();
    }
    }
}
}
// Merge only the tile-local top-k union, never the full centroid matrix.
template<int K,int Items> __global__ void partition_merge(const float* input,const int* ids,
    int width,float* output,int* output_ids) {
    using Sort=cub::BlockRadixSort<float,256,Items,int>;
    __shared__ typename Sort::TempStorage temp;
    float values[Items];int indices[Items];
    #pragma unroll
    for(int i=0;i<Items;++i) {
        const int c=threadIdx.x*Items+i;
        const size_t source=size_t(blockIdx.x)*width+c;
        indices[i]=c<width?ids[source]:-1;
        values[i]=indices[i]>=0?input[source]:INFINITY;
    }
    Sort(temp).Sort(values,indices);
    #pragma unroll
    for(int i=0;i<Items;++i) {
        const int c=threadIdx.x*Items+i;
        if(c<K) {output[size_t(blockIdx.x)*K+c]=values[i];output_ids[size_t(blockIdx.x)*K+c]=indices[i];}
    }
}
template<int K> void partition_merge_launch(const float* input,const int* ids,int batch,int width,
    float* output,int* output_ids,cudaStream_t stream=0) {
    if(width<=256)partition_merge<K,1><<<batch,256,0,stream>>>(input,ids,width,output,output_ids);
    else if(width<=512)partition_merge<K,2><<<batch,256,0,stream>>>(input,ids,width,output,output_ids);
    else if(width<=1024)partition_merge<K,4><<<batch,256,0,stream>>>(input,ids,width,output,output_ids);
    else if(width<=2048)partition_merge<K,8><<<batch,256,0,stream>>>(input,ids,width,output,output_ids);
    else if(width<=4096)partition_merge<K,16><<<batch,256,0,stream>>>(input,ids,width,output,output_ids);
    else throw std::invalid_argument("experimental partition merge supports at most 4096 entries");
}
template<int K> void partition_topk(const float* distances,const float* upper,int batch,int n,
    float* scratch,int* scratch_ids,float* output,int* output_ids,cudaStream_t stream=0) {
    int tiles=(n+4095)/4096;
    coarse_small_k::reduce<K,false><<<dim3(tiles,batch),128,0,stream>>>(distances,nullptr,upper,n,tiles,scratch,scratch_ids);
    coarse_small_k::reduce<K,false,false,false><<<dim3(tiles,batch),128,0,stream>>>(distances,nullptr,upper,n,tiles,scratch,scratch_ids);
    partition_merge_launch<K>(scratch,scratch_ids,batch,tiles*K,output,output_ids,stream);
}

// Identical distance arithmetic to the histogram producer; no L2 materialization.
template<int K> void partition_topk_l2(const float* scores,const float* qnorm,const float* cnorm,
    const float* upper,int batch,int n,float* scratch,int* scratch_ids,float* output,int* output_ids,cudaStream_t stream=0) {
    const int tiles=(n+4095)/4096;
    coarse_small_k::reduce<K,false,true><<<dim3(tiles,batch),128,0,stream>>>(scores,nullptr,upper,n,tiles,scratch,scratch_ids,qnorm,cnorm);
    coarse_small_k::reduce<K,false,true,false><<<dim3(tiles,batch),128,0,stream>>>(scores,nullptr,upper,n,tiles,scratch,scratch_ids,qnorm,cnorm);
    partition_merge_launch<K>(scratch,scratch_ids,batch,tiles*K,output,output_ids,stream);
}

#include "search/kernels/coarse/topk/coarse_histogram_topk.cuh"
#include "search/kernels/coarse/coarse_policy.h"
#include "utils/check_cuda_error.cuh"
namespace paraselect::coarse {
static __global__ void histogram_partition_widen_ids(const int* input, int64_t* output, size_t count) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count) output[i]=input[i];
}
inline size_t histogram_partition_workspace_bytes(int b,int n,int k) {
    return size_t(b)*((n+4095)/4096)*k*8 + size_t(b)*(1044+k*4) + sizeof(int);
}
inline void prepare_histogram_partition_workspace(CoarseHistogramTopKWorkspace& w,int b,int n,int k) {
    const size_t entries=size_t(b)*((n+4095)/4096)*k;
    w.capacity=b; w.nlist=n; w.k=k;
    try {
        CUDA_CHECK(cudaMalloc(&w.histogram.counts,size_t(b)*256*sizeof(unsigned)));
        CUDA_CHECK(cudaMalloc(&w.histogram.minimum,size_t(b)*4*sizeof(float)));
        w.histogram.maximum=w.histogram.minimum+b; w.histogram.lower=w.histogram.maximum+b;
        w.histogram.upper=w.histogram.lower+b;
        CUDA_CHECK(cudaMalloc(&w.histogram.selected_counts,size_t(b)*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&w.histogram.status,sizeof(int)));
        CUDA_CHECK(cudaMalloc(&w.candidates,entries*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&w.candidate_ids,entries*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&w.sorted_candidate_ids,size_t(b)*k*sizeof(int)));
    } catch(...) { coarse_histogram_topk_release(w); throw; }
}
inline void coarse_histogram_partition_l2(const float* scores,const float* qn,const float* cn,
    int batch,int n,int k,float* values,int64_t* ids,CoarseHistogramTopKWorkspace& w,cudaStream_t stream) {
    if(w.capacity<batch || w.nlist!=n || w.k<k || !w.candidates || !w.candidate_ids || !w.sorted_candidate_ids)
        throw std::invalid_argument("histogram_partition_smallk requires matching prepared workspace");
    coarse_distance_histogram_select_l2(scores,qn,cn,batch,n,k,w.histogram,stream);
    switch(k) {
        case 8: partition_topk_l2<8>(scores,qn,cn,w.histogram.upper,batch,n,w.candidates,w.candidate_ids,values,w.sorted_candidate_ids,stream);break;
        case 16: partition_topk_l2<16>(scores,qn,cn,w.histogram.upper,batch,n,w.candidates,w.candidate_ids,values,w.sorted_candidate_ids,stream);break;
        case 32: partition_topk_l2<32>(scores,qn,cn,w.histogram.upper,batch,n,w.candidates,w.candidate_ids,values,w.sorted_candidate_ids,stream);break;
        default: throw std::invalid_argument("unsupported histogram_partition_smallk k");
    }
    histogram_partition_widen_ids<<<(size_t(batch)*k+255)/256,256,0,stream>>>(w.sorted_candidate_ids,ids,size_t(batch)*k);
}
inline void prepare_coarse_histogram_workspace(int reducer,CoarseHistogramTopKWorkspace& w,int b,int n,int k,
    size_t score_budget_bytes = coarse_histogram_auto_score_budget) {
    if(reducer==IVF_COARSE_REDUCER_HISTOGRAM_PARTITION_SMALLK) prepare_histogram_partition_workspace(w,b,n,k);
    else if(reducer==IVF_COARSE_REDUCER_HISTOGRAM_SORT) coarse_histogram_topk_prepare(w,b,n,k,score_budget_bytes);
    else throw std::invalid_argument("not a histogram reducer");
}
inline size_t coarse_histogram_workspace_bytes(int reducer,int b,int n,int k,
    size_t score_budget_bytes = coarse_histogram_auto_score_budget) {
    return reducer==IVF_COARSE_REDUCER_HISTOGRAM_PARTITION_SMALLK
        ? histogram_partition_workspace_bytes(b,n,k) : coarse_histogram_topk_workspace_bytes(b,n,k,score_budget_bytes);
}
}
