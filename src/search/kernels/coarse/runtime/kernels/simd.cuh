#pragma once
#include <math_constants.h>
#include "simd_shape.h"
// Shared exact SIMD tile selection and merge; the legacy route retains scalar loads.
namespace paraselect::coarse::current::kernels {
template<class T> __device__ float2 vector_pair(const T* p);
template<> __device__ float2 vector_pair<float>(const float* p) {
    return *reinterpret_cast<const float2*>(p);
}
template<> __device__ float2 vector_pair<__half>(const __half* p) {
    return __half22float2(*reinterpret_cast<const __half2*>(p));
}

namespace simd_fused {
template<class T> __device__ float scalar(T v) { return float(v); }
template<> __device__ float scalar<__half>(__half v) { return __half2float(v); }
template<class T,int Tile,int Group=8,bool Paired=false,int QueryRows=1>
__global__ void distance_local_sort(const T* queries,const T* centers,
    const float* qnorm,const float* cnorm,int n,int k,float* values,idx_t* indices) {
    constexpr int Threads=128,Items=Tile/Threads;
    using Sort=cub::BlockRadixSort<float,Threads,Items,idx_t>;
    __shared__ typename Sort::TempStorage temp;
    __shared__ float scores[QueryRows][Tile];
    __shared__ float query[QueryRows][128];
    int t=threadIdx.x,q=blockIdx.y*QueryRows,begin=blockIdx.x*Tile;
#pragma unroll
    for(int r=0;r<QueryRows;++r)query[r][t]=scalar(queries[size_t(q+r)*128+t]);
    __syncthreads();
    for(int row=t/Group;row<Tile;row+=Threads/Group) {
        int c=begin+row;float sum[QueryRows]={};
        if(c<n) {
            if constexpr(Paired) {
#pragma unroll
                for(int d=(t%Group)*2;d<128;d+=Group*2) {
                    float2 y=vector_pair(centers+size_t(c)*128+d);
#pragma unroll
                    for(int r=0;r<QueryRows;++r) {
                        sum[r]=fmaf(query[r][d],y.x,sum[r]);
                        sum[r]=fmaf(query[r][d+1],y.y,sum[r]);
                    }
                }
            } else {
#pragma unroll
                for(int d=t%Group;d<128;d+=Group) {
                    const float y=scalar(centers[size_t(c)*128+d]);
#pragma unroll
                    for(int r=0;r<QueryRows;++r)sum[r]=fmaf(query[r][d],y,sum[r]);
                }
            }
        }
#pragma unroll
        for(int r=0;r<QueryRows;++r) {
#pragma unroll
            for(int delta=Group/2;delta;delta/=2)sum[r]+=__shfl_xor_sync(0xffffffff,sum[r],delta,Group);
            if(t%Group==0)scores[r][row]=c<n?qnorm[q+r]*qnorm[q+r]+cnorm[c]*cnorm[c]-2.f*sum[r]:CUDART_INF_F;
        }
    }
    __syncthreads();
#pragma unroll
    for(int r=0;r<QueryRows;++r) {
        float keys[Items];idx_t ids[Items];
#pragma unroll
        for(int j=0;j<Items;++j){int row=t*Items+j;keys[j]=scores[r][row];ids[j]=begin+row<n?begin+row:-1;}
        Sort(temp).SortBlockedToStriped(keys,ids);
#pragma unroll
        for(int j=0;j<Items;++j){int rank=j*Threads+t;if(rank<k){size_t off=(size_t(q+r)*gridDim.x+blockIdx.x)*k+rank;values[off]=keys[j];indices[off]=ids[j];}}
        // The two queries share sort scratch, but never candidate/output slots.
        if constexpr(QueryRows>1)__syncthreads();
    }
}

template<int Tile=SimdShape::merge_tile>
__global__ void merge_local_sort(const float* input,const idx_t* input_ids,int length,int k,
                                float* output,idx_t* output_ids) {
    constexpr int Threads=128,Items=Tile/Threads;
    using Sort=cub::BlockRadixSort<float,Threads,Items,idx_t>;
    __shared__ typename Sort::TempStorage temp;
    float keys[Items];idx_t ids[Items];int t=threadIdx.x,q=blockIdx.y;
#pragma unroll
    for(int j=0;j<Items;++j){int pos=blockIdx.x*Tile+t*Items+j;size_t off=size_t(q)*length+pos;keys[j]=pos<length?input[off]:CUDART_INF_F;ids[j]=pos<length?input_ids[off]:-1;}
    Sort(temp).SortBlockedToStriped(keys,ids);
#pragma unroll
    for(int j=0;j<Items;++j){int rank=j*Threads+t;if(rank<k){size_t off=(size_t(q)*gridDim.x+blockIdx.x)*k+rank;output[off]=keys[j];output_ids[off]=ids[j];}}
}

inline void finish(const paraselect::coarse::detail::Execution& a,const SimdShape& shape,
                   float* s0,float* s1,idx_t* i0,idx_t* i1) {
    int length=shape.stride;
    while(true) {
        int groups=SimdShape::merge_blocks(length);bool last=groups==1;
        float* dst=last?a.d_top_nprobe_dist:s1;
        idx_t* did=last?a.d_top_nprobe_index:i1;
        merge_local_sort<<<dim3(groups,a.n_query),128,0,a.stream>>>(s0,i0,length,a.nprobe,dst,did);
        if(last)break;
        length=groups*a.nprobe;std::swap(s0,s1);std::swap(i0,i1);
    }
}
inline void paired(const paraselect::coarse::detail::Execution& a,
                   float* s0,float* s1,idx_t* i0,idx_t* i1) {
    if(a.nprobe>128)throw std::invalid_argument("SIMD paired tile requires k<=128");
    const SimdShape shape(a.n_query,a.nlist,a.nprobe);
    // Only the measured large-center regime benefits from sharing a CTA.
    if(a.n_query==2 && a.fp16_coarse && a.nlist>=524288) {
        distance_local_sort<__half,SimdShape::tile,8,true,2><<<dim3(shape.tiles,1),128,0,a.stream>>>(
            a.d_query_fp16,a.d_cluster_centers_fp16,a.d_query_norm,a.d_cluster_centers_norm,a.nlist,a.nprobe,s0,i0);
    } else {
        const dim3 grid(shape.tiles,a.n_query);
        if(a.fp16_coarse)distance_local_sort<__half,SimdShape::tile,8,true><<<grid,128,0,a.stream>>>(
            a.d_query_fp16,a.d_cluster_centers_fp16,a.d_query_norm,a.d_cluster_centers_norm,a.nlist,a.nprobe,s0,i0);
        else distance_local_sort<float,SimdShape::tile,8,true><<<grid,128,0,a.stream>>>(
            a.d_query,a.d_cluster_centers,a.d_query_norm,a.d_cluster_centers_norm,a.nlist,a.nprobe,s0,i0);
    }
    finish(a,shape,s0,s1,i0,i1);
}
}
}
