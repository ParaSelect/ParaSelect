#pragma once
#include "search/kernels/coarse/coarse.h"
#include "search/kernels/coarse/stage/coarse_fp16.cuh"
#include "search/kernels/coarse/topk/adaptive_threshold_topk/adaptive_threshold_topk_fp32.h"
#include "search/kernels/coarse/topk/adaptive_threshold_topk/fused/topk_select.cuh"
#include "cuda_kernels/v3_fp32/topk_select.cuh"
#include "simd.cuh"
namespace paraselect::coarse::current::kernels {
template<class T,int Group=8,int QueryRows=1>
__global__ void simt_dot_pair(const T* queries,const T* centers,int n,float* dots) {
    int t=threadIdx.x,lane=t%Group,c=blockIdx.x*(blockDim.x/Group)+t/Group;
    int q=blockIdx.y*QueryRows;float sum[QueryRows]={};
    if(c<n) {
#pragma unroll
        for(int d=lane*2;d<128;d+=Group*2) {
            float2 y=vector_pair(centers+size_t(c)*128+d);
#pragma unroll
            for(int r=0;r<QueryRows;++r) {
                float2 x=vector_pair(queries+size_t(q+r)*128+d);
                sum[r]=fmaf(x.x,y.x,sum[r]);sum[r]=fmaf(x.y,y.y,sum[r]);
            }
        }
    }
#pragma unroll
    for(int r=0;r<QueryRows;++r) {
#pragma unroll
        for(int delta=Group/2;delta;delta/=2)sum[r]+=__shfl_xor_sync(0xffffffff,sum[r],delta,Group);
        if(lane==0 && c<n)dots[size_t(q+r)*n+c]=sum[r];
    }
}

__global__ void finish_adaptive_split(float* values,idx_t* ids,const idx_t* local,
                                    int k,int parts,int width,int total) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<total){int q=i/k;idx_t selected=ids[i];ids[i]=local[size_t(q)*parts*k+selected]+(selected/k)*width;values[i]=-values[i];}
}
__global__ void restore_fused_values(float* values, int total) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<total) values[i]=-values[i];
}
// Merge the already sorted partition heads. The previous rank-count kernel
// binary-searched every other list for every candidate; this emits one output
// per iteration while each lane owns one partition head.
inline void materialized_dot(const paraselect::coarse::detail::Execution& a) {
    // a.dim is the storage width: 128 for padded 96/100/128 inputs, the native width above 128.
    if(a.fp16_coarse)coarse_stage_fp16_gemm(a.d_query_fp16,a.d_cluster_centers_fp16,a.n_query,a.dim,a.nlist,a.d_inner_product,a.cublas_handle,a.stream);
    else{const float alpha=1,beta=0;if(cublasSgemm(a.cublas_handle,CUBLAS_OP_T,CUBLAS_OP_N,a.nlist,a.n_query,a.dim,&alpha,a.d_cluster_centers,a.dim,a.d_query,a.dim,&beta,a.d_inner_product,a.nlist)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cublasSgemm");}
}
// Each round masks only the previous round in owned GEMM scratch. The next
// GEMM refreshes it; inputs and centroid/query norms remain immutable.
__global__ void exclude_fused_round(float* dots,const idx_t* ids,int width,
                                    int output_stride,size_t total) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total;
        i+=size_t(blockDim.x)*gridDim.x) {
        size_t row=i/2048;
        dots[row*width+ids[row*output_stride+i%2048]]=-INFINITY;
    }
}
// Final output handling stays inside the last selection kernel. Scratch is
// exclusively owned by this prepared runtime and fully rewritten before every run.
struct FinishOutput {
    float* values;
    idx_t* ids;
    const idx_t* local_ids;
    int output_stride, count, parts, width;
};
template<class Kernel>
__launch_bounds__(Kernel::NUM_THREADS,Kernel::TARGET_OCCUPANCY,1)
__global__ void topk_finish_kernel(__grid_constant__ const TopkSelectArgs args,
        __grid_constant__ const typename Kernel::TmaParams map, FinishOutput out) {
    Kernel::topk_select_kernel_devfunc(args,map);
    __syncthreads();
    const size_t row=blockIdx.x;
    for(int j=threadIdx.x;j<out.count;j+=blockDim.x) {
        size_t pos=row*out.output_stride+j;
        if(out.local_ids) {
            idx_t selected=out.ids[pos];
            out.ids[pos]=out.local_ids[row*out.parts*out.output_stride+selected]
                +(selected/out.output_stride)*out.width;
        }
        out.values[pos]=-out.values[pos];
    }
}
template<class Config>
inline void select_and_finish(const TopkSelectArgs& args,FinishOutput out) {
    using Kernel=topk_select_fp32::TopkSelectKernelFP32<Config>;
    constexpr size_t smem=sizeof(typename Kernel::SharedMemoryPlanFP32);
    static_assert(smem<=99*1024);
    auto kernel=topk_finish_kernel<Kernel>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    typename Kernel::TmaParams map{Kernel::make_topk_tensor_map(args)};
    topk_finish_kernel<Kernel><<<args.batch_size,Kernel::NUM_THREADS,smem,args.stream>>>(args,map,out);
}
inline void owned_merge_and_finish(const paraselect::coarse::detail::Execution& a,
                                   int parts,float* scores,const idx_t* local_ids) {
    TopkSelectArgs args{};
    args.batch_size=a.n_query;args.vocab_size=parts*a.nprobe;args.input=scores;
    args.stride_input_batch=parts*a.nprobe;
    args.stride_output_value_batch=a.nprobe;args.stride_output_index_batch=a.nprobe;
    args.sorted_value=true;args.return_value=true;args.idx_oob_fill_value=-1;
    args.value_oob_fill_value=-INFINITY;args.abort_when_nan_found=true;args.stream=a.stream;
    int device=0;CUDA_CHECK(cudaGetDevice(&device));cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop,device));args.shared_memory_size_per_sm=prop.sharedMemPerBlockOptin;
    auto run=[&]<int Capacity>() {
        using Config=TopkSelectConfig<float,int64_t,true,false,true,Capacity,128,1,2048,2048,2>;
        for(int start=0;start<a.nprobe;start+=2048) {
            if(start) {
                size_t count=size_t(a.n_query)*2048;
                exclude_fused_round<<<std::min(size_t(65535),(count+255)/256),256,0,a.stream>>>(
                    scores,a.d_top_nprobe_index+start-2048,args.vocab_size,a.nprobe,count);
            }
            args.topk=std::min(2048,a.nprobe-start);
            args.output_value=a.d_top_nprobe_dist+start;args.output_index=a.d_top_nprobe_index+start;
            if(start+args.topk==a.nprobe) {
                select_and_finish<Config>(args,{a.d_top_nprobe_dist,a.d_top_nprobe_index,local_ids,
                    a.nprobe,a.nprobe,parts,a.nlist/parts});
            } else topk_select_fp32::run_topk_select_kernel<Config>(args);
        }
    };
    if(a.nprobe<=512)run.template operator()<512>();
    else run.template operator()<2048>();
}
// Each logical row is one contiguous centroid partition.
template<int Parts, bool VectorNorm=false, int Threads=128, bool MaterializeScores=false>
inline void fused_partition_select(const paraselect::coarse::detail::Execution& a,
                                   float* values, idx_t* ids, bool final_output=false) {
    constexpr int parts=Parts;
    if constexpr(MaterializeScores) {
        if(a.nprobe<=2048)throw std::invalid_argument("materialized-round variant requires k>2048");
    }
    if constexpr(Threads==256) {
        if(a.nprobe>512) throw std::invalid_argument("256-thread variant requires k<=512");
    }
    if constexpr(VectorNorm) {
        if((a.nlist/parts)%512) throw std::invalid_argument("vector norms require full 512-element segments");
    }
    topk_select_fused::FusedArgs args{};
    args.batch_size=a.n_query*parts; args.vocab_size=a.nlist/parts; args.topk=std::min(a.nprobe,2048);
    args.input=a.d_inner_product; args.output_value=values; args.output_index=ids;
    args.stride_input_batch=a.nlist/parts;
    args.stride_output_value_batch=a.nprobe; args.stride_output_index_batch=a.nprobe;
    args.sorted_value=true; args.return_value=true; args.idx_oob_fill_value=-1;
    args.value_oob_fill_value=-INFINITY; args.abort_when_nan_found=true;
    args.query_norm=a.d_query_norm; args.centroid_norm=a.d_cluster_centers_norm;
    args.stream=a.stream; args.final_output=final_output;
    int device=0; CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
    args.shared_memory_size_per_sm=prop.sharedMemPerBlockOptin;
    if(a.nprobe<=512) {
        using Config=TopkSelectConfig<float,int64_t,true,false,true,512,Threads,1,Threads*16,2048,2>;
        topk_select_fused::run_topk_select_kernel<Config,Parts,VectorNorm,MaterializeScores>(args);
    } else {
        // The 256-thread / 2048-capacity layout exceeds SM120 shared memory.
        using Config=TopkSelectConfig<float,int64_t,true,false,true,2048,128,1,2048,2048,2>;
        static_assert(sizeof(typename topk_select_fused::TopkSelectKernelFP32<Config,Parts,VectorNorm,MaterializeScores>::SharedMemoryPlanFP32)<=99*1024);
        topk_select_fused::run_topk_select_kernel<Config,Parts,VectorNorm,MaterializeScores>(args);
        for(int start=2048;start<a.nprobe;start+=2048) {
            const size_t total=size_t(args.batch_size)*2048;
            exclude_fused_round<<<std::min(size_t(65535),(total+255)/256),256,0,a.stream>>>(
                a.d_inner_product,ids+start-2048,args.vocab_size,a.nprobe,total);
            args.topk=std::min(2048,a.nprobe-start);
            args.output_value=values+start; args.output_index=ids+start;
            if constexpr(MaterializeScores) {
                if(final_output)select_and_finish<Config>(args,{values+start,ids+start,nullptr,a.nprobe,int(args.topk),1,a.nlist});
                else topk_select_fp32::run_topk_select_kernel<Config>(args);
            } else topk_select_fused::run_topk_select_kernel<Config,Parts,VectorNorm>(args);
        }
    }
}

}
