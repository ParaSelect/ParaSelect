#pragma once
// Included inside current::pc. Only retained SM120 producer specializations.
#include "wmma_body.cuh"
template<bool Owned, bool RejectEmpty, bool CenterL1>
__global__ __launch_bounds__(256,2) void swizzle_producer(
    const __half* q,const __half* c,const float* qn,const float* cn,
    int batch,int nlist,int parts,int capacity,const float* thresholds,
    float* values,int* ids,int* counts,int tile) {
    wmma_pipeline_body<32,false,64,true,true,true,true,Owned,RejectEmpty,
        false,true,false,true,true,false,CenterL1,false>(
        q,c,qn,cn,batch,nlist,parts,capacity,thresholds,values,ids,counts,tile);
}
__global__ __launch_bounds__(256,2) void register_producer(
    const __half* q,const __half* c,const float* qn,const float* cn,
    int batch,int nlist,int parts,int capacity,const float* thresholds,
    float* values,int* ids,int* counts,int tile) {
    wmma_pipeline_body<32,false,64,true,true,false,false,false,false,false,false,false>(
        q,c,qn,cn,batch,nlist,parts,capacity,thresholds,values,ids,counts,tile);
}
using Producer = void (*)(const detail::Execution&,const Plan&,const float*,float*,int*,int*);
template<bool Owned,bool RejectEmpty,bool CenterL1>
void launch_swizzle(const detail::Execution& a,const Plan& p,const float* t,float* v,int* ids,int* counts) {
    constexpr auto bytes=sizeof(WmmaPipelineStorage<32,64,true,true,true>)+64*(2*sizeof(float)+sizeof(int));
    const int tile=p.nlist/p.parts/p.splits;
    swizzle_producer<Owned,RejectEmpty,CenterL1><<<dim3((a.n_query+63)/64,p.nlist/tile),256,bytes,a.stream>>>(
        a.d_query_fp16,a.d_cluster_centers_fp16,a.d_query_norm,a.d_cluster_centers_norm,
        a.n_query,p.nlist,p.parts,p.capacity,t,v,ids,counts,tile);
}
void launch_register(const detail::Execution& a,const Plan& p,const float* t,float* v,int* ids,int* counts) {
    constexpr auto bytes=sizeof(WmmaPipelineStorage<32,64,true,true>);
    const int tile=p.nlist/p.parts/p.splits;
    register_producer<<<dim3((a.n_query+63)/64,p.nlist/tile),256,bytes,a.stream>>>(
        a.d_query_fp16,a.d_cluster_centers_fp16,a.d_query_norm,a.d_cluster_centers_norm,
        a.n_query,p.nlist,p.parts,p.capacity,t,v,ids,counts,tile);
}
template<bool Owned,bool RejectEmpty,bool CenterL1>
Producer prepare_swizzle() {
    constexpr auto bytes=sizeof(WmmaPipelineStorage<32,64,true,true,true>)+64*(2*sizeof(float)+sizeof(int));
    CUDA_CHECK(cudaFuncSetAttribute(swizzle_producer<Owned,RejectEmpty,CenterL1>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    return launch_swizzle<Owned,RejectEmpty,CenterL1>;
}
template<bool Owned,bool RejectEmpty>
Producer prepare_cache(bool l1) {
    return l1?prepare_swizzle<Owned,RejectEmpty,true>():prepare_swizzle<Owned,RejectEmpty,false>();
}
Producer prepare_producer(const Plan& p) {
    if(p.register_producer) {
        CUDA_CHECK(cudaFuncSetAttribute(register_producer,cudaFuncAttributeMaxDynamicSharedMemorySize,
            sizeof(WmmaPipelineStorage<32,64,true,true>)));
        return launch_register;
    }
    const bool l1=p.layout==Layout::SwizzleL1;
    if(p.splits==1)return p.reject_empty?prepare_cache<true,true>(l1):prepare_cache<true,false>(l1);
    return p.reject_empty?prepare_cache<false,true>(l1):prepare_cache<false,false>(l1);
}
