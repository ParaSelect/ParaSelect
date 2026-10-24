#pragma once
#include <cuda_fp16.h>
#include <mma.h>
#include <cfloat>
#include <math_constants.h>
#include "quantization/turboquant/turboquant_distance.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_query_lut_scan.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_threshold_status.cuh"

namespace paraselect::search::turboquant {
// Each warp handles candidates for one query; preserve the common pool contract.
__device__ inline void mma_append(bool pass,int query,int row,float value,
                                  PQAdcThresholdCandidatePoolView out){
 const unsigned mask=__ballot_sync(0xffffffffu,pass);int count=__popc(mask);
 if(!count)return;int leader=__ffs(mask)-1,lane=threadIdx.x&31,base=0;
 if(lane==leader){
  if(out.accepted_counts)atomicAdd(out.accepted_counts+query,(unsigned long long)count);
  base=atomicAdd(out.total_count,count);int stored=max(0,min(count,out.capacity-base));
  if(stored)atomicAdd(out.query_counts+query,stored);
  if(stored!=count)detail::record_threshold_status(out.status,1);
 }
 base=__shfl_sync(0xffffffffu,base,leader);
 if(pass){int slot=base+__popc(mask&((1u<<lane)-1));if(slot>=0&&slot<out.capacity){out.indices[slot]=row;out.queries[slot]=query;out.distances[slot]=value;}}
}
__device__ inline float mma_refine(SignResidualDeviceView v,int q,int c,int row,float norm){
 const uint4 code=((const uint4*)v.codes)[row];const unsigned words[4]={code.x,code.y,code.z,code.w};float dots[4]={};
 #pragma unroll
 for(int w=0;w<4;w++){
  #pragma unroll 4
  for(int b=0;b<32;b++){int d=w*32+b;float x=v.queries[size_t(q)*128+d]-v.centroids[size_t(c)*128+d];dots[w]+=((words[w]>>b)&1)?x:-x;}
 }
 return reconstructed_distance(norm,v.scales[row],(dots[0]+dots[1])+(dots[2]+dots[3]),128);
}
// Experimental SIFT128 path. Queries remain FP32 in the index. The half operand
// is used only as a conservative admission prefilter; emitted scores use FP32.
__global__ void mma_scan(SignResidualDeviceView v,IVFProbeScheduleView schedule,
 IVFProbeTileScheduleView tiles,const long long* offsets,const int* counts,
 const float* threshold,PQAdcThresholdCandidatePoolView out){
 using namespace nvcuda;
 constexpr int S=136;
 __shared__ __align__(32) half a[32*S],b[64*S];
 __shared__ __align__(32) float products[32*64];
 __shared__ float norms[32],errors[32],magnitudes[32];
 __shared__ float direct[129];
 int entry=blockIdx.x,row_begin=0,row_end;
 if(tiles.entry_tile_offsets){int lo=0,hi=schedule.entry_count;while(lo+1<hi){int m=(lo+hi)/2;if(tiles.entry_tile_offsets[m]<=int(blockIdx.x))lo=m;else hi=m;}entry=lo;row_begin=(blockIdx.x-tiles.entry_tile_offsets[entry])*tiles.tile_rows;}
 int cluster=schedule.clusters[entry],visitors=schedule.counts[entry],start=schedule.starts[entry];
 row_end=min(counts[cluster],row_begin+(tiles.entry_tile_offsets?tiles.tile_rows:counts[cluster]));
 const int lane=threadIdx.x&31,warp=threadIdx.x/32;
 if(visitors<16){
  for(int j=0;j<visitors;j++){int q=schedule.queries[start+j];prepare_residual(v,q,cluster,128,direct);
   for(int r=row_begin;r<row_end;r+=256){int row=int(offsets[cluster])+r+threadIdx.x;bool valid=r+threadIdx.x<row_end;float value=valid?distance(v,direct,row):CUDART_INF_F;
    mma_append(valid&&value<=threshold[q],q,row,value,out);}
   __syncthreads();
  }return;
 }
 for(int q0=0;q0<visitors;q0+=32){
  // One warp per query; four separate norm reductions match prepare_residual.
  for(int j=warp;j<32;j+=8){bool valid=q0+j<visitors;int q=valid?schedule.queries[start+q0+j]:0;float sq[4],error=0,mag=0;
   #pragma unroll
   for(int k=0;k<4;k++){int d=lane+32*k;float x=valid?v.queries[size_t(q)*128+d]-v.centroids[size_t(cluster)*128+d]:0;half h=__float2half_rn(x);a[j*S+d]=h;sq[k]=__fmaf_rn(x,x,0.f);error+=fabsf(x-__half2float(h));mag+=fabsf(x);}
   #pragma unroll
   for(int step=16;step>0;step>>=1){
    #pragma unroll
    for(int k=0;k<4;k++)sq[k]+=__shfl_down_sync(0xffffffffu,sq[k],step);
    error+=__shfl_down_sync(0xffffffffu,error,step);mag+=__shfl_down_sync(0xffffffffu,mag,step);
   }
   if(lane==0){norms[j]=((sq[0]+sq[1])+sq[2])+sq[3];errors[j]=error;magnitudes[j]=mag;}
  }
  __syncthreads();
  for(int r0=row_begin;r0<row_end;r0+=64){
   for(int i=threadIdx.x;i<64*16;i+=256){int r=r0+i/16,byte=i%16;unsigned bits=r<row_end?((const unsigned char*)v.codes)[(offsets[cluster]+r)*16+byte]:0;unsigned packed[4];
    #pragma unroll
    for(int k=0;k<4;k++){unsigned h0=(bits>>(2*k))&1?0x3c00:0xbc00,h1=(bits>>(2*k+1))&1?0x3c00:0xbc00;packed[k]=h0|(h1<<16);}
    ((uint4*)b)[((i/16)*S+byte*8)/8]=make_uint4(packed[0],packed[1],packed[2],packed[3]);
   }
   __syncthreads();
   wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> fa;
   wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> fb;
   wmma::fragment<wmma::accumulator,16,16,16,float> fc;wmma::fill_fragment(fc,0.f);
   #pragma unroll
   for(int k=0;k<128;k+=16){wmma::load_matrix_sync(fa,a+(warp/4)*16*S+k,S);wmma::load_matrix_sync(fb,b+(warp%4)*16*S+k,S);wmma::mma_sync(fc,fa,fb,fc);}
   wmma::store_matrix_sync(products+(warp/4)*16*64+(warp%4)*16,fc,64,wmma::mem_row_major);__syncthreads();
   for(int i=threadIdx.x;i<32*64;i+=256){int j=i/64,r=r0+i%64;bool valid=q0+j<visitors&&r<row_end;int q=valid?schedule.queries[start+q0+j]:0,row=int(offsets[cluster])+r;float value=CUDART_INF_F;bool pass=false;
    if(valid){float scale=v.scales[row],dot=products[i];float approx=reconstructed_distance(norms[j],scale,dot,128);
     // Bound half conversion plus both FP32 accumulation paths and epilogues.
     float guard=2*fabsf(scale)*(errors[j]+256*FLT_EPSILON*magnitudes[j])+16*FLT_EPSILON*(fabsf(norms[j])+128*scale*scale+2*fabsf(scale*dot));
     if(!isfinite(approx)||approx-guard<=threshold[q]){value=mma_refine(v,q,cluster,row,norms[j]);pass=value<=threshold[q];}
    }
    mma_append(pass,q,row,value,out);
   }
   __syncthreads();
  }
  __syncthreads();
 }
}
} // namespace paraselect::search::turboquant
