#pragma once
#include <cuda/barrier>
#include <climits>
#include "pq_codeword_scan.cuh"
#include "pq_threshold_status.cuh"
// Internal U8 scan inputs; quantization belongs to the hybrid wrapper.
namespace paraselect::search::detail {
struct PQCodewordView {
 const uint8_t* codebook;
 const uint8_t* queries;
 const float4* query_affine;
 const float4* book_affine;
};
template<int BQ, int ProducerThreads=(BQ==8?64:128)>
__global__ void u8_pipeline(PQCodewordView input,PQQueryLutView lut,IVFProbeScheduleView schedule,IVFProbeTileScheduleView tiles,const uint8_t* codes,const long long* offsets,const int* counts,const float* thresholds,int nq,int np,PQAdcThresholdCandidatePoolView output,int probe_begin,int probe_end,unsigned long long* physical_rows=nullptr){
 static_assert(BQ>=8 && BQ<=64 && BQ%8==0);
 static_assert(ProducerThreads==32||ProducerThreads==64||ProducerThreads==128);
 constexpr int BV=32,S=144,P=ProducerThreads,T=P+BQ/8*32;
 const float4 book_affine=*input.book_affine;
 int entry=blockIdx.x,begin=0;
 if(tiles.entry_tile_offsets){int lo=0,hi=schedule.entry_count;while(lo<hi){int mid=(lo+hi)/2;if(tiles.entry_tile_offsets[mid+1]<=int(blockIdx.x))lo=mid+1;else hi=mid;}entry=lo;if(entry>=schedule.entry_count)return;begin=(int(blockIdx.x)-tiles.entry_tile_offsets[entry])*tiles.tile_rows;}
 int cluster=schedule.clusters[entry],visitors=schedule.counts[entry];
 int end=tiles.entry_tile_offsets?min(counts[cluster],begin+tiles.tile_rows):counts[cluster];long long base=offsets[cluster];
 if(base<0||base+end>INT_MAX){if(threadIdx.x==0)record_threshold_status(output.status,4);return;}
 if(begin>=end)return;
 __shared__ __align__(32) uint8_t v[2][BV*S];
 __shared__ int ids[BQ],sums[2][BV];
 __shared__ float4 score_affine[BQ];
 __shared__ float qcut_cache[BQ];
 __shared__ float bias[2][BV];
 __shared__ cuda::barrier<cuda::thread_scope_block> ready[2],free[2];
 int t=threadIdx.x,lane=t%32;
 if(t==0)for(int b=0;b<2;b++){init(&ready[b],T);init(&free[b],T);}__syncthreads();
 for(int first=0;first<visitors;first+=BQ){
  if(t==0&&physical_rows)atomicAdd(physical_rows,static_cast<unsigned long long>(end-begin));
  if(t<BQ){int ref=schedule.starts[entry]+first+t;int id=first+t<visitors?schedule.queries[ref]:-1;int rank=first+t<visitors?schedule.probe_ranks[ref]:-1;
   if(first+t<visitors&&(id<0||id>=nq||rank<0||rank>=np))record_threshold_status(output.status,3);
   ids[t]=id>=0&&id<nq&&rank>=probe_begin&&rank<probe_end?id:-1;
   const auto qa=ids[t]>=0?input.query_affine[ids[t]]:make_float4(0,0,0,0);
   const float base=qa.x*book_affine.y*qa.z+128.0f*qa.y*book_affine.y;
   const float center=ids[t]>=0?lut.query_centroid_distances[size_t(ids[t])*np+rank]:0;
   // Fold query-only terms once; vector correction and U8 sum remain per row.
   score_affine[t]=make_float4(-2.f*qa.x*book_affine.x,
      -2.f*qa.y*book_affine.x,center-2.f*base,0.f);
   qcut_cache[t]=ids[t]>=0?thresholds[ids[t]]:0;}
  __syncthreads();int nt=(end-begin+BV-1)/BV;
  unsigned qreg[4][2];
  if(t>=P){int qr=(t-P)/32*8;
   // Query operands belong to this consumer warp. Resolve their immutable
   // schedule references directly instead of depending on producer metadata.
   int qi=first+qr+lane/4,ref=schedule.starts[entry]+qi;
   int id=qi<visitors?schedule.queries[ref]:-1;
   int rank=qi<visitors?schedule.probe_ranks[ref]:-1;
   if(id<0||id>=nq||rank<probe_begin||rank>=probe_end)id=-1;
#pragma unroll
   for(int k=0;k<4;k++){
    qreg[k][0]=id>=0?*reinterpret_cast<const unsigned*>(input.queries+size_t(id)*128+k*32+(lane%4)*4):0;
    qreg[k][1]=id>=0?*reinterpret_cast<const unsigned*>(input.queries+size_t(id)*128+k*32+(lane%4)*4+16):0;
   }
  }
  __syncthreads();
  if(t<P){
   unsigned packed[128/P];
#pragma unroll
   for(int w=0;w<128/P;++w){int i=t+w*P;packed[w]=begin+i/4<end?reinterpret_cast<const unsigned*>(codes+size_t(base+begin)*16)[i]:0;}
   for(int tile=0;tile<nt;tile++){int b=tile%2,start=begin+tile*BV;if(tile>=2)free[b].arrive_and_wait();
#pragma unroll
    for(int w=0;w<128/P;++w){int i=t+w*P;
    int row=i/4;bool valid=start+row<end;
#pragma unroll
    for(int j=0;j<4;++j){int m=(i%4)*4+j,code=(packed[w]>>(8*j))&255;
     unsigned dst=static_cast<unsigned>(__cvta_generic_to_shared(v[b]+row*S+m*8));const void* src=input.codebook+(m*256+code)*8;
     asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;"::"r"(dst),"l"(src),"r"(valid?8:0):"memory");}
    }
    asm volatile("cp.async.commit_group;":::"memory");
    int next=start+BV;
#pragma unroll
    for(int w=0;w<128/P;++w){int i=t+w*P;packed[w]=next+i/4<end?reinterpret_cast<const unsigned*>(codes+size_t(base+next)*16)[i]:0;
    asm volatile("" : "+r"(packed[w]) :: "memory");}
    asm volatile("cp.async.wait_group 0;":::"memory");
    asm volatile("barrier.sync 1, %0;"::"n"(P):"memory");
    if(t<BV){int sum=0;for(int d=0;d<128;d+=4)sum+=__vsadu4(*reinterpret_cast<unsigned*>(v[b]+t*S+d),0);sums[b][t]=sum;
     int id=int(base+start+t);bias[b][t]=start+t<end?(lut.query_lut_bias.u16?fmaf(float(lut.query_lut_bias.u16[id]),lut.query_lut_bias.scale,lut.query_lut_bias.offset):lut.query_lut_bias.fp32[id]):0;}
    asm volatile("barrier.sync 1, %0;"::"n"(P):"memory");
    (void)ready[b].arrive();
   }
   for(int tile=max(0,nt-2);tile<nt;tile++)free[tile%2].arrive_and_wait();
  }else{
   int warp=(t-P)/32,qr=warp*8;
   for(int tile=0;tile<nt;tile++){int b=tile%2,start=begin+tile*BV;ready[b].arrive_and_wait();
     int accumulators[2][4]={};
#pragma unroll
     for(int k=0;k<4;k++){
#pragma unroll
      for(int h=0;h<2;h++){
       auto& acc=accumulators[h];
      const uint8_t* ptr=v[b]+(h*16+lane/4)*S+k*32+(lane%4)*4;
      unsigned a0=*reinterpret_cast<const unsigned*>(ptr),a1=*reinterpret_cast<const unsigned*>(ptr+8*S),a2=*reinterpret_cast<const unsigned*>(ptr+16),a3=*reinterpret_cast<const unsigned*>(ptr+8*S+16);
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+r"(acc[0]),"+r"(acc[1]),"+r"(acc[2]),"+r"(acc[3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(qreg[k][0]),"r"(qreg[k][1]));
      }
     }
#pragma unroll
    for(int h=0;h<2;h++){
     auto& acc=accumulators[h];
#pragma unroll
     for(int j=0;j<4;j++){int qi=qr+(lane%4)*2+j%2,row=h*16+lane/4+(j/2)*8,id=ids[qi],index=int(base+start+row);bool valid=id>=0&&start+row<end;float distance=0;
      if(valid){const auto affine=score_affine[qi];
       const float offset=fmaf(affine.y,float(sums[b][row]),affine.z)+bias[b][row];
       distance=fmaf(affine.x,float(acc[j]),offset);valid=distance<=qcut_cache[qi];}
     unsigned mask=__ballot_sync(0xffffffff,valid);int slot=0;if(mask){if(lane==0)slot=atomicAdd(output.total_count,__popc(mask));slot=__shfl_sync(0xffffffff,slot,0)+__popc(mask&((1u<<lane)-1));if(valid){if(output.accepted_counts)atomicAdd(output.accepted_counts+id,1ULL);if(slot<output.capacity){output.indices[slot]=index;output.queries[slot]=id;output.distances[slot]=distance;atomicAdd(output.query_counts+id,1);}else record_threshold_status(output.status,1);}}
    }
    }
    (void)free[b].arrive();
   }
  }
  __syncthreads();
 }
}
}
