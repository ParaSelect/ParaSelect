#pragma once
// Center rows are normally loaded directly in the fragment layout. The
// coalesced variant reads a tile already packed in the MMA lane order, so one
// uint2 load supplies the same B fragment without shuffles.
template<bool Coalesced>
__device__ __forceinline__ void load_center_fragment(
    const __half* centers, int begin, int warp, int row, int k, int col,
    int lane, unsigned& b0, unsigned& b1) {
 if constexpr (!Coalesced) {
  const __half* bp=centers+size_t(begin+warp*8+row)*128+k+col;
  b0=*reinterpret_cast<const unsigned*>(bp);
  b1=*reinterpret_cast<const unsigned*>(bp+8);
 } else {
  const size_t tile_index=(size_t(begin)/8+warp)*8+size_t(k)/16;
  const __half* tile=centers+tile_index*128;
  const uint2 packed=*reinterpret_cast<const uint2*>(tile+lane*4);
  b0=packed.x;
  b1=packed.y;
 }
}

// D128 assignment top1 kernel, 64 queries per CTA. MMA accumulator stays in
// registers; ldmatrix loads each query sub-tile as four packed registers.
// Output tie rule is min ID.
template<bool Coalesced>
__global__ void fused_gemm_local_top1(
 const __half* queries,const __half* centers,const float* query_norm,const float* center_norm,
 int batch,int chunk,int pool_stride,float* pool_scores,float* pool_ids) {
 __shared__ __align__(32) __half query[64*136];
 __shared__ float minima[8][64];
 __shared__ int ids[8][64];
 const int t=threadIdx.x, lane=t%32, warp=t/32, row=lane/4, col=(lane%4)*2;
 const int base=blockIdx.x*64;
 for(int i=t;i<64*136;i+=256) query[i]=(base+i/136<batch && i%136<128)?queries[size_t(base+i/136)*128+i%136]:__float2half(0.f);
 __syncthreads();
 float best[8]; int bid[8];
#pragma unroll
 for(int i=0;i<8;++i){best[i]=-INFINITY;bid[i]=INT_MAX;}
 for(int begin=blockIdx.y*chunk;begin<(blockIdx.y+1)*chunk;begin+=64){
  float d[4][4]={};
  for(int k=0;k<128;k+=16){
   unsigned b0,b1;
   load_center_fragment<Coalesced>(centers,begin,warp,row,k,col,lane,b0,b1);
   #pragma unroll
   for(int g=0;g<4;++g){
    unsigned a0,a1,a2,a3;
    unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(query+(g*16+lane%16)*136+k+(lane/16)*8));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "r"(addr));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
     : "+f"(d[g][0]),"+f"(d[g][1]),"+f"(d[g][2]),"+f"(d[g][3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
   }
  }
  #pragma unroll
  for(int j=0;j<2;++j){
   int id=begin+warp*8+col+j;float cn=center_norm[id];
   cn=__shfl_sync(0xffffffffu,cn,lane&3);
   #pragma unroll
   for(int i=0;i<8;++i){
    float dot=d[i/2][(i%2)*2+j];
    float s=__fsub_rn(__fadd_rn(dot,dot),__fmul_rn(cn,cn));
    if(s>best[i]||(s==best[i]&&id<bid[i])){best[i]=s;bid[i]=id;}
   }
  }
 }
 #pragma unroll
 for(int i=0;i<8;++i){
  #pragma unroll
  for(int delta=2;delta>0;delta>>=1){float s=__shfl_down_sync(0xffffffff,best[i],delta,4);int id=__shfl_down_sync(0xffffffff,bid[i],delta,4);if(lane%4+delta<4&&(s>best[i]||(s==best[i]&&id<bid[i]))){best[i]=s;bid[i]=id;}}
  if(lane%4==0){int r=row+(i%2)*8+(i/2)*16;minima[warp][r]=best[i];ids[warp][r]=bid[i];}
 }
 __syncthreads();
 if(t<64&&base+t<batch){float s=-INFINITY;int id=INT_MAX;
  #pragma unroll
  for(int w=0;w<8;++w){float v=minima[w][t];int c=ids[w][t];if(v>s||(v==s&&c<id)){s=v;id=c;}}
  size_t off=size_t(base+t)*pool_stride+blockIdx.y;pool_scores[off]=s;pool_ids[off]=__int_as_float(id);
 }
}

// Large-batch variant: 128 queries per CTA.
template<bool Coalesced>
__global__ void fused_gemm_local_top1_128(
 const __half* queries,const __half* centers,const float* query_norm,const float* center_norm,
 int batch,int chunk,int pool_stride,float* pool_scores,float* pool_ids) {
 __shared__ __align__(32) __half query[128*136];
 __shared__ float minima[8][128];
 __shared__ int ids[8][128];
 const int t=threadIdx.x, lane=t%32, warp=t/32, row=lane/4, col=(lane%4)*2;
 const int base=blockIdx.x*128;
 for(int i=t;i<128*136;i+=256) query[i]=(base+i/136<batch && i%136<128)?queries[size_t(base+i/136)*128+i%136]:__float2half(0.f);
 __syncthreads();
 float best[16]; int bid[16];
#pragma unroll
 for(int i=0;i<16;++i){best[i]=-INFINITY;bid[i]=INT_MAX;}
 for(int begin=blockIdx.y*chunk;begin<(blockIdx.y+1)*chunk;begin+=64){
  float d[8][4]={};
  for(int k=0;k<128;k+=16){
   unsigned b0,b1;
   load_center_fragment<Coalesced>(centers,begin,warp,row,k,col,lane,b0,b1);
   #pragma unroll
   for(int g=0;g<8;++g){
    unsigned a0,a1,a2,a3;
    unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(query+(g*16+lane%16)*136+k+(lane/16)*8));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "r"(addr));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
     : "+f"(d[g][0]),"+f"(d[g][1]),"+f"(d[g][2]),"+f"(d[g][3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
   }
  }
  #pragma unroll
  for(int j=0;j<2;++j){
   int id=begin+warp*8+col+j;float cn=center_norm[id];
   cn=__shfl_sync(0xffffffffu,cn,lane&3);
   #pragma unroll
   for(int i=0;i<16;++i){
    float dot=d[i/2][(i%2)*2+j];
    float s=__fsub_rn(__fadd_rn(dot,dot),__fmul_rn(cn,cn));
    if(s>best[i]||(s==best[i]&&id<bid[i])){best[i]=s;bid[i]=id;}
   }
  }
 }
 #pragma unroll
 for(int i=0;i<16;++i){
  #pragma unroll
  for(int delta=2;delta>0;delta>>=1){float s=__shfl_down_sync(0xffffffff,best[i],delta,4);int id=__shfl_down_sync(0xffffffff,bid[i],delta,4);if(lane%4+delta<4&&(s>best[i]||(s==best[i]&&id<bid[i]))){best[i]=s;bid[i]=id;}}
  if(lane%4==0){int r=row+(i%2)*8+(i/2)*16;minima[warp][r]=best[i];ids[warp][r]=bid[i];}
 }
 __syncthreads();
 if(t<128&&base+t<batch){float s=-INFINITY;int id=INT_MAX;
  #pragma unroll
  for(int w=0;w<8;++w){float v=minima[w][t];int c=ids[w][t];if(v>s||(v==s&&c<id)){s=v;id=c;}}
  size_t off=size_t(base+t)*pool_stride+blockIdx.y;pool_scores[off]=s;pool_ids[off]=__int_as_float(id);
 }
}
