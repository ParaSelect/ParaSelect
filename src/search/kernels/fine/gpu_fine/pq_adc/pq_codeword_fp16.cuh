#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace pq_fp16_detail {
__device__ inline void copy_async(void* dst,const void* src){
    unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(address),"l"(src) : "memory");
}
__device__ inline void wait(){asm volatile("cp.async.wait_group 0;" ::: "memory");}

// One consumer warp: 16 vector rows, eight queries, cached query fragments.
__device__ inline void multiply(const half* vectors,int stride,float (&c)[4],const unsigned (&queries)[8][2]){
    int lane=threadIdx.x&31;
#pragma unroll
    for(int k=0;k<128;k+=16){
        unsigned a0,a1,a2,a3;
        unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(vectors+(lane%16)*stride+k+(lane/16)*8));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "r"(address));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(queries[k/16][0]),"r"(queries[k/16][1]));
    }
}

// Four producer warps gather four consecutive subspaces per thread.
template<int Stride>
__device__ inline void decode(half* dst,const half* book,int valid_rows,unsigned packed){
    int row=threadIdx.x/4,m0=(threadIdx.x%4)*4;
#pragma unroll 1
    for(int j=0;j<4;++j){
        int m=m0+j,index=packed&255;packed>>=8;
        if(row<valid_rows)copy_async(dst+row*Stride+m*8,book+(m*256+index)*8);
        else *reinterpret_cast<uint4*>(dst+row*Stride+m*8)=make_uint4(0,0,0,0);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}
} // namespace pq_fp16_detail
