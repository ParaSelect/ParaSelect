#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace paraselect::quantization::pq {

static __global__ void encode_serial_kernel(
    const float* __restrict__ data,
    int n,
    int dim,
    int pq_m,
    int pq_k,
    int sub_dim,
    const float* __restrict__ codebooks,
    uint8_t* __restrict__ codes)
{
    const size_t total = (size_t)n * pq_m;
    for (size_t linear = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += (size_t)blockDim.x * gridDim.x) {
        const int i = (int)(linear / pq_m);
        const int m = (int)(linear - (size_t)i * pq_m);
        const float* x = data + (size_t)i * dim + (size_t)m * sub_dim;
        const float* cb = codebooks + (size_t)m * pq_k * sub_dim;
        int best = 0;
        float best_sq = INFINITY;
        for (int k = 0; k < pq_k; ++k) {
            const float* c = cb + (size_t)k * sub_dim;
            float sq = 0.0f;
            for (int d = 0; d < sub_dim; ++d) {
                const float diff = x[d] - c[d];
                sq += diff * diff;
            }
            if (sq < best_sq) {
                best_sq = sq;
                best = k;
            }
        }
        codes[(size_t)i * pq_m + m] = (uint8_t)best;
    }
}

// One warp owns one subvector. Transposed [M,dsub,K] codebooks give adjacent
// lanes adjacent codewords. Each lane retains the original dimension order.
// Equal distances choose the smallest codeword ID, as in the serial baseline.
template<int SubDim, int K=256>
__global__ void encode_warp_kernel(const float* __restrict__ data, int n, int m_count,
    const float* __restrict__ transposed_codebooks, uint8_t* __restrict__ codes) {
    const int lane=threadIdx.x&31;
    const int warps=blockDim.x/32;
    const size_t total=size_t(n)*m_count;
    for(size_t item=size_t(blockIdx.x)*warps+threadIdx.x/32; item<total;
        item+=size_t(gridDim.x)*warps) {
        const int m=item%m_count;
        const float* x=data+item*SubDim;
        const float* cb=transposed_codebooks+size_t(m)*SubDim*K;
        float best=INFINITY;int best_id=0;
        float values[SubDim];
#pragma unroll
        for(int d=0;d<SubDim;++d)values[d]=x[d];
#pragma unroll
        for(int k=lane;k<K;k+=32) {
            float sq=0;
#pragma unroll
            for(int d=0;d<SubDim;++d) {
                const float delta=values[d]-cb[d*K+k];sq+=delta*delta;
            }
            if(sq<best || (sq==best && k<best_id)){best=sq;best_id=k;}
        }
#pragma unroll
        for(int step=16;step;step>>=1) {
            float other=__shfl_down_sync(0xffffffff,best,step);
            int id=__shfl_down_sync(0xffffffff,best_id,step);
            if(lane+step<32 && (other<best || (other==best && id<best_id))){best=other;best_id=id;}
        }
        if(lane==0)codes[item]=static_cast<uint8_t>(best_id);
    }
}

} // namespace paraselect::quantization::pq
