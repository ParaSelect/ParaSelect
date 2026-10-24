#pragma once
// SM120-only experiment. test_wmma_fragment_layout checks the accumulator
// lane/slot mapping against all 256 coordinates before performance runs.
template<bool CtaOwned=false, bool RejectEmpty=false, int Groups, class Fragment>
__device__ __forceinline__ void admit_wmma_registers(
    const Fragment (&acc)[Groups], const float* norms, const float* qnorm,
    const float* thresholds, int qb, int batch, int base, int parts, int part,
    int capacity, float* values, int* ids, int* counts, int metadata_base=0, int* local_counts=nullptr, int center_warp=-1, int only_group=-1) {
    const int lane=threadIdx.x%32, warp=center_warp<0?threadIdx.x/32:center_warp, quad_lane=lane%4;
    #pragma unroll
    for(int g=0;g<Groups;++g) {
        if(only_group>=0 && g!=only_group)continue;
        #pragma unroll
        for(int half=0;half<2;++half) {
            const int q=qb+g*16+lane/4+half*8;
            const float qn=q<batch?qnorm[q-metadata_base]:0.f;
            const float threshold=q<batch?thresholds[q-metadata_base]:-INFINITY;
            float distance[4];
            unsigned hits=0;
            #pragma unroll
            for(int j=0;j<4;++j) {
                const int col=quad_lane*2+(j&1)+(j/2)*8;
                const int slot=half*2+(j&1)+(j/2)*4;
                const float cn=norms[warp*16+col];
                distance[j]=qn*qn+cn*cn-2.f*acc[g].x[slot];
                if(q<batch && distance[j]<=threshold)hits|=1u<<j;
            }
            // Uniform across the full warp, including padded query rows.
            // No candidate means no prefix, allocation or output work.
            if constexpr(RejectEmpty)if(!__any_sync(0xffffffff,hits!=0))continue;
            const int local_count=__popc(hits);
            int prefix=local_count;
            int v=__shfl_up_sync(0xffffffff,prefix,1,4);
            if(quad_lane>=1)prefix+=v;
            v=__shfl_up_sync(0xffffffff,prefix,2,4);
            if(quad_lane>=2)prefix+=v;
            const int total=__shfl_sync(0xffffffff,prefix,3,4);
            int offset=0;
            if(quad_lane==0 && total) {
                if constexpr(CtaOwned)offset=atomicAdd(local_counts+q-qb,total);
                else offset=atomicAdd(counts+size_t(q)*parts+part,total);
            }
            offset=__shfl_sync(0xffffffff,offset,0,4)+prefix-local_count;
            #pragma unroll
            for(int j=0;j<4;++j) {
                if(hits&(1u<<j)) {
                    if(offset<capacity) {
                        const size_t dst=(size_t(q)*parts+part)*capacity+offset;
                        values[dst]=-distance[j];
                        ids[dst]=base+warp*16+quad_lane*2+(j&1)+(j/2)*8;
                    }
                    ++offset;
                }
            }
        }
    }
}
