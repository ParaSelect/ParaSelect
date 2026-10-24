#pragma once

// Experimental p256/c32/k512 consumer. Empty pool slots are never read.
// Radix narrowing finds the exact cutoff; only the final 512 pairs are sorted.
template<bool CacheCounts=false,bool ParallelPivot=false,bool Dense=false>
__global__ void select_counted_512(const float* pool, const int* pool_ids,
    const int* counts, float* output, idx_t* output_ids) {
    using Sort=cub::BlockRadixSort<float,256,2,int>;
    union Scratch {
        typename Sort::TempStorage sort;
        unsigned dense[Dense?8192:1];
    };
    __shared__ Scratch scratch;
    __shared__ unsigned buckets[256], warp_a[8], warp_b[8], lower, upper;
    __shared__ int remaining, shift, invalid, total_valid;
    __shared__ int cached_counts[CacheCounts?256:1];
    __shared__ float selected[512];
    __shared__ int selected_ids[512];
    const int tid=threadIdx.x, lane=tid%32, warp=tid/32, q=blockIdx.x;
    const size_t base=size_t(q)*8192;
    int count=counts[size_t(q)*256+tid];
    const int original_count=count;
    if constexpr(CacheCounts)cached_counts[tid]=count;
    const unsigned bad=__any_sync(0xffffffff,count>32);
    #pragma unroll
    for(int offset=16;offset;offset/=2)count+=__shfl_down_sync(0xffffffff,count,offset);
    if(lane==0){warp_a[warp]=count;warp_b[warp]=bad;}
    __syncthreads();
    if(tid==0) {
        unsigned total=0, overflow=0;
        for(int w=0;w<8;++w){total+=warp_a[w];overflow|=warp_b[w];}
        invalid=overflow || total<512;remaining=512;total_valid=total;
    }
    __syncthreads();
    // The caller overwrites invalid rows through its exact fallback.
    if(invalid)return;
    if constexpr(Dense) {
        int prefix=original_count;
        #pragma unroll
        for(int offset=1;offset<32;offset*=2) {
            const int previous=__shfl_up_sync(0xffffffff,prefix,offset);
            if(lane>=offset)prefix+=previous;
        }
        int preceding=0;
        for(int w=0;w<warp;++w)preceding+=warp_a[w];
        // Reuse the output-ID staging area until final candidate gathering.
        selected_ids[tid]=preceding+prefix-original_count;
        __syncthreads();
    }
    unsigned keys[32], local_min=~0u, local_max=0;
    #pragma unroll
    for(int i=0;i<32;++i) {
        const int part=warp+i*8;
        const int length=CacheCounts?cached_counts[part]:counts[size_t(q)*256+part];
        unsigned key=0;
        if(lane<length) {
            const unsigned bits=__float_as_uint(pool[base+tid+i*256]);
            key=(bits&0x80000000u)?~bits:(bits^0x80000000u);
            if constexpr(Dense)scratch.dense[selected_ids[part]+lane]=key;
            local_min=min(local_min,key);local_max=max(local_max,key);
        }
        keys[i]=key;
    }
    #pragma unroll
    for(int offset=16;offset;offset/=2) {
        local_min=min(local_min,__shfl_down_sync(0xffffffff,local_min,offset));
        local_max=max(local_max,__shfl_down_sync(0xffffffff,local_max,offset));
    }
    if(lane==0){warp_a[warp]=local_min;warp_b[warp]=local_max;}
    __syncthreads();
    if(tid==0) {
        lower=~0u;upper=0;
        for(int w=0;w<8;++w){lower=min(lower,warp_a[w]);upper=max(upper,warp_b[w]);}
    }
    __syncthreads();
    while(lower<upper) {
        buckets[tid]=0;
        if(tid==0)shift=max(0,24-__clz(upper-lower));
        __syncthreads();
        if constexpr(Dense) {
            for(int i=tid;i<total_valid;i+=256) {
                const unsigned key=scratch.dense[i];
                if(key>=lower && key<=upper)atomicAdd(buckets+((key-lower)>>shift),1u);
            }
        } else {
            #pragma unroll
            for(int i=0;i<32;++i)if(keys[i]>=lower && keys[i]<=upper)
                atomicAdd(buckets+((keys[i]-lower)>>shift),1u);
        }
        __syncthreads();
        if constexpr(ParallelPivot) {
            // Snapshot before the barrier: the winning thread updates shared
            // bounds/rank, while other threads still evaluate their intervals.
            const unsigned wanted=remaining,lo0=lower,hi0=upper;
            const int bits=shift;
            const unsigned size=buckets[255-tid];
            unsigned prefix=size;
            #pragma unroll
            for(int offset=1;offset<32;offset*=2) {
                const unsigned previous=__shfl_up_sync(0xffffffff,prefix,offset);
                if(lane>=offset)prefix+=previous;
            }
            if(lane==31)warp_a[warp]=prefix;
            __syncthreads();
            for(int w=0;w<warp;++w)prefix+=warp_a[w];
            const unsigned above=prefix-size;
            if(prefix>=wanted && above<wanted) {
                remaining=wanted-above;
                const unsigned lo=lo0+(unsigned(255-tid)<<bits);
                const unsigned long long hi=static_cast<unsigned long long>(lo)+(1u<<bits)-1;
                lower=lo;upper=hi<hi0?unsigned(hi):hi0;
            }
        } else if(tid==0) {
            unsigned above=0;
            for(int b=255;b>=0;--b) {
                const unsigned size=buckets[b];
                if(above+size>=unsigned(remaining)) {
                    remaining-=above;
                    const unsigned lo=lower+(unsigned(b)<<shift);
                    const unsigned long long hi=static_cast<unsigned long long>(lo)+(1u<<shift)-1;
                    lower=lo;upper=hi<upper?unsigned(hi):upper;
                    break;
                }
                above+=size;
            }
        }
        __syncthreads();
    }
    const unsigned cutoff=lower;
    unsigned greater=0,equal=0;
    #pragma unroll
    for(int i=0;i<32;++i){greater+=keys[i]>cutoff;equal+=keys[i]==cutoff;}
    unsigned prefix_g=greater,prefix_e=equal;
    #pragma unroll
    for(int offset=1;offset<32;offset*=2) {
        const unsigned g=__shfl_up_sync(0xffffffff,prefix_g,offset);
        const unsigned e=__shfl_up_sync(0xffffffff,prefix_e,offset);
        if(lane>=offset){prefix_g+=g;prefix_e+=e;}
    }
    if(lane==31){warp_a[warp]=prefix_g;warp_b[warp]=prefix_e;}
    __syncthreads();
    unsigned total_g=0,offset_g=prefix_g-greater,offset_e=prefix_e-equal;
    for(int w=0;w<8;++w) {
        total_g+=warp_a[w];
        if(w<warp){offset_g+=warp_a[w];offset_e+=warp_b[w];}
    }
    #pragma unroll
    for(int i=0;i<32;++i) {
        const unsigned key=keys[i];
        if(key<cutoff)continue;
        const unsigned dst=key>cutoff?offset_g++:total_g+offset_e++;
        if(dst<512) {
            selected[dst]=__uint_as_float((key&0x80000000u)?(key^0x80000000u):~key);
            selected_ids[dst]=pool_ids[base+tid+i*256];
        }
    }
    __syncthreads();
    float values[2];int ids[2];
    #pragma unroll
    for(int i=0;i<2;++i){values[i]=selected[tid*2+i];ids[i]=selected_ids[tid*2+i];}
    // All dense-key readers have completed before sort reuses the union.
    Sort(scratch.sort).SortDescending(values,ids);
    #pragma unroll
    for(int i=0;i<2;++i) {
        output[size_t(q)*512+tid*2+i]=-values[i];
        output_ids[size_t(q)*512+tid*2+i]=ids[i];
    }
}

template<bool Dense>
inline void dispatch_counted_512(const paraselect::coarse::detail::Execution& a,
    const float* pool,const int* pool_ids,const int* counts,bool cache,bool pivot) {
    const int batch=a.n_query;
    if(cache) {
        if(pivot)select_counted_512<true,true,Dense><<<batch,256,0,a.stream>>>(pool,pool_ids,counts,a.d_top_nprobe_dist,a.d_top_nprobe_index);
        else select_counted_512<true,false,Dense><<<batch,256,0,a.stream>>>(pool,pool_ids,counts,a.d_top_nprobe_dist,a.d_top_nprobe_index);
    } else {
        if(pivot)select_counted_512<false,true,Dense><<<batch,256,0,a.stream>>>(pool,pool_ids,counts,a.d_top_nprobe_dist,a.d_top_nprobe_index);
        else select_counted_512<false,false,Dense><<<batch,256,0,a.stream>>>(pool,pool_ids,counts,a.d_top_nprobe_dist,a.d_top_nprobe_index);
    }
}
