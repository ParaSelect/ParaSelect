#pragma once

__device__ __forceinline__ unsigned ordered_score(float value) {
    unsigned bits=__float_as_uint(value);
    return bits ^ ((bits&0x80000000u)?0xffffffffu:0x80000000u);
}

// Select exactly k records into the caller's output arrays. Histogram rank
// refinement needs no k-sized shared state; ties consume an explicit quota.
__global__ void gather_exact_admitted(const float* pool, const int* ids,
    const int* counts, int parts, int capacity, int k, float* output, idx_t* output_ids) {
    __shared__ int histogram[256], rank, equal_seen, output_count;
    __shared__ unsigned prefix;
    const int q=blockIdx.x, tid=threadIdx.x, lane=tid%32;
    const size_t base=size_t(q)*parts*capacity, out=size_t(q)*k;
    int total=0;bool invalid=false;
    for(int p=0;p<parts;++p) {
        int count=counts[size_t(q)*parts+p];
        invalid|=count>capacity;total+=min(count,capacity);
    }
    if(invalid || total<k) {
        for(int i=tid;i<k;i+=blockDim.x){output[out+i]=-INFINITY;output_ids[out+i]=-1;}
        return;
    }
    if(tid==0){prefix=0;rank=k;equal_seen=output_count=0;}
    __syncthreads();
    unsigned mask=0;
    for(int shift=24;shift>=0;shift-=8) {
        histogram[tid]=0;
        __syncthreads();
        for(int p=0;p<parts;++p) {
            int count=counts[size_t(q)*parts+p];
            for(int j=tid;j<count;j+=blockDim.x) {
                unsigned bits=ordered_score(pool[base+p*capacity+j]);
                if((bits&mask)==prefix)atomicAdd(histogram+((bits>>shift)&255),1);
            }
        }
        __syncthreads();
        if(tid==0) {
            int above=0;
            for(int bucket=255;bucket>=0;--bucket) {
                if(above+histogram[bucket]>=rank) {
                    rank-=above;prefix|=unsigned(bucket)<<shift;break;
                }
                above+=histogram[bucket];
            }
        }
        __syncthreads();
        mask|=0xffu<<shift;
    }
    for(int p=0;p<parts;++p) {
        int count=counts[size_t(q)*parts+p];
        for(int chunk=0;chunk<count;chunk+=blockDim.x) {
            int j=chunk+tid;float value=0.f;bool hit=false;
            if(j<count) {
                value=pool[base+p*capacity+j];unsigned bits=ordered_score(value);
                hit=bits>prefix || (bits==prefix && atomicAdd(&equal_seen,1)<rank);
            }
            unsigned hits=__ballot_sync(0xffffffff,hit);
            int offset=lane==0?atomicAdd(&output_count,__popc(hits)):0;
            offset=__shfl_sync(0xffffffff,offset,0)+__popc(hits&((1u<<lane)-1));
            if(hit){output[out+offset]=value;output_ids[out+offset]=ids[base+p*capacity+j];}
        }
    }
}

template<int RunItems,bool ToPool,bool Final=false>
__global__ __launch_bounds__(RunItems/8) void sort_output_runs(float* values, idx_t* ids, float* pool,
    int* pool_ids,int k,int stride) {
    constexpr int BlockThreads=RunItems/8;
    using Sort=cub::BlockRadixSort<float,BlockThreads,8,int>;
    __shared__ typename Sort::TempStorage temp;
    const size_t base=size_t(blockIdx.x)*k+blockIdx.y*RunItems;
    float keys[8];int indices[8];
    #pragma unroll
    for(int i=0;i<8;++i){keys[i]=values[base+threadIdx.x+i*BlockThreads];indices[i]=int(ids[base+threadIdx.x+i*BlockThreads]);}
    Sort(temp).SortDescendingBlockedToStriped(keys,indices);
    #pragma unroll
    for(int i=0;i<8;++i) {
        if constexpr(ToPool) {
            size_t out=size_t(blockIdx.x)*stride+blockIdx.y*RunItems+threadIdx.x+i*BlockThreads;
            pool[out]=keys[i];pool_ids[out]=indices[i];
        } else {values[base+threadIdx.x+i*BlockThreads]=Final?-keys[i]:keys[i];ids[base+threadIdx.x+i*BlockThreads]=indices[i];}
    }
}

__device__ __forceinline__ int descending_merge_path(const float* a,int na,
    const float* b,int nb,int diagonal) {
    int lo=max(0,diagonal-nb),hi=min(diagonal,na);
    while(lo<hi) {
        int i=(lo+hi)/2,j=diagonal-i;
        if(j>0 && i<na && b[j-1]<=a[i])lo=i+1;else hi=i;
    }
    return lo;
}

// Ping-pong between the existing pool and final output; no second pool.
// Each CTA independently owns one 1024-element diagonal of a merge pair.
template<bool FromOutput,bool Final=false>
__global__ void merge_output_runs(float* pool, int* pool_ids, float* output,
    idx_t* output_ids, int stride,int k,int run) {
    constexpr int Tile=1024;
    __shared__ float keys[Tile],merged[Tile];
    __shared__ int ids[Tile],merged_ids[Tile],a0,b0,na,nb;
    int q=blockIdx.y,tile=blockIdx.x*Tile;
    int pair=tile/(2*run)*(2*run),diagonal=tile-pair;
    const size_t source_base=size_t(q)*(FromOutput?k:stride)+pair;
    const float* source=(FromOutput?output:pool)+source_base;
    if(threadIdx.x==0) {
        a0=descending_merge_path(source,run,source+run,run,diagonal);b0=diagonal-a0;
        int a1=descending_merge_path(source,run,source+run,run,diagonal+Tile);
        na=a1-a0;nb=Tile-na;
    }
    __syncthreads();
    for(int i=threadIdx.x;i<na;i+=blockDim.x) {
        keys[i]=source[a0+i];
        if constexpr(FromOutput)ids[i]=int(output_ids[source_base+a0+i]);
        else ids[i]=pool_ids[source_base+a0+i];
    }
    for(int i=threadIdx.x;i<nb;i+=blockDim.x) {
        keys[na+i]=source[run+b0+i];
        if constexpr(FromOutput)ids[na+i]=int(output_ids[source_base+run+b0+i]);
        else ids[na+i]=pool_ids[source_base+run+b0+i];
    }
    __syncthreads();
    int first=threadIdx.x*4,i=descending_merge_path(keys,na,keys+na,nb,first),j=first-i;
    #pragma unroll
    for(int t=0;t<4;++t) {
        bool take_a=i<na && (j>=nb || keys[i]>=keys[na+j]);
        int src=take_a?i++:na+j++;
        merged[first+t]=keys[src];merged_ids[first+t]=ids[src];
    }
    __syncthreads();
    const size_t destination=size_t(q)*(FromOutput?stride:k)+tile;
    for(int t=threadIdx.x;t<Tile;t+=blockDim.x) {
        if constexpr(FromOutput){pool[destination+t]=merged[t];pool_ids[destination+t]=merged_ids[t];}
        else {output[destination+t]=Final?-merged[t]:merged[t];output_ids[destination+t]=merged_ids[t];}
    }
}

template<int RunItems>
inline void sort_selected_runs(const paraselect::coarse::detail::Execution& a,
    float* pool,int* ids,int stride) {
    const int k=a.nprobe,batch=a.n_query;
    if constexpr(RunItems==8192) {
        if(k==RunItems) {
            sort_output_runs<RunItems,false,true><<<dim3(batch,1),RunItems/8,0,a.stream>>>(a.d_top_nprobe_dist,a.d_top_nprobe_index,pool,ids,k,stride);
            return;
        }
    }
    int levels=0;for(int run=RunItems;run<k;run*=2)++levels;
    // Start in the buffer that makes the last merge write final output.
    bool in_output=levels%2==0;
    if(in_output)sort_output_runs<RunItems,false><<<dim3(batch,k/RunItems),RunItems/8,0,a.stream>>>(a.d_top_nprobe_dist,a.d_top_nprobe_index,pool,ids,k,stride);
    else sort_output_runs<RunItems,true><<<dim3(batch,k/RunItems),RunItems/8,0,a.stream>>>(a.d_top_nprobe_dist,a.d_top_nprobe_index,pool,ids,k,stride);
    for(int run=RunItems;run<k;run*=2) {
        if(in_output)merge_output_runs<true><<<dim3(k/1024,batch),256,0,a.stream>>>(pool,ids,a.d_top_nprobe_dist,a.d_top_nprobe_index,stride,k,run);
        else if(run*2==k)merge_output_runs<false,true><<<dim3(k/1024,batch),256,0,a.stream>>>(pool,ids,a.d_top_nprobe_dist,a.d_top_nprobe_index,stride,k,run);
        else merge_output_runs<false><<<dim3(k/1024,batch),256,0,a.stream>>>(pool,ids,a.d_top_nprobe_dist,a.d_top_nprobe_index,stride,k,run);
        in_output=!in_output;
    }
}

inline void sort_reusing_outputs(const paraselect::coarse::detail::Execution& a,
    float* pool,int* ids,const int* counts,int parts,int capacity,int run_items=2048) {
    const int k=a.nprobe,batch=a.n_query,stride=parts*capacity;
    if((k!=8192 && k!=16384) || stride<k || batch<=0)
        throw std::invalid_argument("reuse sort requires k8192/16384 and sufficient pool stride");
    if(run_items!=2048 && run_items!=4096 && run_items!=8192)
        throw std::invalid_argument("reuse sort run must be 2048, 4096 or 8192");
    gather_exact_admitted<<<batch,256,0,a.stream>>>(pool,ids,counts,parts,capacity,k,a.d_top_nprobe_dist,a.d_top_nprobe_index);
    if(run_items==8192)sort_selected_runs<8192>(a,pool,ids,stride);
    else if(run_items==4096)sort_selected_runs<4096>(a,pool,ids,stride);
    else sort_selected_runs<2048>(a,pool,ids,stride);
}
