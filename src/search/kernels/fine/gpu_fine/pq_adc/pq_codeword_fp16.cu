#include "pq_codeword_scan.cuh"
#include "pq_codeword_fp16.cuh"
#include "pq_codeword_schedule.cuh"
#include "pq_threshold_status.cuh"
#include "utils/check_cuda_error.cuh"
#include <climits>
#include <cuda/barrier>
#include <stdexcept>

namespace paraselect::search {
namespace {
__global__ void convert_book(const float* source, half* target, int dim) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<32768){int subdim=dim/16,d=i%8;target[i]=d<subdim?__float2half_rn(source[(i/8)*subdim+d]):__float2half_rn(0.0f);}
}

template<int BQ>
__global__ __launch_bounds__((4+BQ/8)*32,3)
void fp16_scan(const float* queries,const half* book,PQQueryLutView lut,
    IVFProbeScheduleView schedule,const uint8_t* codes,const long long* offsets,
    const int* counts,const float* thresholds,int nq,int np,
    PQAdcThresholdCandidatePoolView output,int probe_begin,int probe_end,int raw_dim){
    constexpr int BV=32,S=136,Buffers=2,ProducerWarps=4,Threads=(4+BQ/8)*32;
    static_assert(BQ>=32&&BQ<=64&&BQ%8==0);
    int entry=blockIdx.x,cluster=schedule.clusters[entry],visitors=schedule.counts[entry];
    int end=counts[cluster];long long base=offsets[cluster];
    if(base<0||base+end>INT_MAX){if(threadIdx.x==0)detail::record_threshold_status(output.status,4);return;}
    if(end<=0)return;
    // Query fragments live in registers during scanning; the same shared
    // storage then holds the producer's two decoded vector tiles.
    __shared__ __align__(16) half storage[Buffers*BV*S];
    auto v=reinterpret_cast<half (*)[BV*S]>(storage);
    __shared__ int qids[BQ];
    __shared__ float query_distance[BQ],query_cut[BQ];
    __shared__ volatile float correction[Buffers][BV];
    __shared__ cuda::barrier<cuda::thread_scope_block> ready[Buffers],free[Buffers];
    int t=threadIdx.x,warp=t/32,lane=t%32;
    if(t==0)for(int b=0;b<Buffers;++b){init(&ready[b],Threads);init(&free[b],Threads);}
    __syncthreads();
    for(int first=0;first<visitors;first+=BQ){
        if(t<BQ){
            int ref=schedule.starts[entry]+first+t;
            int id=first+t<visitors?schedule.queries[ref]:-1;
            int rank=first+t<visitors?schedule.probe_ranks[ref]:-1;
            if(first+t<visitors&&(id<0||id>=nq||rank<0||rank>=np))detail::record_threshold_status(output.status,3);
            qids[t]=id>=0&&id<nq&&rank>=probe_begin&&rank<probe_end?id:-1;
            query_distance[t]=qids[t]>=0?lut.query_centroid_distances[size_t(id)*np+rank]:0;
            query_cut[t]=qids[t]>=0?thresholds[id]:0;
        }
        __syncthreads();
        for(int i=t;i<BQ*64;i+=blockDim.x){
            int qi=i/64,d=(i%64)*2,id=qids[qi],subdim=raw_dim/16,component=d%8;
            float2 value=id>=0&&component<subdim?*reinterpret_cast<const float2*>(queries+size_t(id)*raw_dim+(d/8)*subdim+component):make_float2(0,0);
            *reinterpret_cast<half2*>(storage+qi*S+d)=__floats2half2_rn(value.x,value.y);
        }
        __syncthreads();
        unsigned query_fragments[8][2];
        if(warp>=ProducerWarps){
            int qr=(warp-ProducerWarps)*8;
#pragma unroll
            for(int k=0;k<128;k+=16){
                unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(storage+(qr+lane%8)*S+k+((lane/8)%2)*8));
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(query_fragments[k/16][0]),"=r"(query_fragments[k/16][1]) : "r"(address));
            }
        }
        __syncthreads();
        int cached_id[2]={};float cached_distance[2]={},cached_cut[2]={};
        if(warp>=ProducerWarps){
            int qr=(warp-ProducerWarps)*8+(lane%4)*2;
#pragma unroll
            for(int j=0;j<2;++j){cached_id[j]=qids[qr+j];cached_distance[j]=query_distance[qr+j];cached_cut[j]=query_cut[qr+j];}
        }
        int nt=(end+BV-1)/BV;
        if(warp<ProducerWarps){
            unsigned next_indices[1]={};
            if(t/4<min(BV,end))next_indices[0]=reinterpret_cast<const unsigned*>(codes+base*16)[t];
            for(int tile=0;tile<nt;++tile){
                int buffer=tile%Buffers,start=tile*BV;
                if(tile>=Buffers)free[buffer].arrive_and_wait();
                unsigned early_bias=0;
                if(t<BV&&start+t<end){
                    int index=int(base+start+t);
                    if(lut.query_lut_bias.u16)asm volatile("ld.global.u16 %0, [%1];" : "=r"(early_bias) : "l"(lut.query_lut_bias.u16+index) : "memory");
                    else asm volatile("ld.global.u32 %0, [%1];" : "=r"(early_bias) : "l"(lut.query_lut_bias.fp32+index) : "memory");
                }
                pq_fp16_detail::decode<S>(v[buffer],book,min(BV,end-start),next_indices[0]);
                if(t<BV){
                    float bias=0;
                    if(start+t<end)bias=lut.query_lut_bias.u16?fmaf(float(early_bias),lut.query_lut_bias.scale,lut.query_lut_bias.offset):__uint_as_float(early_bias);
                    correction[buffer][t]=bias;
                }
                int next_start=start+BV;
                next_indices[0]=next_start+t/4<end?reinterpret_cast<const unsigned*>(codes+(base+next_start)*16)[t]:0;
                asm volatile("" : "+r"(next_indices[0]) :: "memory");
                pq_fp16_detail::wait();
                (void)ready[buffer].arrive();
            }
            for(int tile=max(0,nt-Buffers);tile<nt;++tile)free[tile%Buffers].arrive_and_wait();
        }else{
            for(int tile=0;tile<nt;++tile){
                int buffer=tile%Buffers,start=tile*BV;
                ready[buffer].arrive_and_wait();
#pragma unroll
                for(int part=0;part<2;++part){
                    int vr=part*16;
                    float bias[2]={correction[buffer][vr+lane/4],correction[buffer][vr+lane/4+8]},acc[4]={};
                    pq_fp16_detail::multiply(v[buffer]+vr*S,S,acc,query_fragments);
                    if(part==1)(void)free[buffer].arrive();
#pragma unroll
                    for(int j=0;j<4;++j){
                        int row=start+vr+lane/4+(j/2)*8,id=cached_id[j%2],index=int(base+row);
                        bool valid=id>=0&&row<end;float distance=0;
                        if(valid){distance=cached_distance[j%2]+bias[j/2]-2*acc[j];valid=distance<=cached_cut[j%2];}
                        unsigned mask=__ballot_sync(0xffffffffu,valid);int slot=0;
                        if(mask){
                            if(lane==0)slot=atomicAdd(output.total_count,__popc(mask));
                            slot=__shfl_sync(0xffffffffu,slot,0)+__popc(mask&((1u<<lane)-1));
                            if(valid){
                                if(output.accepted_counts)atomicAdd(output.accepted_counts+id,1ULL);
                                if(slot<output.capacity){output.indices[slot]=index;output.queries[slot]=id;output.distances[slot]=distance;atomicAdd(output.query_counts+id,1);}
                                else detail::record_threshold_status(output.status,1);
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}
} // namespace

static void scan_fp16_book(const float* queries,const half* book,const PQQueryLutView& lut,
    const IVFProbeScheduleView& schedule,const uint8_t* codes,const long long* offsets,
    const int* counts,const float* thresholds,int nq,int np,
    const PQAdcThresholdCandidatePoolView& output,cudaStream_t stream,
    int probe_begin,int probe_end,int query_tile,int raw_dim){
    if(!queries||!book||(raw_dim!=96&&raw_dim!=128)||!codes||!offsets||!counts||!thresholds||nq<=0||np<=0||
       !lut.enabled()||lut.kind!=PQQueryLutKind::ResidualDot||lut.sign_codec.codes||
       !schedule.clusters||!schedule.starts||!schedule.counts||!schedule.queries||!schedule.probe_ranks||
       !output.indices||!output.distances||!output.queries||!output.query_counts||!output.total_count||!output.status||output.capacity<=0||
       query_tile<32||query_tile>64||query_tile%8)
        throw std::invalid_argument("invalid FP16 M16/D96-or-D128 codeword scan");
    if(probe_end<0)probe_end=np;
    if(probe_begin<0||probe_begin>probe_end||probe_end>np)throw std::invalid_argument("invalid FP16 probe range");
    if(schedule.entry_count<=0||probe_begin==probe_end)return;
#define LAUNCH(Q) case Q: CUDA_CHECK(cudaFuncSetAttribute(fp16_scan<Q>,cudaFuncAttributePreferredSharedMemoryCarveout,64)); fp16_scan<Q><<<schedule.entry_count,(4+Q/8)*32,0,stream>>>(queries,book,lut,schedule,codes,offsets,counts,thresholds,nq,np,output,probe_begin,probe_end,raw_dim); break
    switch(query_tile){LAUNCH(32);LAUNCH(40);LAUNCH(48);LAUNCH(56);LAUNCH(64);}
#undef LAUNCH
    CUDA_LAUNCH_CHECK("FP16 codeword scan");
}
static void scan_hybrid(const float* queries,const float* codebook,const PQQueryLutView& lut,
    const IVFProbeScheduleView& schedule,const uint8_t* codes,const long long* offsets,const int* counts,
    const float* thresholds,int nq,int np,const PQAdcThresholdCandidatePoolView& output,cudaStream_t stream,
    int probe_begin,int probe_end,unsigned long long* physical_rows,int raw_dim){
    if(schedule.entry_count<=0)return;
    if(nq<=0||np<=0||size_t(nq)*np>INT_MAX)throw std::invalid_argument("invalid hybrid reference bound");
    constexpr int buckets=detail::CodewordBuckets;
    int *sizes=nullptr,*storage=nullptr;
    half* book=nullptr;
    if(!codebook||(raw_dim!=96&&raw_dim!=128))throw std::invalid_argument("invalid hybrid codebook shape");
    // Reuse the builder's existing summary readback. No TC work means no
    // secondary schedule, allocation, or host/device round trip is needed.
    if(schedule.max_list_queries>=0&&schedule.max_list_queries<detail::CodewordMinQueries){
        pq_adc_query_lut_segmented_scan(lut,schedule,nullptr,codes,offsets,counts,
            thresholds,nq,np,16,PQAdcListScanRowMode::Serial,PQAdcLutReadMode::Global,
            physical_rows,output,stream,PQAdcSharedLutLayout::Compact,probe_begin,probe_end);
        return;
    }
    CUDA_CHECK(cudaMallocAsync(&sizes,(buckets+1)*sizeof(int),stream));
    try{
        CUDA_CHECK(cudaMemsetAsync(sizes,0,(buckets+1)*sizeof(int),stream));
        detail::split_codeword_schedule<<<(schedule.entry_count+255)/256,256,0,stream>>>(
            schedule,nullptr,nullptr,nullptr,sizes,counts,nullptr);
        CUDA_LAUNCH_CHECK("count hybrid list schedule");
        int host_sizes[buckets+1],starts[buckets+1]={};
        CUDA_CHECK(cudaMemcpyAsync(host_sizes,sizes,sizeof(host_sizes),cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if(host_sizes[buckets])throw std::runtime_error("hybrid schedule requires contiguous list references");
        size_t total=0;
        for(int b=0;b<buckets;++b){
            starts[b]=int(total);total+=host_sizes[b];
            if(total>size_t(nq)*np)throw std::runtime_error("hybrid entries exceed query/probe bound");
        }
        if(total){
            CUDA_CHECK(cudaMallocAsync(&storage,3*total*sizeof(int),stream));
            CUDA_CHECK(cudaMemcpyAsync(sizes,starts,sizeof(starts),cudaMemcpyHostToDevice,stream));
            detail::split_codeword_schedule<<<(schedule.entry_count+255)/256,256,0,stream>>>(
                schedule,storage,storage+total,storage+2*total,sizes,counts,physical_rows);
            CUDA_LAUNCH_CHECK("emit hybrid list schedule");
            bool has_tc=false;
            for(int b=0;b<buckets-1;++b)has_tc|=host_sizes[b]>0;
            if(has_tc){
                CUDA_CHECK(cudaMallocAsync(&book,32768*sizeof(half),stream));
                convert_book<<<128,256,0,stream>>>(codebook,book,raw_dim);
            }
            for(int b=0;b<buckets;++b){
                if(!host_sizes[b])continue;
                IVFProbeScheduleView part{storage+starts[b],storage+total+starts[b],storage+2*total+starts[b],
                    schedule.queries,schedule.probe_ranks,host_sizes[b]};
                if(b<buckets-1)scan_fp16_book(queries,book,lut,part,codes,offsets,counts,
                    thresholds,nq,np,output,stream,probe_begin,probe_end,32+8*b,raw_dim);
                else pq_adc_query_lut_segmented_scan(lut,part,nullptr,codes,offsets,counts,thresholds,nq,np,16,
                    PQAdcListScanRowMode::Serial,PQAdcLutReadMode::Global,physical_rows,output,stream,
                    PQAdcSharedLutLayout::Compact,probe_begin,probe_end);
            }
        }
    }catch(...){if(book)cudaFreeAsync(book,stream);if(storage)cudaFreeAsync(storage,stream);cudaFreeAsync(sizes,stream);throw;}
    if(book)CUDA_CHECK(cudaFreeAsync(book,stream));
    if(storage)CUDA_CHECK(cudaFreeAsync(storage,stream));
    CUDA_CHECK(cudaFreeAsync(sizes,stream));
}
void pq_codeword_scan_hybrid(const float* queries,const float* codebook,const PQQueryLutView& lut,
    const IVFProbeScheduleView& schedule,const uint8_t* codes,const long long* offsets,const int* counts,
    const float* thresholds,int nq,int np,const PQAdcThresholdCandidatePoolView& output,cudaStream_t stream,
    int probe_begin,int probe_end,unsigned long long* physical_rows,int raw_dim){
    scan_hybrid(queries,codebook,lut,schedule,codes,offsets,counts,thresholds,nq,np,output,stream,probe_begin,probe_end,physical_rows,raw_dim);
}
} // namespace paraselect::search
