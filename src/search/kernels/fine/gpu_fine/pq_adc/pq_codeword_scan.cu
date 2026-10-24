#include "pq_codeword_scan.cuh"
#include "pq_codeword_schedule.cuh"
#include "pq_threshold_status.cuh"
#include "utils/check_cuda_error.cuh"
#include <climits>
#include <cfloat>
#include <stdexcept>
#include "pq_codeword_u8_pipeline.cuh"

namespace paraselect::search {
namespace {
template<int Width>
__global__ void quantize(const float* source, uint8_t* target, float4* affine) {
    __shared__ float lo[256],hi[256];
    __shared__ int sums[256];
    int t=threadIdx.x;
    source+=size_t(blockIdx.x)*Width; target+=size_t(blockIdx.x)*Width;
    float low=FLT_MAX, high=-FLT_MAX;
    for(int i=t;i<Width;i+=256) { low=fminf(low,source[i]); high=fmaxf(high,source[i]); }
    lo[t]=low;hi[t]=high;__syncthreads();
    for(int stride=128;stride>0;stride/=2) {
        if(t<stride){lo[t]=fminf(lo[t],lo[t+stride]);hi[t]=fmaxf(hi[t],hi[t+stride]);}
        __syncthreads();
    }
    float scale=fmaxf((hi[0]-lo[0])/255.0f,1e-9f);
    int sum=0;
    for(int i=t;i<Width;i+=256) {
        int v=max(0,min(255,__float2int_rn((source[i]-lo[0])/scale)));
        target[i]=uint8_t(v);sum+=v;
    }
    sums[t]=sum;__syncthreads();
    for(int stride=128;stride>0;stride/=2) {
        if(t<stride) sums[t]+=sums[t+stride];
        __syncthreads();
    }
    if(t==0) affine[blockIdx.x]=make_float4(scale,lo[0],float(sums[0]),0);
}

} // namespace

static void scan_u8(const detail::PQCodewordView& input, const PQQueryLutView& lut,
    const IVFProbeScheduleView& schedule,
    const uint8_t* codes, const long long* offsets, const int* counts,
    const float* thresholds, int nq, int np,
    const PQAdcThresholdCandidatePoolView& output, cudaStream_t stream,
    int probe_begin, int probe_end, unsigned long long* physical_code_rows, int query_tile) {
    if(!input.codebook || !input.queries || !input.query_affine || !input.book_affine ||
       !lut.enabled() || lut.kind!=PQQueryLutKind::ResidualDot || lut.sign_codec.codes ||
       !schedule.clusters || !schedule.starts || !schedule.counts || !schedule.queries || !schedule.probe_ranks ||
       !codes || !offsets || !counts || !thresholds || nq<=0 || np<=0 ||
       output.capacity<=0 || !output.indices || !output.distances || !output.queries ||
       !output.query_counts || !output.total_count || !output.status)
        throw std::invalid_argument("invalid M16/D128 codeword scan");
    if(probe_end<0) probe_end=np;
    if(probe_begin<0 || probe_begin>probe_end || probe_end>np)
        throw std::invalid_argument("invalid codeword scan probe range");
    if(schedule.entry_count<=0 || probe_begin==probe_end)return;
    if(query_tile<32 || query_tile>64 || query_tile%8)
        throw std::invalid_argument("invalid internal U8 query tile");
#define LAUNCH_U8(Q) case Q: CUDA_CHECK(cudaFuncSetAttribute(detail::u8_pipeline<Q>,cudaFuncAttributePreferredSharedMemoryCarveout,50)); detail::u8_pipeline<Q><<<schedule.entry_count,128+Q/8*32,0,stream>>>(input,lut,schedule,{},codes,offsets,counts,thresholds,nq,np,output,probe_begin,probe_end,physical_code_rows); break
    switch(query_tile){LAUNCH_U8(32);LAUNCH_U8(40);LAUNCH_U8(48);LAUNCH_U8(56);LAUNCH_U8(64);}
#undef LAUNCH_U8
    CUDA_LAUNCH_CHECK("U8 pipeline scan");
}
void pq_codeword_scan_u8_hybrid(
    const float* queries,const float* codebook,const PQQueryLutView& lut,
    const IVFProbeScheduleView& schedule,const uint8_t* codes,const long long* offsets,const int* counts,
    const float* thresholds,int nq,int np,const PQAdcThresholdCandidatePoolView& output,cudaStream_t stream,
    int probe_begin,int probe_end,unsigned long long* physical_rows) {
    if(schedule.entry_count<=0)return;
    if(!queries||!codebook||nq<=0||np<=0||size_t(nq)*np>INT_MAX)
        throw std::invalid_argument("invalid U8 hybrid input");
    if(schedule.max_list_queries>=0&&schedule.max_list_queries<detail::CodewordMinQueries){
        pq_adc_query_lut_segmented_scan(lut,schedule,nullptr,codes,offsets,counts,
            thresholds,nq,np,16,PQAdcListScanRowMode::Serial,PQAdcLutReadMode::Global,
            physical_rows,output,stream,PQAdcSharedLutLayout::Compact,probe_begin,probe_end);
        return;
    }
    int *sizes=nullptr,*storage=nullptr;
    void* scratch=nullptr;
    CUDA_CHECK(cudaMallocAsync(&sizes,(detail::CodewordBuckets+1)*sizeof(int),stream));
    try {
        CUDA_CHECK(cudaMemsetAsync(sizes,0,(detail::CodewordBuckets+1)*sizeof(int),stream));
        detail::split_codeword_schedule<<<(schedule.entry_count+255)/256,256,0,stream>>>(
            schedule,nullptr,nullptr,nullptr,sizes,counts,nullptr);
        CUDA_LAUNCH_CHECK("count U8 hybrid schedule");
        int host[detail::CodewordBuckets+1],starts[detail::CodewordBuckets+1]={};
        CUDA_CHECK(cudaMemcpyAsync(host,sizes,sizeof(host),cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if(host[detail::CodewordBuckets])throw std::runtime_error("U8 hybrid requires contiguous list references");
        size_t total=0;
        for(int b=0;b<detail::CodewordBuckets;++b){starts[b]=int(total);total+=host[b];
            if(total>size_t(nq)*np)throw std::runtime_error("U8 hybrid entries exceed references");}
        if(total){
            CUDA_CHECK(cudaMallocAsync(&storage,3*total*sizeof(int),stream));
            CUDA_CHECK(cudaMemcpyAsync(sizes,starts,sizeof(starts),cudaMemcpyHostToDevice,stream));
            detail::split_codeword_schedule<<<(schedule.entry_count+255)/256,256,0,stream>>>(
                schedule,storage,storage+total,storage+2*total,sizes,counts,nullptr);
            CUDA_LAUNCH_CHECK("emit U8 hybrid schedule");
            detail::PQCodewordView input{};
            if(host[0]+host[1]+host[2]+host[3]+host[4]){
                size_t bytes=32768+size_t(nq)*128;
                CUDA_CHECK(cudaMallocAsync(&scratch,bytes+(size_t(nq)+1)*sizeof(float4),stream));
                auto* book=static_cast<uint8_t*>(scratch);auto* query=book+32768;
                auto* affine=reinterpret_cast<float4*>(book+bytes);
                quantize<32768><<<1,256,0,stream>>>(codebook,book,affine);
                quantize<128><<<nq,256,0,stream>>>(queries,query,affine+1);
                CUDA_LAUNCH_CHECK("U8 hybrid affine conversion");
                input={book,query,affine+1,affine};
            }
            for(int b=0;b<detail::CodewordBuckets;++b)if(host[b]){
                IVFProbeScheduleView part{storage+starts[b],storage+total+starts[b],storage+2*total+starts[b],
                    schedule.queries,schedule.probe_ranks,host[b]};
                if(b==detail::CodewordBuckets-1) {
                    pq_adc_query_lut_segmented_scan(lut,part,nullptr,codes,offsets,counts,thresholds,nq,np,16,
                        PQAdcListScanRowMode::Serial,PQAdcLutReadMode::Global,physical_rows,output,stream,
                        PQAdcSharedLutLayout::Compact,probe_begin,probe_end);
                } else {
                    const int bq = 32 + 8*b;
                    scan_u8(input,lut,part,codes,offsets,counts,thresholds,nq,np,output,
                        stream,probe_begin,probe_end,physical_rows,bq);
                }
            }
        }
    }catch(...){if(scratch)cudaFreeAsync(scratch,stream);if(storage)cudaFreeAsync(storage,stream);cudaFreeAsync(sizes,stream);throw;}
    if(scratch)CUDA_CHECK(cudaFreeAsync(scratch,stream));
    if(storage)CUDA_CHECK(cudaFreeAsync(storage,stream));
    CUDA_CHECK(cudaFreeAsync(sizes,stream));
}
}  // namespace paraselect::search
