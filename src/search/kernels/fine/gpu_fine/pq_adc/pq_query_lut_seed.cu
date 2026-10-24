#include "pq_distance_histogram.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_query_lut_seed.cuh"
#include "utils/check_cuda_error.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_query_lut.cuh"
#include <cub/cub.cuh>
#include "search/kernels/fine/gpu_fine/pq_adc/pq_query_lut_score.cuh"
#include <climits>
#include <algorithm>
#include <math_constants.h>
#include <stdexcept>
#include <type_traits>

namespace paraselect::search {
namespace {
constexpr int K = 512;
constexpr int Threads = 256;
constexpr uint64_t Invalid = ~uint64_t{0};

__device__ uint64_t pack(float distance, int row) {
    const unsigned bits = __float_as_uint(distance);
    const unsigned ordered = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
    return (uint64_t(ordered) << 32) | unsigned(row);
}
__device__ float unpack(uint64_t key) {
    const unsigned ordered = key >> 32;
    return __uint_as_float(ordered & 0x80000000u ? ordered ^ 0x80000000u : ~ordered);
}

// Repeatedly retain top-512 of (previous top-512 + next 512 rows).
// Discarded rows are dominated by 512 rows from the same seed list.
template<int M, bool U8, bool QUERY_LUT_BIAS_U16, bool FourAcc, bool PlainL2 = false>
__global__ void list_top512(PQQueryLutView view, const idx_t* probes,
    const uint8_t* codes, const long long* offsets, const int* counts,
    int nprobe, int seed_probes, int partitions, uint64_t* scratch, int* status,
    unsigned long long* scanned_rows) {
    using Sort = cub::BlockRadixSort<uint64_t, Threads, 4>;
    __shared__ typename Sort::TempStorage temp;
    __shared__ uint64_t retained[K];
    const int pair = blockIdx.x / partitions;
    const int partition = blockIdx.x % partitions;
    const int q = pair / seed_probes, probe = pair % seed_probes;
    const idx_t list = probes[size_t(q) * nprobe + probe];
    const int count = list < 0 ? 0 : counts[list];
    const long long offset = list < 0 ? 0 : offsets[list];
    for (int i = threadIdx.x; i < K; i += Threads) retained[i] = Invalid;
    if (threadIdx.x == 0 && partition == 0 && scanned_rows) atomicAdd(scanned_rows, (unsigned long long)count);
    __syncthreads();
    for (int base = partition * K; base < count; base += partitions * K) {
        uint64_t keys[4];
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int rank = threadIdx.x * 4 + item;
            keys[item] = Invalid;
            if (rank < K) keys[item] = retained[rank];
            else {
                const int local = base + rank - K;
                const long long row = offset + local;
                if (local < count && row >= 0 && row <= INT_MAX) {
                    const float distance = pq_query_lut_seed_score<M,U8,QUERY_LUT_BIAS_U16,FourAcc,PlainL2>(view,codes,q,probe,nprobe,int(row));
                    if (isfinite(distance)) keys[item] = pack(distance,int(row));
                    else atomicMax(status, 4);
                } else if (local < count) atomicMax(status, 4);
            }
        }
        __syncthreads();
        Sort(temp).Sort(keys);
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int rank = threadIdx.x * 4 + item;
            if (rank < K) retained[rank] = keys[item];
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < K; i += Threads)
        scratch[size_t(blockIdx.x) * K + i] = retained[i];
}

template<int Items>
__global__ void merge_top512(const uint64_t* scratch, int seed_probes,
                            int* ids, float* distances) {
    using Sort = cub::BlockRadixSort<uint64_t, Threads, Items>;
    __shared__ typename Sort::TempStorage temp;
    uint64_t keys[Items];
    const int q = blockIdx.x;
#pragma unroll
    for (int item = 0; item < Items; ++item) {
        const int rank = threadIdx.x * Items + item;
        keys[item] = rank < seed_probes * K ? scratch[(size_t(q)*seed_probes*K)+rank] : Invalid;
    }
    Sort(temp).Sort(keys);
#pragma unroll
    for (int item = 0; item < Items; ++item) {
        const int rank = threadIdx.x * Items + item;
        if (rank < K) {
            ids[size_t(q)*K+rank] = keys[item] == Invalid ? -1 : int(unsigned(keys[item]));
            distances[size_t(q)*K+rank] = keys[item] == Invalid ? CUDART_INF_F : unpack(keys[item]);
        }
    }
}

// Reduce groups of eight local prefixes before the final per-query merge.
// This bounds shared memory independently of the number of scan partitions.
__global__ void merge_seed_groups(const uint64_t* source, int sources, int groups,
                                  uint64_t* output) {
    using Sort = cub::BlockRadixSort<uint64_t, Threads, 16>;
    __shared__ typename Sort::TempStorage temp;
    const int q = blockIdx.x / groups;
    const int group = blockIdx.x % groups;
    uint64_t keys[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int rank = group * 8 * K + threadIdx.x * 16 + i;
        keys[i] = rank < sources * K ? source[size_t(q) * sources * K + rank] : Invalid;
    }
    Sort(temp).Sort(keys);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int rank = threadIdx.x * 16 + i;
        if (rank < K) output[size_t(blockIdx.x) * K + rank] = keys[i];
    }
}

// Runtime-width exact merge. Unlike a K-specialized sorting network this keeps
// code size bounded. Equal keys are taken from the left input first.
__device__ uint64_t merged_at(const uint64_t* a, int na,
                              const uint64_t* b, int nb, int rank) {
    int lo = max(0, rank - nb), hi = min(rank, na);
    while (lo < hi) {
        const int i = (lo + hi) / 2, j = rank - i;
        if (j > 0 && i < na && b[j-1] >= a[i]) lo = i + 1;
        else hi = i;
    }
    const int j = rank - lo;
    const uint64_t av = lo < na ? a[lo] : Invalid;
    const uint64_t bv = j < nb ? b[j] : Invalid;
    return min(av, bv);
}

template<int M, bool U8, bool QUERY_LUT_BIAS_U16, bool FourAcc, bool PlainL2 = false>
__global__ void list_topk_runtime(PQQueryLutView view, const idx_t* probes,
    const uint8_t* codes, const long long* offsets, const int* counts,
    int nprobe, int seeds, int k, int query_begin, uint64_t* scratch, uint64_t* work,
    int* status, unsigned long long* scanned_rows) {
    using Sort = cub::BlockRadixSort<uint64_t, Threads, 2>;
    __shared__ typename Sort::TempStorage temp;
    __shared__ uint64_t tile[512];
    const int pair = blockIdx.x, q = query_begin + pair / seeds, probe = pair % seeds;
    const idx_t list = probes[size_t(q) * nprobe + probe];
    const int count = list < 0 ? 0 : counts[list];
    const long long offset = list < 0 ? 0 : offsets[list];
    uint64_t* a = scratch + size_t(pair) * k;
    uint64_t* b = work + size_t(pair) * k;
    for (int i=threadIdx.x; i<k; i+=Threads) a[i]=Invalid;
    if (!threadIdx.x && scanned_rows) atomicAdd(scanned_rows,(unsigned long long)count);
    __syncthreads();
    for (int base=0; base<count; base+=512) {
        uint64_t keys[2];
#pragma unroll
        for(int i=0;i<2;++i) {
            const int local=base+threadIdx.x*2+i;
            const long long row=offset+local;
            keys[i]=Invalid;
            if(local<count) {
                if(row<0 || row>INT_MAX) atomicMax(status,4);
                else {
                    const float d=pq_query_lut_seed_score<M,U8,QUERY_LUT_BIAS_U16,FourAcc,PlainL2>(view,codes,q,probe,nprobe,int(row));
                    if(isfinite(d)) keys[i]=pack(d,int(row)); else atomicMax(status,4);
                }
            }
        }
        Sort(temp).Sort(keys);
        tile[threadIdx.x*2]=keys[0]; tile[threadIdx.x*2+1]=keys[1];
        __syncthreads();
        for(int i=threadIdx.x;i<k;i+=Threads) b[i]=merged_at(a,k,tile,512,i);
        __syncthreads();
        uint64_t* swap=a;a=b;b=swap;
    }
    if(a!=scratch+size_t(pair)*k)
        for(int i=threadIdx.x;i<k;i+=Threads) scratch[size_t(pair)*k+i]=a[i];
}

__global__ void merge_runtime(const uint64_t* input, uint64_t* output,
                             int sources, int k) {
    const int groups=(sources+1)/2, q=blockIdx.x/groups, group=blockIdx.x%groups;
    const uint64_t* a=input+(size_t(q)*sources+2*group)*k;
    const int nb=2*group+1<sources?k:0;
    for(int i=threadIdx.x;i<k;i+=Threads)
        output[(size_t(q)*groups+group)*k+i]=merged_at(a,k,a+k,nb,i);
}
__global__ void unpack_runtime(const uint64_t* keys, int k, int* ids, float* distances) {
    const int q=blockIdx.x;
    for(int i=threadIdx.x;i<k;i+=Threads) {
        const size_t slot=size_t(q)*k+i;
        ids[slot]=keys[slot]==Invalid?-1:int(unsigned(keys[slot]));
        distances[slot]=keys[slot]==Invalid?CUDART_INF_F:unpack(keys[slot]);
    }
}

template<int M>
void launch_runtime(const PQQueryLutView& v, const idx_t* probes,
    const uint8_t* codes, const long long* offsets, const int* counts,
    int queries,int nprobe,int seeds,int k,int query_begin,bool four,uint64_t* scratch,uint64_t* work,
    int* status,unsigned long long* rows,cudaStream_t stream) {
    const auto dispatch_score = [&](auto plain_tag) {
    constexpr bool PlainL2 = decltype(plain_tag)::value;
#define RUN(U,P,F) list_topk_runtime<M,U,P,F,PlainL2><<<queries*seeds,Threads,0,stream>>>(v,probes,codes,offsets,counts,nprobe,seeds,k,query_begin,scratch,work,status,rows)
    if(v.quantized()) {
        if(v.quantized_query_lut_bias()) {if(four){RUN(true,true,true);}else{RUN(true,true,false);}}
        else {if(four){RUN(true,false,true);}else{RUN(true,false,false);}}
    } else {
        if(v.quantized_query_lut_bias()){RUN(false,true,false);}else{RUN(false,false,false);}
    }
#undef RUN
    };
    if (v.kind == PQQueryLutKind::PlainL2) dispatch_score(std::true_type{});
    else dispatch_score(std::false_type{});
}

template<int M>
void launch(const PQQueryLutView& v, const idx_t* probes, const uint8_t* codes,
    const long long* offsets, const int* counts, int queries, int nprobe, int seeds, int partitions,
    bool four, uint64_t* scratch, int* status, unsigned long long* rows, cudaStream_t stream) {
    const auto dispatch_score = [&](auto plain_tag) {
    constexpr bool PlainL2 = decltype(plain_tag)::value;
#define SEED(U,P,F) list_top512<M,U,P,F,PlainL2><<<queries*seeds*partitions,Threads,0,stream>>>(v,probes,codes,offsets,counts,nprobe,seeds,partitions,scratch,status,rows)
    if (v.quantized()) {
        if (v.quantized_query_lut_bias()) { if (four) { SEED(true,true,true); } else { SEED(true,true,false); } }
        else { if (four) { SEED(true,false,true); } else { SEED(true,false,false); } }
    } else {
        if (v.quantized_query_lut_bias()) { SEED(false,true,false); }
        else { SEED(false,false,false); }
    }
#undef SEED
    };
    if (v.kind == PQQueryLutKind::PlainL2) dispatch_score(std::true_type{});
    else dispatch_score(std::false_type{});
}

template<int M, bool U8, bool QUERY_LUT_BIAS_U16, bool FourAcc, bool PlainL2 = false, int Lists = 1>
__global__ void list_distance_histogram(
    PQQueryLutView view, const idx_t* probes, const uint8_t* codes,
    const long long* offsets, const int* counts, int nprobe, int seed_probes,
    int nlist, unsigned int* output, int bins, float log_min, float log_step) {
    extern __shared__ unsigned int local[];
    for (int bin = threadIdx.x; bin < bins; bin += blockDim.x) local[bin] = 0;
    __syncthreads();
    const int groups = (seed_probes + Lists - 1) / Lists;
    const int q = blockIdx.x / groups;
    const int first = (blockIdx.x % groups) * Lists;
    const int owner = Lists == 1 ? 0 : threadIdx.x / 32;
    for (int item = owner; item < Lists; item += Lists == 1 ? 1 : Threads / 32) {
        const int probe = first + item;
        if (probe >= seed_probes) continue;
        const idx_t list = probes[size_t(q) * nprobe + probe];
        const bool valid_list = list >= 0 && list < nlist;
        const int count = valid_list ? counts[list] : 0;
        const long long offset = valid_list ? offsets[list] : 0;
        const int lane = Lists == 1 ? threadIdx.x : threadIdx.x % 32;
        for (int row_local = lane; row_local < count; row_local += Lists == 1 ? Threads : 32) {
            const long long row = offset + row_local;
            if (row < 0 || row > INT_MAX) continue;
            const float distance = pq_query_lut_seed_score<M,U8,QUERY_LUT_BIAS_U16,FourAcc,PlainL2>(
                view, codes, q, probe, nprobe, static_cast<int>(row));
            if (!isfinite(distance)) continue;
            const int bin = pq_distance_histogram_bin(distance, bins, log_min, log_step);
            atomicAdd(local + bin, 1u);
        }
    }
    __syncthreads();
    for (int bin = threadIdx.x; bin < bins; bin += blockDim.x) {
        const unsigned int value = local[bin];
        if (value) atomicAdd(output + size_t(q) * bins + bin, value);
    }
}

template<int M, int Lists>
void launch_histogram(const PQQueryLutView& view, const idx_t* probes,
    const uint8_t* codes, const long long* offsets, const int* counts,
    int queries, int nprobe, int seeds, int nlist, bool four, unsigned int* output,
    int bins, float log_min, float log_step, cudaStream_t stream) {
    const auto dispatch_score = [&](auto plain_tag) {
    constexpr bool PlainL2 = decltype(plain_tag)::value;
#define HIST(U,P,F) \
    list_distance_histogram<M,U,P,F,PlainL2,Lists><<<queries * ((seeds + Lists - 1) / Lists), Threads, bins * sizeof(unsigned int), stream>>>( \
        view, probes, codes, offsets, counts, nprobe, seeds, nlist, output, bins, log_min, log_step)
    if (view.quantized()) {
        if (view.quantized_query_lut_bias()) {
            if (four) { HIST(true, true, true); }
            else { HIST(true, true, false); }
        } else {
            if (four) { HIST(true, false, true); }
            else { HIST(true, false, false); }
        }
    } else {
        if (view.quantized_query_lut_bias()) { HIST(false, true, false); }
        else { HIST(false, false, false); }
    }
#undef HIST
    };
    if (view.kind == PQQueryLutKind::PlainL2) dispatch_score(std::true_type{});
    else dispatch_score(std::false_type{});
}
}

void pq_query_lut_seed_top512(const PQQueryLutView& view, const idx_t* probes,
    const uint8_t* codes, const long long* offsets, const int* counts,
    int queries, int probe_count, int seed_probes, int pq_m,
    bool four_accumulators, uint64_t* scratch, uint64_t* merge_scratch, size_t scratch_keys,
    int* output_ids, float* output_distances, int* status,
    unsigned long long* scanned_rows, cudaStream_t stream) {
    if (!view.enabled() || view.sign_codec.codes || !probes || !codes || !offsets || !counts ||
        queries <= 0 || seed_probes <= 0 || seed_probes > 32 || seed_probes > probe_count ||
        !scratch || !merge_scratch || size_t(queries)*seed_probes*K > scratch_keys || !output_ids || !output_distances || !status)
        throw std::invalid_argument("invalid query-LUT seed top-512 request");
    // Scratch is allocated once by the threshold workspace. Split long lists
    // without enlarging it; each partition keeps an exact local top-512.
    // At most 64 local prefixes, so one grouping round leaves <= 8 finalists.
    const int partitions = int(std::min({size_t(4), size_t(std::max(1, 64 / seed_probes)),
        scratch_keys / (size_t(queries)*seed_probes*K)}));
    if (pq_m == 16) launch<16>(view,probes,codes,offsets,counts,queries,probe_count,seed_probes,partitions,four_accumulators,scratch,status,scanned_rows,stream);
    else if (pq_m == 20) launch<20>(view,probes,codes,offsets,counts,queries,probe_count,seed_probes,partitions,four_accumulators,scratch,status,scanned_rows,stream);
    else if (pq_m == 64) launch<64>(view,probes,codes,offsets,counts,queries,probe_count,seed_probes,partitions,four_accumulators,scratch,status,scanned_rows,stream);
    else throw std::invalid_argument("query-LUT seed requires M=16, 20 or 64");
    CUDA_LAUNCH_CHECK("query-LUT seed list top512");
    int sources = seed_probes * partitions;
    const uint64_t* finalists = scratch;
    if (sources > 8) {
        const int groups = (sources + 7) / 8;
        merge_seed_groups<<<queries*groups,Threads,0,stream>>>(scratch,sources,groups,merge_scratch);
        CUDA_LAUNCH_CHECK("query-LUT seed group top512");
        sources = groups;
        finalists = merge_scratch;
    }
    if (sources <= 2) merge_top512<4><<<queries,Threads,0,stream>>>(finalists,sources,output_ids,output_distances);
    else if (sources <= 4) merge_top512<8><<<queries,Threads,0,stream>>>(finalists,sources,output_ids,output_distances);
    else merge_top512<16><<<queries,Threads,0,stream>>>(finalists,sources,output_ids,output_distances);
    CUDA_LAUNCH_CHECK("query-LUT seed merge top512");
}

void pq_query_lut_seed_topk(const PQQueryLutView& view, const idx_t* probes,
    const uint8_t* codes, const long long* offsets, const int* counts,
    int queries, int probe_count, int seeds, int m, int k,
    bool four, uint64_t* scratch, uint64_t* work, size_t scratch_keys,
    int* ids, float* distances, int* status,
    unsigned long long* rows, cudaStream_t stream) {
    if (!view.enabled() || view.sign_codec.codes || !probes || !codes || !offsets || !counts ||
        queries <= 0 || seeds <= 0 || seeds > probe_count || k <= 0 || k > 8192 ||
        !scratch || !work || size_t(seeds) * k > scratch_keys || !ids || !distances || !status ||
        (m != 16 && m != 20 && m != 64))
        throw std::invalid_argument("invalid query-LUT seed top-k request or insufficient prepared scratch");
    if (k == 512 && seeds <= 32 && size_t(queries) * seeds * k <= scratch_keys) {
        pq_query_lut_seed_top512(view,probes,codes,offsets,counts,queries,probe_count,seeds,m,
            four,scratch,work,scratch_keys,ids,distances,status,rows,stream);
        return;
    }
    // Scratch capacity limits concurrent queries, not the valid seed coverage.
    // Global query indices address the original LUT/probe rows; temporary merges
    // stay packed within each chunk. No views, buffers or host sync are needed.
    const int query_chunk = int(std::min(size_t(queries),
        std::min(scratch_keys / (size_t(seeds) * k), size_t(INT_MAX / seeds))));
    for (int begin = 0; begin < queries; begin += query_chunk) {
        const int count = std::min(query_chunk, queries - begin);
        if (m == 16) launch_runtime<16>(view,probes,codes,offsets,counts,count,probe_count,seeds,k,begin,four,scratch,work,status,rows,stream);
        else if (m == 20) launch_runtime<20>(view,probes,codes,offsets,counts,count,probe_count,seeds,k,begin,four,scratch,work,status,rows,stream);
        else launch_runtime<64>(view,probes,codes,offsets,counts,count,probe_count,seeds,k,begin,four,scratch,work,status,rows,stream);
        CUDA_LAUNCH_CHECK("query-LUT runtime seed scan");
        int sources = seeds;
        uint64_t* a = scratch;
        uint64_t* b = work;
        while (sources > 1) {
            merge_runtime<<<count*((sources+1)/2),Threads,0,stream>>>(a,b,sources,k);
            sources = (sources + 1) / 2;
            std::swap(a,b);
        }
        unpack_runtime<<<count,Threads,0,stream>>>(a,k,ids+size_t(begin)*k,distances+size_t(begin)*k);
        CUDA_LAUNCH_CHECK("query-LUT runtime seed merge");
    }
}

void pq_query_lut_seed_histogram(
    const PQQueryLutView& view, const idx_t* probes,
    const uint8_t* codes, const long long* offsets, const int* counts,
    int queries, int probe_count, int seed_probes, int nlist, int pq_m,
    bool four_accumulators, unsigned int* histogram, int histogram_bins,
    float log_min, float log_step, cudaStream_t stream, float mean_list_rows) {
    if (!view.enabled() || view.sign_codec.codes || !probes || !codes ||
        !offsets || !counts || queries <= 0 || seed_probes <= 0 ||
        seed_probes > probe_count || !histogram || histogram_bins <= 0 ||
        histogram_bins > PQDistanceHistogramMaxBins) {
        throw std::invalid_argument("invalid query-LUT seed histogram request");
    }
    const int list_group = pq_seed_histogram_list_group(queries, seed_probes, mean_list_rows);
#define HISTOGRAM_GROUP(M, G) launch_histogram<M,G>(view, probes, codes, offsets, counts, queries, \
    probe_count, seed_probes, nlist, four_accumulators, histogram, histogram_bins, \
    log_min, log_step, stream)
#define HISTOGRAM(M) do { \
    if (list_group == 32) { HISTOGRAM_GROUP(M,32); } \
    else if (list_group == 8) { HISTOGRAM_GROUP(M,8); } \
    else { HISTOGRAM_GROUP(M,1); } \
} while (0)
    if (pq_m == 16) { HISTOGRAM(16); }
    else if (pq_m == 20) { HISTOGRAM(20); }
    else if (pq_m == 64) { HISTOGRAM(64); }
    else {
        throw std::invalid_argument("query-LUT seed histogram requires M=16, 20 or 64");
    }
#undef HISTOGRAM
#undef HISTOGRAM_GROUP
    CUDA_LAUNCH_CHECK("query-LUT seed distance histogram");
}

}

namespace paraselect::search {
namespace {
template<bool CorrectionU16>
__global__ void sampled_seed_histogram_kernel(
    PQQueryLutView view,const uint8_t* codes,const int* rows,const int* groups,
    int row_count,int samples,int group_count,int m,PQSeedHistogramWorkspace workspace) {
    const int q=blockIdx.y;
    __shared__ unsigned histogram[1024];
    __shared__ unsigned max_bits;
    for(int bin=threadIdx.x;bin<workspace.bins;bin+=blockDim.x)histogram[bin]=0;
    if(!threadIdx.x)max_bits=0;
    __syncthreads();
    for(int s=blockIdx.x*1024+threadIdx.x;s<min(samples,int(blockIdx.x+1)*1024);s+=blockDim.x) {
        const int i=int(static_cast<unsigned long long>(s)*row_count/samples);
        const float distance=pq_query_lut_fp32_score<CorrectionU16>(view,codes,q,groups[i],group_count,rows[i],m);
        if(!isfinite(distance)){atomicMax(workspace.status,1);continue;}
        const float log_distance=log2f(fmaxf(distance,exp2f(workspace.log_min)));
        int bin=min(workspace.bins-1,max(0,int(floorf((log_distance-workspace.log_min)/workspace.log_step))));
        while(bin<workspace.bins-1 && distance>exp2f(workspace.log_min+float(bin+1)*workspace.log_step))++bin;
        atomicAdd(histogram+bin,1u);
        atomicMax(&max_bits,__float_as_uint(fmaxf(distance,0.0f)));
    }
    __syncthreads();
    for(int bin=threadIdx.x;bin<workspace.bins;bin+=blockDim.x)
        if(histogram[bin])atomicAdd(workspace.counts+size_t(q)*workspace.bins+bin,histogram[bin]);
    if(!threadIdx.x)atomicMax(reinterpret_cast<unsigned*>(workspace.max_distance+q),max_bits);
}
__global__ void seed_histogram_threshold(int queries,int rank,PQSeedHistogramWorkspace w) {
    const int q=blockIdx.x*blockDim.x+threadIdx.x;
    if(q>=queries)return;
    int count=0;
    for(int bin=0;bin<w.bins;++bin) {
        count+=w.counts[size_t(q)*w.bins+bin];
        if(count<rank)continue;
        w.thresholds[q]=bin==w.bins-1?w.max_distance[q]:
            fminf(w.max_distance[q],exp2f(w.log_min+float(bin+1)*w.log_step));
        return;
    }
    atomicMax(w.status,2); // Insufficient finite seed rows is an error, never a fallback.
}
}
void pq_query_lut_seed_histogram_rows(
    const PQQueryLutView& view,const uint8_t* codes,const int* rows,const int* groups,
    int row_count,int samples,int group_count,int queries,int m,int rank,
    const PQSeedHistogramWorkspace& w,cudaStream_t stream) {
    if(!view.enabled() || view.quantized() || !codes || !rows || !groups ||
       row_count<samples || samples<rank || rank<=0 || group_count<=0 || queries<=0 || m<=0 ||
       !w.counts || !w.max_distance || !w.thresholds || !w.status || w.bins<=0 || w.bins>1024 ||
       !std::isfinite(w.log_min) || !std::isfinite(w.log_step) || w.log_step<=0)
        throw std::invalid_argument("invalid sampled seed histogram request");
    CUDA_CHECK(cudaMemsetAsync(w.counts,0,size_t(queries)*w.bins*sizeof(unsigned),stream));
    CUDA_CHECK(cudaMemsetAsync(w.max_distance,0,queries*sizeof(float),stream));
    CUDA_CHECK(cudaMemsetAsync(w.status,0,sizeof(int),stream));
    const dim3 grid((samples+1023)/1024,queries);
    if(view.quantized_query_lut_bias())
        sampled_seed_histogram_kernel<true><<<grid,256,0,stream>>>(view,codes,rows,groups,row_count,samples,group_count,m,w);
    else
        sampled_seed_histogram_kernel<false><<<grid,256,0,stream>>>(view,codes,rows,groups,row_count,samples,group_count,m,w);
    seed_histogram_threshold<<<(queries+255)/256,256,0,stream>>>(queries,rank,w);
    CUDA_LAUNCH_CHECK("sampled seed histogram threshold");
}
} // namespace paraselect::search
