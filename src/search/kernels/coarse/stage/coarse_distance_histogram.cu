#include "coarse_distance_histogram.cuh"
#include "utils/check_cuda_error.cuh"
#include <cub/block/block_reduce.cuh>
#include <cfloat>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace paraselect::coarse {
namespace {
using Reduce = cub::BlockReduce<float, 256>;
struct Minimum { __device__ float operator()(float a, float b) const { return fminf(a,b); } };
struct Maximum { __device__ float operator()(float a, float b) const { return fmaxf(a,b); } };
template<bool Overflow=false>
__device__ float edge(float lo, float hi, int bin) {
    return bin == CoarseDistanceHistogram::bins - 1 ? (Overflow ? INFINITY : hi)
        : fmaf((hi - lo) / CoarseDistanceHistogram::bins, float(bin + 1), lo);
}
template<bool Overflow=false>
__device__ int distance_bin(float d,float lo,float hi) {
    if constexpr(Overflow) {
        if(d<=lo)return 0;
        if(d>hi)return 255;
    }
    int bin=hi>lo ? min(255,max(0,int((d-lo)/(hi-lo)*256.f))) : 0;
    while(bin<255 && d>edge<Overflow>(lo,hi,bin))++bin;
    while(bin>0 && d<=edge<Overflow>(lo,hi,bin-1))--bin;
    return bin;
}
__device__ void atomic_extreme(float* address,float value,bool minimum) {
    auto bits=reinterpret_cast<unsigned*>(address);
    unsigned old=*bits;
    while(minimum ? value<__uint_as_float(old) : value>__uint_as_float(old)) {
        const unsigned expected=old;
        old=atomicCAS(bits,expected,__float_as_uint(value));
        if(old==expected)break;
    }
}
constexpr int range_tile=8192;
__global__ void init_tiled(int queries,CoarseDistanceHistogram w) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(queries)*256)return;
    w.counts[i]=0;
    if(i%256==0) {
        const int q=i/256;w.minimum[q]=FLT_MAX;w.maximum[q]=-FLT_MAX;
        w.lower[q]=w.upper[q]=-INFINITY;
    }
}
template<bool ConvertL2>
__global__ void range_tiled(const float* distances,int n,CoarseDistanceHistogram w,
                            const float* query_norm,const float* centroid_norm) {
    __shared__ Reduce::TempStorage temp;
    const int q=blockIdx.y,start=blockIdx.x*range_tile,end=min(n,start+range_tile);
    float lo=FLT_MAX,hi=-FLT_MAX;
    for(int c=start+threadIdx.x;c<end;c+=blockDim.x) {
        float d=distances[size_t(q)*n+c];
        if constexpr(ConvertL2) {
            d=coarse_l2_distance(d,query_norm[q],centroid_norm[c]);
        }
        if(!isfinite(d))atomicExch(w.status,1);
        lo=fminf(lo,d);hi=fmaxf(hi,d);
    }
    const float minimum=Reduce(temp).Reduce(lo,Minimum());
    if(!threadIdx.x)atomic_extreme(w.minimum+q,minimum,true);
    __syncthreads();
    const float maximum=Reduce(temp).Reduce(hi,Maximum());
    if(!threadIdx.x)atomic_extreme(w.maximum+q,maximum,false);
}
template<bool ConvertL2,bool Overflow=false,bool Sampled=false>
__global__ void histogram_tiled(const float* distances,int n,CoarseDistanceHistogram w,
                                const float* query_norm,const float* centroid_norm,
                                const int* complete=nullptr, CoarseHistogramQuerySubset subset={}) {
    __shared__ unsigned histogram[256];
    const int queries=subset.count ? *subset.count : gridDim.y;
    for(int row=blockIdx.y;row<queries;row+=gridDim.y) {
    const int q=subset.ids ? subset.ids[row] : row;
    if(complete && complete[q])continue;
    const int samples=Sampled ? min(n,8192) : n;
    const int first=blockIdx.x*range_tile;
    const float lo=w.minimum[q],hi=w.maximum[q];
    histogram[threadIdx.x]=0;__syncthreads();
    // A sliced input may not retain cudaMalloc's alignment.
    const bool aligned=(reinterpret_cast<uintptr_t>(distances)%16==0) &&
        (!ConvertL2 || reinterpret_cast<uintptr_t>(centroid_norm)%16==0);
    for(int start=first;start<samples;start+=gridDim.x*range_tile) {
    const int end=min(samples,start+range_tile);
    if(!Sampled && n%4==0 && aligned) {
        for(int c=start+4*threadIdx.x;c<end;c+=4*blockDim.x) {
            const float4 values=*reinterpret_cast<const float4*>(distances+size_t(q)*n+c);
            float4 norms{};
            if constexpr(ConvertL2)norms=*reinterpret_cast<const float4*>(centroid_norm+c);
            const float ds[4]={values.x,values.y,values.z,values.w};
            const float ns[4]={norms.x,norms.y,norms.z,norms.w};
            #pragma unroll
            for(int i=0;i<4;++i) {
                float d=ds[i];
                if constexpr(ConvertL2)d=coarse_l2_distance(d,query_norm[q],ns[i]);
                if(isfinite(d))atomicAdd(histogram+distance_bin<Overflow>(d,lo,hi),1u);
                else if constexpr(Overflow)atomicExch(w.status,1);
            }
        }
    } else for(int i=start+threadIdx.x;i<end;i+=blockDim.x) {
        const int c=Sampled ? coarse_histogram_sample_index(i,n,samples) : i;
        float d=distances[size_t(q)*n+c];
        if constexpr(ConvertL2)d=coarse_l2_distance(d,query_norm[q],centroid_norm[c]);
        if(isfinite(d))atomicAdd(histogram+distance_bin<Overflow>(d,lo,hi),1u);
        else if constexpr(Overflow)atomicExch(w.status,1);
    }
    }
    __syncthreads();
    const unsigned count=histogram[threadIdx.x];
    if(count)atomicAdd(w.counts+size_t(q)*256+threadIdx.x,count);
    __syncthreads();
    }
}
__global__ void sampled_range(const float* scores,const float* query_norm,
    const float* centroid_norm,int n,CoarseDistanceHistogram w,const int* complete=nullptr,
    CoarseHistogramQuerySubset subset={}) {
    __shared__ Reduce::TempStorage temp;
    const int queries=subset.count ? *subset.count : gridDim.x;
    for(int row=blockIdx.x;row<queries;row+=gridDim.x) {
    const int q=subset.ids ? subset.ids[row] : row,samples=min(n,1024);
    if(complete && complete[q])continue;
    float lo=FLT_MAX,hi=-FLT_MAX;
    w.counts[size_t(q)*256+threadIdx.x]=0;
    for(int i=threadIdx.x;i<samples;i+=blockDim.x) {
        const int c=coarse_histogram_sample_index(i,n,samples);
        const float d=coarse_l2_distance(scores[size_t(q)*n+c],query_norm[q],centroid_norm[c]);
        lo=fminf(lo,d);hi=fmaxf(hi,d);
    }
    const float minimum=Reduce(temp).Reduce(lo,Minimum());
    if(!threadIdx.x)w.minimum[q]=minimum;
    __syncthreads();
    const float maximum=Reduce(temp).Reduce(hi,Maximum());
    if(!threadIdx.x) {
        w.maximum[q]=maximum;
        w.lower[q]=w.upper[q]=-INFINITY;
    }
    __syncthreads();
    }
}
__global__ void build_kernel(const float* distances, int nlist, CoarseDistanceHistogram w) {
    __shared__ Reduce::TempStorage temp;
    __shared__ float lo, hi;
    __shared__ unsigned histogram[CoarseDistanceHistogram::bins];
    const int q = blockIdx.x;
    float local_min = FLT_MAX, local_max = -FLT_MAX;
    for (int c = threadIdx.x; c < nlist; c += blockDim.x) {
        const float d = distances[size_t(q) * nlist + c];
        if (!isfinite(d)) atomicExch(w.status, 1);
        local_min = fminf(local_min, d); local_max = fmaxf(local_max, d);
    }
    const float minimum = Reduce(temp).Reduce(local_min, Minimum());
    if (!threadIdx.x) lo = minimum;
    __syncthreads();
    const float maximum = Reduce(temp).Reduce(local_max, Maximum());
    if (!threadIdx.x) hi = maximum;
    histogram[threadIdx.x] = 0;
    __syncthreads();
    for (int c = threadIdx.x; c < nlist; c += blockDim.x) {
        const float d = distances[size_t(q) * nlist + c];
        if (!isfinite(d)) continue;
        atomicAdd(histogram + distance_bin(d,lo,hi), 1u);
    }
    __syncthreads();
    w.counts[size_t(q) * 256 + threadIdx.x] = histogram[threadIdx.x];
    if (!threadIdx.x) {
        w.minimum[q] = lo; w.maximum[q] = hi;
        w.lower[q] = w.upper[q] = -INFINITY;
    }
}
template<bool Boundary,bool Overflow=false>
__global__ void select_kernel(int queries, int target, const int* complete, CoarseDistanceHistogram w,
    CoarseHistogramQuerySubset subset={}) {
    const int count=subset.count ? *subset.count : queries;
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<count;row+=gridDim.x*blockDim.x) {
    const int q=subset.ids ? subset.ids[row] : row;
    if (complete && complete[q]) continue;
    int total = 0, bin = 0;
    for (; bin < 255; ++bin) {
        total += w.counts[size_t(q) * 256 + bin];
        if (total >= target) break;
    }
    if (bin == 255) total += w.counts[size_t(q) * 256 + bin];
    w.lower[q] = Boundary ? (bin ? edge<Overflow>(w.minimum[q], w.maximum[q], bin-1) : -INFINITY) : w.upper[q];
    w.upper[q] = edge<Overflow>(w.minimum[q], w.maximum[q], bin);
    w.selected_counts[q] = total;
    }
}
void validate(CoarseDistanceHistogram w) {
    if (!w.counts || !w.minimum || !w.maximum || !w.lower || !w.upper ||
        !w.selected_counts || !w.status) throw std::invalid_argument("invalid coarse seed histogram workspace");
}
}
void coarse_distance_histogram_build(const float* distances, int queries, int nlist,
                                CoarseDistanceHistogram w, cudaStream_t stream) {
    validate(w);
    if (!distances || queries <= 0 || nlist <= 0) throw std::invalid_argument("invalid coarse histogram shape");
    build_kernel<<<queries, 256, 0, stream>>>(distances, nlist, w);
    CUDA_LAUNCH_CHECK("coarse seed histogram build");
}
void coarse_distance_histogram_boundary(int queries, int target, CoarseDistanceHistogram w, cudaStream_t stream) {
    validate(w);
    if(queries<=0 || target<=0) throw std::invalid_argument("invalid coarse histogram rank");
    select_kernel<true><<<(queries+255)/256,256,0,stream>>>(queries,target,nullptr,w);
    CUDA_LAUNCH_CHECK("coarse histogram boundary");
}

namespace {
template<bool ConvertL2>
void build_tiled(const float* distances,const float* query_norm,const float* centroid_norm,
                 int queries,int nlist,CoarseDistanceHistogram w,cudaStream_t stream) {
    validate(w);
    if(!distances || queries<=0 || queries>65535 || nlist<=0)
        throw std::invalid_argument("invalid tiled coarse histogram shape");
    if constexpr(ConvertL2) {
        if(!query_norm || !centroid_norm)
            throw std::invalid_argument("missing coarse L2 norms");
    }
    init_tiled<<<queries,256,0,stream>>>(queries,w);
    const dim3 grid((nlist+range_tile-1)/range_tile,queries);
    range_tiled<ConvertL2><<<grid,256,0,stream>>>(distances,nlist,w,query_norm,centroid_norm);
    histogram_tiled<ConvertL2><<<grid,256,0,stream>>>(distances,nlist,w,query_norm,centroid_norm);
    CUDA_LAUNCH_CHECK("tiled coarse histogram build");
}
}
void coarse_distance_histogram_build_tiled(const float* distances,int queries,int nlist,
                                          CoarseDistanceHistogram w,cudaStream_t stream) {
    build_tiled<false>(distances,nullptr,nullptr,queries,nlist,w,stream);
}
void coarse_distance_histogram_build_l2_tiled(const float* scores,const float* query_norm,
    const float* centroid_norm,int queries,int nlist,CoarseDistanceHistogram w,cudaStream_t stream) {
    build_tiled<true>(scores,query_norm,centroid_norm,queries,nlist,w,stream);
}
void coarse_distance_histogram_select_l2(const float* scores,const float* query_norm,
    const float* centroid_norm,int queries,int nlist,int target,
    CoarseDistanceHistogram w,cudaStream_t stream,const int* complete,CoarseHistogramQuerySubset subset) {
    validate(w);
    if(!scores || !query_norm || !centroid_norm || queries<=0 || queries>65535 ||
       nlist<=0 || target<=0 || target>nlist)
        throw std::invalid_argument("invalid sampled coarse histogram shape");
    const int blocks=subset.ids ? std::min(queries,8) : queries;
    sampled_range<<<blocks,256,0,stream>>>(scores,query_norm,centroid_norm,nlist,w,complete,subset);
    const dim3 grid((nlist+range_tile-1)/range_tile,blocks);
    histogram_tiled<true,true><<<grid,256,0,stream>>>(scores,nlist,w,query_norm,centroid_norm,complete,subset);
    select_kernel<true,true><<<(blocks+255)/256,256,0,stream>>>(queries,target,complete,w,subset);
    CUDA_LAUNCH_CHECK("sampled-range coarse histogram select");
}

void coarse_distance_histogram_estimate_l2(const float* scores,const float* query_norm,
    const float* centroid_norm,int queries,int nlist,int target,
    CoarseDistanceHistogram w,cudaStream_t stream) {
    validate(w);
    if(!scores || !query_norm || !centroid_norm || queries<=0 || queries>65535 ||
       nlist<=0 || target<=0 || target>nlist)
        throw std::invalid_argument("invalid estimated coarse histogram shape");
    const int samples=std::min(nlist,8192);
    const float expected=float(target)*samples/nlist;
    const int rank=std::min(samples,int(std::ceil(expected+2.f*std::sqrt(expected)+1.f)));
    sampled_range<<<queries,256,0,stream>>>(scores,query_norm,centroid_norm,nlist,w);
    histogram_tiled<true,true,true><<<dim3(1,queries),256,0,stream>>>(scores,nlist,w,query_norm,centroid_norm);
    select_kernel<true,true><<<(queries+255)/256,256,0,stream>>>(queries,rank,nullptr,w);
    CUDA_LAUNCH_CHECK("estimated coarse histogram select");
}

}
