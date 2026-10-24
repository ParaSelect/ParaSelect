#include "search/bruteforce_respq/bruteforce_respq_search.cuh"
#include "search/ivf_common/ivf_host_rerank.h"
#include "transform/opq_transform.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_query_lut_score.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_query_lut_seed.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_candidate_cap_select.cuh"
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace paraselect::search {
namespace {
void check(cudaError_t e) {
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template<class T> struct Buffer {
    T* p=nullptr;
    explicit Buffer(size_t n) { if(n) check(cudaMalloc(&p,n*sizeof(T))); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    void upload(const T* h,size_t n) { if(n) check(cudaMemcpy(p,h,n*sizeof(T),cudaMemcpyHostToDevice)); }
};
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t) {
    return std::chrono::duration<double,std::milli>(Clock::now()-t).count();
}

__global__ void make_query_lut_bias(const uint8_t* codes,const float* book,const float* centers,
    const int* rows,const int* groups,int n,int m,int k,int dim,float* correction) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const int row=rows[i],group=groups[i],sub=dim/m;
    float sum=0;
    for(int d=0;d<dim;++d) {
        const float residual=book[((d/sub)*k+codes[size_t(row)*m+d/sub])*sub+d%sub];
        sum += residual*residual + 2.0f*centers[group*dim+d]*residual;
    }
    correction[row]=sum;
}
template<bool Threshold, bool CorrectionU16>
__global__ void scan(const uint8_t* codes,const int* rows,const int* groups,
    PQQueryLutView view,int n,int m,int k,int nlist,float* distances,int* ids,
    const float* thresholds=nullptr,int* pool_queries=nullptr,int* counts=nullptr,
    int* total=nullptr,int* status=nullptr) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x,q=blockIdx.y;
    float distance=INFINITY;
    if(i<n) distance=pq_query_lut_fp32_score<CorrectionU16>(view,codes,q,groups[i],nlist,rows[i],m,k);
    if constexpr (!Threshold) {
        if(i<n) {distances[size_t(q)*n+i]=distance; ids[size_t(q)*n+i]=rows[i];}
    } else {
        if(i<n && !isfinite(distance)) atomicMax(status,1);
        const unsigned mask=__ballot_sync(0xffffffffu,i<n && distance<=thresholds[q]);
        const int lane=threadIdx.x&31;
        int begin=0;
        if(lane==0 && mask) {
            begin=atomicAdd(total,__popc(mask));
            atomicAdd(counts+q,__popc(mask));
        }
        begin=__shfl_sync(0xffffffffu,begin,0);
        if(mask & (1u<<lane)) {
            const int slot=begin+__popc(mask & ((1u<<lane)-1u));
            ids[slot]=rows[i]; distances[slot]=distance; pool_queries[slot]=q;
        }
    }
}

// Shared ResPQ scoring needs a center term for every active list. Compute it
// once per scan chunk and reuse it in seed and scan; it performs no selection.
__global__ void adc_centroid_bias(const float* queries,const float* centers,
                                float* distances,int dim,int groups) {
    const int lane=threadIdx.x%32;
    const int group=(blockIdx.x*blockDim.x+threadIdx.x)/32,q=blockIdx.y;
    float value=0;
    if(group<groups) for(int d=lane;d<dim;d+=32) {
        float x=queries[size_t(q)*dim+d]-centers[size_t(group)*dim+d];
        value=fmaf(x,x,value);
    }
    for(int offset=16;offset;offset/=2)value+=__shfl_down_sync(0xffffffffu,value,offset);
    if(group<groups && lane==0)distances[size_t(q)*groups+group]=value;
}
struct SeedWorkspace {
    Buffer<float> max_distance,thresholds;
    Buffer<unsigned> histogram;
    Buffer<int> status;
    SeedWorkspace(int batch,int bins)
        : max_distance(batch),thresholds(batch),histogram(size_t(batch)*bins),status(1) {}
    PQSeedHistogramWorkspace view(int bins) {
        return {histogram.p,max_distance.p,thresholds.p,status.p,bins,-8.0f,32.0f/bins};
    }
};

struct CandidateWorkspace {
    Buffer<float> pool_dist,out_dist;
    Buffer<int> pool_ids,pool_queries,counts,total,sorted_ids,offsets,out_counts,status;
    Buffer<uint64_t> keys_in,keys_out;
    size_t sort_bytes;
    Buffer<uint8_t> sort_temp;
    CandidateWorkspace(int b,size_t capacity,int take)
        : pool_dist(capacity),out_dist(size_t(b)*take),pool_ids(capacity),pool_queries(capacity),
          counts(b),total(1),sorted_ids(capacity),offsets(b+1),out_counts(b),status(1),
          keys_in(capacity),keys_out(capacity),
          sort_bytes(pq_candidate_cap_select_temp_bytes(int(capacity),b)),sort_temp(sort_bytes) {}
};

__global__ void shortlist(const int* sorted,int* out,int n,int take) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<take) out[size_t(blockIdx.y)*take+i]=sorted[size_t(blockIdx.y)*n+i];
}
} // namespace

void bruteforce_respq_search(const BruteForceResPQSearchRequest& r, IVFSearchStats* stats) {
    const auto& index=r.index;
    const auto& pq=index.storage.pq;
    const auto& storage=index.storage.assets.exact;
    const int dim=r.queries.dim,m=pq.subvectors,k=pq.codebook_size;
    validate_ivf_index_shape(index.coarse, index.lists);
    if (index.coarse.dim != dim)
        throw std::invalid_argument("BruteForceResPQ query dimension does not match the IVF index");
    if(!pq.codes || !pq.codebook || m<=0 || k!=256 ||
       dim<=0 || dim%m || !r.queries.fp32 || r.queries.count<=0 ||
       !r.output.indices || !r.output.distances || r.batch_size<=0 || r.scan_batch<=0 ||
       r.topk<=0 || r.n_rerank<r.topk || index.lists.n_vectors>INT_MAX ||
       storage.kind!=IVFExactStorageKind::Host ||
       (!index.storage.host_rows && storage.host.count!=index.lists.n_vectors) ||
       storage.host.dim!=dim)
        throw std::invalid_argument("invalid BruteForceResPQ request (host exact storage and row-major PQ required)");
    if(index.storage.host_rows) {
        if(storage.host.source_rows || storage.host.count<=0 || storage.host.count>INT_MAX)
            throw std::invalid_argument("invalid shared host storage for materialized BF");
        for(idx_t row=0;row<index.lists.n_vectors;++row)
            if(index.storage.host_rows[row]<0 || index.storage.host_rows[row]>=storage.host.count ||
               index.lists.reordered_ids[row]!=index.storage.host_rows[row])
                throw std::invalid_argument("materialized BF row mapping mismatch");
    }
    const bool use_float=!(storage.host.u8 && r.queries.u8);
    if(use_float && !storage.host.fp32) throw std::invalid_argument("BruteForceResPQ rerank dtype mismatch");
    const auto& rc=index.storage.assets.query_lut_bias;
    if(!rc.empty() && (!rc.enabled() || rc.count!=static_cast<int64_t>(index.lists.n_vectors)))
        throw std::invalid_argument("BruteForceResPQ row correction shape mismatch");
    if(r.seed_histogram_bins!=0 && r.seed_histogram_bins!=256 &&
       r.seed_histogram_bins!=512 && r.seed_histogram_bins!=1024)
        throw std::invalid_argument("seed histogram bins must be 0,256,512,1024");
    if(r.seed_rows<=0) throw std::invalid_argument("seed row budget must be positive");
    const bool use_histogram=r.seed_histogram_bins!=0;
    IVFSearchStats local;
    auto prepare_begin=Clock::now();
    // Enumerate physical rows once, retaining their codec group and scan row ID.
    std::vector<int> rows,groups;
    std::vector<float> active_centers;
    for(int g=0;g<index.lists.nlist;++g) {
        const int compact_group=int(active_centers.size()/dim);
        const size_t before=rows.size();
        for(idx_t row=index.lists.offsets[g];row<index.lists.offsets[g+1];++row) {
            rows.push_back(static_cast<int>(row)); groups.push_back(compact_group);
        }
        if(rows.size()!=before)active_centers.insert(active_centers.end(),
            index.coarse.centroids+size_t(g)*dim,index.coarse.centroids+size_t(g+1)*dim);
    }
    const int n=static_cast<int>(rows.size());
    const int requested_batch=std::min(r.batch_size,r.queries.count);
    const int take=std::min(r.n_rerank,n),nlist=int(active_centers.size()/dim);
    Buffer<uint8_t> codes(n ? size_t(index.lists.n_vectors)*m : 0);
    Buffer<float> book(size_t(m)*k*(dim/m)),centers(size_t(nlist)*dim),rot_centers(index.storage.assets.residual_rotation?size_t(nlist)*dim:0);
    Buffer<int> device_rows(n),device_groups(n);
    Buffer<float> rc_float(n && (rc.fp32 || rc.empty()) ? index.lists.n_vectors:0);
    Buffer<uint16_t> rc_u16(n && rc.u16 ? rc.count:0);
    if(n) codes.upload(pq.codes,size_t(index.lists.n_vectors)*m);
    book.upload(pq.codebook,size_t(k)*dim); centers.upload(active_centers.data(),active_centers.size());
    device_rows.upload(rows.data(),n); device_groups.upload(groups.data(),n);
    QueryLutBias device_rc=rc;
    if(n && rc.fp32) rc_float.upload(rc.fp32,rc.count);
    if(n && rc.u16) rc_u16.upload(rc.u16,rc.count);
    device_rc.fp32=rc_float.p; device_rc.u16=rc_u16.p;
    device_rc.count=index.lists.n_vectors;
    std::unique_ptr<paraselect::transform::OPQTransform> transform;
    const float* scan_centers=centers.p;
    if(index.storage.assets.residual_rotation) {
        transform=std::make_unique<paraselect::transform::OPQTransform>(dim);
        transform->set_matrix(std::vector<float>(index.storage.assets.residual_rotation,index.storage.assets.residual_rotation+size_t(dim)*dim));
        transform->prepare(); if(nlist)transform->apply_device(centers.p,rot_centers.p,nlist,0);
        scan_centers=rot_centers.p;
    }

    // The fixed scan budget applies identically to full sort and histogram.
    // Every row is visited; capacity only bounds concurrent queries.
    const size_t max_scan_cells=std::max(size_t(1)<<26,size_t(n));
    const int batch=n ? std::min({requested_batch,r.scan_batch,int(max_scan_cells/n)})
                      : std::min(requested_batch,r.scan_batch);
    const int seed_rows=std::min(n,std::max(take,r.seed_rows));
    const size_t cells=size_t(batch)*n;
    const size_t reference_cells=(!use_histogram || r.verify_candidates)?cells:0;
    Buffer<int> offsets(batch+1);
    Buffer<float> query(size_t(batch)*dim),rot_query(index.storage.assets.residual_rotation?size_t(batch)*dim:0),lut(size_t(batch)*m*k),dots(size_t(batch)*nlist);
    Buffer<float> distances(reference_cells),sorted_distances(reference_cells);
    Buffer<int> ids(reference_cells),sorted_ids(reference_cells),candidates(size_t(batch)*take);
    std::vector<int> host_offsets(batch+1);
    for(int q=0;q<=batch;++q) host_offsets[q]=q*n;
    offsets.upload(host_offsets.data(),batch+1);
    std::unique_ptr<SeedWorkspace> hw;
    std::unique_ptr<CandidateWorkspace> cw;
    if(use_histogram && n) {
        hw=std::make_unique<SeedWorkspace>(batch,r.seed_histogram_bins);
        cw=std::make_unique<CandidateWorkspace>(batch,cells,take);
    }
    size_t temp_bytes=0;
    if(n) {
        if(rc.empty()) make_query_lut_bias<<<(n+255)/256,256>>>(codes.p,book.p,scan_centers,device_rows.p,device_groups.p,n,m,k,dim,rc_float.p);
        check(cudaGetLastError());
        if(reference_cells) check(cub::DeviceSegmentedRadixSort::SortPairs(nullptr,temp_bytes,distances.p,sorted_distances.p,ids.p,sorted_ids.p,int(cells),batch,offsets.p,offsets.p+1));
    }
    Buffer<uint8_t> temp(temp_bytes);
    std::vector<int> host_candidates(size_t(batch)*take),out_rows(size_t(batch)*r.topk);
    ivf_prewarm_host_rerank(r.rerank_threads,batch,false);
    check(cudaDeviceSynchronize());
    local.numeric_fields["bruteforce_prepare_ms"]=elapsed(prepare_begin);
    local.numeric_fields["bruteforce_scanned_rows"]=n;
    local.numeric_fields["scan_batch"]=batch;
    local.numeric_fields["score_workspace_bytes"]=double(size_t(batch)*(nlist+m*k)*sizeof(float));
    local.numeric_fields["request_batch_size"]=r.batch_size;
    double score_prepare_ms=0,scan_ms=0,select_ms=0,rerank_ms=0,seed_ms=0,verify_ms=0;
    long long candidate_total=0;
    int verified_queries=0;
    const bool progress = std::getenv("PARASELECT_BF_QUERY_PROGRESS") != nullptr;
    if(progress) { std::fprintf(stderr,"[BF_QUERY_BEGIN]\n"); std::fflush(stderr); }
    auto start=Clock::now();
    for(int request_begin=0;request_begin<r.queries.count;request_begin+=requested_batch) {
        const int request_n=std::min(requested_batch,r.queries.count-request_begin);
        auto batch_start=Clock::now();
        double batch_validation_ms=0;
        if(n) {
          for(int scan_begin=0;scan_begin<request_n;scan_begin+=batch) {
            const int b=std::min(batch,request_n-scan_begin);
            const int begin=request_begin+scan_begin;
            auto t=Clock::now();
            query.upload(r.queries.fp32+size_t(begin)*dim,size_t(b)*dim);
            const float* scan_query=query.p;
            if(transform) { transform->apply_device(query.p,rot_query.p,b,0); scan_query=rot_query.p; }
            pq_build_query_lut_gpu(scan_query,book.p,b,m,dim/m,lut.p,0);
            adc_centroid_bias<<<dim3((nlist+7)/8,b),256>>>(scan_query,scan_centers,dots.p,dim,nlist);
            check(cudaDeviceSynchronize());score_prepare_ms+=elapsed(t);
            PQQueryLutView view;
            view.query_codeword_dot_lut=lut.p;
            view.query_centroid_distances=dots.p;
            view.query_lut_bias=device_rc;
            if(hw) {
                t=Clock::now();
                pq_query_lut_seed_histogram_rows(view,codes.p,device_rows.p,device_groups.p,
                    n,seed_rows,nlist,b,m,take,hw->view(r.seed_histogram_bins),0);
                int status=0;check(cudaMemcpy(&status,hw->status.p,sizeof(int),cudaMemcpyDeviceToHost));
                if(status)throw std::runtime_error("invalid sampled seed histogram");
                seed_ms+=elapsed(t);
            }
                float* output_dist=r.output.distances+size_t(begin)*r.topk;
                const auto run_scan=[&](bool threshold,float* out_dist,int* out_ids) {
                    const dim3 grid((n+255)/256,b);
                    if(threshold) {
                        if(device_rc.u16) scan<true,true><<<grid,256>>>(codes.p,device_rows.p,device_groups.p,view,n,m,k,nlist,out_dist,out_ids,hw->thresholds.p,cw->pool_queries.p,cw->counts.p,cw->total.p,cw->status.p);
                        else scan<true,false><<<grid,256>>>(codes.p,device_rows.p,device_groups.p,view,n,m,k,nlist,out_dist,out_ids,hw->thresholds.p,cw->pool_queries.p,cw->counts.p,cw->total.p,cw->status.p);
                    } else {
                        if(device_rc.u16) scan<false,true><<<grid,256>>>(codes.p,device_rows.p,device_groups.p,view,n,m,k,nlist,out_dist,out_ids);
                        else scan<false,false><<<grid,256>>>(codes.p,device_rows.p,device_groups.p,view,n,m,k,nlist,out_dist,out_ids);
                    }
                    check(cudaGetLastError());
                };
                if(hw) {
                    check(cudaMemset(cw->counts.p,0,b*sizeof(int)));
                    check(cudaMemset(cw->total.p,0,sizeof(int)));
                    check(cudaMemset(cw->status.p,0,sizeof(int)));
                    t=Clock::now(); run_scan(true,cw->pool_dist.p,cw->pool_ids.p);
                    int total=0,status=0;
                    check(cudaMemcpy(&total,cw->total.p,sizeof(int),cudaMemcpyDeviceToHost));
                    check(cudaMemcpy(&status,cw->status.p,sizeof(int),cudaMemcpyDeviceToHost));
                    if(status) throw std::runtime_error("nonfinite full-scan ADC distance");
                    scan_ms+=elapsed(t); candidate_total+=total; t=Clock::now();
                    pq_candidate_cap_select(cw->pool_ids.p,cw->pool_dist.p,cw->pool_queries.p,cw->counts.p,
                        total,b,take,take,cw->keys_in.p,cw->keys_out.p,cw->sorted_ids.p,cw->offsets.p,
                        candidates.p,cw->out_dist.p,cw->out_counts.p,cw->sort_temp.p,cw->sort_bytes,0);
                } else {
                    t=Clock::now(); run_scan(false,distances.p,ids.p);
                    check(cudaDeviceSynchronize()); scan_ms+=elapsed(t); t=Clock::now();
                    check(cub::DeviceSegmentedRadixSort::SortPairs(temp.p,temp_bytes,distances.p,sorted_distances.p,ids.p,sorted_ids.p,b*n,b,offsets.p,offsets.p+1));
                    shortlist<<<dim3((take+255)/256,b),256>>>(sorted_ids.p,candidates.p,n,take);
                    candidate_total+=static_cast<long long>(b)*n;
                }
                check(cudaGetLastError());
                check(cudaMemcpy(host_candidates.data(),candidates.p,size_t(b)*take*sizeof(int),cudaMemcpyDeviceToHost));
                select_ms+=elapsed(t);
                if(hw && r.verify_candidates) {
                    t=Clock::now(); run_scan(false,distances.p,ids.p);
                    check(cub::DeviceSegmentedRadixSort::SortPairs(temp.p,temp_bytes,distances.p,sorted_distances.p,ids.p,sorted_ids.p,b*n,b,offsets.p,offsets.p+1));
                    std::vector<float> expected(size_t(b)*take),actual(size_t(b)*take);
                    check(cudaMemcpy2D(expected.data(),take*sizeof(float),sorted_distances.p,n*sizeof(float),take*sizeof(float),b,cudaMemcpyDeviceToHost));
                    check(cudaMemcpy(actual.data(),cw->out_dist.p,size_t(b)*take*sizeof(float),cudaMemcpyDeviceToHost));
                    if(expected!=actual) throw std::runtime_error("histogram admission lost ADC top-k");
                    verified_queries+=b;
                    const double validation_ms=elapsed(t);
                    verify_ms+=validation_ms;
                    batch_validation_ms+=validation_ms;
                }
                t=Clock::now();
                ivf_rerank_host(storage.host,r.queries.u8?r.queries.u8+size_t(begin)*dim:nullptr,
                    r.queries.fp32+size_t(begin)*dim,use_float,host_candidates.data(),b,take,take,r.topk,
                    out_rows.data(),output_dist,r.rerank_threads,false,r.rerank_prefetch,nullptr,nullptr,
                    index.storage.host_rows,
                    index.storage.host_rows ? index.lists.n_vectors : 0);
                rerank_ms+=elapsed(t);
                for(int q=0;q<b;++q) for(int j=0;j<r.topk;++j) {
                    const size_t i=size_t(q)*r.topk+j;
                    const int row=out_rows[i];
                    r.output.indices[size_t(begin)*r.topk+i]=row>=0?(index.storage.host_rows?idx_t(row):index.lists.reordered_ids[row]):idx_t(-1);
                    if(row<0) output_dist[i]=INFINITY;
                }
            }
        } else {
            std::fill_n(r.output.indices+size_t(request_begin)*r.topk,size_t(request_n)*r.topk,idx_t(-1));
            std::fill_n(r.output.distances+size_t(request_begin)*r.topk,size_t(request_n)*r.topk,INFINITY);
        }
        local.batch_timings.push_back({int(local.batch_timings.size()),request_begin,request_n,
                                      elapsed(batch_start)-batch_validation_ms});
    }
    if(progress) { std::fprintf(stderr,"[BF_QUERY_END]\n"); std::fflush(stderr); }
    local.total_time_ms=elapsed(start)-verify_ms;
    local.coarse_search_ms=0;
    local.avg_batch_time_ms=local.total_time_ms/local.batch_timings.size();
    local.cpu_fine_ms=rerank_ms;
    local.stage_timings={{"score_prepare",score_prepare_ms},{"seed_histogram",seed_ms},{"bruteforce_scan",scan_ms},{"bruteforce_select",select_ms},{"rerank",rerank_ms}};
    local.numeric_fields["bruteforce_scan_ms"]=scan_ms;
    local.numeric_fields["coarse_ms"]=0;
    local.numeric_fields["score_prepare_ms"]=score_prepare_ms;
    local.numeric_fields["seed_ms"]=seed_ms;
    local.numeric_fields["seed_histogram_bins"]=r.seed_histogram_bins;
    local.numeric_fields["seed_scanned_rows"]=use_histogram?seed_rows:0;
    local.string_fields["seed_source"]="rows";
    local.string_fields["ivfrespq_coarse_topk_policy"]="none";
    local.numeric_fields["candidate_count_avg"]=double(candidate_total)/r.queries.count;
    local.numeric_fields["verified_queries"]=verified_queries;
    local.string_fields["scan_order"]="original";
    local.string_fields["selection"]=use_histogram?"seed_histogram":"full_sort";
    local.numeric_fields["bruteforce_select_ms"]=select_ms;
    local.numeric_fields["rerank_ms"]=rerank_ms;
    local.numeric_fields["bruteforce_prepare_amortized_qps"]=1000*r.queries.count/(local.total_time_ms+local.numeric_fields["bruteforce_prepare_ms"]);
    local.string_fields["search_family"]="bruteforce";
    local.string_fields["search_codec"]="respq";
    local.string_fields["ivfrespq_pipeline"]="serial";
    local.string_fields["adc_mode"]="query_lut";
    local.string_fields["lut_quantization"]="fp32";
    if(stats) *stats=std::move(local);
}
} // namespace paraselect::search
