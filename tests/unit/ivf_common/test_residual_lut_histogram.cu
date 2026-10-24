#include "search/kernels/fine/gpu_fine/pq_adc/pq_query_lut_scan.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_query_lut.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_adc_workspace.cuh"
#include "search/ivfrespq/stage/residual_pq_gpu.cuh"
#include "utils/check_cuda_error.cuh"
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cstdlib>
#include <set>
using namespace paraselect::search;
template<class T> struct Buffer {
    T* p; size_t n;
    Buffer(size_t count):n(count){CUDA_CHECK(cudaMalloc(&p,n*sizeof(T)));}
    ~Buffer(){cudaFree(p);}
    void put(const std::vector<T>& x){CUDA_CHECK(cudaMemcpy(p,x.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T>x(n);CUDA_CHECK(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return x;}
};
void check(int m, int dsub) {
    const int nq=2,np=3,nlist=4,perlist=512,n=nlist*perlist,dim=m*dsub,k=512;
    std::vector<float> q(nq*dim),c(nlist*dim),cb(m*256*dsub),bias(n),coarse(nq*np);
    std::vector<uint8_t> codes(n*m);
    std::vector<idx_t> probes={0,1,2,3,2,1};
    for(int i=0;i<nq*dim;++i)q[i]=(i*7)%9;
    for(int i=0;i<nlist*dim;++i)c[i]=(i*3)%5;
    for(int i=0;i<m*256*dsub;++i)cb[i]=(i*11+i/256)%7-3;
    for(int r=0;r<n;++r)for(int j=0;j<m;++j)codes[r*m+j]=(r*37+j*13+r/256)%256;
    for(int r=0;r<n;++r){for(int j=0;j<m;++j)for(int d=0;d<dsub;++d){float v=cb[(j*256+codes[r*m+j])*dsub+d];bias[r]+=v*v+2*c[(r/perlist)*dim+j*dsub+d]*v;}}
    std::vector<std::vector<std::pair<float,int>>> oracle(nq);
    for(int qi=0;qi<nq;++qi)for(int p=0;p<np;++p){int cl=probes[qi*np+p];for(int d=0;d<dim;++d){float x=q[qi*dim+d]-c[cl*dim+d];coarse[qi*np+p]+=x*x;}
        for(int r=cl*perlist;r<(cl+1)*perlist;++r){float dist=0;for(int j=0;j<m;++j)for(int d=0;d<dsub;++d){float x=q[qi*dim+j*dsub+d]-c[cl*dim+j*dsub+d]-cb[(j*256+codes[r*m+j])*dsub+d];dist+=x*x;}oracle[qi].emplace_back(dist,r);}}
    Buffer<float> dq(q.size()),dc(c.size()),dcb(cb.size()),db(bias.size()),dd(coarse.size()),lut(nq*m*256);
    Buffer<uint8_t> code(codes.size());Buffer<idx_t> dp(probes.size());Buffer<long long> offsets(nlist);Buffer<int>counts(nlist);
    dq.put(q);dc.put(c);dcb.put(cb);db.put(bias);dd.put(coarse);code.put(codes);dp.put(probes);offsets.put({0,512,1024,1536});counts.put({512,512,512,512});
    pq_build_query_lut_gpu(dq.p,dcb.p,nq,m,dsub,lut.p,nullptr);
    PQQueryLutView view;view.query_codeword_dot_lut=lut.p;view.query_centroid_distances=dd.p;view.query_lut_bias={db.p,nullptr,0,0,n};
    PQAdcThresholdParams params;params.seed_probes=2;params.seed_rank=k;params.n_rerank=k;params.candidate_k=k;params.candidate_capacity=8192;params.policy=PQThresholdPolicy::DistanceHistogram;params.histogram_bins=512;
    setenv("PARASELECT_PQ_THRESHOLD_GLOBAL_POOL_CAPACITY","32768",1);
    PQAdcThresholdWorkspace ws;pq_adc_threshold_workspace_reserve(&ws,nq,8192,2);
    ResidualPQGroupedWorkspace grouped;
    ResidualPQGpuSearchSpec spec;spec.pq_m=m;spec.pq_k=256;spec.pq_dim_padded=dim;spec.pq_d_sub=dsub;spec.pq_d_sub_padded=dsub;spec.global_k=k;spec.threshold=params;
    for(int mode=0;mode<2;++mode){
        if(mode==0 && m==32) continue;
        // Reuse the expanded workspace at 512 bins after the upward sweep.
        for(int bins : {512,1024,2048,4096,8192,512}) {
        if(mode && bins!=512) continue;
        params.histogram_bins=bins;
        if(mode==0)pq_query_lut_threshold_search_gpu(view,dp.p,code.p,offsets.p,counts.p,nq,nlist,np,m,params,nullptr,nullptr,nullptr,&ws,nullptr);
        else pq_adc_threshold_search_gpu(PQAdcDistanceMode::Residual,dq.p,dc.p,dp.p,code.p,offsets.p,counts.p,dcb.p,nq,dim,nlist,np,spec,nullptr,nullptr,nullptr,&ws,&grouped,nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<int>ids(nq*params.candidate_capacity);std::vector<float>dist(nq*params.candidate_capacity);
        CUDA_CHECK(cudaMemcpy(ids.data(),ws.d_candidate_idx,ids.size()*sizeof(int),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dist.data(),ws.d_candidate_dist,dist.size()*sizeof(float),cudaMemcpyDeviceToHost));
        for(int qi=0;qi<nq;++qi) {
            std::sort(oracle[qi].begin(),oracle[qi].end());
            std::set<int> seen;
            for(int j=0;j<k;++j) {
                const int at=qi*params.candidate_capacity+j;
                const std::pair<float,int> candidate{dist[at],ids[at]};
                if(dist[at]!=oracle[qi][j].first || !seen.insert(ids[at]).second ||
                   !std::binary_search(oracle[qi].begin(),oracle[qi].end(),candidate))
                    throw std::runtime_error("histogram selection differs from reconstruction top512");
            }
        }
        }
    }
    pq_adc_threshold_workspace_release(&ws);
}
int main(){check(16,8);check(20,4);check(32,4);std::cout<<"Both ADC histogram paths match reconstruction oracle and top512\n";}
