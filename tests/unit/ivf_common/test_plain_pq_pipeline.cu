#include "search/ivfrespq/ivfrespq_search.cuh"
#include "utils/pch.h"
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iostream>
using namespace paraselect::search;

void check(int m, int dim, bool fp32, int rerank = 512, int sparse_seed = 0) {
    constexpr int nlist=4,nq=19,topk=10;
    const int rows=std::max(1024,rerank*2);
    std::vector<float> base(rows*dim), query(nq*dim), centers(nlist*dim,0), book(m*256*(dim/m));
    std::vector<uint8_t> base_u8(rows*dim),query_u8(nq*dim),codes(rows*m);
    std::vector<idx_t> ids(rows), offsets={0,rows/4,rows/2,rows*3/4,rows};
    if (sparse_seed == 2) offsets[1] = 0;
    for(int r=0;r<rows;++r){ids[r]=r;for(int j=0;j<m;++j){
        int c=(r*37+j*11+r/256)%256;codes[r*m+j]=c;
        for(int d=0;d<dim/m;++d)base[r*dim+j*(dim/m)+d]=base_u8[r*dim+j*(dim/m)+d]=c;
    }}
    for(int j=0;j<m;++j)for(int c=0;c<256;++c)for(int d=0;d<dim/m;++d)book[(j*256+c)*(dim/m)+d]=c;
    for(int q=0;q<nq;++q)for(int d=0;d<dim;++d)query[q*dim+d]=query_u8[q*dim+d]=base[(q*47%rows)*dim+d];
    std::vector<idx_t> expected(nq*topk),out(nq*topk);
    std::vector<float> expected_d(nq*topk),out_d(nq*topk);
    for(int q=0;q<nq;++q){std::vector<std::pair<float,int>> ranked;
        for(int r=0;r<rows;++r){float dist=0;for(int d=0;d<dim;++d){float x=query[q*dim+d]-base[r*dim+d];dist+=x*x;}ranked.emplace_back(dist,r);}
        std::sort(ranked.begin(),ranked.end());for(int k=0;k<topk;++k){expected[q*topk+k]=ranked[k].second;expected_d[q*topk+k]=ranked[k].first;}}
    IVFResPQSearchRequest req;
    req.index.coarse={centers.data(),nlist,dim};
    req.index.lists={offsets.data(),ids.data(),nlist,rows};
    req.index.storage.pq.codes=codes.data();req.index.storage.pq.codebook=book.data();
    req.index.storage.pq.subvectors=m;req.index.storage.pq.codebook_size=256;
    req.index.storage.assets.exact=IVFExactStorageView::from_host(
        {fp32 ? base.data() : nullptr, fp32 ? nullptr : base_u8.data(), rows, dim});
    req.queries={query.data(),query_u8.data(),nq,dim};req.params={nlist,topk,8,L2_DISTANCE,false};
    req.output={out.data(),out_d.data(),nullptr};
    req.options.pq_distance_mode=PQAdcDistanceMode::Plain;
    req.options.n_rerank=rerank;req.options.gpu.candidate_count=rerank;req.options.float_rerank=fp32;
    req.options.cpu.rerank_threads=3;req.options.cpu.rerank_prefetch=4;
    req.options.adc_strategy=PQAdcSearchStrategy::Threshold;
    req.options.adc_mode=ResidualPQAdcMode::QueryLut;
    req.options.query_lut_quantization=PQAdcLutQuantizationMode::U8Affine;
    req.options.adc_threshold.seed_probes=sparse_seed ? 1 : nlist;
    req.options.adc_threshold.seed_rank=rerank;
    req.options.adc_threshold.candidate_k=rerank;
    req.options.adc_threshold.candidate_capacity=rows;
    req.options.adc_threshold.lut_read_mode=PQAdcLutReadMode::Global;
    for(auto layout:{PQAdcThresholdScanLayout::PairCentric,PQAdcThresholdScanLayout::ListCentric})
      for(bool overlap:{false,true}){
        req.options.adc_threshold.scan_layout=layout;req.options.overlap_cpu_pipeline=overlap;
        std::cerr << "pipeline m=" << m << " exact=" << (fp32 ? "fp32" : "u8")
                  << " sparse_seed=" << sparse_seed
                  << " layout=" << int(layout) << " overlap=" << overlap << std::endl;
        IVFSearchStats stats;ivfrespq_search(req,&stats);
        if(out!=expected||out_d!=expected_d)throw std::runtime_error("plain pipeline differs from exhaustive exact topk");
        const auto mode=stats.string_fields.at("ivfrespq_pipeline");
        if(mode!=(overlap?"chunk_overlap":"serial"))throw std::runtime_error("requested pipeline was not executed");
        if(stats.string_fields.at("pq_quantizer")!="plain")throw std::runtime_error("missing quantizer metadata");
      }
}
int main() {
    // The seed prefix can contain fewer than K rows, or be empty, while
    // later probed lists contain the true nearest neighbours.
    check(16,96,true,512,1);
    check(16,96,true,512,2);
    check(20,100,false,2048,1);
    check(20,100,false,2048,2);
    check(16,128,false);
    check(20,100,true);
    check(16,128,false,2048);
    check(20,100,true,2048);
    check(20,100,false,2048);
    check(16,96,true,2048);
    std::cout << "PASS: plain M16/M20 pair/list serial/overlap exact results and tail batches\n";
}
