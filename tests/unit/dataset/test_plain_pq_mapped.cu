#include "clustering/kmeans/kmeans_cache_recipe.h"
#include "index/ivf_cache.h"
#include "search/ivfpq/ivfpq_index.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <unistd.h>
using namespace paraselect;
int main(){
    char pattern[]="/tmp/paraselect-plain-mapped-XXXXXX";
    const char* temp=mkdtemp(pattern);if(!temp)return 1;const std::filesystem::path root(temp);
    try {
        constexpr int n=512,dim=100,m=20,k=256;
        std::vector<uint8_t> base(n*dim);
        for(int r=0;r<n;++r)for(int d=0;d<dim;++d)base[r*dim+d]=(r*37+d*11+r/7)%251;
        const auto path=root/"base.u8bin";
        {std::ofstream out(path,std::ios::binary);const uint32_t shape[]={n,dim};out.write((const char*)shape,8);out.write((const char*)base.data(),base.size());}
        setenv("PARASELECT_SPACEV_BASE",path.c_str(),1);setenv("PARASELECT_SPACEV_QUERY",path.c_str(),1);setenv("PARASELECT_SPACEV_GT_1M",path.c_str(),1);
        const auto cache_path=root/"cache";setenv("IVF_CLUSTER_CACHE_DIR",cache_path.c_str(),1);
        auto spec=dataset::describe_dataset("spacev","1M");spec.n_vectors=n;spec.dim=dim;dataset::DatasetSource source(spec);
        KMeansSpec km;dataset::ClusterCache cluster_cache(cache_path.string());
        std::vector<float> centers(2*dim,0);std::vector<idx_t> perm(n);for(int i=0;i<n;++i)perm[i]=n-1-i;
        const idx_t offsets[]={0,256,512};const int counts[]={256,256};
        cluster_cache.save(dataset::dataset_key_name(spec),{n,2,dim,make_kmeans_cache_recipe(km)},centers.data(),perm.data(),counts,offsets,2);
        ivf_cache::PlainPQParams pq;pq.m=m;pq.k=k;pq.iters=2;pq.train_n=n;pq.encode_chunk=137;
        setenv("PARASELECT_CACHE_POLICY","build",1);
        auto mapped=ivf_cache::ensure_pq_mapped(source,n,2,km,pq);
        if(!mapped.pq.hit||mapped.pq.request.pq_type!="plain"||mapped.pq.request.cluster_layout_id!=mapped.clusters.layout_id)
            throw std::runtime_error("wrong mapped asset identity");
        // Independent nearest-centroid check in original (unpadded) subspaces.
        for(int r=0;r<n;++r)for(int j=0;j<m;++j){
            float best=INFINITY,actual=0;
            for(int c=0;c<k;++c){float distance=0;for(int d=0;d<dim/m;++d){float x=base[perm[r]*dim+j*(dim/m)+d]-mapped.pq.codebook[(j*k+c)*(dim/m)+d];distance+=x*x;}best=std::min(best,distance);if(c==mapped.pq.codes[r*m+j])actual=distance;}
            if(actual>best+0.01f)throw std::runtime_error("mapped code violates original-space encoding or reorder");
        }
        setenv("PARASELECT_CACHE_POLICY","require",1);
        auto hit=ivf_cache::ensure_pq_mapped(source,n,2,km,pq);
        auto owning=ivf_cache::ensure_pq(source,n,2,km,pq);
        if(hit.pq.entry_dir!=mapped.pq.entry_dir||owning.pq.entry_dir!=mapped.pq.entry_dir||
            !std::equal(owning.pq.codes.begin(),owning.pq.codes.end(),mapped.pq.codes.data()))throw std::runtime_error("mapped/owning cache mismatch");
        // The public plain index must reach the same optimized pipeline.
        paraselect::search::IVFPQBuildConfig config(source);
         config.n_vectors = n; config.nlist = 2;
        config.kmeans = km; config.pq_m = m; config.pq_k = k;
        config.pq_iters = 2; config.pq_train_n = n;
        config.fine_storage = paraselect::search::IVFPQFineStorage::Resident;
        config.exact_rerank = paraselect::search::IVFPQExactRerank::Flat;
        config.n_rerank = n; config.cpu_rerank_threads = 2;
        config.adc_strategy = paraselect::search::PQAdcSearchStrategy::Threshold;
        config.threshold.seed_probes = 2; config.threshold.seed_rank = n;
        config.threshold.candidate_capacity = n; config.threshold.candidate_k = n;
        config.threshold.lut_read_mode = paraselect::search::PQAdcLutReadMode::Global;
        config.threshold.scan_layout = paraselect::search::PQAdcThresholdScanLayout::ListCentric;
        auto index = paraselect::search::IVFPQIndex::build(config);
        constexpr int nq = 3, topk = 10;
        std::vector<float> queries(nq*dim);for(int i=0;i<nq*dim;++i)queries[i]=base[i];
        std::vector<idx_t> result(nq*topk);std::vector<float> distances(nq*topk);
        paraselect::search::IVFSearchStats stats;
        index.search({queries.data(),base.data(),nq,dim},{2,topk,2,L2_DISTANCE,false},
                     {result.data(),distances.data(),nullptr},&stats);
        if(stats.string_fields.at("ivfrespq_pipeline")!="chunk_overlap" || stats.string_fields.at("pq_quantizer")!="plain")
            throw std::runtime_error("public plain index bypassed shared overlap pipeline");
        for(int q=0;q<nq;++q){std::vector<float> sorted;
            for(int r=0;r<n;++r){float d=0;for(int j=0;j<dim;++j){float x=queries[q*dim+j]-base[r*dim+j];d+=x*x;}sorted.push_back(d);}
            std::sort(sorted.begin(),sorted.end());for(int j=0;j<topk;++j)
                if(distances[q*topk+j]!=sorted[j])throw std::runtime_error("public plain search differs from exhaustive rerank");}
        std::filesystem::remove_all(root);std::cout<<"PASS: M20/dsub5 training, reordered mapped plain codes, require-hit and owning parity\n";
    }catch(const std::exception& e){std::filesystem::remove_all(root);std::cerr<<e.what()<<'\n';return 1;}
}
