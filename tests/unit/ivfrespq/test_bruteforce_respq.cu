#include "search/bruteforce_respq/bruteforce_respq_search.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace paraselect::search;

// Independently reconstructible PQ vectors: check exhaustive coverage, rather
// than merely comparing two implementations of the same ADC arithmetic.
int main() {
    constexpr int nlist=128, rows_per_list=4, n=nlist*rows_per_list, m=16, k=256, nq=9, topk=10;
    for(int dim : {96,128}) {
        int sub=dim/m;
        std::vector<float> centers(nlist*dim), book(m*k*sub), base(n*dim), query(nq*dim);
        std::vector<uint8_t> codes(n*m);
        std::vector<idx_t> offsets(nlist+1), ids(n);
        for(int c=0;c<=nlist;++c) offsets[c]=c*rows_per_list;
        for(int c=0;c<nlist;++c) for(int d=0;d<dim;++d) centers[c*dim+d]=c*.25f;
        for(int j=0;j<m;++j) for(int v=0;v<k;++v) for(int d=0;d<sub;++d)
            book[(j*k+v)*sub+d]=v*.0625f;
        for(int row=0;row<n;++row) {
            ids[row]=n-row-1;
            for(int j=0;j<m;++j) codes[row*m+j]=row%4;
            for(int d=0;d<dim;++d) base[row*dim+d]=centers[(row/4)*dim+d]+(row%4)*.0625f;
        }
        for(int q=0;q<nq;++q) for(int d=0;d<dim;++d) query[q*dim+d]=q*3.1f+.017f;
        std::vector<idx_t> reference(nq*topk), actual(nq*topk);
        std::vector<float> distances(nq*topk);
        for(int q=0;q<nq;++q) {
            std::vector<std::pair<float,idx_t>> truth;
            for(int row=0;row<n;++row) {
                float d2=0;for(int d=0;d<dim;++d){float x=query[q*dim+d]-base[row*dim+d];d2+=x*x;}
                truth.emplace_back(d2,ids[row]);
            }
            std::sort(truth.begin(),truth.end());
            for(int j=0;j<topk;++j)reference[q*topk+j]=truth[j].second;
        }
        BruteForceResPQSearchRequest r;
        r.index.coarse={centers.data(),nlist,dim};
        r.index.lists={offsets.data(),ids.data(),nlist,n};
        r.index.storage.pq={codes.data(),nullptr,book.data(),m,k};
        r.index.storage.assets.exact=IVFExactStorageView::from_host({base.data(),nullptr,n,dim});
        r.queries={query.data(),nullptr,nq,dim};r.output={actual.data(),distances.data(),nullptr};
        r.topk=topk;r.n_rerank=32;r.seed_rows=32;
        std::vector<float> correction(n);
        std::vector<uint16_t> correction_u16(n);
        for(int row=0;row<n;++row) {
            float residual=(row%4)*.0625f,center=(row/4)*.25f;
            correction[row]=dim*(residual*residual+2*center*residual);
            correction_u16[row]=uint16_t(correction[row]*32);
        }
        for(int mode : {0,1,2}) {
        r.index.storage.assets.query_lut_bias = mode==0 ? QueryLutBias{} : mode==1
            ? QueryLutBias{correction.data(),nullptr,0,0,n}
            : QueryLutBias{nullptr,correction_u16.data(),1.0f/32,0,n};
        for(int batch : {1,8,10000}) for(int bins : {0,256,512}) {
            r.batch_size=batch;r.scan_batch=3;r.seed_histogram_bins=bins;r.verify_candidates=bins!=0;
            IVFSearchStats stats;bruteforce_respq_search(r,&stats);
            if(actual!=reference)throw std::runtime_error("exhaustive top10 differs from CPU oracle");
            if(stats.coarse_search_ms!=0 || stats.string_fields["seed_source"]!="rows")
                throw std::runtime_error("brute force unexpectedly depends on coarse selection");
            if(bins && stats.numeric_fields["seed_scanned_rows"]!=32)
                throw std::runtime_error("seed sample budget was not respected");
            if(bins && stats.numeric_fields["verified_queries"]!=nq)
                throw std::runtime_error("histogram ADC top-k verification incomplete");
        }
        }
    }
    std::cout<<"PASS no coarse, sampled rows, exhaustive top10, seed ADC verification, tail batches\n";
}
