#include "search/kernels/coarse/coarse_prepare.h"
#include "search/ivf_common/ivf_coarse_ctx.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>
#include <stdexcept>
#include "utils/check_cuda_error.cuh"

// Exercise the shared executor independently of the FP16 L2 dispatcher.
void test_cosine_context() {
    constexpr int dim=128,nlist=257,nq=3,k=8;
    std::vector<float> centers(nlist*dim,0),queries(nq*dim,0);
    for(int c=0;c<nlist;++c){centers[c*dim]=1;centers[c*dim+1]=float(c)/128;}
    for(int q=0;q<nq;++q){queries[q*dim]=1;queries[q*dim+1]=float(q)/2;}
    for(bool fp16:{false,true})for(int tile:{0,96}) {
        paraselect::search::IVFCoarseConfig cfg;
        cfg.dim=dim;cfg.nlist=nlist;cfg.max_batch=nq;cfg.max_nprobe=k;
        cfg.distance=COSINE_DISTANCE;cfg.fp16_coarse=fp16;cfg.cluster_tile_size=tile;
        paraselect::search::IVFCoarseCtx coarse;coarse.init(cfg,centers.data());
        idx_t* ids;float* values;CUDA_CHECK(cudaMalloc(&ids,nq*k*sizeof(idx_t)));
        CUDA_CHECK(cudaMalloc(&values,nq*k*sizeof(float)));
        paraselect::search::IVFCoarseTiming timing;
        coarse.search_device_into(queries.data(),nq,k,ids,nullptr,&timing,values);
        std::vector<idx_t> actual_ids(nq*k);std::vector<float> actual(nq*k);
        CUDA_CHECK(cudaMemcpy(actual_ids.data(),ids,nq*k*sizeof(idx_t),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(actual.data(),values,nq*k*sizeof(float),cudaMemcpyDeviceToHost));
        for(int q=0;q<nq;++q) {
            std::vector<float> reference(nlist),selected(k);std::vector<bool> seen(nlist,false);
            const float x=queries[q*dim+1];
            for(int c=0;c<nlist;++c){const float y=centers[c*dim+1];reference[c]=1-(1+x*y)/(std::sqrt(1+x*x)*std::sqrt(1+y*y));}
            for(int j=0;j<k;++j){auto id=actual_ids[q*k+j];
                if(id<0 || id>=nlist || seen[id] || std::abs(actual[q*k+j]-reference[id])>2e-5f)
                    throw std::runtime_error("cosine output ID/distance mismatch");
                seen[id]=true;selected[j]=actual[q*k+j];}
            std::sort(reference.begin(),reference.end());std::sort(selected.begin(),selected.end());
            for(int j=0;j<k;++j)if(std::abs(selected[j]-reference[j])>2e-5f)
                throw std::runtime_error("cosine top-k mismatch");
        }
        if(!(timing.gemm_ms>0 && timing.topk_ms>0))throw std::runtime_error("cosine stage timing missing");
        CUDA_CHECK(cudaFree(ids));CUDA_CHECK(cudaFree(values));
    }
    std::cout<<"PASS cosine FP16/FP32 full and tiled selection with timing\n";
}

void test_centroid_updates() {
    constexpr int dim=128,nlist=256,nq=2,k=8;
    for(bool fp16:{false,true}) for(int tile:{0,96}) for(int policy:{
        int(IVF_COARSE_REDUCER_HISTOGRAM_SORT),int(IVF_COARSE_REDUCER_HISTOGRAM_PARTITION_SMALLK)}) {
        if(policy==IVF_COARSE_REDUCER_HISTOGRAM_PARTITION_SMALLK && !fp16) continue;
        std::vector<float> centers(nlist*dim,0),queries(nq*dim,0);
        for(int c=0;c<nlist;++c)centers[c*dim]=float(c*4);
        queries[0]=10;queries[dim]=50;
        paraselect::search::IVFCoarseConfig cfg;
        cfg.dim=dim;cfg.nlist=nlist;cfg.max_batch=nq;cfg.max_nprobe=k;
        cfg.fp16_coarse=fp16;
        cfg.l2_topk_policy=policy;cfg.cluster_tile_size=tile;
        paraselect::search::IVFCoarseCtx coarse;
        coarse.init(cfg,centers.data());
        idx_t* ids=nullptr;float* distances=nullptr;
        cudaStream_t consumer;
        CUDA_CHECK(cudaStreamCreateWithFlags(&consumer,cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&ids,nq*k*sizeof(idx_t)));
        CUDA_CHECK(cudaMalloc(&distances,nq*k*sizeof(float)));
        auto check=[&](const std::vector<float>& expected) {
            coarse.search_device_into(queries.data(),nq,k,ids,consumer,nullptr,distances);
            std::vector<idx_t> actual_ids(nq*k);
            std::vector<float> actual_distances(nq*k);
            CUDA_CHECK(cudaMemcpyAsync(actual_ids.data(),ids,nq*k*sizeof(idx_t),cudaMemcpyDeviceToHost,consumer));
            CUDA_CHECK(cudaMemcpyAsync(actual_distances.data(),distances,nq*k*sizeof(float),cudaMemcpyDeviceToHost,consumer));
            CUDA_CHECK(cudaStreamSynchronize(consumer));
            for(int q=0;q<nq;++q) {
                std::vector<float> reference(nlist),selected;
                for(int c=0;c<nlist;++c) {
                    const float delta=queries[q*dim]-expected[c*dim];
                    reference[c]=delta*delta;
                }
                for(int j=0;j<k;++j) {
                    const auto id=actual_ids[q*k+j];
                    if(id<0 || id>=nlist || actual_distances[q*k+j]!=reference[id])
                        throw std::runtime_error("coarse centroid update distance mismatch fp16="+std::to_string(fp16));
                    selected.push_back(actual_distances[q*k+j]);
                }
                std::sort(reference.begin(),reference.end());
                std::sort(selected.begin(),selected.end());
                if(!std::equal(selected.begin(),selected.end(),reference.begin()))
                    throw std::runtime_error("coarse centroid update top-k mismatch");
            }
        };
        check(centers);
        centers[0]=-64;centers[120*dim]=10;centers[121*dim]=-8;
        const auto expected=centers;
        centers[220*dim]=10; // Not in the update set; must remain unchanged on device.
        coarse.update_centroids(centers.data(),{121,0,120,120});
        check(expected);
        coarse.update_centroids(centers.data(),{});
        check(expected);
        CUDA_CHECK(cudaFree(ids));CUDA_CHECK(cudaFree(distances));
        CUDA_CHECK(cudaStreamDestroy(consumer));
    }
}

void test_mixed_topk_capacity() {
    using namespace paraselect;
    setenv("PARASELECT_COARSE_STRATEGY", "auto", 1);
    for (int batch : {1, 256}) {
        if (coarse::select_coarse_reducer(-1,batch,32768,4096,true,false) != IVF_COARSE_REDUCER_HISTOGRAM_SORT ||
            coarse::select_coarse_reducer(-1,batch,32768,2048,true,false) == IVF_COARSE_REDUCER_HISTOGRAM_SORT ||
            coarse::select_coarse_reducer(-1,batch,32768,4096,false,false) == IVF_COARSE_REDUCER_HISTOGRAM_SORT)
            throw std::runtime_error("prepared coarse policy does not respect workspace capability/k");
    }
    constexpr int dim=128,nlist=8192,max_batch=3,max_k=4096;
    std::vector<float> centers(nlist*dim,0),queries(max_batch*dim,0);
    for (int c=0; c<nlist; ++c) centers[c*dim]=float(c%2048)/16.f;
    for (int q=0; q<max_batch; ++q) queries[q*dim]=(q+1)*0.25f;
    for (bool fp16 : {false,true}) for (int tile : {0,1536}) {
        search::IVFCoarseConfig cfg;
        cfg.dim=dim;cfg.nlist=nlist;cfg.max_batch=max_batch;cfg.max_nprobe=max_k;
        cfg.fp16_coarse=fp16;cfg.cluster_tile_size=tile;cfg.select_l2_topk_per_request=true;
        search::IVFCoarseCtx ctx;
        setenv("PARASELECT_COARSE_STRATEGY","auto",1);
        ctx.init(cfg,centers.data());
        setenv("PARASELECT_COARSE_STRATEGY","retired-invalid-name",1);
        float* distances=nullptr;idx_t* ids=nullptr;
        CUDA_CHECK(cudaMalloc(&distances,size_t(max_batch)*max_k*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ids,size_t(max_batch)*max_k*sizeof(idx_t)));
        for (int k : {8,4096,17,64,4096}) for (int batch : {3,1,2}) {
            ctx.search_device_into(queries.data(),batch,k,ids,nullptr,nullptr,distances);
            std::vector<float> actual(size_t(batch)*k);
            std::vector<idx_t> indices(actual.size());
            CUDA_CHECK(cudaMemcpy(actual.data(),distances,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(indices.data(),ids,indices.size()*sizeof(idx_t),cudaMemcpyDeviceToHost));
            for (int q=0;q<batch;++q) {
                std::vector<float> reference(nlist);
                for(int c=0;c<nlist;++c) {
                    const float delta=queries[q*dim]-centers[c*dim];
                    reference[c]=delta*delta;
                }
                std::vector<bool> seen(nlist,false);
                for(int i=0;i<k;++i) {
                    const auto id=indices[size_t(q)*k+i];
                    if(id<0 || id>=nlist || seen[id] || actual[size_t(q)*k+i]!=reference[id])
                        throw std::runtime_error("mixed coarse k/batch output mismatch");
                    seen[id]=true;
                }
                std::sort(reference.begin(),reference.end());
                std::sort(actual.begin()+q*k,actual.begin()+(q+1)*k);
                if(!std::equal(actual.begin()+q*k,actual.begin()+(q+1)*k,reference.begin()))
                    throw std::runtime_error("mixed coarse top-k differs from CPU oracle");
            }
        }
        cudaFree(distances);cudaFree(ids);
    }
    unsetenv("PARASELECT_COARSE_STRATEGY");
    std::cout << "PASS mixed coarse k/batch/centroid tiles within prepared capacity\n";
}

void test_large_histogram_context() {
    constexpr int dim=128,nlist=131072,batch=4097,k=8;
    std::vector<float> centers(size_t(nlist)*dim,0),queries(size_t(batch)*dim,0);
    for(int c=0;c<nlist;++c)centers[size_t(c)*dim]=float(c%32);
    paraselect::search::IVFCoarseConfig cfg;
    cfg.dim=dim;cfg.nlist=nlist;cfg.max_batch=batch;cfg.max_nprobe=k;
    cfg.fp16_coarse=true;cfg.l2_topk_policy=IVF_COARSE_REDUCER_HISTOGRAM_SORT;
    cfg.select_l2_topk_per_request=false;
    size_t before=0,after=0,total=0;CUDA_CHECK(cudaMemGetInfo(&before,&total));
    paraselect::search::IVFCoarseCtx coarse;coarse.init(cfg,centers.data());
    CUDA_CHECK(cudaMemGetInfo(&after,&total));
    if(before-after>(2560ULL<<20))throw std::runtime_error("large histogram context reserves excessive memory");
    auto view=coarse.search_device(queries.data(),batch,k,nullptr,nullptr);
    std::vector<idx_t> ids(size_t(batch)*k);
    CUDA_CHECK(cudaMemcpy(ids.data(),view.indices,ids.size()*sizeof(idx_t),cudaMemcpyDeviceToHost));
    for(int q=0;q<batch;++q) {
        std::vector<idx_t> row(ids.begin()+size_t(q)*k,ids.begin()+size_t(q+1)*k);
        std::sort(row.begin(),row.end());
        if(std::adjacent_find(row.begin(),row.end())!=row.end())throw std::runtime_error("duplicate context top-k IDs");
        for(auto id:row)if(id<0 || id>=nlist || id%32)throw std::runtime_error("large context top-k mismatch");
    }
    std::cout<<"PASS production histogram bs=4097 nlist=131072 reserved="<<(before-after)<<" bytes\n";
}

// A context capacity must not determine the policy for a smaller request.
void test_batch_boundaries() {
    constexpr int dim=128,nlist=8192,max_batch=257,max_k=65;
    std::vector<float> centers(size_t(nlist)*dim,0),queries(size_t(max_batch)*dim,0);
    for(int c=0;c<nlist;++c) centers[size_t(c)*dim]=float(c%32);
    for(int q=0;q<max_batch;++q) queries[size_t(q)*dim]=float(q%32);
    for(bool fp16:{false,true}) {
        paraselect::search::IVFCoarseConfig cfg;
        cfg.dim=dim;cfg.nlist=nlist;cfg.max_batch=max_batch;cfg.max_nprobe=max_k;
        cfg.fp16_coarse=fp16;cfg.select_l2_topk_per_request=true;
        setenv("PARASELECT_COARSE_STRATEGY","auto",1);
        paraselect::search::IVFCoarseCtx ctx;ctx.init(cfg,centers.data());
        setenv("PARASELECT_COARSE_STRATEGY","invalid-after-prepare",1);
        for(int k:{8,64,65}) for(int batch:{1,255,256,257,7}) {
            std::vector<idx_t> ids(size_t(batch)*k);
            ctx.search(queries.data(),batch,k,ids.data());
            for(int q=0;q<batch;++q) {
                auto first=ids.begin()+size_t(q)*k,last=first+k;
                for(auto it=first;it!=last;++it)
                    if(*it<0 || *it>=nlist || *it%32!=q%32)
                        throw std::runtime_error("batch boundary selected non-nearest centroid");
                std::sort(first,last);
                if(std::adjacent_find(first,last)!=last)
                    throw std::runtime_error("batch boundary returned duplicate centroid");
            }
        }
    }
    unsetenv("PARASELECT_COARSE_STRATEGY");
    std::cout<<"PASS actual batch 1/255/256/257/7 and k 8/64/65 with frozen request\n";
}

void test_unified_context() {
    constexpr int dim=128,nlist=131072,max_batch=8,max_k=513;
    std::vector<float> centers(size_t(nlist)*dim,0),queries(size_t(max_batch)*dim,0);
    for(int c=0;c<nlist;++c) {
        centers[size_t(c)*dim]=float(c%256)/16;
        centers[size_t(c)*dim+1]=float(c/256)/16;
    }
    for(int q=0;q<max_batch;++q) {
        queries[size_t(q)*dim]=float(q)/8;
        queries[size_t(q)*dim+1]=float(q)/16;
    }
    paraselect::search::IVFCoarseConfig cfg;
    cfg.dim=dim;cfg.nlist=nlist;cfg.max_batch=max_batch;cfg.max_nprobe=max_k;
    cfg.fp16_coarse=true;cfg.select_l2_topk_per_request=true;
    setenv("PARASELECT_COARSE_STRATEGY","auto",1);
    paraselect::search::IVFCoarseCtx ctx;ctx.init(cfg,centers.data());
    setenv("PARASELECT_COARSE_STRATEGY","invalid-after-prepare",1);
    for(int pass=0;pass<2;++pass) {
        for(auto shape : {std::pair<int,int>{8,65},{1,513},{7,96},{3,8},{8,17},{8,65}}) {
            const auto [batch,k]=shape;
            std::vector<idx_t> ids(size_t(batch)*k);
            ctx.search(queries.data(),batch,k,ids.data());
            for(int q=0;q<batch;++q) {
                std::vector<float> reference(nlist),selected(k);
                std::vector<bool> seen(nlist,false);
                for(int c=0;c<nlist;++c) {
                    const float x=queries[size_t(q)*dim]-centers[size_t(c)*dim];
                    const float y=queries[size_t(q)*dim+1]-centers[size_t(c)*dim+1];
                    reference[c]=x*x+y*y;
                }
                for(int j=0;j<k;++j) {
                    const auto id=ids[size_t(q)*k+j];
                    if(id<0 || id>=nlist || seen[id])throw std::runtime_error("unified context invalid ID");
                    seen[id]=true;selected[j]=reference[id];
                }
                std::partial_sort(reference.begin(),reference.begin()+k,reference.end());
                std::sort(selected.begin(),selected.end());
                if(!std::equal(selected.begin(),selected.end(),reference.begin()))
                    throw std::runtime_error("unified context differs from exact oracle");
            }
        }
        centers[0]=12;centers[1]=12;centers[123*dim]=0;centers[123*dim+1]=0;
        ctx.update_centroids(centers.data(),{0,123});
    }
    unsetenv("PARASELECT_COARSE_STRATEGY");
    std::cout<<"PASS unified FP16 context actual batch/k transitions and centroid updates\n";
}

int main() try {
    constexpr int kDim = 4;
    constexpr int kNlist = 8;
    constexpr int kQueryCount = 3;
    constexpr int kProbeCount = 4;
    std::vector<float> centroids(kNlist * kDim);
    std::vector<float> queries(kQueryCount * kDim);
    for (int row = 0; row < kNlist; ++row) {
        for (int dim = 0; dim < kDim; ++dim) {
            centroids[row * kDim + dim] =
                static_cast<float>(row * 7 + dim * 3);
        }
    }
    for (int row = 0; row < kQueryCount; ++row) {
        for (int dim = 0; dim < kDim; ++dim) {
            queries[row * kDim + dim] =
                static_cast<float>(row * 11 + dim * 2 + 1);
        }
    }

    paraselect::search::IVFCoarseConfig config;
    config.dim = kDim;
    config.nlist = kNlist;
    config.max_batch = kQueryCount;
    config.max_nprobe = kProbeCount;
    config.distance = L2_DISTANCE;
    paraselect::search::IVFCoarseCtx coarse;
    coarse.init(config, centroids.data());

    std::vector<idx_t> host(kQueryCount * kProbeCount);
    coarse.search(
        queries.data(), kQueryCount, kProbeCount, host.data(), nullptr);

    cudaStream_t consumer = nullptr;
    if (cudaStreamCreate(&consumer) != cudaSuccess) return 1;
    const auto view = coarse.search_device(
        queries.data(), kQueryCount, kProbeCount, consumer, nullptr);
    std::vector<idx_t> device(host.size());
    if (cudaMemcpyAsync(
            device.data(), view.indices, device.size() * sizeof(idx_t),
            cudaMemcpyDeviceToHost, consumer) != cudaSuccess ||
        cudaStreamSynchronize(consumer) != cudaSuccess) {
        cudaStreamDestroy(consumer);
        return 1;
    }
    if (view.query_count != kQueryCount ||
        view.probe_count != kProbeCount || device != host) {
        std::cerr << "host and device coarse probe outputs differ\n";
        cudaStreamDestroy(consumer);
        return 1;
    }
    idx_t* destination = nullptr;
    if (cudaMalloc(&destination, device.size() * sizeof(idx_t)) != cudaSuccess) {
        cudaStreamDestroy(consumer);
        return 1;
    }
    const auto owned_view = coarse.search_device_into(
        queries.data(), kQueryCount, kProbeCount, destination, consumer,
        nullptr);
    std::fill(device.begin(), device.end(), -1);
    const bool copy_failed = cudaMemcpyAsync(
        device.data(), owned_view.indices, device.size() * sizeof(idx_t),
        cudaMemcpyDeviceToHost, consumer) != cudaSuccess;
    const bool sync_failed = cudaStreamSynchronize(consumer) != cudaSuccess;
    cudaFree(destination);
    cudaStreamDestroy(consumer);
    if (copy_failed || sync_failed || device != host) {
        std::cerr << "coarse device destination output differs\n";
        return 1;
    }
    test_cosine_context();
    test_centroid_updates();
    test_mixed_topk_capacity();
    test_large_histogram_context();
    test_batch_boundaries();
    test_unified_context();
    return 0;
} catch(const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
