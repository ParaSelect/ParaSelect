#pragma once
#include "utils/check_cuda_error.cuh"
namespace paraselect::coarse::detail {
class StageTimer {
    cudaStream_t stream_;
    cudaEvent_t start_=nullptr,stop_=nullptr;
public:
    StageTimer(cudaStream_t stream,bool enabled):stream_(stream) {
        if(!enabled)return;
        CUDA_CHECK(cudaEventCreate(&start_));
        try { CUDA_CHECK(cudaEventCreate(&stop_)); CUDA_CHECK(cudaEventRecord(start_,stream_)); }
        catch(...) { cudaEventDestroy(start_);if(stop_)cudaEventDestroy(stop_);throw; }
    }
    ~StageTimer(){if(start_)cudaEventDestroy(start_);if(stop_)cudaEventDestroy(stop_);}
    StageTimer(const StageTimer&)=delete;
    StageTimer& operator=(const StageTimer&)=delete;
    void add(double* destination) {
        if(!destination)return;
        CUDA_CHECK(cudaEventRecord(stop_,stream_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,start_,stop_));*destination+=ms;
        CUDA_CHECK(cudaEventRecord(start_,stream_));
    }
};
}
