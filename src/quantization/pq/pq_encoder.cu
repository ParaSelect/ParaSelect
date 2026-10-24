#include "quantization/pq/pq_encoder.cuh"
#include "quantization/pq/pq_encode.cuh"
#include "utils/check_cuda_error.cuh"
#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>
#include <vector>

namespace paraselect::quantization::pq {
PQEncoderGPU::PQEncoderGPU(const float* book,int dim,int m,int k,size_t max_rows)
    :dim_(dim),m_(m),k_(k),subdim_(m>0?dim/m:0),max_rows_(max_rows),
     warp_(k==256 && subdim_==8) {
    if(!book || dim<=0 || m<=0 || dim%m || k<=0 || k>256 ||
       max_rows==0 || max_rows>INT_MAX ||
       max_rows>std::numeric_limits<size_t>::max()/size_t(dim)/sizeof(float))
        throw std::invalid_argument("invalid PQ encoder shape or capacity");
    const size_t book_values=size_t(dim)*k;
    std::vector<float> transposed;
    if(warp_) {
        transposed.resize(book_values);
        for(int i=0;i<m;++i)for(int c=0;c<k;++c)for(int d=0;d<subdim_;++d)
            transposed[(size_t(i)*subdim_+d)*k+c]=book[(size_t(i)*k+c)*subdim_+d];
        book=transposed.data();
    }
    try {
        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUDA_CHECK(cudaMalloc(&data_,max_rows*dim*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&codebook_,book_values*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&codes_,max_rows*m));
        CUDA_CHECK(cudaMemcpyAsync(codebook_,book,book_values*sizeof(float),cudaMemcpyHostToDevice,stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    } catch(...) {release();throw;}
}
void PQEncoderGPU::release() noexcept {
    if(stream_)cudaStreamSynchronize(stream_);
    cudaFree(data_);cudaFree(codebook_);cudaFree(codes_);
    if(stream_)cudaStreamDestroy(stream_);
    data_=nullptr;codebook_=nullptr;codes_=nullptr;stream_=nullptr;
}
PQEncoderGPU::~PQEncoderGPU(){release();}
void PQEncoderGPU::encode(const float* data,int rows,uint8_t* codes) {
    if(!data || !codes || rows<=0 || size_t(rows)>max_rows_)
        throw std::invalid_argument("PQ encode exceeds prepared capacity or has invalid pointers");
    CUDA_CHECK(cudaMemcpyAsync(data_,data,size_t(rows)*dim_*sizeof(float),cudaMemcpyHostToDevice,stream_));
    const size_t work=size_t(rows)*m_;
    if(warp_) {
        const int blocks=int(std::min<size_t>((work+7)/8,65535));
        encode_warp_kernel<8><<<blocks,256,0,stream_>>>(data_,rows,m_,codebook_,codes_);
    } else {
        const int blocks=int(std::min<size_t>((work+255)/256,65535));
        encode_serial_kernel<<<blocks,256,0,stream_>>>(data_,rows,dim_,m_,k_,subdim_,codebook_,codes_);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(codes,codes_,work,cudaMemcpyDeviceToHost,stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}
}
