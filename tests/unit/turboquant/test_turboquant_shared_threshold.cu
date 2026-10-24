#include "quantization/turboquant/residual_encoding.cuh"
#include "quantization/turboquant/turboquant_distance.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace paraselect::search;
static void ck(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
__global__ void score(SignResidualDeviceView view, float* out) {
    __shared__ float residual[129];
    turboquant::prepare_residual(view, blockIdx.x, blockIdx.x, 128, residual);
    for(int i=threadIdx.x;i<600;i+=blockDim.x)
        out[blockIdx.x*600+i]=turboquant::distance(view,residual,blockIdx.x*600+i);
}
static unsigned mix(unsigned v) {v^=v>>16;v*=0x7feb352du;v^=v>>15;v*=0x846ca68bu;return v^(v>>16);}
static std::vector<double> rotate(std::vector<double> v) {
    for(int i=0;i<128;++i) if(mix(17u^i)&1u) v[i]=-v[i];
    for(int w=1;w<128;w*=2) for(int b=0;b<128;b+=2*w) for(int d=0;d<w;++d) {
        double a=v[b+d],c=v[b+d+w];v[b+d]=a+c;v[b+d+w]=a-c;
    }
    for(auto& x:v)x/=std::sqrt(128.);return v;
}
int main() {
    std::vector<unsigned char> base(1200*128);
    std::vector<float> centers(2*128),queries(2*128);
    for(size_t i=0;i<base.size();++i)base[i]=mix(i+91)%256;
    for(int i=0;i<256;++i){centers[i]=32+(mix(i+3)%160);queries[i]=mix(i+900)%256;}
    long long offsets[]={0,600,1200};
    auto encoding = make_residual_codec(ResidualCodecKind::Sign1Bit,
        {base.data(),centers.data(),offsets,1200,2,128,2});
    std::vector<std::uint32_t> cached_codes(1200 * 4);
    bool rejected_partial_cache = false;
    try {
        (void)make_residual_codec(
            ResidualCodecKind::Sign1Bit,
            {base.data(), centers.data(), offsets, 1200, 2, 128, 2,
             cached_codes.data(), nullptr});
    } catch (const std::invalid_argument&) {
        rejected_partial_cache = true;
    }
    if (!rejected_partial_cache)
        throw std::runtime_error("partial TurboQuant cache was accepted");
    if (make_residual_codec(ResidualCodecKind::PQ, {}))
        throw std::runtime_error("PQ must retain its existing workspace");
    float *dq=nullptr,*dd=nullptr;ck(cudaMalloc(&dq,256*sizeof(float)));ck(cudaMalloc(&dd,1200*sizeof(float)));
    ck(cudaMemcpy(dq,queries.data(),256*sizeof(float),cudaMemcpyHostToDevice));
    score<<<2,256>>>(encoding->prepare_queries(dq,2,nullptr),dd);ck(cudaGetLastError());
    std::vector<float> got(1200);ck(cudaMemcpy(got.data(),dd,1200*sizeof(float),cudaMemcpyDeviceToHost));
    double worst=0;
    for(int row=0;row<1200;++row){int c=row/600;std::vector<double>x(128),q(128);
        for(int d=0;d<128;++d){x[d]=double(base[row*128+d])-centers[c*128+d];q[d]=double(queries[c*128+d])-centers[c*128+d];}
        x=rotate(x);q=rotate(q);double a=0;for(double z:x)a+=std::abs(z);a/=128;
        double expected=0;for(int d=0;d<128;++d){double delta=q[d]-(x[d]>=0?a:-a);expected+=delta*delta;}
        double err=std::abs(got[row]-expected)/std::max(1.,expected);worst=std::max(worst,err);
        if(err>2e-5)throw std::runtime_error("asymmetric score differs from CPU decoded L2");
    }
    ck(cudaFree(dq));ck(cudaFree(dd));printf("FP32 query / CPU decoded L2: PASS max_relative_error=%g\n",worst);
}
