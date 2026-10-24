#include "pq_bounded_reservation.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s\n",cudaGetErrorString(e)); exit(1); } } while(0)
__global__ void check(int n,int cap,int stride,int* count,int* out) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) {
        if(i%stride==0) {
            int slot=paraselect::search::detail::reserve_bounded_warp(count,cap);
            if(slot>=0) out[slot]=i;
        }
    }
}
int main() {
    int *count,*out; CK(cudaMalloc(&count,sizeof(int))); CK(cudaMalloc(&out,2050*sizeof(int)));
    int cases=0;
    for(int n: {0,1,17,31,32,33,255,257,65539})
    for(int cap: {0,1,17,31,32,33,512,2048})
    for(int stride: {1,2,7,33}) {
        CK(cudaMemset(count,0,sizeof(int))); CK(cudaMemset(out,255,2050*sizeof(int)));
        check<<<64,256>>>(n,cap,stride,count,out+1); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        int got; std::vector<int> values(2050); CK(cudaMemcpy(&got,count,sizeof(int),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(values.data(),out,2050*sizeof(int),cudaMemcpyDeviceToHost));
        if(got!=std::min(cap,(n+stride-1)/stride)||values[0]!=-1||values[cap+1]!=-1) return 2;
        std::vector<int> seen(n,0);
        for(int j=1;j<=got;++j) {int v=values[j]; if(v<0||v>=n||v%stride||seen[v]++) return 3;}
        ++cases;
    }
    CK(cudaFree(count)); CK(cudaFree(out)); printf("PASS %d cases: count, bounds, uniqueness, accepted-only\n",cases);
}
