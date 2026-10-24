#include "pq_threshold_status.cuh"
#include <cstdio>
#include <cuda_runtime.h>
__global__ void mixed(int* status) {
    if(threadIdx.x&1) paraselect::search::detail::record_threshold_status(status,3);
    else paraselect::search::detail::record_threshold_status(status,1);
}
int main() {
    int* d; if(cudaMalloc(&d,sizeof(int))!=cudaSuccess) return 2;
    int initial=1, result=0;
    cudaMemcpy(d,&initial,sizeof(int),cudaMemcpyHostToDevice);
    mixed<<<32,256>>>(d);
    if(cudaMemcpy(&result,d,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess) return 2;
    cudaFree(d);
    if(result!=3) {printf("FAIL: overflow masked invalid cluster: status=%d\n",result);return 1;}
    puts("PASS: invalid cluster survives earlier overflow and concurrent overflow reports");
}
