#include "utils/cuda_fill.cuh"

namespace {

template <typename T>
__global__ void fill_value_kernel(T* __restrict__ data, T value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = value;
}

template <typename T>
void launch_fill_value_impl(T* data, T value, int n, cudaStream_t stream) {
    if (!data || n <= 0) return;
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    fill_value_kernel<T><<<grid, block, 0, stream>>>(data, value, n);
}

}  // namespace

__global__ void fill_kernel(float* __restrict__ d_data, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_data[idx] = value;
}

__global__ void fill_int_kernel(int* __restrict__ d_data, int value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_data[idx] = value;
}

namespace paraselect {
namespace cuda_utils {

void launch_fill_value(float* data, float value, int n, cudaStream_t stream) {
    launch_fill_value_impl<float>(data, value, n, stream);
}

void launch_fill_value(int* data, int value, int n, cudaStream_t stream) {
    launch_fill_value_impl<int>(data, value, n, stream);
}

void launch_fill_value(idx_t* data, idx_t value, int n, cudaStream_t stream) {
    launch_fill_value_impl<idx_t>(data, value, n, stream);
}

}  // namespace cuda_utils
}  // namespace paraselect
