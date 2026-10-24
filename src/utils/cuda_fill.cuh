#ifndef PARASELECT_CUDA_FILL_CUH
#define PARASELECT_CUDA_FILL_CUH

#include "utils/pch.h"

namespace paraselect {
namespace cuda_utils {

void launch_fill_value(float* data, float value, int n, cudaStream_t stream = 0);
void launch_fill_value(int* data, int value, int n, cudaStream_t stream = 0);
void launch_fill_value(idx_t* data, idx_t value, int n, cudaStream_t stream = 0);

}  // namespace cuda_utils
}  // namespace paraselect

__global__ void fill_kernel(float* __restrict__ d_data, float value, int n);
__global__ void fill_int_kernel(int* __restrict__ d_data, int value, int n);

#endif  // PARASELECT_CUDA_FILL_CUH
