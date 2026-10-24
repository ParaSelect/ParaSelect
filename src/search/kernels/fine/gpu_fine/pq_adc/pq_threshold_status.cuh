#pragma once
#include <cuda_runtime.h>
namespace paraselect::search::detail {
__device__ __forceinline__ void record_threshold_status(int* status,int value) {
    // 0=success, 1=hard truncation, >=2=fatal input/seed errors.
    atomicMax(status,value);
}
}
