#pragma once
#include <cuda_runtime.h>

namespace paraselect::search::detail {
// Called only by accepted lanes. The currently active subset reserves one
// contiguous span; overflow lanes return -1. No distance ordering is imposed.
__device__ __forceinline__ int reserve_bounded_warp(int* counter, int capacity) {
    const unsigned mask = __activemask();
    const int lane = threadIdx.x & 31;
    const int leader = __ffs(mask) - 1;
    const int rank = __popc(mask & ((1u << lane) - 1u));
    const int requested = __popc(mask);
    int base = 0, granted = 0;
    if (lane == leader) {
        base = atomicAdd(counter, 0);
        while (base < capacity) {
            granted = min(requested, capacity - base);
            const int prior = atomicCAS(counter, base, base + granted);
            if (prior == base) break;
            base = prior;
        }
        if (base >= capacity) granted = 0;
    }
    base = __shfl_sync(mask, base, leader);
    granted = __shfl_sync(mask, granted, leader);
    return rank < granted ? base + rank : -1;
}
}  // namespace paraselect::search::detail
