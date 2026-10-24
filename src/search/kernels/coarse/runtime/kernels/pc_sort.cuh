#pragma once
inline cudaError_t sort_admitted_pairs(bool segmented, void* temp, size_t& bytes,
    cub::DoubleBuffer<float>& keys, cub::DoubleBuffer<int>& ids, int items, int batch,
    const int* starts, const int* ends, cudaStream_t stream) {
    if(segmented)return cub::DeviceSegmentedSort::SortPairsDescending(
        temp,bytes,keys,ids,items,batch,starts,ends,stream);
    return cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp,bytes,keys,ids,items,batch,starts,ends,0,32,stream);
}

// Pack admitted records into the other sort
// buffer, preserving each query's fixed segment address and global IDs.
__global__ void compact_admitted_for_sort(const float* values, const int* ids,
    const int* counts, int parts, int capacity, int k, float* packed_values,
    int* packed_ids, int* ends) {
    const int q = blockIdx.x, stride = parts * capacity;
    const size_t base = size_t(q) * stride;
    int offset = 0;
    for (int p = 0; p < parts; ++p) {
        // Overflow rows are already marked for exact fallback. Do not read
        // beyond their bounded owner pool while preparing the sort.
        const int count = min(counts[size_t(q) * parts + p], capacity);
        for (int j = threadIdx.x; j < count; j += blockDim.x) {
            packed_values[base + offset + j] = values[base + p * capacity + j];
            packed_ids[base + offset + j] = ids[base + p * capacity + j];
        }
        offset += count;
    }
    // Underfill rows also fall back. Initialize their first k outputs so the
    // common gather never reads uninitialized data before fallback overwrites.
    for (int j = offset + threadIdx.x; j < k; j += blockDim.x) {
        packed_values[base + j] = -INFINITY;
        packed_ids[base + j] = -1;
    }
    if (threadIdx.x == 0) ends[q] = int(base) + max(offset, k);
}
