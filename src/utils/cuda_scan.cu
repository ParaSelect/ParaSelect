#include "utils/cuda_scan.cuh"

namespace {

template <typename CountT>
__global__ void inclusive_scan_typed_kernel(
    const CountT* __restrict__ input,
    int* __restrict__ output,
    int* __restrict__ block_sums,
    int n)
{
    extern __shared__ int s_data[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    s_data[tid] = (idx < n) ? static_cast<int>(input[idx]) : 0;
    __syncthreads();

    int new_val = 0;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        new_val = s_data[tid];
        if (tid >= stride) new_val += s_data[tid - stride];
        __syncthreads();
        s_data[tid] = new_val;
    }
    __syncthreads();

    if (tid == blockDim.x - 1 && block_sums) {
        block_sums[bid] = s_data[tid];
    }
    if (idx < n) {
        output[idx] = s_data[tid];
    }
}

__global__ void merge_scan_blocks_kernel(
    const int* __restrict__ block_sums,
    int* __restrict__ output,
    int n,
    int block_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int block_id = idx / block_size;
    int prefix_sum = 0;
    for (int i = 0; i < block_id; ++i) {
        prefix_sum += block_sums[i];
    }
    output[idx] += prefix_sum;
}

template <typename CountT>
void compute_prefix_sum_typed(
    const CountT* d_count,
    int* d_offset,
    int n,
    cudaStream_t stream)
{
    cudaMemsetAsync(d_offset, 0, sizeof(int), stream);
    if (n <= 0) return;

    const int block_size = 256;
    const int n_blocks = (n + block_size - 1) / block_size;
    const int shared_mem_size = block_size * sizeof(int);

    if (n_blocks == 1) {
        inclusive_scan_typed_kernel<<<1, block_size, shared_mem_size, stream>>>(
            d_count, d_offset + 1, nullptr, n);
    } else {
        int* d_block_sums = nullptr;
        cudaMalloc(&d_block_sums, (size_t)n_blocks * sizeof(int));
        inclusive_scan_typed_kernel<<<n_blocks, block_size, shared_mem_size, stream>>>(
            d_count, d_offset + 1, d_block_sums, n);
        merge_scan_blocks_kernel<<<n_blocks, block_size, 0, stream>>>(
            d_block_sums, d_offset + 1, n, block_size);
        cudaFree(d_block_sums);
    }
}

}  // namespace

__global__ void inclusive_scan_block_kernel(
    const int* __restrict__ d_input,
    int* __restrict__ d_output,
    int* __restrict__ d_block_sums,
    int n)
{
    extern __shared__ int s_data[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    s_data[tid] = (idx < n) ? d_input[idx] : 0;
    __syncthreads();

    int new_val = 0;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        new_val = s_data[tid];
        if (tid >= stride) new_val += s_data[tid - stride];
        __syncthreads();
        s_data[tid] = new_val;
    }
    __syncthreads();

    if (tid == blockDim.x - 1 && d_block_sums) d_block_sums[bid] = s_data[tid];
    if (idx < n) d_output[idx] = s_data[tid];
}

__global__ void inclusive_scan_add_block_sums_kernel(
    int* __restrict__ d_output,
    const int* __restrict__ d_block_prefix_sums,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int block_id = blockIdx.x;
    if (block_id > 0) d_output[idx] += d_block_prefix_sums[block_id - 1];
}

__global__ void inclusive_scan_kernel(
    const int* __restrict__ d_input,
    int* __restrict__ d_output,
    int n)
{
    extern __shared__ int s_data[];
    int tid = threadIdx.x;
    int idx = threadIdx.x;

    s_data[tid] = (idx < n) ? d_input[idx] : 0;
    __syncthreads();

    int new_val = 0;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        new_val = s_data[tid];
        if (tid >= stride) new_val += s_data[tid - stride];
        __syncthreads();
        s_data[tid] = new_val;
    }
    __syncthreads();

    if (idx < n) d_output[idx] = s_data[tid];
}

void compute_prefix_sum(const int* d_count, int* d_offset, int n, cudaStream_t stream) {
    compute_prefix_sum_typed<int>(d_count, d_offset, n, stream);
}

void compute_prefix_sum(const idx_t* d_count, int* d_offset, int n, cudaStream_t stream) {
    compute_prefix_sum_typed<idx_t>(d_count, d_offset, n, stream);
}

void compute_prefix_sum_scratch(const int* d_count, int* d_offset, int n,
                                int* d_block_sums, cudaStream_t stream) {
    cudaMemsetAsync(d_offset, 0, sizeof(int), stream);
    if (n <= 0) return;

    const int block_size = 256;
    const int n_blocks = (n + block_size - 1) / block_size;
    const int shared_mem_size = block_size * sizeof(int);

    if (n_blocks == 1) {
        inclusive_scan_typed_kernel<<<1, block_size, shared_mem_size, stream>>>(
            d_count, d_offset + 1, nullptr, n);
        return;
    }
    if (!d_block_sums) {
        // No scratch supplied: keep the synchronizing fallback so callers that
        // did not preallocate still work.
        compute_prefix_sum_typed<int>(d_count, d_offset, n, stream);
        return;
    }
    inclusive_scan_typed_kernel<<<n_blocks, block_size, shared_mem_size, stream>>>(
        d_count, d_offset + 1, d_block_sums, n);
    merge_scan_blocks_kernel<<<n_blocks, block_size, 0, stream>>>(
        d_block_sums, d_offset + 1, n, block_size);
}
