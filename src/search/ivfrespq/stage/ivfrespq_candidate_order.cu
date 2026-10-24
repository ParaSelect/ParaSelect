#include "search/ivfrespq/stage/ivfrespq_candidate_order.cuh"
#include "utils/check_cuda_error.cuh"

#include <climits>
#include <stdexcept>

namespace {

__device__ __forceinline__ int local_idx_sort_key(int idx) {
    return idx >= 0 ? idx : INT_MAX;
}

__global__ void order_candidates_by_local_idx_kernel(
    int* __restrict__ candidate_idx,
    float* __restrict__ candidate_dist,
    int batch_n,
    int candidate_k,
    int sort_n,
    int candidate_stride)
{
    const int q = blockIdx.x;
    if (q >= batch_n) return;

    extern __shared__ unsigned char smem[];
    int* s_idx = reinterpret_cast<int*>(smem);
    float* s_dist = reinterpret_cast<float*>(s_idx + sort_n);

    int* q_idx = candidate_idx + (size_t)q * candidate_stride;
    float* q_dist = candidate_dist ? candidate_dist + (size_t)q * candidate_stride : nullptr;

    for (int i = threadIdx.x; i < sort_n; i += blockDim.x) {
        if (i < candidate_k) {
            s_idx[i] = q_idx[i];
            s_dist[i] = q_dist ? q_dist[i] : 0.0f;
        } else {
            s_idx[i] = -1;
            s_dist[i] = 0.0f;
        }
    }
    __syncthreads();

    for (int k = 2; k <= sort_n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < sort_n; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const bool asc = ((i & k) == 0);
                    const int ki = local_idx_sort_key(s_idx[i]);
                    const int kj = local_idx_sort_key(s_idx[ixj]);
                    const bool do_swap = asc ? (ki > kj) : (ki < kj);
                    if (do_swap) {
                        const int ti = s_idx[i];
                        s_idx[i] = s_idx[ixj];
                        s_idx[ixj] = ti;
                        const float td = s_dist[i];
                        s_dist[i] = s_dist[ixj];
                        s_dist[ixj] = td;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = threadIdx.x; i < candidate_k; i += blockDim.x) {
        q_idx[i] = s_idx[i];
        if (q_dist) q_dist[i] = s_dist[i];
    }
}

int next_power_of_two(int x) {
    int n = 1;
    while (n < x) n <<= 1;
    return n;
}

}  // namespace

void ivfrespq_order_candidates_by_local_idx_gpu(
    int* d_candidate_idx,
    float* d_candidate_dist,
    int batch_n,
    int candidate_k,
    cudaStream_t stream,
    int candidate_stride)
{
    if (!d_candidate_idx || batch_n <= 0 || candidate_k <= 1) return;

    if (candidate_stride == 0) candidate_stride = candidate_k;
    if (candidate_stride < candidate_k) {
        throw std::invalid_argument("candidate ordering: stride is smaller than prefix width");
    }
    const int sort_n = next_power_of_two(candidate_k);
    if (sort_n > 2048) {
        throw std::invalid_argument(
            "ivfrespq_order_candidates_by_local_idx_gpu: candidate_k > 2048 is unsupported");
    }

    const int threads = sort_n < 256 ? 256 : (sort_n > 1024 ? 1024 : sort_n);
    const size_t shmem = (size_t)sort_n * (sizeof(int) + sizeof(float));
    order_candidates_by_local_idx_kernel<<<batch_n, threads, shmem, stream>>>(
        d_candidate_idx, d_candidate_dist, batch_n, candidate_k, sort_n, candidate_stride);
    CUDA_LAUNCH_CHECK("ivfrespq_order_candidates_by_local_idx_kernel");
}
