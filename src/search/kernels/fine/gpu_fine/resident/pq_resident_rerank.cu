#include "search/kernels/fine/gpu_fine/resident/pq_resident_rerank.cuh"
#include "layout/ivf_interleaved_layout.cuh"
#include "utils/check_cuda_error.cuh"

#include <cfloat>
#include <stdexcept>

namespace {

template <typename VectorT>
__global__ void pq_resident_rerank_full_kernel(
    int k,
    int dim,
    const float* __restrict__ d_queries,
    const int* __restrict__ d_candidate_idx,
    int candidate_count,
    int candidate_row_stride,
    const VectorT* __restrict__ d_interleaved_vectors,
    const int* __restrict__ d_compact_to_padded,
    int n_vectors,
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_local_idx)
{
    extern __shared__ unsigned char smem[];
    float* s_dist = reinterpret_cast<float*>(smem);
    int* s_idx = reinterpret_cast<int*>(s_dist + candidate_count);

    const int q = blockIdx.x;
    const float* query = d_queries + (size_t)q * dim;
    constexpr int kGroup = paraselect::kInterleavedGroupSize;
    const size_t words_per_group = (size_t)kGroup * (size_t)dim;

    for (int pos = threadIdx.x; pos < candidate_count; pos += blockDim.x) {
        const int local = d_candidate_idx[(size_t)q * candidate_row_stride + pos];
        float dist = FLT_MAX;
        int out_idx = -1;
        if (local >= 0 && local < n_vectors) {
            const int padded = d_compact_to_padded[local];
            if (padded >= 0) {
                const int group = padded / kGroup;
                const int lane = padded - group * kGroup;
                const VectorT* vec_group =
                    d_interleaved_vectors + (size_t)group * words_per_group;
                float sum = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    const float diff = query[d] -
                        static_cast<float>(vec_group[(size_t)d * kGroup + lane]);
                    sum += diff * diff;
                }
                dist = sum;
                out_idx = local;
            }
        }
        s_dist[pos] = dist;
        s_idx[pos] = out_idx;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float* out_d = d_topk_dist + (size_t)q * k;
        int* out_i = d_topk_local_idx + (size_t)q * k;
        for (int out = 0; out < k; ++out) {
            float best_d = FLT_MAX;
            int best_pos = -1;
            int best_i = -1;
            for (int pos = 0; pos < candidate_count; ++pos) {
                const float d = s_dist[pos];
                const int idx = s_idx[pos];
                if (idx >= 0 && (d < best_d || (d == best_d && idx < best_i))) {
                    best_d = d;
                    best_i = idx;
                    best_pos = pos;
                }
            }
            out_d[out] = best_d;
            out_i[out] = best_i;
            if (best_pos >= 0) {
                s_dist[best_pos] = FLT_MAX;
                s_idx[best_pos] = -1;
            }
        }
    }
}

}  // namespace

void launch_pq_resident_rerank_full(
    int k,
    int dim,
    const float* d_queries,
    const int* d_candidate_idx,
    int candidate_count,
    int candidate_row_stride,
    const float* d_interleaved_vectors,
    const uint8_t* d_interleaved_vectors_u8,
    const int* d_compact_to_padded,
    int n_vectors,
    int batch_n,
    float* d_topk_dist,
    int* d_topk_local_idx,
    cudaStream_t stream)
{
    if (batch_n <= 0) return;
    if (k <= 0 || candidate_count <= 0 || dim <= 0 ||
        candidate_row_stride < candidate_count) {
        throw std::invalid_argument("pq resident rerank: invalid shape");
    }
    if (k > candidate_count) {
        throw std::invalid_argument("pq resident rerank: topk exceeds candidate count");
    }
    if (!d_queries || !d_candidate_idx ||
        ((d_interleaved_vectors != nullptr) == (d_interleaved_vectors_u8 != nullptr)) ||
        !d_compact_to_padded ||
        !d_topk_dist || !d_topk_local_idx) {
        throw std::invalid_argument("pq resident rerank: null input");
    }
    const int threads = 256;
    const size_t shmem = (size_t)candidate_count * (sizeof(float) + sizeof(int));
    if (d_interleaved_vectors_u8) {
        pq_resident_rerank_full_kernel<uint8_t><<<batch_n, threads, shmem, stream>>>(
            k, dim, d_queries, d_candidate_idx, candidate_count, candidate_row_stride,
            d_interleaved_vectors_u8, d_compact_to_padded, n_vectors,
            d_topk_dist, d_topk_local_idx);
    } else {
        pq_resident_rerank_full_kernel<float><<<batch_n, threads, shmem, stream>>>(
            k, dim, d_queries, d_candidate_idx, candidate_count, candidate_row_stride,
            d_interleaved_vectors, d_compact_to_padded, n_vectors,
            d_topk_dist, d_topk_local_idx);
    }
    CUDA_LAUNCH_CHECK("pq_resident_rerank_full_kernel");
}
