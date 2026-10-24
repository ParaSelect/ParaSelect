#include "search/kernels/fine/gpu_fine/resident/resident_chunked_l2_topk.cuh"
#include "layout/ivf_interleaved_layout.cuh"
#include "search/kernels/topk/warpsort/warpsort_utils.cuh"
#include "search/kernels/topk/warpsort/warpsort.cuh"

#include <cfloat>
#include <stdexcept>

namespace {

#ifndef kWarpSize
#define kWarpSize 32
#endif

__device__ __forceinline__ int resident_u8_lane_id() {
    return threadIdx.x & 31;
}

template<int Capacity, bool Ascending, int WarpsPerBlock>
__global__ __launch_bounds__(256, 1)
void resident_u8_l2_topk_chunk_scan_kernel(
    const uint8_t* __restrict__ d_query_group,
    const uint8_t* __restrict__ d_cluster_vector,
    const int* __restrict__ d_probe_vector_offset,
    const int* __restrict__ d_probe_vector_offset_compact,
    const int* __restrict__ d_probe_vector_count,
    const int* __restrict__ d_task_row,
    const int* __restrict__ d_task_cluster,
    const int* __restrict__ d_task_first_block,
    const int* __restrict__ d_task_num_blocks,
    float* __restrict__ d_query_norm,
    const float* __restrict__ d_cluster_vector_norm,
    int n_tasks,
    int n_probes,
    int n_dim,
    int k,
    float* __restrict__ d_candidate_dist,
    idx_t* __restrict__ d_candidate_index)
{
    const int task = blockIdx.x;
    if (task >= n_tasks) return;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = resident_u8_lane_id();
    if (warp_id >= WarpsPerBlock) return;

    const int row = d_task_row[task];
    const int query_id = row / n_probes;
    const int cid = d_task_cluster[task];
    const int vector_offset = d_probe_vector_offset[cid];
    const int compact_offset = d_probe_vector_offset_compact[cid];
    const int vector_count = d_probe_vector_count[cid];
    const int first_block = d_task_first_block[task];
    const int num_blocks = d_task_num_blocks[task];

    const uint8_t* query = d_query_group + (size_t)query_id * n_dim;
    const float query_norm = d_query_norm[query_id];
    constexpr int kGroup = paraselect::kInterleavedGroupSize;
    const size_t words_per_block = (size_t)kGroup * n_dim;
    const uint8_t* cluster_base = d_cluster_vector + (size_t)(vector_offset / kGroup) * words_per_block;

    using WarpSortBase = paraselect::warpsort::WarpSort<Capacity, Ascending, float, idx_t>;
    const float dummy_val = WarpSortBase::kDummy();
    paraselect::warpsort::WarpSortFiltered<Capacity, Ascending, float, idx_t> queue(k);

    for (int chunk_block = warp_id; chunk_block < num_blocks; chunk_block += WarpsPerBlock) {
        const int block_in_cluster = first_block + chunk_block;
        const int vec_idx_padded = vector_offset + block_in_cluster * kWarpSize + lane;
        const bool valid_vec = vec_idx_padded < vector_offset + vector_count;
        const uint8_t* data = cluster_base + (size_t)block_in_cluster * words_per_block;
        int dot = 0;
        const int dimBlocks = (n_dim / kWarpSize) * kWarpSize;
        for (int dBase = 0; dBase < dimBlocks; dBase += kWarpSize) {
            const int qReg = (int)query[dBase + lane];
            for (int i = 0; i < kWarpSize / 4; ++i, data += 4 * kGroup) {
                const int v0 = (int)data[0 * kGroup + lane];
                const int v1 = (int)data[1 * kGroup + lane];
                const int v2 = (int)data[2 * kGroup + lane];
                const int v3 = (int)data[3 * kGroup + lane];
                dot += __shfl_sync(0xffffffff, qReg, i * 4 + 0) * v0;
                dot += __shfl_sync(0xffffffff, qReg, i * 4 + 1) * v1;
                dot += __shfl_sync(0xffffffff, qReg, i * 4 + 2) * v2;
                dot += __shfl_sync(0xffffffff, qReg, i * 4 + 3) * v3;
            }
        }
        if (dimBlocks < n_dim) {
            const int dLoad = dimBlocks + lane;
            const int qReg = (dLoad < n_dim) ? (int)query[dLoad] : 0;
            for (int dSub = 0; dSub < n_dim - dimBlocks; ++dSub, data += kGroup) {
                dot += __shfl_sync(0xffffffff, qReg, dSub) * (int)data[lane];
            }
        }
        if (!valid_vec) {
            queue.add(dummy_val, -1);
        } else {
            const float data_norm = d_cluster_vector_norm[vec_idx_padded];
            const float dist = query_norm * query_norm + data_norm * data_norm - 2.0f * (float)dot;
            const idx_t compact_id = compact_offset + (vec_idx_padded - vector_offset);
            queue.add(dist, compact_id);
        }
    }
    __syncwarp();
    queue.done();
    __syncwarp();

    extern __shared__ unsigned char shmem[];
    float* warp_dist = reinterpret_cast<float*>(shmem);
    idx_t* warp_idx = reinterpret_cast<idx_t*>(warp_dist + WarpsPerBlock * Capacity);
    queue.store(warp_dist + (size_t)warp_id * Capacity, warp_idx + (size_t)warp_id * Capacity);
    __syncthreads();

    if (warp_id == 0) {
        paraselect::warpsort::WarpSortFiltered<Capacity, Ascending, float, idx_t> merge_queue(k);
        for (int w = 0; w < WarpsPerBlock; ++w) {
            for (int r_base = 0; r_base < k; r_base += kWarpSize) {
                const int r = r_base + lane;
                float d = dummy_val;
                idx_t id = (idx_t)-1;
                if (r < k) {
                    d = warp_dist[(size_t)w * Capacity + r];
                    id = warp_idx[(size_t)w * Capacity + r];
                }
                const bool keep = id >= 0 && d < merge_queue.threshold();
                merge_queue.add(keep ? d : dummy_val, keep ? id : (idx_t)-1);
            }
        }
        __syncwarp();
        merge_queue.done();
        __syncwarp();
        merge_queue.store(d_candidate_dist + (size_t)task * k, d_candidate_index + (size_t)task * k);
    }
}

}  // namespace

void launch_resident_u8_l2_topk_chunk_scan(
    int k,
    dim3 block,
    int dim,
    const uint8_t* d_queries_u8,
    const uint8_t* d_vectors_u8,
    const int* d_padded_offsets,
    const int* d_compact_offsets,
    const int* d_counts,
    const int* d_task_row,
    const int* d_task_cluster,
    const int* d_task_first_block,
    const int* d_task_num_blocks,
    float* d_query_norm,
    const float* d_norms,
    int n_tasks,
    int n_probes,
    float* d_candidate_dist,
    idx_t* d_candidate_index,
    cudaStream_t stream)
{
    if (n_tasks <= 0) return;
    constexpr int kWarps = 8;
    auto shmem = [](int cap) { return (size_t)kWarps * cap * (sizeof(float) + sizeof(idx_t)); };
    if (k <= 64) {
        resident_u8_l2_topk_chunk_scan_kernel<64, true, kWarps><<<n_tasks, block, shmem(64), stream>>>(
            d_queries_u8, d_vectors_u8, d_padded_offsets, d_compact_offsets, d_counts,
            d_task_row, d_task_cluster, d_task_first_block, d_task_num_blocks,
            d_query_norm, d_norms, n_tasks, n_probes, dim, k, d_candidate_dist, d_candidate_index);
    } else if (k <= 128) {
        resident_u8_l2_topk_chunk_scan_kernel<128, true, kWarps><<<n_tasks, block, shmem(128), stream>>>(
            d_queries_u8, d_vectors_u8, d_padded_offsets, d_compact_offsets, d_counts,
            d_task_row, d_task_cluster, d_task_first_block, d_task_num_blocks,
            d_query_norm, d_norms, n_tasks, n_probes, dim, k, d_candidate_dist, d_candidate_index);
    } else if (k <= 256) {
        resident_u8_l2_topk_chunk_scan_kernel<256, true, kWarps><<<n_tasks, block, shmem(256), stream>>>(
            d_queries_u8, d_vectors_u8, d_padded_offsets, d_compact_offsets, d_counts,
            d_task_row, d_task_cluster, d_task_first_block, d_task_num_blocks,
            d_query_norm, d_norms, n_tasks, n_probes, dim, k, d_candidate_dist, d_candidate_index);
    } else {
        throw std::runtime_error("resident u8 IVFFlat supports topk <= 256");
    }
}
