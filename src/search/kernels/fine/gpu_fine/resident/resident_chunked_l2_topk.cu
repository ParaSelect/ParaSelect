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

__device__ __forceinline__ int resident_lane_id() {
    return threadIdx.x & 31;
}

__global__ void resident_count_row_chunks_kernel(
    const int* __restrict__ d_entry_cluster_id,
    const int* __restrict__ d_entry_query_start,
    const int* __restrict__ d_entry_query_count,
    const int* __restrict__ d_entry_queries,
    const int* __restrict__ d_entry_probe_indices,
    const int* __restrict__ d_counts,
    int n_entry,
    int n_probes,
    int chunk_blocks,
    int* __restrict__ d_row_chunk_count)
{
    const int entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry >= n_entry) return;
    const int cid = d_entry_cluster_id[entry];
    const int vector_count = d_counts[cid];
    const int vector_blocks = (vector_count + kWarpSize - 1) / kWarpSize;
    const int chunks = (vector_blocks + chunk_blocks - 1) / chunk_blocks;
    const int q_start = d_entry_query_start[entry];
    const int q_count = d_entry_query_count[entry];
    for (int i = 0; i < q_count; ++i) {
        const int pos = q_start + i;
        const int q = d_entry_queries[pos];
        const int p = d_entry_probe_indices[pos];
        atomicAdd(d_row_chunk_count + q * n_probes + p, chunks);
    }
}

__global__ void resident_fill_row_chunk_tasks_kernel(
    const int* __restrict__ d_entry_cluster_id,
    const int* __restrict__ d_entry_query_start,
    const int* __restrict__ d_entry_query_count,
    const int* __restrict__ d_entry_queries,
    const int* __restrict__ d_entry_probe_indices,
    const int* __restrict__ d_counts,
    int n_entry,
    int n_probes,
    int chunk_blocks,
    int max_tasks,
    int* __restrict__ d_row_chunk_write_pos,
    int* __restrict__ d_task_row,
    int* __restrict__ d_task_cluster,
    int* __restrict__ d_task_first_block,
    int* __restrict__ d_task_num_blocks)
{
    const int entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry >= n_entry) return;
    const int cid = d_entry_cluster_id[entry];
    const int vector_count = d_counts[cid];
    const int vector_blocks = (vector_count + kWarpSize - 1) / kWarpSize;
    const int q_start = d_entry_query_start[entry];
    const int q_count = d_entry_query_count[entry];
    for (int i = 0; i < q_count; ++i) {
        const int pos = q_start + i;
        const int q = d_entry_queries[pos];
        const int p = d_entry_probe_indices[pos];
        const int row = q * n_probes + p;
        for (int first = 0; first < vector_blocks; first += chunk_blocks) {
            const int out = atomicAdd(d_row_chunk_write_pos + row, 1);
            if (out < max_tasks) {
                d_task_row[out] = row;
                d_task_cluster[out] = cid;
                d_task_first_block[out] = first;
                d_task_num_blocks[out] = min(chunk_blocks, vector_blocks - first);
            }
        }
    }
}

template<int Capacity, bool Ascending, int WarpsPerBlock>
__global__ __launch_bounds__(256, 1)
void resident_l2_topk_chunk_scan_kernel(
    float* __restrict__ d_query_group,
    const float* __restrict__ d_cluster_vector,
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
    const int lane = resident_lane_id();
    if (warp_id >= WarpsPerBlock) return;

    const int row = d_task_row[task];
    const int query_id = row / n_probes;
    const int cid = d_task_cluster[task];
    const int vector_offset = d_probe_vector_offset[cid];
    const int compact_offset = d_probe_vector_offset_compact[cid];
    const int vector_count = d_probe_vector_count[cid];
    const int first_block = d_task_first_block[task];
    const int num_blocks = d_task_num_blocks[task];

    const float* query = d_query_group + (size_t)query_id * n_dim;
    const float query_norm = d_query_norm[query_id];
    constexpr int kGroup = paraselect::kInterleavedGroupSize;
    const size_t words_per_block = (size_t)kGroup * n_dim;
    const float* cluster_base = d_cluster_vector + (size_t)(vector_offset / kGroup) * words_per_block;

    using WarpSortBase = paraselect::warpsort::WarpSort<Capacity, Ascending, float, idx_t>;
    const float dummy_val = WarpSortBase::kDummy();
    paraselect::warpsort::WarpSortFiltered<Capacity, Ascending, float, idx_t> queue(k);

    for (int chunk_block = warp_id; chunk_block < num_blocks; chunk_block += WarpsPerBlock) {
        const int block_in_cluster = first_block + chunk_block;
        const int vec_idx_padded = vector_offset + block_in_cluster * kWarpSize + lane;
        const bool valid_vec = vec_idx_padded < vector_offset + vector_count;
        const float* data = cluster_base + (size_t)block_in_cluster * words_per_block;
        float dot = 0.0f;
        const int dimBlocks = (n_dim / kWarpSize) * kWarpSize;
        for (int dBase = 0; dBase < dimBlocks; dBase += kWarpSize) {
            const float qReg = query[dBase + lane];
            for (int i = 0; i < kWarpSize / 4; ++i, data += 4 * kGroup) {
                const float v0 = data[0 * kGroup + lane];
                const float v1 = data[1 * kGroup + lane];
                const float v2 = data[2 * kGroup + lane];
                const float v3 = data[3 * kGroup + lane];
                dot += __shfl_sync(0xffffffff, qReg, i * 4 + 0) * v0;
                dot += __shfl_sync(0xffffffff, qReg, i * 4 + 1) * v1;
                dot += __shfl_sync(0xffffffff, qReg, i * 4 + 2) * v2;
                dot += __shfl_sync(0xffffffff, qReg, i * 4 + 3) * v3;
            }
        }
        if (dimBlocks < n_dim) {
            const int dLoad = dimBlocks + lane;
            const float qReg = (dLoad < n_dim) ? query[dLoad] : 0.0f;
            for (int dSub = 0; dSub < n_dim - dimBlocks; ++dSub, data += kGroup) {
                dot += __shfl_sync(0xffffffff, qReg, dSub) * data[lane];
            }
        }
        if (!valid_vec) {
            queue.add(dummy_val, -1);
        } else {
            const float data_norm = d_cluster_vector_norm[vec_idx_padded];
            const float dist = query_norm * query_norm + data_norm * data_norm - 2.0f * dot;
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

template<int Capacity, bool Ascending, int WarpsPerBlock>
__global__ __launch_bounds__(256, 1)
void resident_reduce_row_chunks_kernel(
    const int* __restrict__ d_row_chunk_offset,
    const float* __restrict__ d_candidate_dist,
    const idx_t* __restrict__ d_candidate_index,
    int n_rows,
    int k,
    float* __restrict__ d_probe_dist,
    idx_t* __restrict__ d_probe_index)
{
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int warp_id = threadIdx.x >> 5;
    const int lane = resident_lane_id();
    if (warp_id >= WarpsPerBlock) return;
    const int begin = d_row_chunk_offset[row];
    const int end = d_row_chunk_offset[row + 1];

    using WarpSortBase = paraselect::warpsort::WarpSort<Capacity, Ascending, float, idx_t>;
    const float dummy_val = WarpSortBase::kDummy();
    paraselect::warpsort::WarpSortFiltered<Capacity, Ascending, float, idx_t> queue(k);
    for (int src = begin + warp_id; src < end; src += WarpsPerBlock) {
        for (int r_base = 0; r_base < k; r_base += kWarpSize) {
            const int r = r_base + lane;
            float d = dummy_val;
            idx_t id = (idx_t)-1;
            if (r < k) {
                d = d_candidate_dist[(size_t)src * k + r];
                id = d_candidate_index[(size_t)src * k + r];
            }
            const bool keep = id >= 0 && d < queue.threshold();
            queue.add(keep ? d : dummy_val, keep ? id : (idx_t)-1);
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
        merge_queue.store(d_probe_dist + (size_t)row * k, d_probe_index + (size_t)row * k);
    }
}

}  // namespace

void launch_resident_count_row_chunks(
    const int* d_entry_cluster_id,
    const int* d_entry_query_start,
    const int* d_entry_query_count,
    const int* d_entry_queries,
    const int* d_entry_probe_indices,
    const int* d_counts,
    int n_entry,
    int n_probes,
    int chunk_blocks,
    int* d_row_chunk_count,
    cudaStream_t stream)
{
    dim3 block(256, 1, 1);
    dim3 grid((n_entry + block.x - 1) / block.x, 1, 1);
    resident_count_row_chunks_kernel<<<grid, block, 0, stream>>>(
        d_entry_cluster_id, d_entry_query_start, d_entry_query_count, d_entry_queries,
        d_entry_probe_indices, d_counts, n_entry, n_probes, chunk_blocks, d_row_chunk_count);
}

void launch_resident_fill_row_chunk_tasks(
    const int* d_entry_cluster_id,
    const int* d_entry_query_start,
    const int* d_entry_query_count,
    const int* d_entry_queries,
    const int* d_entry_probe_indices,
    const int* d_counts,
    int n_entry,
    int n_probes,
    int chunk_blocks,
    int max_tasks,
    int* d_row_chunk_write_pos,
    int* d_task_row,
    int* d_task_cluster,
    int* d_task_first_block,
    int* d_task_num_blocks,
    cudaStream_t stream)
{
    dim3 block(256, 1, 1);
    dim3 grid((n_entry + block.x - 1) / block.x, 1, 1);
    resident_fill_row_chunk_tasks_kernel<<<grid, block, 0, stream>>>(
        d_entry_cluster_id, d_entry_query_start, d_entry_query_count, d_entry_queries,
        d_entry_probe_indices, d_counts, n_entry, n_probes, chunk_blocks, max_tasks,
        d_row_chunk_write_pos, d_task_row, d_task_cluster, d_task_first_block, d_task_num_blocks);
}

void launch_resident_l2_topk_chunk_scan(
    int k,
    dim3 block,
    int dim,
    float* d_queries,
    const float* d_vectors,
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
        resident_l2_topk_chunk_scan_kernel<64, true, kWarps><<<n_tasks, block, shmem(64), stream>>>(
            d_queries, d_vectors, d_padded_offsets, d_compact_offsets, d_counts,
            d_task_row, d_task_cluster, d_task_first_block, d_task_num_blocks,
            d_query_norm, d_norms, n_tasks, n_probes, dim, k, d_candidate_dist, d_candidate_index);
    } else if (k <= 128) {
        resident_l2_topk_chunk_scan_kernel<128, true, kWarps><<<n_tasks, block, shmem(128), stream>>>(
            d_queries, d_vectors, d_padded_offsets, d_compact_offsets, d_counts,
            d_task_row, d_task_cluster, d_task_first_block, d_task_num_blocks,
            d_query_norm, d_norms, n_tasks, n_probes, dim, k, d_candidate_dist, d_candidate_index);
    } else if (k <= 256) {
        resident_l2_topk_chunk_scan_kernel<256, true, kWarps><<<n_tasks, block, shmem(256), stream>>>(
            d_queries, d_vectors, d_padded_offsets, d_compact_offsets, d_counts,
            d_task_row, d_task_cluster, d_task_first_block, d_task_num_blocks,
            d_query_norm, d_norms, n_tasks, n_probes, dim, k, d_candidate_dist, d_candidate_index);
    } else {
        throw std::runtime_error("resident IVFFlat supports topk <= 256");
    }
}

void launch_resident_reduce_row_chunks(
    int k,
    dim3 block,
    const int* d_row_chunk_offset,
    const float* d_candidate_dist,
    const idx_t* d_candidate_index,
    int n_rows,
    float* d_probe_dist,
    idx_t* d_probe_index,
    cudaStream_t stream)
{
    if (n_rows <= 0) return;
    constexpr int kWarps = 8;
    auto shmem = [](int cap) { return (size_t)kWarps * cap * (sizeof(float) + sizeof(idx_t)); };
    if (k <= 64) {
        resident_reduce_row_chunks_kernel<64, true, kWarps><<<n_rows, block, shmem(64), stream>>>(
            d_row_chunk_offset, d_candidate_dist, d_candidate_index, n_rows, k, d_probe_dist, d_probe_index);
    } else if (k <= 128) {
        resident_reduce_row_chunks_kernel<128, true, kWarps><<<n_rows, block, shmem(128), stream>>>(
            d_row_chunk_offset, d_candidate_dist, d_candidate_index, n_rows, k, d_probe_dist, d_probe_index);
    } else if (k <= 256) {
        resident_reduce_row_chunks_kernel<256, true, kWarps><<<n_rows, block, shmem(256), stream>>>(
            d_row_chunk_offset, d_candidate_dist, d_candidate_index, n_rows, k, d_probe_dist, d_probe_index);
    } else {
        throw std::runtime_error("resident IVFFlat supports topk <= 256");
    }
}
