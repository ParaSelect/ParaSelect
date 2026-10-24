/**
 * gemv_page_kernel.cuh
 *
 *  entry  kernel().
 *  Interleaved  [block][dim][lane] .
 * :
 *   Phase 1: Page -> Page Buffer (n_entry * max_queries_per_block * k)
 *   Phase 2: Page Buffer -> Probe Buffer (n_queries * n_probes * k)
 *   Phase 3: Probe Buffer -> Final Buffer (n_queries * k)
 */
#pragma once

#include "layout/ivf_interleaved_layout.cuh"
#include "topk/warpsort/warpsort_utils.cuh"
#include "topk/warpsort/warpsort.cuh"

#ifndef kWarpSize
#define kWarpSize 32
#endif

namespace paraselect {

using namespace paraselect::warpsort_utils;
using namespace paraselect::warpsort;

/* ============================================================================
 *  Kernel:  Probe
 * ============================================================================ */
inline __global__ void init_probe_buffer_kernel(
    float* d_probe_dist, int* d_probe_index, int* d_mutex,
    int total_probe_elements, int total_mutex, float dummy_val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_probe_elements) {
        d_probe_dist[idx] = dummy_val;
        d_probe_index[idx] = -1;
    }
    if (idx < total_mutex) {
        d_mutex[idx] = 0;
    }
}

/* ============================================================================
 * Phase 1:  (Interleaved L2 , Page )
 * ============================================================================ */
template<int Capacity, bool Ascending, int QueriesPerBlock>
inline __global__ __launch_bounds__(256, 2)
void gemv_page_l2_topk_kernel(
    const float* __restrict__ d_query,
    const float* __restrict__ d_vectors,
    const int*   __restrict__ d_cluster_page_table,
    const int*   __restrict__ d_probe_vec_offset_compact,
    const int*   __restrict__ d_probe_vec_count,
    const int*   __restrict__ d_entry_cluster_id,
    const int*   __restrict__ d_entry_page_idx, //  Entry  cluster
    const int*   __restrict__ d_entry_query_start,
    const int*   __restrict__ d_entry_query_count,
    const int*   __restrict__ d_entry_queries,
    const float* __restrict__ d_query_norm,
    int n_entry, int n_dim, int k,
    float* __restrict__ d_page_topk_dist,
    int*   __restrict__ d_page_topk_index
) {
    const int entry_id = blockIdx.x;
    if (entry_id >= n_entry) return;

    const int cluster_id   = d_entry_cluster_id[entry_id];
    const int logical_page = d_entry_page_idx[entry_id];
    const int compact_off  = d_probe_vec_offset_compact[cluster_id];
    const int vec_count    = d_probe_vec_count[cluster_id];
    const int phys_page_id = d_cluster_page_table[entry_id];

    const int eq_start = d_entry_query_start[entry_id];
    const int eq_count = d_entry_query_count[entry_id];

    const int local_qid = threadIdx.x / kWarpSize;
    const int lane      = threadIdx.x % kWarpSize;
    const bool active   = (local_qid < eq_count);

    __shared__ float s_query_norm[QueriesPerBlock];

    int query_gid = -1;
    if (active) {
        const int idx = eq_start + local_qid;
        query_gid  = d_entry_queries[idx];
        if (lane == 0) s_query_norm[local_qid] = d_query_norm[query_gid];
    }
    __syncthreads();
    if (!active) return;

    const float* q_ptr = d_query + (size_t)query_gid * n_dim;

    constexpr int kG = kInterleavedGroupSize; //  32
    const size_t wpb = (size_t)kG * n_dim;
    const int kPageVecs = 256;
    const size_t page_bytes_floats = (size_t)kPageVecs * n_dim;

    //
    const float* page_base = d_vectors + (size_t)phys_page_id * page_bytes_floats;

    using WarpSortBase = paraselect::warpsort::WarpSort<Capacity, Ascending, float, int>;
    const float dummy_val = WarpSortBase::kDummy();
    paraselect::warpsort::WarpSortFiltered<Capacity, Ascending, float, int> queue(k);

    //  Cluster
    const int page_vec_start = logical_page * kPageVecs;
    const int vec_in_page = min(kPageVecs, vec_count - page_vec_start);
    const int n_iter = (vec_in_page + kWarpSize - 1) / kWarpSize;
    const int dimBlocks = (n_dim / kWarpSize) * kWarpSize;

    for (int iter = 0; iter < n_iter; ++iter) {
        const int vid_in_page = iter * kWarpSize + lane;
        const bool valid = vid_in_page < vec_in_page;
        const float* data = page_base + (size_t)iter * wpb; //  interleaved  block

        float dot = 0.0f;
        float vnorm_sq = 0.0f;

        // [block][dim][lane]  Coalesced
        for (int dBase = 0; dBase < dimBlocks; dBase += kWarpSize) {
            const float qReg = q_ptr[dBase + lane];
            #pragma unroll 8
            for (int i = 0; i < kWarpSize / 4; ++i, data += 4 * kG) {
                float v0 = data[0 * kG + lane];
                float v1 = data[1 * kG + lane];
                float v2 = data[2 * kG + lane];
                float v3 = data[3 * kG + lane];
                float q0 = __shfl_sync(0xffffffff, qReg, i * 4 + 0);
                float q1 = __shfl_sync(0xffffffff, qReg, i * 4 + 1);
                float q2 = __shfl_sync(0xffffffff, qReg, i * 4 + 2);
                float q3 = __shfl_sync(0xffffffff, qReg, i * 4 + 3);
                dot += q0 * v0 + q1 * v1 + q2 * v2 + q3 * v3;
                vnorm_sq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
            }
        }
        if (dimBlocks < n_dim) {
            const int dLeft = n_dim - dimBlocks;
            const float qReg = (lane < dLeft) ? q_ptr[dimBlocks + lane] : 0.0f;
            for (int i = 0; i < (dLeft + 3) / 4; ++i, data += 4 * kG) {
                const int d0 = i * 4;
                float v0 = (d0 + 0 < dLeft) ? data[0 * kG + lane] : 0.0f;
                float v1 = (d0 + 1 < dLeft) ? data[1 * kG + lane] : 0.0f;
                float v2 = (d0 + 2 < dLeft) ? data[2 * kG + lane] : 0.0f;
                float v3 = (d0 + 3 < dLeft) ? data[3 * kG + lane] : 0.0f;
                dot += __shfl_sync(0xffffffff, qReg, d0 + 0) * v0;
                dot += __shfl_sync(0xffffffff, qReg, d0 + 1) * v1;
                dot += __shfl_sync(0xffffffff, qReg, d0 + 2) * v2;
                dot += __shfl_sync(0xffffffff, qReg, d0 + 3) * v3;
                vnorm_sq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
            }
        }

        if (valid) {
            float qn = s_query_norm[local_qid];
            float l2 = qn * qn + vnorm_sq - 2.0f * dot;
            //  compact index
            int compact_idx = compact_off + page_vec_start + vid_in_page;
            queue.add(l2, compact_idx);
        } else {
            queue.add(dummy_val, -1);
        }
    }

    size_t page_out_offset = ((size_t)entry_id * QueriesPerBlock + local_qid) * k;
    float* row_dist = d_page_topk_dist + page_out_offset;
    int*   row_idx  = d_page_topk_index + page_out_offset;

    __syncwarp();
    queue.done();
    __syncwarp();
    queue.store(row_dist, row_idx);
}

/* ============================================================================
 * Phase 2: Page -> Probe
 * ============================================================================ */
template<int Capacity, bool Ascending, int QueriesPerBlock>
inline __global__ __launch_bounds__(256, 2)
void reduce_page_to_probe_kernel(
    const float* __restrict__ d_page_topk_dist,
    const int*   __restrict__ d_page_topk_index,
    const int*   __restrict__ d_entry_query_start,
    const int*   __restrict__ d_entry_query_count,
    const int*   __restrict__ d_entry_queries,
    const int*   __restrict__ d_entry_probe_indices,
    int n_entry, int n_probes, int k,
    float* __restrict__ d_probe_topk_dist,
    int*   __restrict__ d_probe_topk_index,
    int*   __restrict__ d_mutex
) {
    // []: Shared Memory
    // ,,
    __shared__ float s_dist[QueriesPerBlock * Capacity];
    __shared__ int   s_idx [QueriesPerBlock * Capacity];

    const int entry_id = blockIdx.x;
    const int local_qid = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;

    if (entry_id >= n_entry) return;
    const int eq_start = d_entry_query_start[entry_id];
    const int eq_count = d_entry_query_count[entry_id];

    if (local_qid >= eq_count) return;

    const int query_gid = d_entry_queries[eq_start + local_qid];
    const int probe_slot = d_entry_probe_indices[eq_start + local_qid];

    size_t page_offset = ((size_t)entry_id * QueriesPerBlock + local_qid) * k;
    const float* my_dist = d_page_topk_dist + page_offset;
    const int* my_idx = d_page_topk_index + page_offset;

    size_t probe_offset = ((size_t)query_gid * n_probes + probe_slot) * k;
    float* target_dist = d_probe_topk_dist + probe_offset;
    int* target_idx = d_probe_topk_index + probe_offset;
    int* mutex = d_mutex + (query_gid * n_probes + probe_slot);

    using WarpSortBase = paraselect::warpsort::WarpSort<Capacity, Ascending, float, int>;
    const float dummy_val = WarpSortBase::kDummy();
    paraselect::warpsort::WarpSortFiltered<Capacity, Ascending, float, int> queue(k);

    const bool is_master = (lane == 0);

    // 1.
    if (is_master) { while (atomicCAS(mutex, 0, 1) != 0) {} }
    __syncwarp();

    // 2. ( volatile  L1  L2)
    for (int i = 0; i < k; i += kWarpSize) {
        int idx = i + lane;
        float d = dummy_val;
        int id = -1;
        if (idx < k) {
            d = *(volatile float*)(&target_dist[idx]);
            id = *(volatile int*)(&target_idx[idx]);
            if (id == -1) d = dummy_val;
        }
        queue.add(d, id);
    }

    // 3.  Page
    for (int i = 0; i < k; i += kWarpSize) {
        int idx = i + lane;
        float d = dummy_val;
        int id = -1;
        if (idx < k) {
            d = my_dist[idx];
            id = my_idx[idx];
            if (id == -1) d = dummy_val;
        }
        queue.add(d, id);
    }

    queue.done();

    // 4. []: Shared Memory
    float* smem_dist = s_dist + local_qid * Capacity;
    int*   smem_idx  = s_idx  + local_qid * Capacity;
    queue.store(smem_dist, smem_idx);

    //  Warp  Shared Memory
    __syncwarp();

    // 5. []: volatile  (ST.CG), L1
    for (int i = 0; i < k; i += kWarpSize) {
        int idx = i + lane;
        if (idx < k) {
            *(volatile float*)(&target_dist[idx]) = smem_dist[idx];
            *(volatile int*)(&target_idx[idx])    = smem_idx[idx];
        }
    }

    // 6. , L2
    __threadfence();
    __syncwarp();

    // 7.
    if (is_master) { atomicExch(mutex, 0); }
}
/* ============================================================================
 * Phase 3: Probe -> Query
 * ============================================================================ */
template<int Capacity, bool Ascending>
inline __global__ __launch_bounds__(256, 2)
void reduce_probe_to_query_kernel(
    const float* __restrict__ d_probe_topk_dist,
    const int*   __restrict__ d_probe_topk_index,
    int n_queries, int n_probes, int k,
    float* __restrict__ d_final_dist, int*   __restrict__ d_final_index
) {
    const int query_id = blockIdx.x * (blockDim.x / kWarpSize) + (threadIdx.x / kWarpSize);
    const int lane = threadIdx.x % kWarpSize;
    if (query_id >= n_queries) return;

    using WarpSortBase = paraselect::warpsort::WarpSort<Capacity, Ascending, float, int>;
    const float dummy_val = WarpSortBase::kDummy();
    paraselect::warpsort::WarpSortFiltered<Capacity, Ascending, float, int> queue(k);

    size_t query_base_offset = (size_t)query_id * n_probes * k;

    for (int p = 0; p < n_probes; ++p) {
        const float* p_dist = d_probe_topk_dist + query_base_offset + p * k;
        const int* p_idx = d_probe_topk_index + query_base_offset + p * k;

        for (int i = 0; i < k; i += kWarpSize) {
            int idx = i + lane;
            float d = dummy_val;
            int id = -1;
            if (idx < k) {
                d = p_dist[idx];
                id = p_idx[idx];
                if (id == -1) d = dummy_val;
            }
            queue.add(d, id);
        }
    }
    queue.done();
    queue.store(d_final_dist + query_id * k, d_final_index + query_id * k);
}

/* ============================================================================
 * Host-side launch wrapper
 * ============================================================================ */
inline void launch_gemv_page_l2(
    int n_dim, int n_entry, int n_probes, int k,
    const float* d_query, const float* d_vectors,
    const int* d_cluster_page_table,
    const int* d_pvoc, const int* d_pvc,
    const int* d_ecid, const int* d_epid,
    const int* d_eqs, const int* d_eqc,
    const int* d_eq, const int* d_epi,
    const float* d_qnorm, int n_queries,
    float* d_page_topk_dist, int* d_page_topk_index,
    float* d_probe_topk_dist, int* d_probe_topk_index,
    int* d_mutex,
    float* d_final_dist, int* d_final_index,
    cudaStream_t stream
) {
    if (n_entry <= 0 || n_queries <= 0) return;

    constexpr int QPB = 8;

    // Step 0
    int probe_elements = n_queries * n_probes * k;
    int mutex_elements = n_queries * n_probes;
    int init_blocks = (probe_elements + 255) / 256;
    float dummy_val = paraselect::warpsort_utils::upper_bound<float>();

    init_probe_buffer_kernel<<<init_blocks, 256, 0, stream>>>(
        d_probe_topk_dist, d_probe_topk_index, d_mutex, probe_elements, mutex_elements, dummy_val);

    // Step 1
    dim3 p1_grid(n_entry);
    dim3 p1_block(QPB * kWarpSize);

    #define LAUNCH_P1(Cap) \
        gemv_page_l2_topk_kernel<Cap, true, QPB><<<p1_grid, p1_block, 0, stream>>>( \
            d_query, d_vectors, d_cluster_page_table, d_pvoc, d_pvc, \
            d_ecid, d_epid, d_eqs, d_eqc, d_eq, d_qnorm, \
            n_entry, n_dim, k, d_page_topk_dist, d_page_topk_index)

    if      (k <= 64)  { LAUNCH_P1(64);  }
    else if (k <= 128) { LAUNCH_P1(128); }
    else               { LAUNCH_P1(256); }
    #undef LAUNCH_P1

    // Step 2
    #define LAUNCH_P2(Cap) \
        reduce_page_to_probe_kernel<Cap, true, QPB><<<p1_grid, p1_block, 0, stream>>>( \
            d_page_topk_dist, d_page_topk_index, d_eqs, d_eqc, d_eq, d_epi, \
            n_entry, n_probes, k, d_probe_topk_dist, d_probe_topk_index, d_mutex)

    if      (k <= 64)  { LAUNCH_P2(64);  }
    else if (k <= 128) { LAUNCH_P2(128); }
    else               { LAUNCH_P2(256); }
    #undef LAUNCH_P2

    // Step 3
    int p3_threads = n_queries * kWarpSize;
    int p3_blocks = (p3_threads + 255) / 256;

    #define LAUNCH_P3(Cap) \
        reduce_probe_to_query_kernel<Cap, true><<<p3_blocks, 256, 0, stream>>>( \
            d_probe_topk_dist, d_probe_topk_index, n_queries, n_probes, k, d_final_dist, d_final_index)

    if      (k <= 64)  { LAUNCH_P3(64);  }
    else if (k <= 128) { LAUNCH_P3(128); }
    else               { LAUNCH_P3(256); }
    #undef LAUNCH_P3
}

} // namespace paraselect
