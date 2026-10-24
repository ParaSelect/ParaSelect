#include "search/kernels/fine/gpu_fine/resident/resident_threshold_l2_topk.cuh"

#include "layout/ivf_interleaved_layout.cuh"
#include "search/kernels/topk/warpsort/warpsort.cuh"
#include "search/kernels/topk/warpsort/warpsort_utils.cuh"

#include <cfloat>
#include <stdexcept>

namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ unsigned load_u8x4_threshold(
    const uint8_t* ptr,
    int remaining) {
    unsigned value = 0;
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&value);
    if (remaining > 0) bytes[0] = ptr[0];
    if (remaining > 1) bytes[1] = ptr[1];
    if (remaining > 2) bytes[2] = ptr[2];
    if (remaining > 3) bytes[3] = ptr[3];
    return value;
}

__device__ __forceinline__ int pack_i8x4_threshold_v1(
    const int8_t* data, int dim, int lane) {
    const unsigned packed =
        static_cast<unsigned>(static_cast<uint8_t>(data[(dim + 0) * kWarpSize + lane])) |
        (static_cast<unsigned>(static_cast<uint8_t>(data[(dim + 1) * kWarpSize + lane])) << 8) |
        (static_cast<unsigned>(static_cast<uint8_t>(data[(dim + 2) * kWarpSize + lane])) << 16) |
        (static_cast<unsigned>(static_cast<uint8_t>(data[(dim + 3) * kWarpSize + lane])) << 24);
    return static_cast<int>(packed);
}

__global__ void resident_threshold_count_seed_entries_kernel(
    const int* __restrict__ cluster_offsets,
    const int* __restrict__ probe_indices,
    int nlist,
    int seed_probes,
    int* __restrict__ entry_counts) {
    const int cluster = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster >= nlist) return;
    int count = 0;
    for (int i = cluster_offsets[cluster]; i < cluster_offsets[cluster + 1]; ++i)
        count += probe_indices[i] < seed_probes;
    entry_counts[cluster] = (count + 7) / 8;
}

__global__ void resident_threshold_fill_seed_entries_kernel(
    const int* __restrict__ cluster_offsets,
    const int* __restrict__ cluster_queries,
    const int* __restrict__ probe_indices,
    const int* __restrict__ entry_offsets,
    int nlist,
    int seed_probes,
    int* __restrict__ entry_cluster_id,
    int* __restrict__ entry_query_start,
    int* __restrict__ entry_query_count,
    int* __restrict__ entry_queries,
    int* __restrict__ entry_probe_indices) {
    const int cluster = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster >= nlist) return;
    const int ref_start = cluster_offsets[cluster];
    int selected = 0;
    for (int i = ref_start; i < cluster_offsets[cluster + 1]; ++i) {
        const int probe = probe_indices[i];
        if (probe >= seed_probes) continue;
        entry_queries[ref_start + selected] = cluster_queries[i];
        entry_probe_indices[ref_start + selected] = probe;
        ++selected;
    }
    const int entry_start = entry_offsets[cluster];
    const int entries = (selected + 7) / 8;
    for (int e = 0; e < entries; ++e) {
        const int entry = entry_start + e;
        entry_cluster_id[entry] = cluster;
        entry_query_start[entry] = ref_start + e * 8;
        entry_query_count[entry] = min(8, selected - e * 8);
    }
}

template <int Capacity>
__global__ void resident_threshold_from_probe_topk_kernel(
    const float* __restrict__ probe_dist,
    int n_query,
    int n_probes,
    int seed_probes,
    int k,
    float* __restrict__ thresholds) {
    const int query_id = blockIdx.x;
    if (query_id >= n_query || threadIdx.x != 0) return;
    float best[Capacity];
    for (int rank = 0; rank < k; ++rank) best[rank] = FLT_MAX;
    const int probes = min(seed_probes, n_probes);
    for (int probe = 0; probe < probes; ++probe) {
        const size_t base = ((size_t)query_id * n_probes + probe) * k;
        for (int candidate = 0; candidate < k; ++candidate) {
            const float dist = probe_dist[base + candidate];
            if (dist >= best[k - 1]) continue;
            int rank = k - 1;
            while (rank > 0 && dist < best[rank - 1]) {
                best[rank] = best[rank - 1];
                --rank;
            }
            best[rank] = dist;
        }
    }
    thresholds[query_id] = best[k - 1];
}

template <ResidentDenseDistanceKind DistanceKind>
__global__ __launch_bounds__(256, 1) void resident_threshold_grouped_scan_kernel(
    int dim,
    int n_entry,
    int candidate_capacity,
    const float* __restrict__ queries,
    const uint8_t* __restrict__ queries_u8,
    const float* __restrict__ vectors,
    const uint8_t* __restrict__ vectors_u8,
    const int* __restrict__ padded_offsets,
    const int* __restrict__ compact_offsets,
    const int* __restrict__ counts,
    const int* __restrict__ entry_cluster_id,
    const int* __restrict__ entry_query_start,
    const int* __restrict__ entry_query_count,
    const int* __restrict__ entry_queries,
    const float* __restrict__ query_norms,
    const float* __restrict__ query_dot_scales,
    const float* __restrict__ vector_norms,
    const float* __restrict__ thresholds,
    int* __restrict__ candidate_counts,
    int* __restrict__ overflow,
    float* __restrict__ candidate_dist,
    idx_t* __restrict__ candidate_index) {
    const int entry = blockIdx.x;
    if (entry >= n_entry) return;
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int query_count = entry_query_count[entry];
    const bool active = warp < query_count;
    if (!active) return;

    const int query_id_lane = lane == 0
        ? entry_queries[entry_query_start[entry] + warp]
        : 0;
    const int query_id = __shfl_sync(kFullMask, query_id_lane, 0);
    const float query_norm_lane = lane == 0 ? query_norms[query_id] : 0.0f;
    const float query_norm = __shfl_sync(kFullMask, query_norm_lane, 0);

    const int cluster = entry_cluster_id[entry];
    const int padded_start = padded_offsets[cluster];
    const int compact_start = compact_offsets[cluster];
    const int count = counts[cluster];
    const int vector_blocks = (count + kWarpSize - 1) / kWarpSize;
    const float threshold = thresholds[query_id];

    for (int block_id = 0; block_id < vector_blocks; ++block_id) {
        const int local = block_id * kWarpSize + lane;
        const int padded = padded_start + local;
        const float vector_norm = vector_norms[padded];
        float dot = 0.0f;
        if constexpr (DistanceKind == ResidentDenseDistanceKind::PCAI8Dp4aV1 ||
                      DistanceKind == ResidentDenseDistanceKind::PCAI8Dp4aV2) {
            const int8_t* query =
                reinterpret_cast<const int8_t*>(queries_u8) + (size_t)query_id * dim;
            const size_t values_per_block = (size_t)kWarpSize * dim;
            const int8_t* data = reinterpret_cast<const int8_t*>(vectors_u8) +
                (size_t)(padded_start / kWarpSize + block_id) * values_per_block;
            const int dim4 = dim / 4;
            const int query4_lane = lane < dim4
                ? reinterpret_cast<const int*>(query)[lane]
                : 0;
            int dot_i32 = 0;
            for (int d4 = 0; d4 < dim4; ++d4) {
                const int query4 = __shfl_sync(kFullMask, query4_lane, d4);
                int vector4 = 0;
                if constexpr (DistanceKind == ResidentDenseDistanceKind::PCAI8Dp4aV1) {
                    vector4 = pack_i8x4_threshold_v1(data, d4 * 4, lane);
                } else {
                    vector4 = *reinterpret_cast<const int*>(
                        data + ((size_t)d4 * kWarpSize + lane) * 4);
                }
                dot_i32 = __dp4a(query4, vector4, dot_i32);
            }
            dot = query_dot_scales[query_id] * static_cast<float>(dot_i32);
        } else if constexpr (DistanceKind == ResidentDenseDistanceKind::U8Dp4a) {
            const uint8_t* query = queries_u8 + (size_t)query_id * dim;
            const int dim4 = (dim + 3) / 4;
            const size_t bytes_per_block = (size_t)dim4 * kWarpSize * 4;
            const uint8_t* data = vectors_u8 +
                                  (size_t)(padded_start / kWarpSize + block_id) * bytes_per_block;
            unsigned dot_u32 = 0;
            for (int d4_base = 0; d4_base < dim4; d4_base += kWarpSize) {
                const int lane_d4 = d4_base + lane;
                unsigned q4_lane = 0;
                if (lane_d4 < dim4) {
                    const int d = lane_d4 * 4;
                    const int remaining = dim - d;
                    q4_lane = remaining >= 4
                                  ? *reinterpret_cast<const unsigned*>(query + d)
                                  : load_u8x4_threshold(query + d, remaining);
                }
                const int limit = min(kWarpSize, dim4 - d4_base);
                for (int j = 0; j < limit; ++j) {
                    const unsigned q4 = __shfl_sync(kFullMask, q4_lane, j);
                    const int d4 = d4_base + j;
                    const unsigned v4 = *reinterpret_cast<const unsigned*>(
                        data + ((size_t)d4 * kWarpSize + lane) * 4);
                    dot_u32 = __dp4a(q4, v4, dot_u32);
                }
            }
            dot = (float)dot_u32;
        } else {
            const float* query = queries + (size_t)query_id * dim;
            const size_t values_per_block = (size_t)kWarpSize * dim;
            const int dim_blocks = (dim / kWarpSize) * kWarpSize;
            if constexpr (DistanceKind == ResidentDenseDistanceKind::FP32) {
                const float* data = vectors +
                    (size_t)(padded_start / kWarpSize + block_id) * values_per_block;
                for (int d_base = 0; d_base < dim_blocks; d_base += kWarpSize) {
                    const float qreg = query[d_base + lane];
                    for (int i = 0; i < kWarpSize / 4; ++i, data += 4 * kWarpSize) {
                        const float q0 = __shfl_sync(kFullMask, qreg, i * 4 + 0);
                        const float q1 = __shfl_sync(kFullMask, qreg, i * 4 + 1);
                        const float q2 = __shfl_sync(kFullMask, qreg, i * 4 + 2);
                        const float q3 = __shfl_sync(kFullMask, qreg, i * 4 + 3);
                        dot += q0 * data[0 * kWarpSize + lane];
                        dot += q1 * data[1 * kWarpSize + lane];
                        dot += q2 * data[2 * kWarpSize + lane];
                        dot += q3 * data[3 * kWarpSize + lane];
                    }
                }
                if (dim_blocks < dim) {
                    const int dload = dim_blocks + lane;
                    const float qreg = dload < dim ? query[dload] : 0.0f;
                    for (int d = 0; d < dim - dim_blocks; ++d)
                        dot += __shfl_sync(kFullMask, qreg, d) * data[(size_t)d * kWarpSize + lane];
                }
            } else {
                const int8_t* data = reinterpret_cast<const int8_t*>(vectors_u8) +
                    (size_t)(padded_start / kWarpSize + block_id) * values_per_block;
                for (int d_base = 0; d_base < dim_blocks; d_base += kWarpSize) {
                    const float qreg = query[d_base + lane];
                    for (int i = 0; i < kWarpSize / 4; ++i, data += 4 * kWarpSize) {
                        const float q0 = __shfl_sync(kFullMask, qreg, i * 4 + 0);
                        const float q1 = __shfl_sync(kFullMask, qreg, i * 4 + 1);
                        const float q2 = __shfl_sync(kFullMask, qreg, i * 4 + 2);
                        const float q3 = __shfl_sync(kFullMask, qreg, i * 4 + 3);
                        dot += q0 * static_cast<float>(data[0 * kWarpSize + lane]);
                        dot += q1 * static_cast<float>(data[1 * kWarpSize + lane]);
                        dot += q2 * static_cast<float>(data[2 * kWarpSize + lane]);
                        dot += q3 * static_cast<float>(data[3 * kWarpSize + lane]);
                    }
                }
                if (dim_blocks < dim) {
                    const int dload = dim_blocks + lane;
                    const float qreg = dload < dim ? query[dload] : 0.0f;
                    for (int d = 0; d < dim - dim_blocks; ++d)
                        dot += __shfl_sync(kFullMask, qreg, d) *
                               static_cast<float>(data[(size_t)d * kWarpSize + lane]);
                }
            }
        }
        const float dist =
            (DistanceKind == ResidentDenseDistanceKind::PCAI8Scaled ||
             DistanceKind == ResidentDenseDistanceKind::PCAI8Dp4aV1 ||
             DistanceKind == ResidentDenseDistanceKind::PCAI8Dp4aV2)
            ? query_norm + vector_norm - 2.0f * dot
            : query_norm * query_norm + vector_norm * vector_norm - 2.0f * dot;
        const bool valid = local < count;
        const unsigned mask =
            __ballot_sync(kFullMask, valid && dist <= threshold);
        if (mask == 0) continue;
        const int leader = __ffs(mask) - 1;
        int output_base = 0;
        if (lane == leader)
            output_base = atomicAdd(candidate_counts + query_id, __popc(mask));
        output_base = __shfl_sync(kFullMask, output_base, leader);
        if ((mask & (1u << lane)) == 0) continue;
        const unsigned lower =
            lane == 0 ? 0u : mask & ((1u << lane) - 1u);
        const int output = output_base + __popc(lower);
        if (output < candidate_capacity) {
            const size_t dst = (size_t)query_id * candidate_capacity + output;
            candidate_dist[dst] = dist;
            candidate_index[dst] = compact_start + local;
        } else {
            atomicExch(overflow, 1);
        }
    }
}

__device__ __forceinline__ void append_threshold_candidates(
    bool accepted,
    float distance,
    idx_t index,
    int query_id,
    int candidate_capacity,
    int* candidate_counts,
    int* overflow,
    float* candidate_dist,
    idx_t* candidate_index) {
    const int lane = threadIdx.x % kWarpSize;
    const unsigned mask = __ballot_sync(kFullMask, accepted);
    if (mask == 0) return;
    const int leader = __ffs(mask) - 1;
    int output_base = 0;
    if (lane == leader) {
        const int add = __popc(mask);
        int* counter = candidate_counts + query_id;
        int cur = atomicAdd(counter, 0);
        while (true) {
            const int next = min(cur + add, candidate_capacity);
            if (atomicCAS(counter, cur, next) == cur) {
                output_base = cur;
                if (cur + add > candidate_capacity) atomicExch(overflow, 1);
                break;
            }
            cur = atomicAdd(counter, 0);
        }
    }
    output_base = __shfl_sync(kFullMask, output_base, leader);
    if ((mask & (1u << lane)) == 0) return;
    const unsigned lower = lane == 0 ? 0u : mask & ((1u << lane) - 1u);
    const int output = output_base + __popc(lower);
    if (output < candidate_capacity) {
        const size_t dst = (size_t)query_id * candidate_capacity + output;
        candidate_dist[dst] = distance;
        candidate_index[dst] = index;
    } else {
        atomicExch(overflow, 1);
    }
}

struct PCAI8ThresholdQueryState {
    int id;
    int query4_lane;
    float norm;
    float dot_scale;
    float threshold;
};

template <bool Paired>
__device__ __forceinline__ void scan_pca_i8_dp4a_v3_entry(
    int dim,
    int candidate_capacity,
    const int8_t* __restrict__ vectors,
    int padded_start,
    int compact_start,
    int count,
    const float* __restrict__ vector_norms,
    PCAI8ThresholdQueryState primary,
    PCAI8ThresholdQueryState secondary,
    int* __restrict__ candidate_counts,
    int* __restrict__ overflow,
    float* __restrict__ candidate_dist,
    idx_t* __restrict__ candidate_index) {
    const int lane = threadIdx.x % kWarpSize;
    const int dim4 = dim / 4;
    const int vector_blocks = (count + kWarpSize - 1) / kWarpSize;
    const size_t values_per_block = (size_t)kWarpSize * dim;
    for (int block_id = 0; block_id < vector_blocks; ++block_id) {
        const int local = block_id * kWarpSize + lane;
        const int padded = padded_start + local;
        const int8_t* data = vectors +
            (size_t)(padded_start / kWarpSize + block_id) * values_per_block;
        int primary_dot_i32 = 0;
        int secondary_dot_i32 = 0;
        for (int d4 = 0; d4 < dim4; ++d4) {
            const int vector4 = *reinterpret_cast<const int*>(
                data + ((size_t)d4 * kWarpSize + lane) * 4);
            const int primary_query4 =
                __shfl_sync(kFullMask, primary.query4_lane, d4);
            primary_dot_i32 =
                __dp4a(primary_query4, vector4, primary_dot_i32);
            if constexpr (Paired) {
                const int secondary_query4 =
                    __shfl_sync(kFullMask, secondary.query4_lane, d4);
                secondary_dot_i32 =
                    __dp4a(secondary_query4, vector4, secondary_dot_i32);
            }
        }
        const bool valid = local < count;
        const float vector_norm = vector_norms[padded];
        const float primary_distance = primary.norm + vector_norm -
            2.0f * primary.dot_scale * static_cast<float>(primary_dot_i32);
        append_threshold_candidates(
            valid && primary_distance <= primary.threshold,
            primary_distance, compact_start + local, primary.id,
            candidate_capacity, candidate_counts, overflow,
            candidate_dist, candidate_index);
        if constexpr (Paired) {
            const float secondary_distance = secondary.norm + vector_norm -
                2.0f * secondary.dot_scale *
                    static_cast<float>(secondary_dot_i32);
            append_threshold_candidates(
                valid && secondary_distance <= secondary.threshold,
                secondary_distance, compact_start + local, secondary.id,
                candidate_capacity, candidate_counts, overflow,
                candidate_dist, candidate_index);
        }
    }
}

// Paired U8-DP4A scan: one warp owns two queries of the same entry (primary =
// warp, secondary = warp + 4) and loads each interleaved vector word once,
// feeding both dp4a accumulators. This halves the number of L1/TEX global
// loads per query-vector pair versus the warp-per-query grouped kernel, without
// changing the dp4a or shuffle count. Mirrors scan_pca_i8_dp4a_v3_entry.
template <bool Paired>
__device__ __forceinline__ void scan_u8_dp4a_pair_entry(
    int dim,
    int candidate_capacity,
    const uint8_t* __restrict__ vectors,
    int padded_start,
    int compact_start,
    int count,
    const float* __restrict__ vector_norms,
    const uint8_t* __restrict__ primary_query,
    float primary_norm,
    float primary_threshold,
    int primary_id,
    const uint8_t* __restrict__ secondary_query,
    float secondary_norm,
    float secondary_threshold,
    int secondary_id,
    int* __restrict__ candidate_counts,
    int* __restrict__ overflow,
    float* __restrict__ candidate_dist,
    idx_t* __restrict__ candidate_index) {
    const int lane = threadIdx.x % kWarpSize;
    const int dim4 = (dim + 3) / 4;
    const int vector_blocks = (count + kWarpSize - 1) / kWarpSize;
    const size_t bytes_per_block = (size_t)dim4 * kWarpSize * 4;
    for (int block_id = 0; block_id < vector_blocks; ++block_id) {
        const int local = block_id * kWarpSize + lane;
        const int padded = padded_start + local;
        const uint8_t* data = vectors +
            (size_t)(padded_start / kWarpSize + block_id) * bytes_per_block;
        unsigned primary_dot = 0;
        unsigned secondary_dot = 0;
        for (int d4_base = 0; d4_base < dim4; d4_base += kWarpSize) {
            const int lane_d4 = d4_base + lane;
            unsigned qa = 0, qb = 0;
            if (lane_d4 < dim4) {
                const int d = lane_d4 * 4;
                const int remaining = dim - d;
                qa = remaining >= 4
                         ? *reinterpret_cast<const unsigned*>(primary_query + d)
                         : load_u8x4_threshold(primary_query + d, remaining);
                if constexpr (Paired) {
                    qb = remaining >= 4
                             ? *reinterpret_cast<const unsigned*>(secondary_query + d)
                             : load_u8x4_threshold(secondary_query + d, remaining);
                }
            }
            const int limit = min(kWarpSize, dim4 - d4_base);
            for (int j = 0; j < limit; ++j) {
                const int d4 = d4_base + j;
                const unsigned v4 = *reinterpret_cast<const unsigned*>(
                    data + ((size_t)d4 * kWarpSize + lane) * 4);
                primary_dot =
                    __dp4a(__shfl_sync(kFullMask, qa, j), v4, primary_dot);
                if constexpr (Paired) {
                    secondary_dot =
                        __dp4a(__shfl_sync(kFullMask, qb, j), v4, secondary_dot);
                }
            }
        }
        const bool valid = local < count;
        const float vector_norm = vector_norms[padded];
        const float primary_distance =
            primary_norm * primary_norm + vector_norm * vector_norm -
            2.0f * static_cast<float>(primary_dot);
        append_threshold_candidates(
            valid && primary_distance <= primary_threshold,
            primary_distance, compact_start + local, primary_id,
            candidate_capacity, candidate_counts, overflow,
            candidate_dist, candidate_index);
        if constexpr (Paired) {
            const float secondary_distance =
                secondary_norm * secondary_norm + vector_norm * vector_norm -
                2.0f * static_cast<float>(secondary_dot);
            append_threshold_candidates(
                valid && secondary_distance <= secondary_threshold,
                secondary_distance, compact_start + local, secondary_id,
                candidate_capacity, candidate_counts, overflow,
                candidate_dist, candidate_index);
        }
    }
}

__global__ __launch_bounds__(128, 8)
void resident_threshold_grouped_u8_pair_scan_kernel(
    int dim,
    int n_entry,
    int candidate_capacity,
    const uint8_t* __restrict__ queries_u8,
    const uint8_t* __restrict__ vectors_u8,
    const int* __restrict__ padded_offsets,
    const int* __restrict__ compact_offsets,
    const int* __restrict__ counts,
    const int* __restrict__ entry_cluster_id,
    const int* __restrict__ entry_query_start,
    const int* __restrict__ entry_query_count,
    const int* __restrict__ entry_queries,
    const float* __restrict__ query_norms,
    const float* __restrict__ vector_norms,
    const float* __restrict__ thresholds,
    int* __restrict__ candidate_counts,
    int* __restrict__ overflow,
    float* __restrict__ candidate_dist,
    idx_t* __restrict__ candidate_index) {
    const int entry = blockIdx.x;
    if (entry >= n_entry) return;
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int query_count = entry_query_count[entry];
    if (warp >= query_count) return;

    const int query_start = entry_query_start[entry];
    const int primary_id_lane =
        lane == 0 ? entry_queries[query_start + warp] : 0;
    const int primary_id = __shfl_sync(kFullMask, primary_id_lane, 0);
    const uint8_t* primary_query = queries_u8 + (size_t)primary_id * dim;
    const float primary_norm_lane =
        lane == 0 ? query_norms[primary_id] : 0.0f;
    const float primary_norm = __shfl_sync(kFullMask, primary_norm_lane, 0);
    const float primary_threshold = thresholds[primary_id];

    const int cluster = entry_cluster_id[entry];
    const int padded_start = padded_offsets[cluster];
    const int compact_start = compact_offsets[cluster];
    const int count = counts[cluster];

    const int secondary_slot = warp + 4;
    if (secondary_slot >= query_count) {
        scan_u8_dp4a_pair_entry<false>(
            dim, candidate_capacity, vectors_u8, padded_start, compact_start,
            count, vector_norms, primary_query, primary_norm, primary_threshold,
            primary_id, nullptr, 0.0f, 0.0f, -1, candidate_counts, overflow,
            candidate_dist, candidate_index);
        return;
    }

    const int secondary_id_lane =
        lane == 0 ? entry_queries[query_start + secondary_slot] : 0;
    const int secondary_id = __shfl_sync(kFullMask, secondary_id_lane, 0);
    const uint8_t* secondary_query = queries_u8 + (size_t)secondary_id * dim;
    const float secondary_norm_lane =
        lane == 0 ? query_norms[secondary_id] : 0.0f;
    const float secondary_norm =
        __shfl_sync(kFullMask, secondary_norm_lane, 0);
    const float secondary_threshold = thresholds[secondary_id];
    scan_u8_dp4a_pair_entry<true>(
        dim, candidate_capacity, vectors_u8, padded_start, compact_start, count,
        vector_norms, primary_query, primary_norm, primary_threshold, primary_id,
        secondary_query, secondary_norm, secondary_threshold, secondary_id,
        candidate_counts, overflow, candidate_dist, candidate_index);
}

// Shared-query U8 grouped scan. Identical math/layout to the grouped U8 path,
// but the per-query u8 words are staged in shared memory once per block and
// read as broadcasts instead of being re-derived with one __shfl_sync per d4.
// The shared-query path avoids repeating one shuffle chain per d4 value.
// Opt-in via combo "threshold_grouped_shq"; shape-restricted (dim4<=64).
constexpr int kShqWarps = 8;
constexpr int kShqMaxDim4 = 64;

__global__ __launch_bounds__(256, 1)
void resident_threshold_grouped_shq_scan_kernel(
    int dim,
    int n_entry,
    int candidate_capacity,
    const uint8_t* __restrict__ queries_u8,
    const uint8_t* __restrict__ vectors_u8,
    const int* __restrict__ padded_offsets,
    const int* __restrict__ compact_offsets,
    const int* __restrict__ counts,
    const int* __restrict__ entry_cluster_id,
    const int* __restrict__ entry_query_start,
    const int* __restrict__ entry_query_count,
    const int* __restrict__ entry_queries,
    const float* __restrict__ query_norms,
    const float* __restrict__ vector_norms,
    const float* __restrict__ thresholds,
    int* __restrict__ candidate_counts,
    int* __restrict__ overflow,
    float* __restrict__ candidate_dist,
    idx_t* __restrict__ candidate_index) {
  __shared__ unsigned sq[kShqWarps][kShqMaxDim4];
  const int entry = blockIdx.x;
  if (entry >= n_entry) return;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int query_count = entry_query_count[entry];
  if (warp >= query_count) return;

  const int query_start = entry_query_start[entry];
  const int query_id_lane =
      lane == 0 ? entry_queries[query_start + warp] : 0;
  const int query_id = __shfl_sync(kFullMask, query_id_lane, 0);
  const float query_norm_lane = lane == 0 ? query_norms[query_id] : 0.0f;
  const float query_norm = __shfl_sync(kFullMask, query_norm_lane, 0);
  const float threshold = thresholds[query_id];

  const int dim4 = (dim + 3) / 4;
  const uint8_t* query = queries_u8 + (size_t)query_id * dim;
  for (int d4 = lane; d4 < dim4; d4 += kWarpSize) {
    const int d = d4 * 4;
    const int remaining = dim - d;
    sq[warp][d4] = remaining >= 4
        ? *reinterpret_cast<const unsigned*>(query + d)
        : load_u8x4_threshold(query + d, remaining);
  }
  __syncwarp();

  const int cluster = entry_cluster_id[entry];
  const int padded_start = padded_offsets[cluster];
  const int compact_start = compact_offsets[cluster];
  const int count = counts[cluster];
  const int vector_blocks = (count + kWarpSize - 1) / kWarpSize;
  const size_t bytes_per_block = (size_t)dim4 * kWarpSize * 4;

  for (int block_id = 0; block_id < vector_blocks; ++block_id) {
    const int local = block_id * kWarpSize + lane;
    const int padded = padded_start + local;
    const uint8_t* data = vectors_u8 +
        (size_t)(padded_start / kWarpSize + block_id) * bytes_per_block;
    unsigned dot_u32 = 0;
#pragma unroll 4
    for (int d4 = 0; d4 < dim4; ++d4) {
      const unsigned v4 = *reinterpret_cast<const unsigned*>(
          data + ((size_t)d4 * kWarpSize + lane) * 4);
      dot_u32 = __dp4a(sq[warp][d4], v4, dot_u32);
    }
    const bool valid = local < count;
    const float vector_norm = vector_norms[padded];
    const float dist = query_norm * query_norm + vector_norm * vector_norm -
                       2.0f * static_cast<float>(dot_u32);
    append_threshold_candidates(
        valid && dist <= threshold, dist, compact_start + local, query_id,
        candidate_capacity, candidate_counts, overflow, candidate_dist,
        candidate_index);
  }
}

// Vec4 layout: 4 d4 words per lane are contiguous, so one 16B load feeds four
// dp4a. Keeps the shuffle chain (32 shuffles/vector) for latency hiding.
// Requires dim4 in {4,8,...,32} (dim in [16,128], dim%16==0).
__global__ __launch_bounds__(256, 1)
void resident_threshold_grouped_vec4_scan_kernel(
    int dim,
    int n_entry,
    int candidate_capacity,
    const uint8_t* __restrict__ queries_u8,
    const uint8_t* __restrict__ vectors_u8,
    const int* __restrict__ padded_offsets,
    const int* __restrict__ compact_offsets,
    const int* __restrict__ counts,
    const int* __restrict__ entry_cluster_id,
    const int* __restrict__ entry_query_start,
    const int* __restrict__ entry_query_count,
    const int* __restrict__ entry_queries,
    const float* __restrict__ query_norms,
    const float* __restrict__ vector_norms,
    const float* __restrict__ thresholds,
    int* __restrict__ candidate_counts,
    int* __restrict__ overflow,
    float* __restrict__ candidate_dist,
    idx_t* __restrict__ candidate_index) {
  const int entry = blockIdx.x;
  if (entry >= n_entry) return;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int query_count = entry_query_count[entry];
  if (warp >= query_count) return;

  const int query_start = entry_query_start[entry];
  const int query_id_lane = lane == 0 ? entry_queries[query_start + warp] : 0;
  const int query_id = __shfl_sync(kFullMask, query_id_lane, 0);
  const float query_norm_lane = lane == 0 ? query_norms[query_id] : 0.0f;
  const float query_norm = __shfl_sync(kFullMask, query_norm_lane, 0);
  const float threshold = thresholds[query_id];

  const int dim4 = (dim + 3) / 4;
  const uint8_t* query = queries_u8 + (size_t)query_id * dim;
  unsigned qw = 0;
  if (lane < dim4) {
    const int d = lane * 4;
    const int remaining = dim - d;
    qw = remaining >= 4 ? *reinterpret_cast<const unsigned*>(query + d)
                        : load_u8x4_threshold(query + d, remaining);
  }

  const int cluster = entry_cluster_id[entry];
  const int padded_start = padded_offsets[cluster];
  const int compact_start = compact_offsets[cluster];
  const int count = counts[cluster];
  const int vector_blocks = (count + kWarpSize - 1) / kWarpSize;
  const size_t bytes_per_block = (size_t)dim4 * kWarpSize * 4;
  for (int block_id = 0; block_id < vector_blocks; ++block_id) {
    const int local = block_id * kWarpSize + lane;
    const int padded = padded_start + local;
    const uint8_t* data = vectors_u8 +
        (size_t)(padded_start / kWarpSize + block_id) * bytes_per_block;
    unsigned dot_u32 = 0;
#pragma unroll
    for (int g = 0; g < dim4 / 4; ++g) {
      const uint4 u =
          *reinterpret_cast<const uint4*>(data + ((size_t)g * 32 + lane) * 16);
      dot_u32 = __dp4a(__shfl_sync(kFullMask, qw, g * 4 + 0), u.x, dot_u32);
      dot_u32 = __dp4a(__shfl_sync(kFullMask, qw, g * 4 + 1), u.y, dot_u32);
      dot_u32 = __dp4a(__shfl_sync(kFullMask, qw, g * 4 + 2), u.z, dot_u32);
      dot_u32 = __dp4a(__shfl_sync(kFullMask, qw, g * 4 + 3), u.w, dot_u32);
    }
    const bool valid = local < count;
    const float vector_norm = vector_norms[padded];
    const float dist = query_norm * query_norm + vector_norm * vector_norm -
                       2.0f * static_cast<float>(dot_u32);
    append_threshold_candidates(
        valid && dist <= threshold, dist, compact_start + local, query_id,
        candidate_capacity, candidate_counts, overflow, candidate_dist,
        candidate_index);
  }
}

// Vec4 layout + shared-query broadcast (removes the 32 shuffles/vector).
__global__ __launch_bounds__(256, 1)
void resident_threshold_grouped_vec4_shq_scan_kernel(
    int dim,
    int n_entry,
    int candidate_capacity,
    const uint8_t* __restrict__ queries_u8,
    const uint8_t* __restrict__ vectors_u8,
    const int* __restrict__ padded_offsets,
    const int* __restrict__ compact_offsets,
    const int* __restrict__ counts,
    const int* __restrict__ entry_cluster_id,
    const int* __restrict__ entry_query_start,
    const int* __restrict__ entry_query_count,
    const int* __restrict__ entry_queries,
    const float* __restrict__ query_norms,
    const float* __restrict__ vector_norms,
    const float* __restrict__ thresholds,
    int* __restrict__ candidate_counts,
    int* __restrict__ overflow,
    float* __restrict__ candidate_dist,
    idx_t* __restrict__ candidate_index) {
  __shared__ unsigned sq[kShqWarps][kShqMaxDim4];
  const int entry = blockIdx.x;
  if (entry >= n_entry) return;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int query_count = entry_query_count[entry];
  if (warp >= query_count) return;

  const int query_start = entry_query_start[entry];
  const int query_id_lane = lane == 0 ? entry_queries[query_start + warp] : 0;
  const int query_id = __shfl_sync(kFullMask, query_id_lane, 0);
  const float query_norm_lane = lane == 0 ? query_norms[query_id] : 0.0f;
  const float query_norm = __shfl_sync(kFullMask, query_norm_lane, 0);
  const float threshold = thresholds[query_id];

  const int dim4 = (dim + 3) / 4;
  const uint8_t* query = queries_u8 + (size_t)query_id * dim;
  for (int d4 = lane; d4 < dim4; d4 += kWarpSize) {
    const int d = d4 * 4;
    const int remaining = dim - d;
    sq[warp][d4] = remaining >= 4
        ? *reinterpret_cast<const unsigned*>(query + d)
        : load_u8x4_threshold(query + d, remaining);
  }
  __syncwarp();

  const int cluster = entry_cluster_id[entry];
  const int padded_start = padded_offsets[cluster];
  const int compact_start = compact_offsets[cluster];
  const int count = counts[cluster];
  const int vector_blocks = (count + kWarpSize - 1) / kWarpSize;
  const size_t bytes_per_block = (size_t)dim4 * kWarpSize * 4;
  for (int block_id = 0; block_id < vector_blocks; ++block_id) {
    const int local = block_id * kWarpSize + lane;
    const int padded = padded_start + local;
    const uint8_t* data = vectors_u8 +
        (size_t)(padded_start / kWarpSize + block_id) * bytes_per_block;
    unsigned dot_u32 = 0;
#pragma unroll
    for (int g = 0; g < dim4 / 4; ++g) {
      const uint4 u =
          *reinterpret_cast<const uint4*>(data + ((size_t)g * 32 + lane) * 16);
      dot_u32 = __dp4a(sq[warp][g * 4 + 0], u.x, dot_u32);
      dot_u32 = __dp4a(sq[warp][g * 4 + 1], u.y, dot_u32);
      dot_u32 = __dp4a(sq[warp][g * 4 + 2], u.z, dot_u32);
      dot_u32 = __dp4a(sq[warp][g * 4 + 3], u.w, dot_u32);
    }
    const bool valid = local < count;
    const float vector_norm = vector_norms[padded];
    const float dist = query_norm * query_norm + vector_norm * vector_norm -
                       2.0f * static_cast<float>(dot_u32);
    append_threshold_candidates(
        valid && dist <= threshold, dist, compact_start + local, query_id,
        candidate_capacity, candidate_counts, overflow, candidate_dist,
        candidate_index);
  }
}

// MLP2: each lane owns two vectors from two adjacent 32-vector tiles and keeps
// two independent accumulators, so each d4 step issues two independent global
// loads. Increases in-flight loads per warp (memory-level parallelism) instead
// of reducing instruction count. Same math and layout as the grouped U8 path.
__global__ __launch_bounds__(256, 1)
void resident_threshold_grouped_mlp2_scan_kernel(
    int dim,
    int n_entry,
    int candidate_capacity,
    const uint8_t* __restrict__ queries_u8,
    const uint8_t* __restrict__ vectors_u8,
    const int* __restrict__ padded_offsets,
    const int* __restrict__ compact_offsets,
    const int* __restrict__ counts,
    const int* __restrict__ entry_cluster_id,
    const int* __restrict__ entry_query_start,
    const int* __restrict__ entry_query_count,
    const int* __restrict__ entry_queries,
    const float* __restrict__ query_norms,
    const float* __restrict__ vector_norms,
    const float* __restrict__ thresholds,
    int* __restrict__ candidate_counts,
    int* __restrict__ overflow,
    float* __restrict__ candidate_dist,
    idx_t* __restrict__ candidate_index) {
  const int entry = blockIdx.x;
  if (entry >= n_entry) return;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int query_count = entry_query_count[entry];
  if (warp >= query_count) return;

  const int query_start = entry_query_start[entry];
  const int query_id_lane = lane == 0 ? entry_queries[query_start + warp] : 0;
  const int query_id = __shfl_sync(kFullMask, query_id_lane, 0);
  const float query_norm_lane = lane == 0 ? query_norms[query_id] : 0.0f;
  const float query_norm = __shfl_sync(kFullMask, query_norm_lane, 0);
  const float threshold = thresholds[query_id];
  const uint8_t* query = queries_u8 + (size_t)query_id * dim;

  const int cluster = entry_cluster_id[entry];
  const int padded_start = padded_offsets[cluster];
  const int compact_start = compact_offsets[cluster];
  const int count = counts[cluster];
  const int dim4 = (dim + 3) / 4;
  const size_t bytes_per_block = (size_t)dim4 * kWarpSize * 4;
  const uint8_t* cluster_base =
      vectors_u8 + (size_t)(padded_start / kWarpSize) * bytes_per_block;
  const int padded_tiles = (count + kWarpSize - 1) / kWarpSize;
  const int tile_pairs = (padded_tiles + 1) / 2;

  for (int pair = 0; pair < tile_pairs; ++pair) {
    const int tile_a = pair * 2;
    const int tile_b = pair * 2 + 1;
    const bool has_b = tile_b < padded_tiles;
    const int local_a = tile_a * kWarpSize + lane;
    const int local_b = tile_b * kWarpSize + lane;
    const uint8_t* data_a = cluster_base + (size_t)tile_a * bytes_per_block;
    const uint8_t* data_b = cluster_base + (size_t)tile_b * bytes_per_block;
    unsigned dot_a = 0;
    unsigned dot_b = 0;
    for (int d4_base = 0; d4_base < dim4; d4_base += kWarpSize) {
      const int lane_d4 = d4_base + lane;
      unsigned q4_lane = 0;
      if (lane_d4 < dim4) {
        const int d = lane_d4 * 4;
        const int remaining = dim - d;
        q4_lane = remaining >= 4
            ? *reinterpret_cast<const unsigned*>(query + d)
            : load_u8x4_threshold(query + d, remaining);
      }
      const int limit = min(kWarpSize, dim4 - d4_base);
#pragma unroll
      for (int j = 0; j < limit; ++j) {
        const int d4 = d4_base + j;
        const unsigned q4 = __shfl_sync(kFullMask, q4_lane, j);
        const unsigned v4a = *reinterpret_cast<const unsigned*>(
            data_a + ((size_t)d4 * kWarpSize + lane) * 4);
        dot_a = __dp4a(q4, v4a, dot_a);
        if (has_b) {
          const unsigned v4b = *reinterpret_cast<const unsigned*>(
              data_b + ((size_t)d4 * kWarpSize + lane) * 4);
          dot_b = __dp4a(q4, v4b, dot_b);
        }
      }
    }
    {
      const bool valid = local_a < count;
      const int padded = padded_start + local_a;
      const float vn = valid ? vector_norms[padded] : 0.0f;
      const float dist = query_norm * query_norm + vn * vn -
                         2.0f * static_cast<float>(dot_a);
      append_threshold_candidates(
          valid && dist <= threshold, dist, compact_start + local_a, query_id,
          candidate_capacity, candidate_counts, overflow, candidate_dist,
          candidate_index);
    }
    if (has_b) {
      const bool valid = local_b < count;
      const int padded = padded_start + local_b;
      const float vn = valid ? vector_norms[padded] : 0.0f;
      const float dist = query_norm * query_norm + vn * vn -
                         2.0f * static_cast<float>(dot_b);
      append_threshold_candidates(
          valid && dist <= threshold, dist, compact_start + local_b, query_id,
          candidate_capacity, candidate_counts, overflow, candidate_dist,
          candidate_index);
    }
  }
}

__device__ __forceinline__ void resident_cp_async16(void* smem, const void* gmem) {
  const unsigned saddr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(saddr), "l"(gmem));
}
__device__ __forceinline__ void resident_cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void resident_cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n");
}

// cp.async double-buffered grouped scan. The cluster's 32-vector tiles are
// staged global->shared asynchronously (256 threads x 16B per tile) while the
// previous tile is reduced, hiding global latency that the compiler cannot hide
// on its own. Same math/layout as the grouped U8 path; u8 only.
__global__ __launch_bounds__(256, 1)
void resident_threshold_grouped_cpasync_scan_kernel(
    int dim,
    int n_entry,
    int candidate_capacity,
    const uint8_t* __restrict__ queries_u8,
    const uint8_t* __restrict__ vectors_u8,
    const int* __restrict__ padded_offsets,
    const int* __restrict__ compact_offsets,
    const int* __restrict__ counts,
    const int* __restrict__ entry_cluster_id,
    const int* __restrict__ entry_query_start,
    const int* __restrict__ entry_query_count,
    const int* __restrict__ entry_queries,
    const float* __restrict__ query_norms,
    const float* __restrict__ vector_norms,
    const float* __restrict__ thresholds,
    int* __restrict__ candidate_counts,
    int* __restrict__ overflow,
    float* __restrict__ candidate_dist,
    idx_t* __restrict__ candidate_index) {
  extern __shared__ uint8_t smem[];
  const int entry = blockIdx.x;
  if (entry >= n_entry) return;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int tid = threadIdx.x;
  const int query_count = entry_query_count[entry];
  const bool active = warp < query_count;

  const int query_start = entry_query_start[entry];
  int query_id = 0;
  float query_norm = 0.0f;
  float threshold = 0.0f;
  const uint8_t* query = nullptr;
  if (active) {
    const int query_id_lane = lane == 0 ? entry_queries[query_start + warp] : 0;
    query_id = __shfl_sync(kFullMask, query_id_lane, 0);
    const float query_norm_lane = lane == 0 ? query_norms[query_id] : 0.0f;
    query_norm = __shfl_sync(kFullMask, query_norm_lane, 0);
    threshold = thresholds[query_id];
    query = queries_u8 + (size_t)query_id * dim;
  }

  const int cluster = entry_cluster_id[entry];
  const int padded_start = padded_offsets[cluster];
  const int compact_start = compact_offsets[cluster];
  const int count = counts[cluster];
  const int dim4 = (dim + 3) / 4;
  const size_t tile_bytes = (size_t)dim4 * kWarpSize * 4;
  const uint8_t* cluster_base =
      vectors_u8 + (size_t)(padded_start / kWarpSize) * tile_bytes;
  const int padded_tiles = (count + kWarpSize - 1) / kWarpSize;
  if (padded_tiles <= 0) return;

  auto copy_tile = [&](uint8_t* dst, int tile) {
    const uint8_t* src = cluster_base + (size_t)tile * tile_bytes;
    for (size_t off = (size_t)tid * 16; off < tile_bytes;
         off += (size_t)blockDim.x * 16) {
      resident_cp_async16(dst + off, src + off);
    }
  };

  copy_tile(smem, 0);
  resident_cp_async_commit();
  resident_cp_async_wait_all();
  __syncthreads();

  for (int t = 0; t < padded_tiles; ++t) {
    const int cur = t & 1;
    if (t + 1 < padded_tiles) {
      copy_tile(smem + (size_t)(cur ^ 1) * tile_bytes, t + 1);
      resident_cp_async_commit();
    }
    if (active) {
      const uint8_t* tile_smem = smem + (size_t)cur * tile_bytes;
      const int local = t * kWarpSize + lane;
      unsigned dot_u32 = 0;
      for (int d4_base = 0; d4_base < dim4; d4_base += kWarpSize) {
        const int lane_d4 = d4_base + lane;
        unsigned q4_lane = 0;
        if (lane_d4 < dim4) {
          const int d = lane_d4 * 4;
          const int remaining = dim - d;
          q4_lane = remaining >= 4
              ? *reinterpret_cast<const unsigned*>(query + d)
              : load_u8x4_threshold(query + d, remaining);
        }
        const int limit = min(kWarpSize, dim4 - d4_base);
#pragma unroll
        for (int j = 0; j < limit; ++j) {
          const int d4 = d4_base + j;
          const unsigned q4 = __shfl_sync(kFullMask, q4_lane, j);
          const unsigned v4 = *reinterpret_cast<const unsigned*>(
              tile_smem + ((size_t)d4 * kWarpSize + lane) * 4);
          dot_u32 = __dp4a(q4, v4, dot_u32);
        }
      }
      const bool valid = local < count;
      const float vn = valid ? vector_norms[padded_start + local] : 0.0f;
      const float dist = query_norm * query_norm + vn * vn -
                         2.0f * static_cast<float>(dot_u32);
      append_threshold_candidates(
          valid && dist <= threshold, dist, compact_start + local, query_id,
          candidate_capacity, candidate_counts, overflow, candidate_dist,
          candidate_index);
    }
    if (t + 1 < padded_tiles) resident_cp_async_wait_all();
    __syncthreads();
  }
}

__global__ __launch_bounds__(128, 8)
void resident_pca_i8_dp4a_v3_threshold_grouped_scan_kernel(
    int dim,
    int n_entry,
    int candidate_capacity,
    const int8_t* __restrict__ queries,
    const float* __restrict__ query_dot_scales,
    const int8_t* __restrict__ vectors,
    const int* __restrict__ padded_offsets,
    const int* __restrict__ compact_offsets,
    const int* __restrict__ counts,
    const int* __restrict__ entry_cluster_id,
    const int* __restrict__ entry_query_start,
    const int* __restrict__ entry_query_count,
    const int* __restrict__ entry_queries,
    const float* __restrict__ query_norms,
    const float* __restrict__ vector_norms,
    const float* __restrict__ thresholds,
    int* __restrict__ candidate_counts,
    int* __restrict__ overflow,
    float* __restrict__ candidate_dist,
    idx_t* __restrict__ candidate_index) {
    const int entry = blockIdx.x;
    if (entry >= n_entry) return;
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int query_count = entry_query_count[entry];
    if (warp >= query_count) return;

    const int query_start = entry_query_start[entry];
    const int primary_id_lane =
        lane == 0 ? entry_queries[query_start + warp] : 0;
    const int primary_id = __shfl_sync(kFullMask, primary_id_lane, 0);
    const int secondary_slot = warp + 4;
    const bool has_secondary = secondary_slot < query_count;
    const int dim4 = dim / 4;
    const int8_t* primary_query = queries + (size_t)primary_id * dim;
    const int primary_query4_lane = lane < dim4
        ? reinterpret_cast<const int*>(primary_query)[lane]
        : 0;
    const float primary_norm_lane =
        lane == 0 ? query_norms[primary_id] : 0.0f;
    const float primary_norm =
        __shfl_sync(kFullMask, primary_norm_lane, 0);
    const float primary_scale_lane =
        lane == 0 ? query_dot_scales[primary_id] : 0.0f;
    const float primary_scale =
        __shfl_sync(kFullMask, primary_scale_lane, 0);
    const float primary_threshold = thresholds[primary_id];

    const int cluster = entry_cluster_id[entry];
    const int padded_start = padded_offsets[cluster];
    const int compact_start = compact_offsets[cluster];
    const int count = counts[cluster];
    const PCAI8ThresholdQueryState primary{
        primary_id, primary_query4_lane, primary_norm, primary_scale,
        primary_threshold};
    if (!has_secondary) {
        scan_pca_i8_dp4a_v3_entry<false>(
            dim, candidate_capacity, vectors, padded_start, compact_start,
            count, vector_norms, primary, {}, candidate_counts, overflow,
            candidate_dist, candidate_index);
        return;
    }

    const int secondary_id_lane = lane == 0
        ? entry_queries[query_start + secondary_slot]
        : 0;
    const int secondary_id =
        __shfl_sync(kFullMask, secondary_id_lane, 0);
    const int8_t* secondary_query = queries + (size_t)secondary_id * dim;
    const int secondary_query4_lane = lane < dim4
        ? reinterpret_cast<const int*>(secondary_query)[lane]
        : 0;
    const float secondary_norm_lane =
        lane == 0 ? query_norms[secondary_id] : 0.0f;
    const float secondary_norm =
        __shfl_sync(kFullMask, secondary_norm_lane, 0);
    const float secondary_scale_lane =
        lane == 0 ? query_dot_scales[secondary_id] : 0.0f;
    const float secondary_scale =
        __shfl_sync(kFullMask, secondary_scale_lane, 0);
    const float secondary_threshold = thresholds[secondary_id];
    const PCAI8ThresholdQueryState secondary{
        secondary_id, secondary_query4_lane, secondary_norm, secondary_scale,
        secondary_threshold};
    scan_pca_i8_dp4a_v3_entry<true>(
        dim, candidate_capacity, vectors, padded_start, compact_start, count,
        vector_norms, primary, secondary, candidate_counts, overflow,
        candidate_dist, candidate_index);
}

template <bool UseU8Dp4a>
__global__ __launch_bounds__(256, 1) void resident_threshold_query_scan_kernel(
    int dim,
    int n_tasks,
    int n_probes,
    int candidate_capacity,
    const float* __restrict__ queries,
    const uint8_t* __restrict__ queries_u8,
    const float* __restrict__ vectors,
    const uint8_t* __restrict__ vectors_u8,
    const int* __restrict__ padded_offsets,
    const int* __restrict__ compact_offsets,
    const int* __restrict__ counts,
    const int* __restrict__ task_row,
    const int* __restrict__ task_cluster,
    const int* __restrict__ task_first_block,
    const int* __restrict__ task_num_blocks,
    const float* __restrict__ query_norms,
    const float* __restrict__ vector_norms,
    const float* __restrict__ thresholds,
    int* __restrict__ candidate_counts,
    int* __restrict__ overflow,
    float* __restrict__ candidate_dist,
    idx_t* __restrict__ candidate_index) {
    constexpr int kWarpsPerBlock = 8;
    const int task = blockIdx.x;
    if (task >= n_tasks) return;
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    if (warp >= kWarpsPerBlock) return;

    const int row = task_row[task];
    const int query_id = row / n_probes;
    const int cluster = task_cluster[task];
    const int padded_start = padded_offsets[cluster];
    const int compact_start = compact_offsets[cluster];
    const int count = counts[cluster];
    const int first_block = task_first_block[task];
    const int num_blocks = task_num_blocks[task];
    const float query_norm = query_norms[query_id];
    const float threshold = thresholds[query_id];

    for (int chunk_block = warp; chunk_block < num_blocks;
         chunk_block += kWarpsPerBlock) {
        const int block_id = first_block + chunk_block;
        const int local = block_id * kWarpSize + lane;
        const int padded = padded_start + local;
        float dot = 0.0f;
        if constexpr (UseU8Dp4a) {
            const uint8_t* query = queries_u8 + (size_t)query_id * dim;
            const int dim4 = (dim + 3) / 4;
            const size_t bytes_per_block = (size_t)dim4 * kWarpSize * 4;
            const uint8_t* data =
                vectors_u8 +
                (size_t)(padded_start / kWarpSize + block_id) *
                    bytes_per_block;
            unsigned dot_u32 = 0;
            for (int d4_base = 0; d4_base < dim4;
                 d4_base += kWarpSize) {
                const int lane_d4 = d4_base + lane;
                unsigned q4_lane = 0;
                if (lane_d4 < dim4) {
                    const int d = lane_d4 * 4;
                    const int remaining = dim - d;
                    q4_lane =
                        remaining >= 4
                            ? *reinterpret_cast<const unsigned*>(query + d)
                            : load_u8x4_threshold(query + d, remaining);
                }
                const int limit = min(kWarpSize, dim4 - d4_base);
                for (int j = 0; j < limit; ++j) {
                    const unsigned q4 =
                        __shfl_sync(kFullMask, q4_lane, j);
                    const int d4 = d4_base + j;
                    const unsigned v4 =
                        *reinterpret_cast<const unsigned*>(
                            data +
                            ((size_t)d4 * kWarpSize + lane) * 4);
                    dot_u32 = __dp4a(q4, v4, dot_u32);
                }
            }
            dot = (float)dot_u32;
        } else {
            const float* query = queries + (size_t)query_id * dim;
            const size_t values_per_block = (size_t)kWarpSize * dim;
            const float* data =
                vectors +
                (size_t)(padded_start / kWarpSize + block_id) *
                    values_per_block;
            const int dim_blocks = (dim / kWarpSize) * kWarpSize;
            for (int d_base = 0; d_base < dim_blocks;
                 d_base += kWarpSize) {
                const float qreg = query[d_base + lane];
                for (int i = 0; i < kWarpSize / 4;
                     ++i, data += 4 * kWarpSize) {
                    dot +=
                        __shfl_sync(kFullMask, qreg, i * 4 + 0) *
                        data[0 * kWarpSize + lane];
                    dot +=
                        __shfl_sync(kFullMask, qreg, i * 4 + 1) *
                        data[1 * kWarpSize + lane];
                    dot +=
                        __shfl_sync(kFullMask, qreg, i * 4 + 2) *
                        data[2 * kWarpSize + lane];
                    dot +=
                        __shfl_sync(kFullMask, qreg, i * 4 + 3) *
                        data[3 * kWarpSize + lane];
                }
            }
            if (dim_blocks < dim) {
                const int dload = dim_blocks + lane;
                const float qreg =
                    dload < dim ? query[dload] : 0.0f;
                for (int d = 0; d < dim - dim_blocks;
                     ++d, data += kWarpSize)
                    dot += __shfl_sync(kFullMask, qreg, d) *
                           data[lane];
            }
        }

        const bool valid = local < count;
        const float vector_norm = valid ? vector_norms[padded] : 0.0f;
        const float dist = query_norm * query_norm +
                           vector_norm * vector_norm - 2.0f * dot;
        const unsigned mask =
            __ballot_sync(kFullMask, valid && dist <= threshold);
        if (mask == 0) continue;
        const int leader = __ffs(mask) - 1;
        int output_base = 0;
        if (lane == leader)
            output_base =
                atomicAdd(candidate_counts + query_id, __popc(mask));
        output_base =
            __shfl_sync(kFullMask, output_base, leader);
        if ((mask & (1u << lane)) == 0) continue;
        const unsigned lower =
            lane == 0 ? 0u : mask & ((1u << lane) - 1u);
        const int output = output_base + __popc(lower);
        if (output < candidate_capacity) {
            const size_t dst =
                (size_t)query_id * candidate_capacity + output;
            candidate_dist[dst] = dist;
            candidate_index[dst] = compact_start + local;
        } else {
            atomicExch(overflow, 1);
        }
    }
}

template <int Capacity, int WarpsPerBlock>
__global__ __launch_bounds__(256, 1) void resident_threshold_reduce_kernel(
    int n_query,
    int candidate_capacity,
    int n_total_vectors,
    const int* __restrict__ candidate_counts,
    const float* __restrict__ candidate_dist,
    const idx_t* __restrict__ candidate_index,
    const idx_t* __restrict__ reordered_indices,
    int k,
    float* __restrict__ final_dist,
    idx_t* __restrict__ final_index) {
    const int query_id = blockIdx.x;
    if (query_id >= n_query) return;
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    if (warp >= WarpsPerBlock) return;

    using Sort = paraselect::warpsort::WarpSort<Capacity, true, float, idx_t>;
    const float dummy = Sort::kDummy();
    paraselect::warpsort::WarpSortFiltered<Capacity, true, float, idx_t> queue(k);
    const int count = min(candidate_counts[query_id], candidate_capacity);
    const size_t base = (size_t)query_id * candidate_capacity;
    for (int round_base = warp * kWarpSize; round_base < count;
         round_base += WarpsPerBlock * kWarpSize) {
        const int i = round_base + lane;
        const bool valid = i < count;
        const idx_t id = valid ? candidate_index[base + i] : (idx_t)-1;
        const float dist = valid ? candidate_dist[base + i] : dummy;
        const bool keep = valid && id >= 0 && dist < queue.threshold();
        queue.add(keep ? dist : dummy, keep ? id : (idx_t)-1);
    }
    __syncwarp();
    queue.done();
    __syncwarp();

    extern __shared__ unsigned char shared[];
    float* warp_dist = reinterpret_cast<float*>(shared);
    idx_t* warp_index =
        reinterpret_cast<idx_t*>(warp_dist + WarpsPerBlock * Capacity);
    queue.store(
        warp_dist + (size_t)warp * Capacity,
        warp_index + (size_t)warp * Capacity);
    __syncthreads();

    if (warp != 0) return;
    paraselect::warpsort::WarpSortFiltered<Capacity, true, float, idx_t>
        merge_queue(k);
    for (int source_warp = 0; source_warp < WarpsPerBlock; ++source_warp) {
        for (int rank_base = 0; rank_base < k; rank_base += kWarpSize) {
            const int rank = rank_base + lane;
            float dist = dummy;
            idx_t id = (idx_t)-1;
            if (rank < k) {
                dist = warp_dist[(size_t)source_warp * Capacity + rank];
                id = warp_index[(size_t)source_warp * Capacity + rank];
            }
            const bool keep = id >= 0 && dist < merge_queue.threshold();
            merge_queue.add(keep ? dist : dummy, keep ? id : (idx_t)-1);
        }
    }
    __syncwarp();
    merge_queue.done();
    __syncwarp();
    merge_queue.store(
        warp_dist,
        warp_index);
    __syncwarp();
    for (int rank = lane; rank < k; rank += kWarpSize) {
        const idx_t compact = warp_index[rank];
        final_dist[(size_t)query_id * k + rank] = warp_dist[rank];
        final_index[(size_t)query_id * k + rank] =
            compact >= 0 && compact < n_total_vectors
                ? (reordered_indices ? reordered_indices[compact] : compact)
                : (idx_t)-1;
    }
}

} // namespace

void launch_resident_threshold_count_seed_entries(
    const int* d_cluster_offsets,
    const int* d_probe_indices,
    int nlist,
    int seed_probes,
    int* d_entry_counts,
    cudaStream_t stream) {
    const int block = 256;
    resident_threshold_count_seed_entries_kernel<<<(nlist + block - 1) / block, block, 0, stream>>>(
        d_cluster_offsets, d_probe_indices, nlist, seed_probes,
        d_entry_counts);
}

void launch_resident_threshold_fill_seed_entries(
    const int* d_cluster_offsets,
    const int* d_cluster_queries,
    const int* d_probe_indices,
    const int* d_entry_offsets,
    int nlist,
    int seed_probes,
    int* d_entry_cluster_id,
    int* d_entry_query_start,
    int* d_entry_query_count,
    int* d_entry_queries,
    int* d_entry_probe_indices,
    cudaStream_t stream) {
    const int block = 256;
    resident_threshold_fill_seed_entries_kernel<<<(nlist + block - 1) / block, block, 0, stream>>>(
        d_cluster_offsets, d_cluster_queries, d_probe_indices,
        d_entry_offsets, nlist, seed_probes, d_entry_cluster_id,
        d_entry_query_start, d_entry_query_count, d_entry_queries,
        d_entry_probe_indices);
}

void launch_resident_threshold_from_probe_topk(
    int k,
    int n_query,
    int n_probes,
    int seed_probes,
    const float* d_probe_topk_dist,
    float* d_thresholds,
    cudaStream_t stream) {
    if (k <= 0 || k > 256)
        throw std::invalid_argument(
            "resident threshold IVFFlat supports topk 1..256");
    if (n_query <= 0) return;
    if (k <= 64) {
        resident_threshold_from_probe_topk_kernel<64><<<n_query, 1, 0, stream>>>(
            d_probe_topk_dist, n_query, n_probes, seed_probes, k,
            d_thresholds);
    } else if (k <= 128) {
        resident_threshold_from_probe_topk_kernel<128><<<n_query, 1, 0, stream>>>(
            d_probe_topk_dist, n_query, n_probes, seed_probes, k,
            d_thresholds);
    } else {
        resident_threshold_from_probe_topk_kernel<256><<<n_query, 1, 0, stream>>>(
            d_probe_topk_dist, n_query, n_probes, seed_probes, k,
            d_thresholds);
    }
}
void launch_resident_threshold_query_scan(
    int dim,
    int n_tasks,
    int n_probes,
    int candidate_capacity,
    const float* d_queries,
    const uint8_t* d_queries_u8,
    const float* d_vectors,
    const uint8_t* d_vectors_u8_dp4a,
    const int* d_padded_offsets,
    const int* d_compact_offsets,
    const int* d_counts,
    const int* d_task_row,
    const int* d_task_cluster,
    const int* d_task_first_block,
    const int* d_task_num_blocks,
    const float* d_query_norms,
    const float* d_vector_norms,
    const float* d_thresholds,
    bool use_u8_dp4a,
    int* d_candidate_counts,
    int* d_overflow,
    float* d_candidate_dist,
    idx_t* d_candidate_index,
    cudaStream_t stream) {
    if (n_tasks <= 0) return;
    if (use_u8_dp4a) {
        resident_threshold_query_scan_kernel<true>
            <<<n_tasks, 256, 0, stream>>>(
                dim, n_tasks, n_probes, candidate_capacity, d_queries,
                d_queries_u8, d_vectors, d_vectors_u8_dp4a,
                d_padded_offsets, d_compact_offsets, d_counts, d_task_row,
                d_task_cluster, d_task_first_block, d_task_num_blocks,
                d_query_norms, d_vector_norms, d_thresholds,
                d_candidate_counts, d_overflow, d_candidate_dist,
                d_candidate_index);
    } else {
        resident_threshold_query_scan_kernel<false>
            <<<n_tasks, 256, 0, stream>>>(
                dim, n_tasks, n_probes, candidate_capacity, d_queries,
                d_queries_u8, d_vectors, d_vectors_u8_dp4a,
                d_padded_offsets, d_compact_offsets, d_counts, d_task_row,
                d_task_cluster, d_task_first_block, d_task_num_blocks,
                d_query_norms, d_vector_norms, d_thresholds,
                d_candidate_counts, d_overflow, d_candidate_dist,
                d_candidate_index);
    }
}

void launch_resident_threshold_grouped_scan(
    int dim,
    int n_entry,
    int candidate_capacity,
    const float* d_queries,
    const uint8_t* d_queries_u8,
    const float* d_vectors,
    const uint8_t* d_vectors_u8_dp4a,
    const int* d_padded_offsets,
    const int* d_compact_offsets,
    const int* d_counts,
    const int* d_entry_cluster_id,
    const int* d_entry_query_start,
    const int* d_entry_query_count,
    const int* d_entry_queries,
    const float* d_query_norms,
    const float* d_vector_norms,
    const float* d_thresholds,
    bool use_u8_dp4a,
    int* d_candidate_counts,
    int* d_overflow,
    float* d_candidate_dist,
    idx_t* d_candidate_index,
    cudaStream_t stream) {
    if (n_entry <= 0) return;
    if (use_u8_dp4a) {
        resident_threshold_grouped_scan_kernel<ResidentDenseDistanceKind::U8Dp4a><<<n_entry, 256, 0, stream>>>(
            dim, n_entry, candidate_capacity, d_queries, d_queries_u8,
            d_vectors, d_vectors_u8_dp4a, d_padded_offsets,
            d_compact_offsets, d_counts, d_entry_cluster_id,
            d_entry_query_start, d_entry_query_count, d_entry_queries,
            d_query_norms, nullptr, d_vector_norms, d_thresholds,
            d_candidate_counts, d_overflow, d_candidate_dist,
            d_candidate_index);
    } else {
        resident_threshold_grouped_scan_kernel<ResidentDenseDistanceKind::FP32><<<n_entry, 256, 0, stream>>>(
            dim, n_entry, candidate_capacity, d_queries, d_queries_u8,
            d_vectors, d_vectors_u8_dp4a, d_padded_offsets,
            d_compact_offsets, d_counts, d_entry_cluster_id,
            d_entry_query_start, d_entry_query_count, d_entry_queries,
            d_query_norms, nullptr, d_vector_norms, d_thresholds,
            d_candidate_counts, d_overflow, d_candidate_dist,
            d_candidate_index);
    }
}

void launch_resident_threshold_grouped_u8_pair_scan(
    int dim,
    int n_entry,
    int candidate_capacity,
    const uint8_t* d_queries_u8,
    const uint8_t* d_vectors_u8_dp4a,
    const int* d_padded_offsets,
    const int* d_compact_offsets,
    const int* d_counts,
    const int* d_entry_cluster_id,
    const int* d_entry_query_start,
    const int* d_entry_query_count,
    const int* d_entry_queries,
    const float* d_query_norms,
    const float* d_vector_norms,
    const float* d_thresholds,
    int* d_candidate_counts,
    int* d_overflow,
    float* d_candidate_dist,
    idx_t* d_candidate_index,
    cudaStream_t stream) {
    if (n_entry <= 0) return;
    resident_threshold_grouped_u8_pair_scan_kernel<<<n_entry, 128, 0, stream>>>(
        dim, n_entry, candidate_capacity, d_queries_u8, d_vectors_u8_dp4a,
        d_padded_offsets, d_compact_offsets, d_counts, d_entry_cluster_id,
        d_entry_query_start, d_entry_query_count, d_entry_queries, d_query_norms,
        d_vector_norms, d_thresholds, d_candidate_counts, d_overflow,
        d_candidate_dist, d_candidate_index);
}

bool launch_resident_threshold_grouped_shq_scan(
    int dim,
    int n_entry,
    int candidate_capacity,
    const uint8_t* d_queries_u8,
    const uint8_t* d_vectors_u8_dp4a,
    const int* d_padded_offsets,
    const int* d_compact_offsets,
    const int* d_counts,
    const int* d_entry_cluster_id,
    const int* d_entry_query_start,
    const int* d_entry_query_count,
    const int* d_entry_queries,
    const float* d_query_norms,
    const float* d_vector_norms,
    const float* d_thresholds,
    int* d_candidate_counts,
    int* d_overflow,
    float* d_candidate_dist,
    idx_t* d_candidate_index,
    cudaStream_t stream) {
    if (n_entry <= 0) return true;
    if ((dim + 3) / 4 > kShqMaxDim4) return false;
    resident_threshold_grouped_shq_scan_kernel<<<n_entry, 256, 0, stream>>>(
        dim, n_entry, candidate_capacity, d_queries_u8, d_vectors_u8_dp4a,
        d_padded_offsets, d_compact_offsets, d_counts, d_entry_cluster_id,
        d_entry_query_start, d_entry_query_count, d_entry_queries, d_query_norms,
        d_vector_norms, d_thresholds, d_candidate_counts, d_overflow,
        d_candidate_dist, d_candidate_index);
    return true;
}

bool launch_resident_threshold_grouped_vec4_scan(
    int dim,
    int n_entry,
    int candidate_capacity,
    const uint8_t* d_queries_u8,
    const uint8_t* d_vectors_u8_dp4a,
    const int* d_padded_offsets,
    const int* d_compact_offsets,
    const int* d_counts,
    const int* d_entry_cluster_id,
    const int* d_entry_query_start,
    const int* d_entry_query_count,
    const int* d_entry_queries,
    const float* d_query_norms,
    const float* d_vector_norms,
    const float* d_thresholds,
    int* d_candidate_counts,
    int* d_overflow,
    float* d_candidate_dist,
    idx_t* d_candidate_index,
    cudaStream_t stream) {
    if (n_entry <= 0) return true;
    const int dim4 = (dim + 3) / 4;
    if (dim4 <= 0 || dim4 > 32 || (dim4 % 4) != 0) return false;
    resident_threshold_grouped_vec4_scan_kernel<<<n_entry, 256, 0, stream>>>(
        dim, n_entry, candidate_capacity, d_queries_u8, d_vectors_u8_dp4a,
        d_padded_offsets, d_compact_offsets, d_counts, d_entry_cluster_id,
        d_entry_query_start, d_entry_query_count, d_entry_queries, d_query_norms,
        d_vector_norms, d_thresholds, d_candidate_counts, d_overflow,
        d_candidate_dist, d_candidate_index);
    return true;
}

bool launch_resident_threshold_grouped_vec4_shq_scan(
    int dim,
    int n_entry,
    int candidate_capacity,
    const uint8_t* d_queries_u8,
    const uint8_t* d_vectors_u8_dp4a,
    const int* d_padded_offsets,
    const int* d_compact_offsets,
    const int* d_counts,
    const int* d_entry_cluster_id,
    const int* d_entry_query_start,
    const int* d_entry_query_count,
    const int* d_entry_queries,
    const float* d_query_norms,
    const float* d_vector_norms,
    const float* d_thresholds,
    int* d_candidate_counts,
    int* d_overflow,
    float* d_candidate_dist,
    idx_t* d_candidate_index,
    cudaStream_t stream) {
    if (n_entry <= 0) return true;
    const int dim4 = (dim + 3) / 4;
    if (dim4 <= 0 || dim4 > kShqMaxDim4 || (dim4 % 4) != 0) return false;
    resident_threshold_grouped_vec4_shq_scan_kernel<<<n_entry, 256, 0, stream>>>(
        dim, n_entry, candidate_capacity, d_queries_u8, d_vectors_u8_dp4a,
        d_padded_offsets, d_compact_offsets, d_counts, d_entry_cluster_id,
        d_entry_query_start, d_entry_query_count, d_entry_queries, d_query_norms,
        d_vector_norms, d_thresholds, d_candidate_counts, d_overflow,
        d_candidate_dist, d_candidate_index);
    return true;
}

bool launch_resident_threshold_grouped_mlp2_scan(
    int dim,
    int n_entry,
    int candidate_capacity,
    const uint8_t* d_queries_u8,
    const uint8_t* d_vectors_u8_dp4a,
    const int* d_padded_offsets,
    const int* d_compact_offsets,
    const int* d_counts,
    const int* d_entry_cluster_id,
    const int* d_entry_query_start,
    const int* d_entry_query_count,
    const int* d_entry_queries,
    const float* d_query_norms,
    const float* d_vector_norms,
    const float* d_thresholds,
    int* d_candidate_counts,
    int* d_overflow,
    float* d_candidate_dist,
    idx_t* d_candidate_index,
    cudaStream_t stream) {
    if (n_entry <= 0) return true;
    resident_threshold_grouped_mlp2_scan_kernel<<<n_entry, 256, 0, stream>>>(
        dim, n_entry, candidate_capacity, d_queries_u8, d_vectors_u8_dp4a,
        d_padded_offsets, d_compact_offsets, d_counts, d_entry_cluster_id,
        d_entry_query_start, d_entry_query_count, d_entry_queries, d_query_norms,
        d_vector_norms, d_thresholds, d_candidate_counts, d_overflow,
        d_candidate_dist, d_candidate_index);
    return true;
}

bool launch_resident_threshold_grouped_cpasync_scan(
    int dim,
    int n_entry,
    int candidate_capacity,
    const uint8_t* d_queries_u8,
    const uint8_t* d_vectors_u8_dp4a,
    const int* d_padded_offsets,
    const int* d_compact_offsets,
    const int* d_counts,
    const int* d_entry_cluster_id,
    const int* d_entry_query_start,
    const int* d_entry_query_count,
    const int* d_entry_queries,
    const float* d_query_norms,
    const float* d_vector_norms,
    const float* d_thresholds,
    int* d_candidate_counts,
    int* d_overflow,
    float* d_candidate_dist,
    idx_t* d_candidate_index,
    cudaStream_t stream) {
    if (n_entry <= 0) return true;
    const int dim4 = (dim + 3) / 4;
    const size_t smem_bytes = 2 * (size_t)dim4 * kWarpSize * 4;
    if (smem_bytes == 0 || smem_bytes > 48 * 1024) return false;
    resident_threshold_grouped_cpasync_scan_kernel<<<n_entry, 256, smem_bytes, stream>>>(
        dim, n_entry, candidate_capacity, d_queries_u8, d_vectors_u8_dp4a,
        d_padded_offsets, d_compact_offsets, d_counts, d_entry_cluster_id,
        d_entry_query_start, d_entry_query_count, d_entry_queries, d_query_norms,
        d_vector_norms, d_thresholds, d_candidate_counts, d_overflow,
        d_candidate_dist, d_candidate_index);
    return true;
}

void launch_resident_pca_i8_threshold_grouped_scan(
    int dim,
    int n_entry,
    int candidate_capacity,
    const float* d_weighted_queries,
    const int8_t* d_vectors_i8,
    const int* d_padded_offsets,
    const int* d_compact_offsets,
    const int* d_counts,
    const int* d_entry_cluster_id,
    const int* d_entry_query_start,
    const int* d_entry_query_count,
    const int* d_entry_queries,
    const float* d_query_squared_norms,
    const float* d_vector_squared_norms,
    const float* d_thresholds,
    int* d_candidate_counts,
    int* d_overflow,
    float* d_candidate_dist,
    idx_t* d_candidate_index,
    cudaStream_t stream) {
    if (n_entry <= 0) return;
    resident_threshold_grouped_scan_kernel<ResidentDenseDistanceKind::PCAI8Scaled>
        <<<n_entry, 256, 0, stream>>>(
            dim, n_entry, candidate_capacity, d_weighted_queries, nullptr,
            nullptr, reinterpret_cast<const uint8_t*>(d_vectors_i8),
            d_padded_offsets, d_compact_offsets, d_counts,
            d_entry_cluster_id, d_entry_query_start, d_entry_query_count,
            d_entry_queries, d_query_squared_norms, nullptr, d_vector_squared_norms,
            d_thresholds, d_candidate_counts, d_overflow,
            d_candidate_dist, d_candidate_index);
}

template <ResidentDenseDistanceKind DistanceKind>
void launch_resident_pca_i8_dp4a_threshold_grouped_scan_impl(
    int dim,
    int n_entry,
    int candidate_capacity,
    const int8_t* d_queries_i8,
    const float* d_query_dot_scales,
    const int8_t* d_vectors_i8,
    const int* d_padded_offsets,
    const int* d_compact_offsets,
    const int* d_counts,
    const int* d_entry_cluster_id,
    const int* d_entry_query_start,
    const int* d_entry_query_count,
    const int* d_entry_queries,
    const float* d_query_squared_norms,
    const float* d_vector_squared_norms,
    const float* d_thresholds,
    int* d_candidate_counts,
    int* d_overflow,
    float* d_candidate_dist,
    idx_t* d_candidate_index,
    cudaStream_t stream) {
    if (n_entry <= 0) return;
    if (dim <= 0 || dim % 4 != 0 || dim / 4 > kWarpSize) {
        throw std::invalid_argument(
            "resident PCA threshold dp4a requires a dimension divisible by 4 and at most 128");
    }
    resident_threshold_grouped_scan_kernel<DistanceKind>
        <<<n_entry, 256, 0, stream>>>(
            dim, n_entry, candidate_capacity, nullptr,
            reinterpret_cast<const uint8_t*>(d_queries_i8), nullptr,
            reinterpret_cast<const uint8_t*>(d_vectors_i8),
            d_padded_offsets, d_compact_offsets, d_counts,
            d_entry_cluster_id, d_entry_query_start, d_entry_query_count,
            d_entry_queries, d_query_squared_norms, d_query_dot_scales,
            d_vector_squared_norms, d_thresholds, d_candidate_counts,
            d_overflow, d_candidate_dist, d_candidate_index);
}

#define DEFINE_PCA_DP4A_THRESHOLD_LAUNCH(VERSION) \
void launch_resident_pca_i8_dp4a_v##VERSION##_threshold_grouped_scan( \
    int dim, int n_entry, int candidate_capacity, const int8_t* d_queries_i8, \
    const float* d_query_dot_scales, const int8_t* d_vectors_i8, \
    const int* d_padded_offsets, const int* d_compact_offsets, \
    const int* d_counts, const int* d_entry_cluster_id, \
    const int* d_entry_query_start, const int* d_entry_query_count, \
    const int* d_entry_queries, const float* d_query_squared_norms, \
    const float* d_vector_squared_norms, const float* d_thresholds, \
    int* d_candidate_counts, int* d_overflow, float* d_candidate_dist, \
    idx_t* d_candidate_index, cudaStream_t stream) { \
    launch_resident_pca_i8_dp4a_threshold_grouped_scan_impl< \
        ResidentDenseDistanceKind::PCAI8Dp4aV##VERSION>( \
            dim, n_entry, candidate_capacity, d_queries_i8, d_query_dot_scales, \
            d_vectors_i8, d_padded_offsets, d_compact_offsets, d_counts, \
            d_entry_cluster_id, d_entry_query_start, d_entry_query_count, \
            d_entry_queries, d_query_squared_norms, d_vector_squared_norms, \
            d_thresholds, d_candidate_counts, d_overflow, d_candidate_dist, \
            d_candidate_index, stream); \
}

DEFINE_PCA_DP4A_THRESHOLD_LAUNCH(1)
DEFINE_PCA_DP4A_THRESHOLD_LAUNCH(2)

#undef DEFINE_PCA_DP4A_THRESHOLD_LAUNCH

void launch_resident_pca_i8_dp4a_v3_threshold_grouped_scan(
    int dim,
    int n_entry,
    int candidate_capacity,
    const int8_t* d_queries_i8,
    const float* d_query_dot_scales,
    const int8_t* d_vectors_i8,
    const int* d_padded_offsets,
    const int* d_compact_offsets,
    const int* d_counts,
    const int* d_entry_cluster_id,
    const int* d_entry_query_start,
    const int* d_entry_query_count,
    const int* d_entry_queries,
    const float* d_query_squared_norms,
    const float* d_vector_squared_norms,
    const float* d_thresholds,
    int* d_candidate_counts,
    int* d_overflow,
    float* d_candidate_dist,
    idx_t* d_candidate_index,
    cudaStream_t stream) {
    if (n_entry <= 0) return;
    if (dim <= 0 || dim % 4 != 0 || dim / 4 > kWarpSize) {
        throw std::invalid_argument(
            "resident PCA threshold DP4A V3 requires a dimension "
            "divisible by 4 and at most 128");
    }
    resident_pca_i8_dp4a_v3_threshold_grouped_scan_kernel
        <<<n_entry, 128, 0, stream>>>(
            dim, n_entry, candidate_capacity, d_queries_i8,
            d_query_dot_scales, d_vectors_i8, d_padded_offsets,
            d_compact_offsets, d_counts, d_entry_cluster_id,
            d_entry_query_start, d_entry_query_count, d_entry_queries,
            d_query_squared_norms, d_vector_squared_norms, d_thresholds,
            d_candidate_counts, d_overflow, d_candidate_dist,
            d_candidate_index);
}

void launch_resident_threshold_reduce(
    int k,
    int n_query,
    int candidate_capacity,
    int n_total_vectors,
    const int* d_candidate_counts,
    const float* d_candidate_dist,
    const idx_t* d_candidate_index,
    const idx_t* d_reordered_indices,
    float* d_final_dist,
    idx_t* d_final_index,
    cudaStream_t stream) {
    if (k <= 0 || k > 256)
        throw std::invalid_argument(
            "resident threshold IVFFlat supports topk 1..256");
    if (n_query <= 0) return;
    constexpr int kWarps = 8;
    if (k <= 32) {
        constexpr int kCapacity = 64;
        const size_t shared_bytes =
            (size_t)kWarps * kCapacity * (sizeof(float) + sizeof(idx_t));
        resident_threshold_reduce_kernel<kCapacity, kWarps>
            <<<n_query, 256, shared_bytes, stream>>>(
                n_query, candidate_capacity, n_total_vectors, d_candidate_counts,
                d_candidate_dist, d_candidate_index, d_reordered_indices, k,
                d_final_dist, d_final_index);
    } else if (k <= 64) {
        constexpr int kCapacity = 128;
        const size_t shared_bytes =
            (size_t)kWarps * kCapacity * (sizeof(float) + sizeof(idx_t));
        resident_threshold_reduce_kernel<kCapacity, kWarps>
            <<<n_query, 256, shared_bytes, stream>>>(
                n_query, candidate_capacity, n_total_vectors, d_candidate_counts,
                d_candidate_dist, d_candidate_index, d_reordered_indices, k,
                d_final_dist, d_final_index);
    } else if (k <= 128) {
        constexpr int kCapacity = 256;
        const size_t shared_bytes =
            (size_t)kWarps * kCapacity * (sizeof(float) + sizeof(idx_t));
        resident_threshold_reduce_kernel<kCapacity, kWarps>
            <<<n_query, 256, shared_bytes, stream>>>(
                n_query, candidate_capacity, n_total_vectors, d_candidate_counts,
                d_candidate_dist, d_candidate_index, d_reordered_indices, k,
                d_final_dist, d_final_index);
    } else {
        constexpr int kCapacity = 512;
        const size_t shared_bytes =
            (size_t)kWarps * kCapacity * (sizeof(float) + sizeof(idx_t));
        resident_threshold_reduce_kernel<kCapacity, kWarps>
            <<<n_query, 256, shared_bytes, stream>>>(
                n_query, candidate_capacity, n_total_vectors, d_candidate_counts,
                d_candidate_dist, d_candidate_index, d_reordered_indices, k,
                d_final_dist, d_final_index);
    }
}
