#include "search/kernels/fine/gpu_fine/pq_adc/pq_codeword_scan.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_distance_histogram.cuh"
#include "quantization/turboquant/turboquant_distance.cuh"
#include "turboquant_mma_scan.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_adc_gpu.cuh"
#include "utils/check_cuda_error.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_candidate_cap_select.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_query_lut.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_query_lut_seed.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_query_lut_scan.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_threshold_status.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_threshold_seed_audit.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_threshold_policy.cuh"
#include <algorithm>
#include <cfloat>
#include <climits>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>
#include <type_traits>
#include <math_constants.h>


namespace {

constexpr int kPQK = 256;
constexpr int kLutStride = 257;
constexpr int kCompactLutStride = 256;
constexpr int kScanThreads = 256;
constexpr int kSegmentedScanThreads = 256;
constexpr int kSegmentedPqM = 16;
// Global-u8 LUT is latency-bound; one row per thread reduces live ADC state
// and raises occupancy without changing the scan or candidate semantics.
constexpr int kSegmentedRowsPerThread = 1;
constexpr int kSegmentedTileRows = 512;
constexpr int kDefaultGlobalPoolCapacity = 1 << 26;
// Keep counts and sort offsets in signed int; allocations use size_t.
constexpr int kMaxGlobalPoolCapacity = 1 << 28;

int resolve_global_pool_capacity() {
    const char* value = std::getenv("PARASELECT_PQ_THRESHOLD_GLOBAL_POOL_CAPACITY");
    if (!value || !*value) return kDefaultGlobalPoolCapacity;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 ||
        parsed > kMaxGlobalPoolCapacity ||
        (parsed & (parsed - 1)) != 0) {
        throw std::invalid_argument(
            "PARASELECT_PQ_THRESHOLD_GLOBAL_POOL_CAPACITY must be a positive power of two <= 1<<28");
    }
    return static_cast<int>(parsed);
}

__global__ void compact_seed_probes_kernel(
    const idx_t* top_nprobe,
    int n_query,
    int n_probes,
    int seed_probes,
    idx_t* seed_top_nprobe) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_query * seed_probes;
    if (linear >= total) return;
    const int q = linear / seed_probes;
    const int probe = linear - q * seed_probes;
    seed_top_nprobe[linear] = top_nprobe[(size_t)q * n_probes + probe];
}

__global__ void fill_threshold_inf_kernel(float* thresholds, int n_query) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n_query) thresholds[q] = CUDART_INF_F;
}

__global__ void initialize_threshold_pool_kernel(
    int n_query, int* candidate_count, int* global_count) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear < n_query) candidate_count[linear] = 0;
    if (linear == 0) *global_count = 0;
}

// The compact candidate arrays are query-major.  After a bounded batch select,
// the last retained row is the exact current cutoff for that query.  Keeping
// this update on device is required by the chunked producer/reducer path.
__global__ void update_threshold_from_compact_kernel(
    const float* candidate_dist, const int* candidate_count,
    float* thresholds, int n_query, int k) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_query) return;
    const int count = candidate_count[q];
    if (count >= k) thresholds[q] = candidate_dist[(size_t)q * k + (k - 1)];
}

template <int DSub, bool Residual, bool TurboQuant = false, bool Histogram = false>
__global__ void threshold_adc_scan_kernel(
    const float* queries,
    const float* centroids,
    const idx_t* top_nprobe,
    const uint8_t* codes,
    const long long* cluster_offsets,
    const int* cluster_counts,
    const float* codebook,
    const float* threshold,
    int n_query,
    int dim,
    int n_clusters,
    int n_probes,
    int pq_m,
    int pool_capacity,
    int* pool_idx,
    float* pool_dist,
    int* pool_query,
    int* candidate_count,
    int* global_count,
    int* status,
    unsigned long long* accepted_count,
    paraselect::search::SignResidualDeviceView tq = {},
    int skip_probe_prefix = 0,
    unsigned int* histogram = nullptr, int histogram_bins = 0,
    float log_min = -8.0f, float log_step = 0.0625f,
    int probe_stride = 0) {
    extern __shared__ float lut[];
    unsigned int* local_histogram = reinterpret_cast<unsigned int*>(lut + pq_m * kLutStride);
    if constexpr (Histogram) {
        for (int bin = threadIdx.x; bin < histogram_bins; bin += blockDim.x) local_histogram[bin] = 0;
    }
    const int q = blockIdx.x;
    const int probe = blockIdx.y;
    if (q >= n_query || probe >= n_probes) return;
    if (probe < skip_probe_prefix) return;

    const idx_t cluster = top_nprobe[(size_t)q * (probe_stride ? probe_stride : n_probes) + probe];
    if (cluster < 0 || cluster >= n_clusters) {
        if (threadIdx.x == 0) paraselect::search::detail::record_threshold_status(status, 3);
        return;
    }
    if constexpr (TurboQuant) {
        paraselect::search::turboquant::prepare_residual(tq, q, cluster, dim, lut);
    } else {
    const float* query = queries + (size_t)q * dim;
    const float* centroid = Residual ? centroids + (size_t)cluster * dim : nullptr;
    for (int e = threadIdx.x; e < pq_m * kPQK; e += blockDim.x) {
        const int m = e / kPQK;
        const int code = e - m * kPQK;
        const float* query_sub = query + (size_t)m * DSub;
        const float* centroid_sub = Residual ? centroid + (size_t)m * DSub : nullptr;
        const float* codeword = codebook + ((size_t)m * kPQK + code) * DSub;
        float distance = 0.0f;
#pragma unroll
        for (int d = 0; d < DSub; ++d) {
            const float plain = query_sub[d] - codeword[d];
            const float delta = Residual ? plain - centroid_sub[d] : plain;
            distance = fmaf(delta, delta, distance);
        }
        lut[m * kLutStride + code] = distance;
    }
    }
    __syncthreads();

    const long long begin = cluster_offsets[cluster];
    const int count = cluster_counts[cluster];
    const float cutoff = Histogram ? 0.0f : threshold[q];
    for (int item = threadIdx.x; item < count; item += blockDim.x) {
        const long long vector_index_ll = begin + item;
        bool accepted = false;
        int vector_index = -1;
        float distance = 0.0f;
        if (vector_index_ll < 0 || vector_index_ll > INT_MAX) {
            paraselect::search::detail::record_threshold_status(status, 4);
        } else {
            vector_index = static_cast<int>(vector_index_ll);
            if constexpr (TurboQuant) {
                distance = paraselect::search::turboquant::distance(tq, lut, vector_index);
            } else {
            const uint8_t* vector_code = codes + (size_t)vector_index * pq_m;
            for (int m = 0; m < pq_m; ++m) {
                distance += lut[m * kLutStride + vector_code[m]];
            }
            }
            if constexpr (Histogram) {
                if (isfinite(distance)) {
                    const int bin = paraselect::search::pq_distance_histogram_bin(
                        distance, histogram_bins, log_min, log_step);
                    atomicAdd(local_histogram + bin, 1u);
                }
            } else accepted = distance <= cutoff;
        }
        if constexpr (Histogram) continue;

        // One global reservation per warp keeps the append path parallel while
        // preserving every threshold pass until the batch pool is exhausted.
        const unsigned active_mask = __activemask();
        const unsigned pass_mask = __ballot_sync(active_mask, accepted);
        const int pass_count = __popc(pass_mask);
        if (pass_count > 0) {
            const int leader = __ffs(pass_mask) - 1;
            int base = 0;
            int stored = 0;
            if ((threadIdx.x & 31) == leader) {
                if (accepted_count) atomicAdd(accepted_count + q,
                                               static_cast<unsigned long long>(pass_count));
                base = atomicAdd(global_count, pass_count);
                stored = max(0, min(pass_count, pool_capacity - base));
                if (stored > 0) atomicAdd(candidate_count + q, stored);
                if (stored != pass_count) {
                    paraselect::search::detail::record_threshold_status(status, 1);
                }
            }
            base = __shfl_sync(active_mask, base, leader);
            if (accepted) {
                const int lane = threadIdx.x & 31;
                const int rank = __popc(pass_mask & ((1u << lane) - 1u));
                const int slot = base + rank;
                if (slot >= 0 && slot < pool_capacity) {
                    pool_idx[slot] = vector_index;
                    pool_dist[slot] = distance;
                    pool_query[slot] = q;
                }
            }
            if (base < 0 || base > pool_capacity) {
                paraselect::search::detail::record_threshold_status(status, 1);
            }
        }
    }
    if constexpr (Histogram) {
        __syncthreads();
        for (int bin = threadIdx.x; bin < histogram_bins; bin += blockDim.x) {
            if (local_histogram[bin]) atomicAdd(histogram + (size_t)q * histogram_bins + bin, local_histogram[bin]);
        }
    }
}

template <int DSub>
void launch_threshold_scan(
    paraselect::search::PQAdcDistanceMode distance_mode,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    const float* d_threshold,
    int n_query,
    int dim,
    int n_clusters,
    int n_probes,
    int pq_m,
    int pool_capacity,
    int* d_pool_idx,
    float* d_pool_dist,
    int* d_pool_query,
    int* d_candidate_count,
    int* d_global_count,
    int* d_status,
    unsigned long long* d_accepted_count,
    cudaStream_t stream,
    int skip_probe_prefix = 0) {
    const dim3 grid(n_query, n_probes);
    const size_t shared_bytes = (size_t)pq_m * kLutStride * sizeof(float);
    if (distance_mode == paraselect::search::PQAdcDistanceMode::Residual) {
        threshold_adc_scan_kernel<DSub, true><<<grid, kScanThreads, shared_bytes, stream>>>(
            d_queries, d_centroids, d_top_nprobe, d_pq_codes, d_cluster_offsets,
            d_cluster_counts, d_codebook, d_threshold, n_query, dim, n_clusters,
            n_probes, pq_m, pool_capacity, d_pool_idx, d_pool_dist, d_pool_query,
            d_candidate_count, d_global_count, d_status, d_accepted_count,
            {}, skip_probe_prefix);
    } else {
        threshold_adc_scan_kernel<DSub, false><<<grid, kScanThreads, shared_bytes, stream>>>(
            d_queries, nullptr, d_top_nprobe, d_pq_codes, d_cluster_offsets,
            d_cluster_counts, d_codebook, d_threshold, n_query, dim, n_clusters,
            n_probes, pq_m, pool_capacity, d_pool_idx, d_pool_dist, d_pool_query,
            d_candidate_count, d_global_count, d_status, d_accepted_count,
            {}, skip_probe_prefix);
    }
}

__device__ __forceinline__ int pq_code_byte16(const uint4& words, int m) {
    const uint32_t word = m < 4 ? words.x : m < 8 ? words.y : m < 12 ? words.z : words.w;
    return static_cast<int>((word >> ((m & 3) * 8)) & 255u);
}

// One row of PQ codes held in registers. M=16 is one 16-byte vector; packed
// M=20 rows are only 4-byte aligned and keep a fifth word; wider M (32, 64)
// rows are 16-byte multiples. byte(m) resolves to a register at compile time.
template <int PQ_M>
struct PqCodeWords {
    static_assert(PQ_M == 20 || (PQ_M % 16 == 0 && PQ_M <= 64),
                  "scan code rows support M=16, 20, 32, 48 and 64");
    static constexpr int kVectors = PQ_M == 20 ? 1 : PQ_M / 16;
    uint4 w[kVectors];
    uint32_t tail;
    __device__ __forceinline__ void load(const uint8_t* row) {
        if constexpr (PQ_M == 20) {
            const auto* words = reinterpret_cast<const uint32_t*>(row);
            w[0] = make_uint4(words[0], words[1], words[2], words[3]);
            tail = words[4];
        } else {
#pragma unroll
            for (int v = 0; v < kVectors; ++v) w[v] = reinterpret_cast<const uint4*>(row)[v];
            tail = 0u;
        }
    }
    __device__ __forceinline__ void clear() {
#pragma unroll
        for (int v = 0; v < kVectors; ++v) w[v] = make_uint4(0, 0, 0, 0);
        tail = 0u;
    }
    __device__ __forceinline__ int byte(int m) const {
        if constexpr (PQ_M == 20) {
            if (m >= 16) return static_cast<int>((tail >> ((m & 3) * 8)) & 255u);
        }
        const uint4& v = w[PQ_M == 20 ? 0 : (m >> 4)];
        const int local = m & 15;
        const uint32_t word = local < 4 ? v.x : local < 8 ? v.y : local < 12 ? v.z : v.w;
        return static_cast<int>((word >> ((m & 3) * 8)) & 255u);
    }
};

// Above 32 subspaces the per-query scales no longer fit the register budget of
// the scan kernels; they are read through the read-only path (warp-uniform
// address, so one L1 broadcast per m). Scores and accumulation order are unchanged.
template <int PQ_M>
__device__ __forceinline__ float scan_query_scale(const float* cached, const float* global, int m) {
    if constexpr (PQ_M <= 32) return cached[m];
    else return __ldg(global + m);
}
template <int PQ_M>
constexpr int kScanScaleCache = PQ_M <= 32 ? PQ_M : 1;
template <int PQ_M>
constexpr int kScanMinBlocks = PQ_M <= 32 ? 5 : 4;

// Only for metadata whose address is identical across all active warp lanes.
// Inline PTX communicates uniformity without a leader/shuffle dependency.
template <bool Uniform, typename T>
__device__ __forceinline__ T load_scan_metadata(const T* pointer) {
    if constexpr (!Uniform) return *pointer;
    else {
        static_assert(std::is_same_v<T, int> || std::is_same_v<T, float>);
        T value;
        if constexpr (std::is_same_v<T, int>)
            asm("ldu.global.s32 %0, [%1];" : "=r"(value) : "l"(pointer));
        else
            asm("ldu.global.f32 %0, [%1];" : "=f"(value) : "l"(pointer));
        return value;
    }
}

template <int PQ_M, bool FixedTileWork, bool InterleaveRows, bool SharedLut,
          int SharedLutStride, bool QuantizedLut, bool QuantizedPc,
          bool FilterProbeRange = false, bool TurboQuant = false,
          bool VectorizedScales = false, bool PairSchedule = false,
          bool PlainL2 = false>
__global__ __launch_bounds__(kSegmentedScanThreads, kScanMinBlocks<PQ_M>) void threshold_adc_pq_query_lut_segmented_kernel(
    const float* __restrict__ query_codeword_dot_lut,
    const uint8_t* __restrict__ query_codeword_dot_lut_u8,
    const float* __restrict__ query_lut_scale,
    const float* __restrict__ query_lut_lo_sum,
    const float* __restrict__ query_lut_inflation,
    const float* __restrict__ query_centroid_distances,
    paraselect::search::QueryLutBias query_lut_bias,
    const int* __restrict__ entry_clusters,
    const int* __restrict__ entry_starts,
    const int* __restrict__ entry_counts,
    const int* __restrict__ entry_tile_offsets,
    const int* __restrict__ grouped_queries,
    const int* __restrict__ grouped_probe_ranks,
    const uint8_t* __restrict__ codes,
    const long long* __restrict__ cluster_offsets,
    const int* __restrict__ cluster_counts,
    const float* __restrict__ threshold,
    int n_query,
    int n_probes,
    int entry_count,
    int fixed_tile_rows,
    int pool_capacity,
    int* __restrict__ pool_idx,
    float* __restrict__ pool_dist,
    int* __restrict__ pool_query,
    int* __restrict__ candidate_count,
    int* __restrict__ global_count,
    int* __restrict__ status,
    unsigned long long* __restrict__ accepted_count,
    unsigned long long* __restrict__ physical_code_rows,
    int probe_begin,
    int probe_end,
    paraselect::search::SignResidualDeviceView tq = {},
    const idx_t* pair_probes = nullptr, int list_count = 0) {
    extern __shared__ unsigned char shared_bytes[];
    float* lut = reinterpret_cast<float*>(shared_bytes);
    uint8_t* lut_u8 = shared_bytes;
    const int work = blockIdx.x;
    int entry = work;
    int tile_begin = 0;
    if constexpr (FixedTileWork) {
        int lower = 0;
        int upper = entry_count;
        while (lower < upper) {
            const int middle = lower + (upper - lower) / 2;
            if (entry_tile_offsets[middle + 1] <= work) {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }
        entry = lower;
        tile_begin = (work - entry_tile_offsets[entry]) * fixed_tile_rows;
    }
    if (entry >= entry_count) return;

    static_assert(!PairSchedule || (!FixedTileWork && !FilterProbeRange && !TurboQuant));
    const idx_t cluster_id = PairSchedule ? pair_probes[entry] : entry_clusters[entry];
    if constexpr (PairSchedule) {
        if (cluster_id < 0 || cluster_id >= list_count) {
            if (threadIdx.x == 0) paraselect::search::detail::record_threshold_status(status, 3);
            return;
        }
    }
    const int cluster = static_cast<int>(cluster_id);
    const long long begin = cluster_offsets[cluster];
    const int count = cluster_counts[cluster];
    const int query_start = PairSchedule ? 0 : entry_starts[entry];
    const int query_count = PairSchedule ? 1 : entry_counts[entry];
    const int effective_probe_end = probe_end < 0 ? n_probes : probe_end;
    const int lane = threadIdx.x & 31;
    const int tile_end = FixedTileWork
        ? min(count, tile_begin + fixed_tile_rows)
        : count;
    unsigned selected_visitors = 0;
    if constexpr (FilterProbeRange) {
        // Schedule entries contain at most 32 visitors. Each warp builds the
        // same mask once, before any code/QUERY_LUT_BIAS loads; reuse it for every row tile.
        if (query_count > 32) {
            if (threadIdx.x == 0)
                paraselect::search::detail::record_threshold_status(status, 3);
            return;
        }
        bool selected = false;
        if (lane < query_count) {
            const int probe = grouped_probe_ranks[query_start + lane];
            selected = probe >= probe_begin && probe < effective_probe_end;
        }
        selected_visitors = __ballot_sync(0xffffffffu, selected);
        if (selected_visitors == 0) return;
    }
    if (threadIdx.x == 0 && physical_code_rows) {
        // In the original one-pass path a non-tiled entry scans the list once,
        // even though its row loop advances in tile-sized chunks.
        atomicAdd(
            physical_code_rows,
            static_cast<unsigned long long>(tile_end - tile_begin));
    }
    constexpr int ScanRowsPerIteration =
        kSegmentedScanThreads * kSegmentedRowsPerThread;
    for (int tile = tile_begin; tile < tile_end;
         tile += ScanRowsPerIteration) {
        PqCodeWords<PQ_M> code_words[kSegmentedRowsPerThread];
        float corrections[kSegmentedRowsPerThread];
        int vector_indices[kSegmentedRowsPerThread];
        bool valid_rows[kSegmentedRowsPerThread];
#pragma unroll
        for (int r = 0; r < kSegmentedRowsPerThread; ++r) {
            const int local = tile + threadIdx.x + r * kSegmentedScanThreads;
            const bool valid = local < count;
            const long long vector_index_ll = begin + (valid ? local : 0);
            valid_rows[r] = valid && vector_index_ll >= 0 && vector_index_ll <= INT_MAX;
            vector_indices[r] = valid_rows[r] ? static_cast<int>(vector_index_ll) : -1;
            if (valid && !valid_rows[r]) {
                paraselect::search::detail::record_threshold_status(status, 4);
            }
            if (valid_rows[r]) {
                code_words[r].load(codes + (size_t)vector_indices[r] * PQ_M);
                if constexpr (PlainL2) {
                    corrections[r] = 0.0f;
                } else if constexpr (TurboQuant) {
                    corrections[r] = tq.scales[vector_indices[r]];
                } else if constexpr (QuantizedPc) {
                    corrections[r] = fmaf(
                        static_cast<float>(query_lut_bias.u16[vector_indices[r]]),
                        query_lut_bias.scale,
                        query_lut_bias.offset);
                } else {
                    corrections[r] = query_lut_bias.fp32[vector_indices[r]];
                }
            } else {
                code_words[r].clear();
                corrections[r] = 0.0f;
            }
        }

        if constexpr (!SharedLut && !TurboQuant) {
            // A warp processes one row and reuses it across scheduled visitors.
            if (!__any_sync(0xffffffffu, valid_rows[0])) continue;
        }
        // Pair mode has one fixed visitor per CTA. Compute its coordinates once;
        // list mode keeps the schedule lookup inside the visitor loop.
        const int pair_q = PairSchedule ? entry / n_probes : 0;
        const int pair_probe = PairSchedule ? entry % n_probes : 0;
        unsigned remaining_visitors = selected_visitors;
        for (int visitor = 0;
             FilterProbeRange ? remaining_visitors != 0 : visitor < query_count;
             ++visitor) {
            if constexpr (FilterProbeRange) {
                visitor = __ffs(remaining_visitors) - 1;
                remaining_visitors &= remaining_visitors - 1;
            }
            if constexpr (SharedLut) {
                __syncthreads();
            }
            // PairSchedule has exactly one visitor per CTA; keep q/probe uniform.
            const int ref = query_start + visitor;
            const int q = PairSchedule ? pair_q : load_scan_metadata<VectorizedScales>(grouped_queries + ref);
            const int probe = PairSchedule ? pair_probe : load_scan_metadata<VectorizedScales>(grouped_probe_ranks + ref);
            if (q < 0 || q >= n_query || probe < 0 || probe >= n_probes) {
                if (threadIdx.x == 0) {
                    paraselect::search::detail::record_threshold_status(status, 3);
                }
                if constexpr (SharedLut || TurboQuant) __syncthreads();
                continue;
            }
            const float* query_lut = (QuantizedLut || TurboQuant) ? nullptr :
                query_codeword_dot_lut + (size_t)q * PQ_M *
                    paraselect::search::kPqQueryLutGlobalStride;
            const uint8_t* query_lut_u8 = QuantizedLut
                ? query_codeword_dot_lut_u8 + (size_t)q * PQ_M *
                    paraselect::search::kPqQueryLutGlobalStride
                : nullptr;
            const float* query_scale = QuantizedLut
                ? query_lut_scale + (size_t)q * PQ_M
                : nullptr;
            float query_scales[kScanScaleCache<PQ_M>];
            if constexpr (QuantizedLut && PQ_M <= 32) {
                if constexpr (VectorizedScales) {
                    static_assert(PQ_M % 4 == 0);
#pragma unroll
                    for (int m = 0; m < PQ_M; m += 4) {
                        const float4 values = paraselect::search::load_query_lut_scale4(query_scale + m);
                        query_scales[m] = values.x;
                        query_scales[m + 1] = values.y;
                        query_scales[m + 2] = values.z;
                        query_scales[m + 3] = values.w;
                    }
                } else {
#pragma unroll
                    for (int m = 0; m < PQ_M; ++m) {
                        query_scales[m] = query_scale[m];
                    }
                }
            }
            if constexpr (TurboQuant) {
                paraselect::search::turboquant::prepare_residual(tq, q, cluster, 128, lut);
            } else if constexpr (SharedLut) {
                if constexpr (QuantizedLut) {
                    if constexpr ((SharedLutStride % 4) == 0) {
                        // Compact rows are naturally aligned, so stage the
                        // u8 LUT with 32-bit transactions like ExternalBaseline.
                        const uint8_t* src_u8 = query_lut_u8;
                        uint8_t* dst_u8 = lut_u8;
                        for (int e = threadIdx.x;
                             e < PQ_M * (kPQK / 4);
                             e += blockDim.x) {
                            const int m = e / (kPQK / 4);
                            const int word = e - m * (kPQK / 4);
                            *reinterpret_cast<uint32_t*>(
                                dst_u8 + (size_t)m * SharedLutStride + word * 4) =
                                *reinterpret_cast<const uint32_t*>(
                                    src_u8 + (size_t)m *
                                        paraselect::search::kPqQueryLutGlobalStride +
                                        word * 4);
                        }
                    } else {
                        for (int e = threadIdx.x; e < PQ_M * kPQK;
                             e += blockDim.x) {
                            const int m = e / kPQK;
                            const int code = e - m * kPQK;
                            lut_u8[m * SharedLutStride + code] =
                                query_lut_u8[(size_t)m *
                                    paraselect::search::kPqQueryLutGlobalStride +
                                    code];
                        }
                    }
                } else {
                    for (int e = threadIdx.x; e < PQ_M * kPQK;
                         e += blockDim.x) {
                        const int m = e / kPQK;
                        const int code = e - m * kPQK;
                        lut[m * SharedLutStride + code] =
                            query_lut[(size_t)m *
                                paraselect::search::kPqQueryLutGlobalStride + code];
                    }
                }
            }
            const float coarse_distance =
                PlainL2 ? 0.0f : (TurboQuant ? lut[128] : load_scan_metadata<VectorizedScales>(
                    query_centroid_distances + (size_t)q * n_probes + probe));
            const float cutoff = load_scan_metadata<VectorizedScales>(threshold + q) +
                (QuantizedLut && query_lut_inflation ?
                    load_scan_metadata<VectorizedScales>(query_lut_inflation + q) : 0.0f);
            if constexpr (SharedLut) {
                __syncthreads();
            }

            const float dot_offset = QuantizedLut ?
                load_scan_metadata<VectorizedScales>(query_lut_lo_sum + q) : 0.0f;
            float query_dot_reconstruction[kSegmentedRowsPerThread];
            for (int r = 0; r < kSegmentedRowsPerThread; ++r) {
                query_dot_reconstruction[r] = dot_offset;
            }
            bool accepted[kSegmentedRowsPerThread] = {};
            float distances[kSegmentedRowsPerThread] = {};
            // Reconverge before ballots and shared-LUT barriers below.
            if (valid_rows[0]) {
                if constexpr (TurboQuant) {
                    // Filled by the shared codec scorer below; no PQ LUT lookup.
                } else if constexpr (InterleaveRows) {
    #pragma unroll
                    for (int m = 0; m < PQ_M; ++m) {
    #pragma unroll
                        for (int r = 0; r < kSegmentedRowsPerThread; ++r) {
                            const int code = code_words[r].byte(m);
                            if constexpr (QuantizedLut) {
                                query_dot_reconstruction[r] += scan_query_scale<PQ_M>(query_scales, query_scale, m) *
                                    static_cast<float>(SharedLut
                                        ? lut_u8[m * SharedLutStride + code]
                                        : query_lut_u8[(size_t)m *
                                            paraselect::search::kPqQueryLutGlobalStride + code]);
                            } else {
                                query_dot_reconstruction[r] += SharedLut
                                    ? lut[m * SharedLutStride + code]
                                    : query_lut[(size_t)m *
                                        paraselect::search::kPqQueryLutGlobalStride + code];
                            }
                        }
                    }
                } else {
                    // The production global-u8 rows=1 path is latency-bound by the
                    // single accumulator dependency chain. Keep four independent
                    // accumulators so later LUT loads can overlap earlier FFMA
                    // results. Other layouts/templates retain the original order.
                    if constexpr (kSegmentedRowsPerThread == 1 &&
                                  QuantizedLut && !TurboQuant && !SharedLut) {
                        float acc0 = query_dot_reconstruction[0];
                        float acc1 = 0.0f;
                        float acc2 = 0.0f;
                        float acc3 = 0.0f;
    #pragma unroll
                        for (int m = 0; m < PQ_M; ++m) {
                            const int code = code_words[0].byte(m);
                            const float term = scan_query_scale<PQ_M>(query_scales, query_scale, m) * static_cast<float>(
                                query_lut_u8[(size_t)m *
                                    paraselect::search::kPqQueryLutGlobalStride + code]);
                            if ((m & 3) == 0) acc0 += term;
                            else if ((m & 3) == 1) acc1 += term;
                            else if ((m & 3) == 2) acc2 += term;
                            else acc3 += term;
                        }
                        query_dot_reconstruction[0] = (acc0 + acc1) + (acc2 + acc3);
                    } else {
    #pragma unroll
                        for (int r = 0; r < kSegmentedRowsPerThread; ++r) {
    #pragma unroll
                            for (int m = 0; m < PQ_M; ++m) {
                                const int code = code_words[r].byte(m);
                                if constexpr (QuantizedLut) {
                                    query_dot_reconstruction[r] += scan_query_scale<PQ_M>(query_scales, query_scale, m) *
                                        static_cast<float>(SharedLut
                                            ? lut_u8[m * SharedLutStride + code]
                                            : query_lut_u8[(size_t)m *
                                                paraselect::search::kPqQueryLutGlobalStride + code]);
                                } else {
                                    query_dot_reconstruction[r] += SharedLut
                                        ? lut[m * SharedLutStride + code]
                                        : query_lut[(size_t)m *
                                            paraselect::search::kPqQueryLutGlobalStride + code];
                                }
                            }
                        }
                    }
                }

    #pragma unroll
                for (int r = 0; r < kSegmentedRowsPerThread; ++r) {
                    if constexpr (TurboQuant) {
                        distances[r] = paraselect::search::turboquant::distance(
                            code_words[r].w[0], corrections[r], lut);
                    } else {
                    distances[r] = PlainL2 ? query_dot_reconstruction[r] : coarse_distance + corrections[r] -
                        2.0f * query_dot_reconstruction[r];
                    }
                    accepted[r] = valid_rows[r] &&
                        distances[r] <= cutoff;
                }
            }

            unsigned pass_masks[kSegmentedRowsPerThread];
            int warp_pass_count = 0;
#pragma unroll
            for (int r = 0; r < kSegmentedRowsPerThread; ++r) {
                pass_masks[r] = __ballot_sync(0xffffffffu, accepted[r]);
                warp_pass_count += __popc(pass_masks[r]);
            }
            if (warp_pass_count > 0) {
                int base = 0;
                int stored = 0;
                if (lane == 0) {
                    if (accepted_count) {
                        atomicAdd(
                            accepted_count + q,
                            static_cast<unsigned long long>(warp_pass_count));
                    }
                    base = atomicAdd(global_count, warp_pass_count);
                    stored = max(0, min(warp_pass_count, pool_capacity - base));
                    if (stored > 0) atomicAdd(candidate_count + q, stored);
                    if (stored != warp_pass_count) {
                        paraselect::search::detail::record_threshold_status(status, 1);
                    }
                }
                base = __shfl_sync(0xffffffffu, base, 0);
                int previous = 0;
                const unsigned lanes_before = (1u << lane) - 1u;
#pragma unroll
                for (int r = 0; r < kSegmentedRowsPerThread; ++r) {
                    if (accepted[r]) {
                        const int slot = base + previous +
                            __popc(pass_masks[r] & lanes_before);
                        if (slot >= 0 && slot < pool_capacity) {
                            pool_idx[slot] = vector_indices[r];
                            pool_dist[slot] = distances[r];
                            pool_query[slot] = q;
                        }
                    }
                    previous += __popc(pass_masks[r]);
                }
            }
            if constexpr (TurboQuant) {
                // All visitors must finish reading before the next residual overwrites shared memory.
                __syncthreads();
            }
        }
    }
}

struct PairScanArgs { const float* query_codeword_dot_lut; const uint8_t* query_codeword_dot_lut_u8; const float* query_lut_scale; const float* query_lut_lo_sum; const float* query_lut_inflation; const float* query_centroid_distances; paraselect::search::QueryLutBias query_lut_bias; const uint8_t* codes; const long long* cluster_offsets; const int* cluster_counts; const float* threshold; int n_query,n_probes,list_count,pool_capacity; int* pool_idx; float* pool_dist; int* pool_query; int* candidate_count; int* global_count; int* status; unsigned long long* accepted_count; unsigned long long* physical_code_rows; const idx_t* pair_probes; };

template <int M, bool PlainL2, bool Shared, typename Correction>
__global__ __launch_bounds__(kSegmentedScanThreads, kScanMinBlocks<M>)
void threshold_adc_pair_u8_kernel(
    const uint8_t* __restrict__ lut_u8,
    const float* __restrict__ lut_scale,
    const float* __restrict__ lut_lo_sum,
    const float* __restrict__ lut_inflation,
    const float* __restrict__ centroid_distances,
    const Correction* __restrict__ query_lut_bias,
    float row_scale, float row_offset,
    const idx_t* __restrict__ probes,
    const uint8_t* __restrict__ codes,
    const long long* __restrict__ list_offsets,
    const int* __restrict__ list_counts,
    const float* __restrict__ thresholds,
    int n_query, int n_probes, int list_count, int pool_capacity,
    int* __restrict__ pool_idx, float* __restrict__ pool_dist,
    int* __restrict__ pool_query, int* __restrict__ candidate_count,
    int* __restrict__ global_count, int* __restrict__ status,
    unsigned long long* __restrict__ accepted_count,
    unsigned long long* __restrict__ physical_code_rows, int probes_per_cta) {
    const int q = blockIdx.x;
    if (q >= n_query || (!Shared && blockIdx.y >= n_probes)) return;
    const unsigned full_mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const float threshold_value = lane == 0 ? thresholds[q] : 0.0f;
    const float inflation_value = lane == 0 && lut_inflation ? lut_inflation[q] : 0.0f;
    const float cutoff = __shfl_sync(full_mask, threshold_value, 0) +
        __shfl_sync(full_mask, inflation_value, 0);
    const uint8_t* const query_lut = lut_u8 +
        static_cast<size_t>(q) * M * paraselect::search::kPqQueryLutGlobalStride;
    extern __shared__ uint8_t shared_lut[];
    if constexpr (Shared) {
        for (int word = threadIdx.x; word < M * 64; word += blockDim.x)
            reinterpret_cast<uint32_t*>(shared_lut)[word] =
                reinterpret_cast<const uint32_t*>(query_lut)[word];
        __syncthreads();
    }
    const float* const query_scale = lut_scale + static_cast<size_t>(q) * M;
    float query_scales[kScanScaleCache<M>];
    if constexpr (M <= 32) {
        constexpr int ScaleGroups = (M + 3) / 4;
        float4 scale_vectors[ScaleGroups];
#pragma unroll
        for (int group = 0; group < ScaleGroups; ++group) {
            if (lane == group) {
                scale_vectors[group] = *reinterpret_cast<const float4*>(query_scale + group * 4);
            }
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const int scale_lane = m / 4;
            float value = 0.0f;
            if (lane == scale_lane)
                value = (&scale_vectors[scale_lane].x)[m & 3];
            query_scales[m] = __shfl_sync(full_mask, value, scale_lane);
        }
    }
    const int first_probe = Shared ? blockIdx.y * probes_per_cta : blockIdx.y;
    const int owned_probes = Shared ? min(probes_per_cta, n_probes - first_probe) : 1;
    for (int owned = 0; owned < owned_probes; ++owned) {
        const int probe = first_probe + owned;
        const idx_t cluster_id = probes[static_cast<size_t>(q) * n_probes + probe];
        if (cluster_id < 0 || cluster_id >= list_count) {
            if (threadIdx.x == 0)
                paraselect::search::detail::record_threshold_status(status, 3);
            continue;
        }
        const int cluster = static_cast<int>(cluster_id);
        const long long begin = list_offsets[cluster];
        const int count = list_counts[cluster];
        const float coarse_value = !PlainL2 && lane == 0
            ? centroid_distances[static_cast<size_t>(q) * n_probes + probe]
            : 0.0f;
        const float coarse = __shfl_sync(full_mask, coarse_value, 0);
        for (int item = threadIdx.x; item < count; item += blockDim.x) {
            const long long index_ll = begin + item;
            bool accepted = false;
            int index = -1;
            float distance = 0.0f;
            if (index_ll < 0 || index_ll > INT_MAX) {
                paraselect::search::detail::record_threshold_status(status, 4);
            } else {
                index = static_cast<int>(index_ll);
                PqCodeWords<M> words;
                words.load(codes + static_cast<size_t>(index) * M);
                float acc0 = lut_lo_sum[q], acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    #pragma unroll
                for (int m = 0; m < M; ++m) {
                    const int code = words.byte(m);
                    const float term = scan_query_scale<M>(query_scales, query_scale, m) * static_cast<float>(
                        (Shared ? shared_lut : query_lut)[static_cast<size_t>(m) *
                                      paraselect::search::kPqQueryLutGlobalStride + code]);
                    if ((m & 3) == 0) acc0 += term;
                    else if ((m & 3) == 1) acc1 += term;
                    else if ((m & 3) == 2) acc2 += term;
                    else acc3 += term;
                }
                const float dot = (acc0 + acc1) + (acc2 + acc3);
                float correction;
                if constexpr (PlainL2) {
                    correction = 0.0f;
                } else if constexpr (std::is_same_v<Correction, uint16_t>) {
                    correction = fmaf(static_cast<float>(query_lut_bias[index]), row_scale, row_offset);
                } else {
                    correction = query_lut_bias[index];
                }
                distance = PlainL2 ? dot : coarse + correction - 2.0f * dot;
                accepted = distance <= cutoff;
            }
            const unsigned active = __activemask();
            const unsigned pass = __ballot_sync(active, accepted);
            const int pass_count = __popc(pass);
            if (pass_count == 0) continue;
            const int lane = threadIdx.x & 31;
            const int leader = __ffs(pass) - 1;
            int base = 0;
            if (lane == leader) {
                if (accepted_count)
                    atomicAdd(accepted_count + q, static_cast<unsigned long long>(pass_count));
                base = atomicAdd(global_count, pass_count);
                const int stored = max(0, min(pass_count, pool_capacity - base));
                if (stored > 0) atomicAdd(candidate_count + q, stored);
                if (stored != pass_count)
                    paraselect::search::detail::record_threshold_status(status, 1);
            }
            base = __shfl_sync(active, base, leader);
            if (accepted) {
                const int rank = __popc(pass & ((1u << lane) - 1u));
                const int slot = base + rank;
                if (slot >= 0 && slot < pool_capacity) {
                    pool_idx[slot] = index;
                    pool_dist[slot] = distance;
                    pool_query[slot] = q;
                }
            }
        }
    }
}

template <int PQ_M, bool SharedLut, int SharedLutStride, bool QuantizedLut, bool QuantizedPc, bool VectorizedScales, bool PlainL2 = false>
__global__ __launch_bounds__(kSegmentedScanThreads) void threshold_adc_pq_query_lut_pair_kernel(PairScanArgs args, int probes_per_cta) {
    const float* query_codeword_dot_lut=args.query_codeword_dot_lut; const uint8_t* query_codeword_dot_lut_u8=args.query_codeword_dot_lut_u8; const float* query_lut_scale=args.query_lut_scale; const float* query_lut_lo_sum=args.query_lut_lo_sum; const float* query_lut_inflation=args.query_lut_inflation; const float* query_centroid_distances=args.query_centroid_distances; auto query_lut_bias=args.query_lut_bias; const uint8_t* codes=args.codes; const long long* cluster_offsets=args.cluster_offsets; const int* cluster_counts=args.cluster_counts; const float* threshold=args.threshold; const int n_query=args.n_query,n_probes=args.n_probes,list_count=args.list_count,pool_capacity=args.pool_capacity; int* pool_idx=args.pool_idx; float* pool_dist=args.pool_dist; int* pool_query=args.pool_query; int* candidate_count=args.candidate_count; int* global_count=args.global_count; int* status=args.status; unsigned long long* accepted_count=args.accepted_count; unsigned long long* physical_code_rows=args.physical_code_rows; const idx_t* pair_probes=args.pair_probes; const int entry_count=n_query*n_probes; constexpr bool FixedTileWork=false,InterleaveRows=false,FilterProbeRange=false,TurboQuant=false,PairSchedule=true; const int* entry_tile_offsets=nullptr; const int* entry_clusters=nullptr; const int* entry_starts=nullptr; const int* entry_counts=nullptr; const int* grouped_queries=nullptr; const int* grouped_probe_ranks=nullptr; const int fixed_tile_rows=0; const int probe_begin=0; const int probe_end=n_probes; paraselect::search::SignResidualDeviceView tq={};

    extern __shared__ unsigned char shared_bytes[];
    float* lut = reinterpret_cast<float*>(shared_bytes);
    uint8_t* lut_u8 = shared_bytes;
    const int groups = (n_probes + probes_per_cta - 1) / probes_per_cta;
    const int owner_query = blockIdx.x / groups;
    const int first_probe = (blockIdx.x % groups) * probes_per_cta;
    const float* query_lut = QuantizedLut ? nullptr :
        query_codeword_dot_lut + size_t(owner_query) * PQ_M * paraselect::search::kPqQueryLutGlobalStride;
    const uint8_t* query_lut_u8 = QuantizedLut ?
        query_codeword_dot_lut_u8 + size_t(owner_query) * PQ_M * paraselect::search::kPqQueryLutGlobalStride : nullptr;
    // The query and LUT remain fixed across every row tile and owned probe.
    if constexpr (SharedLut) {
        if constexpr (QuantizedLut) {
            if constexpr ((SharedLutStride % 4) == 0) {
                // Compact rows are naturally aligned, so stage the
                // u8 LUT with 32-bit transactions like ExternalBaseline.
                const uint8_t* src_u8 = query_lut_u8;
                uint8_t* dst_u8 = lut_u8;
                for (int e = threadIdx.x;
                     e < PQ_M * (kPQK / 4);
                     e += blockDim.x) {
                    const int m = e / (kPQK / 4);
                    const int word = e - m * (kPQK / 4);
                    *reinterpret_cast<uint32_t*>(
                        dst_u8 + (size_t)m * SharedLutStride + word * 4) =
                        *reinterpret_cast<const uint32_t*>(
                            src_u8 + (size_t)m *
                                paraselect::search::kPqQueryLutGlobalStride +
                                word * 4);
                }
            } else {
                for (int e = threadIdx.x; e < PQ_M * kPQK;
                     e += blockDim.x) {
                    const int m = e / kPQK;
                    const int code = e - m * kPQK;
                    lut_u8[m * SharedLutStride + code] =
                        query_lut_u8[(size_t)m *
                            paraselect::search::kPqQueryLutGlobalStride +
                            code];
                }
            }
        } else {
            for (int e = threadIdx.x; e < PQ_M * kPQK;
                 e += blockDim.x) {
                const int m = e / kPQK;
                const int code = e - m * kPQK;
                lut[m * SharedLutStride + code] =
                    query_lut[(size_t)m *
                        paraselect::search::kPqQueryLutGlobalStride + code];
            }
        }
        __syncthreads();
    }

    for (int owned_probe = first_probe;
         owned_probe < min(n_probes, first_probe + probes_per_cta); ++owned_probe) {
        const int work = owner_query * n_probes + owned_probe;
        int entry = work;
        int tile_begin = 0;
        if constexpr (FixedTileWork) {
            int lower = 0;
            int upper = entry_count;
            while (lower < upper) {
                const int middle = lower + (upper - lower) / 2;
                if (entry_tile_offsets[middle + 1] <= work) {
                    lower = middle + 1;
                } else {
                    upper = middle;
                }
            }
            entry = lower;
            tile_begin = (work - entry_tile_offsets[entry]) * fixed_tile_rows;
        }
        if (entry >= entry_count) return;

        static_assert(!PairSchedule || (!FixedTileWork && !FilterProbeRange && !TurboQuant));
        const idx_t cluster_id = PairSchedule ? pair_probes[entry] : entry_clusters[entry];
        if constexpr (PairSchedule) {
            if (cluster_id < 0 || cluster_id >= list_count) {
                if (threadIdx.x == 0) paraselect::search::detail::record_threshold_status(status, 3);
                continue;
            }
        }
        const int cluster = static_cast<int>(cluster_id);
        const long long begin = cluster_offsets[cluster];
        const int count = cluster_counts[cluster];
        const int query_start = PairSchedule ? 0 : entry_starts[entry];
        const int query_count = PairSchedule ? 1 : entry_counts[entry];
        const int effective_probe_end = probe_end < 0 ? n_probes : probe_end;
        const int lane = threadIdx.x & 31;
        const int tile_end = FixedTileWork
            ? min(count, tile_begin + fixed_tile_rows)
            : count;
        unsigned selected_visitors = 0;
        if constexpr (FilterProbeRange) {
            // Schedule entries contain at most 32 visitors. Each warp builds the
            // same mask once, before any code/QUERY_LUT_BIAS loads; reuse it for every row tile.
            if (query_count > 32) {
                if (threadIdx.x == 0)
                    paraselect::search::detail::record_threshold_status(status, 3);
                return;
            }
            bool selected = false;
            if (lane < query_count) {
                const int probe = grouped_probe_ranks[query_start + lane];
                selected = probe >= probe_begin && probe < effective_probe_end;
            }
            selected_visitors = __ballot_sync(0xffffffffu, selected);
            if (selected_visitors == 0) return;
        }
        if (threadIdx.x == 0 && physical_code_rows) {
            // In the original one-pass path a non-tiled entry scans the list once,
            // even though its row loop advances in tile-sized chunks.
            atomicAdd(
                physical_code_rows,
                static_cast<unsigned long long>(tile_end - tile_begin));
        }
        constexpr int ScanRowsPerIteration =
            kSegmentedScanThreads * kSegmentedRowsPerThread;
        for (int tile = tile_begin; tile < tile_end;
             tile += ScanRowsPerIteration) {
            PqCodeWords<PQ_M> code_words[kSegmentedRowsPerThread];
            float corrections[kSegmentedRowsPerThread];
            int vector_indices[kSegmentedRowsPerThread];
            bool valid_rows[kSegmentedRowsPerThread];
    #pragma unroll
            for (int r = 0; r < kSegmentedRowsPerThread; ++r) {
                const int local = tile + threadIdx.x + r * kSegmentedScanThreads;
                const bool valid = local < count;
                const long long vector_index_ll = begin + (valid ? local : 0);
                valid_rows[r] = valid && vector_index_ll >= 0 && vector_index_ll <= INT_MAX;
                vector_indices[r] = valid_rows[r] ? static_cast<int>(vector_index_ll) : -1;
                if (valid && !valid_rows[r]) {
                    paraselect::search::detail::record_threshold_status(status, 4);
                }
                if (valid_rows[r]) {
                    code_words[r].load(codes + (size_t)vector_indices[r] * PQ_M);
                    if constexpr (PlainL2) {
                        corrections[r] = 0.0f;
                    } else if constexpr (TurboQuant) {
                        corrections[r] = tq.scales[vector_indices[r]];
                    } else if constexpr (QuantizedPc) {
                        corrections[r] = fmaf(
                            static_cast<float>(query_lut_bias.u16[vector_indices[r]]),
                            query_lut_bias.scale,
                            query_lut_bias.offset);
                    } else {
                        corrections[r] = query_lut_bias.fp32[vector_indices[r]];
                    }
                } else {
                    code_words[r].clear();
                    corrections[r] = 0.0f;
                }
            }

            // Pair mode has one fixed visitor per CTA. Compute its coordinates once;
            // list mode keeps the schedule lookup inside the visitor loop.
            const int pair_q = PairSchedule ? entry / n_probes : 0;
            const int pair_probe = PairSchedule ? entry % n_probes : 0;
            unsigned remaining_visitors = selected_visitors;
            for (int visitor = 0;
                 FilterProbeRange ? remaining_visitors != 0 : visitor < query_count;
                 ++visitor) {
                if constexpr (FilterProbeRange) {
                    visitor = __ffs(remaining_visitors) - 1;
                    remaining_visitors &= remaining_visitors - 1;
                }
                // PairSchedule has exactly one visitor per CTA; keep q/probe uniform.
                const int ref = query_start + visitor;
                const int q = PairSchedule ? pair_q : load_scan_metadata<VectorizedScales>(grouped_queries + ref);
                const int probe = PairSchedule ? pair_probe : load_scan_metadata<VectorizedScales>(grouped_probe_ranks + ref);
                if (q < 0 || q >= n_query || probe < 0 || probe >= n_probes) {
                    if (threadIdx.x == 0) {
                        paraselect::search::detail::record_threshold_status(status, 3);
                    }
                    __syncthreads();
                    continue;
                }
                const float* query_lut = (QuantizedLut || TurboQuant) ? nullptr :
                    query_codeword_dot_lut + (size_t)q * PQ_M *
                        paraselect::search::kPqQueryLutGlobalStride;
                const uint8_t* query_lut_u8 = QuantizedLut
                    ? query_codeword_dot_lut_u8 + (size_t)q * PQ_M *
                        paraselect::search::kPqQueryLutGlobalStride
                    : nullptr;
                const float* query_scale = QuantizedLut
                    ? query_lut_scale + (size_t)q * PQ_M
                    : nullptr;
                float query_scales[kScanScaleCache<PQ_M>];
                if constexpr (QuantizedLut && PQ_M <= 32) {
                    if constexpr (VectorizedScales) {
                        static_assert(PQ_M % 4 == 0);
    #pragma unroll
                        for (int m = 0; m < PQ_M; m += 4) {
                            const float4 values = paraselect::search::load_query_lut_scale4(query_scale + m);
                            query_scales[m] = values.x;
                            query_scales[m + 1] = values.y;
                            query_scales[m + 2] = values.z;
                            query_scales[m + 3] = values.w;
                        }
                    } else {
    #pragma unroll
                        for (int m = 0; m < PQ_M; ++m) {
                            query_scales[m] = query_scale[m];
                        }
                    }
                }
                const float coarse_distance =
                    PlainL2 ? 0.0f : (TurboQuant ? lut[128] : load_scan_metadata<VectorizedScales>(
                        query_centroid_distances + (size_t)q * n_probes + probe));
                const float cutoff = load_scan_metadata<VectorizedScales>(threshold + q) +
                    (QuantizedLut && query_lut_inflation ?
                        load_scan_metadata<VectorizedScales>(query_lut_inflation + q) : 0.0f);

                const float dot_offset = QuantizedLut ?
                    load_scan_metadata<VectorizedScales>(query_lut_lo_sum + q) : 0.0f;
                float query_dot_reconstruction[kSegmentedRowsPerThread];
                for (int r = 0; r < kSegmentedRowsPerThread; ++r) {
                    query_dot_reconstruction[r] = dot_offset;
                }
                if constexpr (TurboQuant) {
                    // Filled by the shared codec scorer below; no PQ LUT lookup.
                } else if constexpr (InterleaveRows) {
    #pragma unroll
                    for (int m = 0; m < PQ_M; ++m) {
    #pragma unroll
                        for (int r = 0; r < kSegmentedRowsPerThread; ++r) {
                            const int code = code_words[r].byte(m);
                            if constexpr (QuantizedLut) {
                                query_dot_reconstruction[r] += scan_query_scale<PQ_M>(query_scales, query_scale, m) *
                                    static_cast<float>(SharedLut
                                        ? lut_u8[m * SharedLutStride + code]
                                        : query_lut_u8[(size_t)m *
                                            paraselect::search::kPqQueryLutGlobalStride + code]);
                            } else {
                                query_dot_reconstruction[r] += SharedLut
                                    ? lut[m * SharedLutStride + code]
                                    : query_lut[(size_t)m *
                                        paraselect::search::kPqQueryLutGlobalStride + code];
                            }
                        }
                    }
                } else {
                    // The production global-u8 rows=1 path is latency-bound by the
                    // single accumulator dependency chain. Keep four independent
                    // accumulators so later LUT loads can overlap earlier FFMA
                    // results. Other layouts/templates retain the original order.
                    if constexpr (kSegmentedRowsPerThread == 1 &&
                                  QuantizedLut && !TurboQuant && !SharedLut) {
                        float acc0 = query_dot_reconstruction[0];
                        float acc1 = 0.0f;
                        float acc2 = 0.0f;
                        float acc3 = 0.0f;
    #pragma unroll
                        for (int m = 0; m < PQ_M; ++m) {
                            const int code = code_words[0].byte(m);
                            const float term = scan_query_scale<PQ_M>(query_scales, query_scale, m) * static_cast<float>(
                                query_lut_u8[(size_t)m *
                                    paraselect::search::kPqQueryLutGlobalStride + code]);
                            if ((m & 3) == 0) acc0 += term;
                            else if ((m & 3) == 1) acc1 += term;
                            else if ((m & 3) == 2) acc2 += term;
                            else acc3 += term;
                        }
                        query_dot_reconstruction[0] = (acc0 + acc1) + (acc2 + acc3);
                    } else {
    #pragma unroll
                        for (int r = 0; r < kSegmentedRowsPerThread; ++r) {
    #pragma unroll
                            for (int m = 0; m < PQ_M; ++m) {
                                const int code = code_words[r].byte(m);
                                if constexpr (QuantizedLut) {
                                    query_dot_reconstruction[r] += scan_query_scale<PQ_M>(query_scales, query_scale, m) *
                                        static_cast<float>(SharedLut
                                            ? lut_u8[m * SharedLutStride + code]
                                            : query_lut_u8[(size_t)m *
                                                paraselect::search::kPqQueryLutGlobalStride + code]);
                                } else {
                                    query_dot_reconstruction[r] += SharedLut
                                        ? lut[m * SharedLutStride + code]
                                        : query_lut[(size_t)m *
                                            paraselect::search::kPqQueryLutGlobalStride + code];
                                }
                            }
                        }
                    }
                }

                bool accepted[kSegmentedRowsPerThread];
                float distances[kSegmentedRowsPerThread];
    #pragma unroll
                for (int r = 0; r < kSegmentedRowsPerThread; ++r) {
                    if constexpr (TurboQuant) {
                        distances[r] = paraselect::search::turboquant::distance(
                            code_words[r].w[0], corrections[r], lut);
                    } else {
                    distances[r] = PlainL2 ? query_dot_reconstruction[r] : coarse_distance + corrections[r] -
                        2.0f * query_dot_reconstruction[r];
                    }
                    accepted[r] = valid_rows[r] &&
                        distances[r] <= cutoff;
                }

                unsigned pass_masks[kSegmentedRowsPerThread];
                int warp_pass_count = 0;
    #pragma unroll
                for (int r = 0; r < kSegmentedRowsPerThread; ++r) {
                    pass_masks[r] = __ballot_sync(0xffffffffu, accepted[r]);
                    warp_pass_count += __popc(pass_masks[r]);
                }
                if (warp_pass_count > 0) {
                    int base = 0;
                    int stored = 0;
                    if (lane == 0) {
                        if (accepted_count) {
                            atomicAdd(
                                accepted_count + q,
                                static_cast<unsigned long long>(warp_pass_count));
                        }
                        base = atomicAdd(global_count, warp_pass_count);
                        stored = max(0, min(warp_pass_count, pool_capacity - base));
                        if (stored > 0) atomicAdd(candidate_count + q, stored);
                        if (stored != warp_pass_count) {
                            paraselect::search::detail::record_threshold_status(status, 1);
                        }
                    }
                    base = __shfl_sync(0xffffffffu, base, 0);
                    int previous = 0;
                    const unsigned lanes_before = (1u << lane) - 1u;
    #pragma unroll
                    for (int r = 0; r < kSegmentedRowsPerThread; ++r) {
                        if (accepted[r]) {
                            const int slot = base + previous +
                                __popc(pass_masks[r] & lanes_before);
                            if (slot >= 0 && slot < pool_capacity) {
                                pool_idx[slot] = vector_indices[r];
                                pool_dist[slot] = distances[r];
                                pool_query[slot] = q;
                            }
                        }
                        previous += __popc(pass_masks[r]);
                    }
                }
                if constexpr (TurboQuant) {
                    // All visitors must finish reading before the next residual overwrites shared memory.
                    __syncthreads();
                }
            }
        }
    }
}

// Experimental large-visitor path. One CTA (one warp) processes four queries
// and eight rows at once. The query LUTs are transposed in shared memory so
// the four lanes for one row read adjacent query slots. This is intentionally
// separate from the production segmented kernel until end-to-end A/B proves
// a benefit.
__global__ __launch_bounds__(256) void threshold_adc_query_row_tile4_kernel(
    const uint8_t* __restrict__ query_lut_u8,
    const float* __restrict__ query_lut_scale,
    const float* __restrict__ query_lut_lo_sum,
    const float* __restrict__ query_lut_inflation,
    const float* __restrict__ query_centroid_distances,
    paraselect::search::QueryLutBias query_lut_bias,
    const int* __restrict__ entry_clusters,
    const int* __restrict__ entry_starts,
    const int* __restrict__ entry_counts,
    const int* __restrict__ grouped_queries,
    const int* __restrict__ grouped_probe_ranks,
    const uint8_t* __restrict__ codes,
    const long long* __restrict__ cluster_offsets,
    const int* __restrict__ cluster_counts,
    const float* __restrict__ thresholds,
    int n_query,
    int n_probes,
    int entry_count,
    int pool_capacity,
    int* __restrict__ pool_idx,
    float* __restrict__ pool_dist,
    int* __restrict__ pool_query,
    int* __restrict__ candidate_count,
    int* __restrict__ global_count,
    int* __restrict__ status,
    unsigned long long* __restrict__ accepted_count,
    unsigned long long* __restrict__ physical_code_rows) {
    constexpr int kGroupQueries = 4;
    constexpr int kM = 16;
    constexpr int kWarpsPerQuery = 2;
    constexpr int kRowsPerLane = 2;
    constexpr int kRowsPerWarp = 32 * kRowsPerLane;
    constexpr int kRowsPerTile = kWarpsPerQuery * kRowsPerWarp;
    constexpr int kQueryLutStride = kM * kPQK + 4;
    constexpr int kLutValues = kGroupQueries * kM * kPQK;
    constexpr int kLutStorageBytes = kGroupQueries * kQueryLutStride;
    extern __shared__ uint8_t smem[];
    // Query lanes are adjacent within each 8-lane row subgroup. Store the
    // staged LUT with query as the innermost dimension so those lanes issue
    // contiguous shared-memory accesses for the same (m, code).
    uint8_t* staged_lut = smem;
    float* staged_scales = reinterpret_cast<float*>(
        staged_lut + kLutStorageBytes);
    uint4* staged_codes = reinterpret_cast<uint4*>(
        staged_scales + kGroupQueries * kM);
    uint16_t* staged_corrections = reinterpret_cast<uint16_t*>(
        staged_codes + kRowsPerTile);
    const int entry = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (entry >= entry_count) return;
    const int cluster = entry_clusters[entry];
    const long long begin = cluster_offsets[cluster];
    const int count = cluster_counts[cluster];
    const int query_start = entry_starts[entry];
    const int query_count = entry_counts[entry];
    if (threadIdx.x == 0 && physical_code_rows)
        atomicAdd(physical_code_rows, static_cast<unsigned long long>(count));

    for (int group = 0; group < query_count; group += kGroupQueries) {
        const int group_size = min(kGroupQueries, query_count - group);
        for (int i = threadIdx.x; i < kLutValues; i += blockDim.x) {
            const int qs = i / (kM * kPQK);
            const int rem = i - qs * kM * kPQK;
            const int m = rem / kPQK;
            const int code = rem - m * kPQK;
            uint8_t value = 0;
            if (qs < group_size) {
                const int q = grouped_queries[query_start + group + qs];
                value = query_lut_u8[(size_t)q * kM * kPQK + (size_t)m * kPQK + code];
            }
            staged_lut[((size_t)m * kPQK + code) * kGroupQueries + qs] = value;
        }
        for (int i = threadIdx.x; i < group_size * kM; i += blockDim.x) {
            const int qs = i / kM;
            const int m = i - qs * kM;
            const int q = grouped_queries[query_start + group + qs];
            staged_scales[qs * kM + m] = query_lut_scale[(size_t)q * kM + m];
        }
        __syncthreads();
        for (int tile = 0; tile < count; tile += kRowsPerTile) {
            // Load each code row once. Two full warps are assigned to each
            // query so every active warp stays query-uniform; the previous
            // 8-lane mapping serialized four query paths within a warp.
            if (threadIdx.x < kRowsPerTile) {
                const int row = tile + threadIdx.x;
                if (row < count) {
                    const int vector_index = static_cast<int>(begin + row);
                    staged_codes[threadIdx.x] = *reinterpret_cast<const uint4*>(
                        codes + (size_t)vector_index * kM);
                    staged_corrections[threadIdx.x] = query_lut_bias.u16[vector_index];
                } else {
                    staged_codes[threadIdx.x] = make_uint4(0, 0, 0, 0);
                    staged_corrections[threadIdx.x] = 0;
                }
            }
            __syncthreads();
            const int qs = warp / kWarpsPerQuery;
            const int row_base = (warp % kWarpsPerQuery) * kRowsPerWarp;
            int local[kRowsPerLane];
            bool valid[kRowsPerLane];
#pragma unroll
            for (int r = 0; r < kRowsPerLane; ++r) {
                local[r] = row_base + r * 32 + lane;
                valid[r] = tile + local[r] < count;
            }
            uint4 code_words[kRowsPerLane];
#pragma unroll
            for (int r = 0; r < kRowsPerLane; ++r) {
                const int shared_row = row_base + r * 32 + lane;
                code_words[r] = staged_codes[shared_row];
            }
            if (qs < group_size) {
                const int q = grouped_queries[query_start + group + qs];
                const int probe = grouped_probe_ranks[query_start + group + qs];
                float dot[kRowsPerLane] = {
                    query_lut_lo_sum[q], query_lut_lo_sum[q]};
#pragma unroll
                for (int m = 0; m < kM; ++m) {
                    const float scale = staged_scales[qs * kM + m];
#pragma unroll
                    for (int r = 0; r < kRowsPerLane; ++r) {
                        const int code = pq_code_byte16(code_words[r], m);
                        dot[r] += scale * static_cast<float>(
                            staged_lut[((size_t)m * kPQK + code) *
                                       kGroupQueries + qs]);
                    }
                }
#pragma unroll
                for (int r = 0; r < kRowsPerLane; ++r) {
                    float distance = 0.0f;
                    if (valid[r]) {
                        const int shared_row = row_base + r * 32 + lane;
                        distance = query_centroid_distances[(size_t)q * n_probes + probe] +
                            static_cast<float>(staged_corrections[shared_row]) *
                                query_lut_bias.scale + query_lut_bias.offset - 2.0f * dot[r];
                    }
                    const bool accepted = valid[r] && distance <=
                        thresholds[q] + (query_lut_inflation ? query_lut_inflation[q] : 0.0f);
                    const unsigned pass = __ballot_sync(0xffffffffu, accepted);
                    const int pass_count = __popc(pass);
                    if (pass_count > 0) {
                        int base = 0;
                        int stored = 0;
                        if (lane == 0) {
                            if (accepted_count) atomicAdd(accepted_count + q,
                                static_cast<unsigned long long>(pass_count));
                            base = atomicAdd(global_count, pass_count);
                            stored = max(0, min(pass_count, pool_capacity - base));
                            if (stored > 0) atomicAdd(candidate_count + q, stored);
                            if (stored != pass_count)
                                paraselect::search::detail::record_threshold_status(status, 1);
                        }
                        base = __shfl_sync(0xffffffffu, base, 0);
                        if (accepted) {
                            const int rank = __popc(pass & ((1u << lane) - 1u));
                            const int slot = base + rank;
                            if (slot >= 0 && slot < pool_capacity) {
                                pool_idx[slot] = static_cast<int>(begin + tile + local[r]);
                                pool_dist[slot] = distance;
                                pool_query[slot] = q;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

// Shared pair/list metadata specialization, selected once before launch.
bool use_optimized_scan_metadata(
    const paraselect::search::PQQueryLutView& view, int m,
    int queries, int probes, paraselect::search::PQAdcLutReadMode mode) {
    return m == 16 && mode == paraselect::search::PQAdcLutReadMode::Global &&
        view.quantized() && view.quantized_query_lut_bias() && !view.sign_codec.codes &&
        reinterpret_cast<uintptr_t>(view.query_lut_scale) % 16 == 0 &&
        queries >= 256 && int64_t(queries) * probes >= 16384;
}

void throw_threshold_status(int status) {
    switch (status) {
        case 1:
            throw std::runtime_error(
                "pq ADC threshold candidate capacity exceeded; increase candidate_capacity");
        case 2:
            throw paraselect::search::PQAdcInsufficientSeedCandidates();
        case 3:
            throw std::runtime_error("pq ADC threshold received an invalid coarse cluster ID");
        case 4:
            throw std::runtime_error("pq ADC threshold vector index exceeds int range");
        case 5:
            throw std::invalid_argument("pq ADC threshold policy is invalid");
        default:
            throw std::runtime_error("pq ADC threshold scan failed with an unknown status");
    }
}

}  // namespace

void paraselect::search::pq_adc_query_lut_pair_scan(
    const PQQueryLutView& pq_query_lut,
    const idx_t* top_probes,
    const uint8_t* codes,
    const long long* list_offsets,
    const int* list_counts,
    const float* thresholds,
    int query_count,
    int list_count,
    int probe_count,
    int pq_m,
    const PQAdcThresholdCandidatePoolView& output,
    cudaStream_t stream,
    PQAdcLutReadMode lut_read_mode,
    PQAdcSharedLutLayout shared_lut_layout,
    int probes_per_cta) {
    if (!pq_query_lut.enabled() || pq_query_lut.sign_codec.codes || !top_probes ||
        !codes || !list_offsets || !list_counts || !thresholds || query_count <= 0 ||
        list_count <= 0 || probe_count <= 0 || pq_m <= 0 || output.capacity <= 0 ||
        !output.indices || !output.distances || !output.queries || !output.query_counts ||
        !output.total_count || !output.status) {
        throw std::invalid_argument("invalid pair query-LUT/QUERY_LUT_BIAS scan request");
    }
    if (probes_per_cta < 1 || probes_per_cta > probe_count ||
        (lut_read_mode != PQAdcLutReadMode::Shared && probes_per_cta != 1))
        throw std::invalid_argument("probe grouping requires shared LUT and 1 <= group <= nprobe");
    // All representation and LUT placement choices resolve before launch.
    if (pq_m != 16 && pq_m != 20 && pq_m != 64)
        throw std::invalid_argument("pair query-LUT/QUERY_LUT_BIAS scan supports M16, M20 or M64");
    if (int64_t(query_count) * probe_count > INT_MAX)
        throw std::invalid_argument("pair scan grid exceeds int range");
    const auto dispatch_score = [&](auto plain_tag) {
    constexpr bool PlainL2 = decltype(plain_tag)::value;
    if ((pq_m == 16 || pq_m == 20 || pq_m == 64) && pq_query_lut.quantized() &&
        (lut_read_mode == PQAdcLutReadMode::Global ||
         shared_lut_layout == PQAdcSharedLutLayout::Compact) &&
        (reinterpret_cast<uintptr_t>(pq_query_lut.query_lut_scale) & 15u) == 0) {
        auto launch_specialized = [&](auto m_tag, const auto* correction, auto shared_tag) {
            constexpr int M = decltype(m_tag)::value;
            threshold_adc_pair_u8_kernel<M, PlainL2, decltype(shared_tag)::value><<<dim3(query_count, (probe_count + probes_per_cta - 1) / probes_per_cta), kSegmentedScanThreads, decltype(shared_tag)::value ? M * 256 : 0, stream>>>(
            pq_query_lut.query_codeword_dot_lut_u8, pq_query_lut.query_lut_scale,
            pq_query_lut.query_lut_lo_sum, pq_query_lut.query_lut_inflation,
            pq_query_lut.query_centroid_distances, correction,
            pq_query_lut.query_lut_bias.scale, pq_query_lut.query_lut_bias.offset,
            top_probes, codes, list_offsets, list_counts, thresholds,
            query_count, probe_count, list_count, output.capacity, output.indices,
            output.distances, output.queries, output.query_counts, output.total_count,
            output.status, output.accepted_counts, nullptr, probes_per_cta);
        };
        const auto launch_m = [&](auto m_tag) {
            const auto launch_read = [&](auto shared_tag) {
                if (pq_query_lut.quantized_query_lut_bias())
                    launch_specialized(m_tag, pq_query_lut.query_lut_bias.u16, shared_tag);
                else
                    launch_specialized(m_tag, pq_query_lut.query_lut_bias.fp32, shared_tag);
            };
            if (lut_read_mode == PQAdcLutReadMode::Shared) launch_read(std::true_type{});
            else launch_read(std::false_type{});
        };
        if (pq_m == 16) launch_m(std::integral_constant<int, 16>{});
        else if (pq_m == 20) launch_m(std::integral_constant<int, 20>{});
        else launch_m(std::integral_constant<int, 64>{});
        CUDA_LAUNCH_CHECK("pair_u8");
        return;
    }
    const auto dispatch_m = [&](auto m_tag, auto metadata_tag) {
    constexpr int M = decltype(m_tag)::value;
    auto launch = [&](auto quantized_lut, auto quantized_query_lut_bias, auto shared, auto stride) {
        constexpr bool U8 = decltype(quantized_lut)::value;
        constexpr bool QUERY_LUT_BIAS_U16 = decltype(quantized_query_lut_bias)::value;
        constexpr bool Shared = decltype(shared)::value;
        constexpr int Stride = decltype(stride)::value;
        const size_t bytes = Shared ? size_t(pq_m) * Stride * (U8 ? 1 : sizeof(float)) : 0;
        constexpr bool Metadata = decltype(metadata_tag)::value && U8 && QUERY_LUT_BIAS_U16 && !Shared;
        threshold_adc_pq_query_lut_pair_kernel<
            M, Shared, Stride, U8, QUERY_LUT_BIAS_U16, Metadata, PlainL2><<<
            query_count * ((probe_count + probes_per_cta - 1) / probes_per_cta), kSegmentedScanThreads, bytes, stream>>>(PairScanArgs{pq_query_lut.query_codeword_dot_lut,pq_query_lut.query_codeword_dot_lut_u8,pq_query_lut.query_lut_scale,pq_query_lut.query_lut_lo_sum,pq_query_lut.query_lut_inflation,pq_query_lut.query_centroid_distances,pq_query_lut.query_lut_bias,codes,list_offsets,list_counts,thresholds,query_count,probe_count,list_count,output.capacity,output.indices,output.distances,output.queries,output.query_counts,output.total_count,output.status,output.accepted_counts,nullptr,top_probes}, probes_per_cta);
    };
    auto dispatch_layout = [&](auto quantized_lut, auto quantized_query_lut_bias) {
        if (lut_read_mode == PQAdcLutReadMode::Global) {
            launch(quantized_lut, quantized_query_lut_bias, std::false_type{},
                   std::integral_constant<int,kCompactLutStride>{});
        } else if (lut_read_mode == PQAdcLutReadMode::Shared) {
            if (shared_lut_layout == PQAdcSharedLutLayout::Compact)
                launch(quantized_lut, quantized_query_lut_bias, std::true_type{},
                       std::integral_constant<int,kCompactLutStride>{});
            else
                launch(quantized_lut, quantized_query_lut_bias, std::true_type{},
                       std::integral_constant<int,kLutStride>{});
        } else throw std::invalid_argument("invalid pair query LUT read mode");
    };
    if (pq_query_lut.quantized()) {
        if (pq_query_lut.quantized_query_lut_bias()) dispatch_layout(std::true_type{}, std::true_type{});
        else dispatch_layout(std::true_type{}, std::false_type{});
    } else {
        if (pq_query_lut.quantized_query_lut_bias()) dispatch_layout(std::false_type{}, std::true_type{});
        else dispatch_layout(std::false_type{}, std::false_type{});
    }
    };
    if (pq_m == 16) {
        const bool optimized = use_optimized_scan_metadata(
            pq_query_lut, pq_m, query_count, probe_count, lut_read_mode);
        if (optimized) dispatch_m(std::integral_constant<int,16>{}, std::true_type{});
        else dispatch_m(std::integral_constant<int,16>{}, std::false_type{});
    } else {
        const auto dispatch_packed = [&](auto m_tag) {
            dispatch_m(m_tag, std::false_type{});
        };
        if (pq_m == 20) dispatch_packed(std::integral_constant<int,20>{});
        else dispatch_packed(std::integral_constant<int,64>{});
    }
    };
    if (pq_query_lut.kind == PQQueryLutKind::PlainL2) dispatch_score(std::true_type{});
    else dispatch_score(std::false_type{});
    CUDA_LAUNCH_CHECK("pair query-LUT/QUERY_LUT_BIAS scan");
}

void paraselect::search::pq_adc_query_lut_segmented_scan(
    const PQQueryLutView& pq_query_lut,
    const IVFProbeScheduleView& schedule,
    const IVFProbeTileScheduleView* tile_schedule,
    const uint8_t* codes,
    const long long* list_offsets,
    const int* list_counts,
    const float* thresholds,
    int query_count,
    int probe_count,
    int pq_m,
    PQAdcListScanRowMode row_mode,
    PQAdcLutReadMode lut_read_mode,
    unsigned long long* physical_code_rows,
    const PQAdcThresholdCandidatePoolView& output,
    cudaStream_t stream,
    PQAdcSharedLutLayout shared_lut_layout,
    int probe_begin,
    int probe_end) {
    if (!pq_query_lut.enabled() || !schedule.clusters || !schedule.starts ||
        !schedule.counts || !schedule.queries || !schedule.probe_ranks ||
        (!codes && !pq_query_lut.sign_codec.codes) || !list_offsets || !list_counts || !thresholds ||
        query_count <= 0 || probe_count <= 0 || output.capacity <= 0 ||
        !output.indices || !output.distances || !output.queries ||
        !output.query_counts || !output.total_count || !output.status) {
        throw std::invalid_argument("invalid segmented query-LUT/QUERY_LUT_BIAS scan request");
    }
    if (pq_m != 16 && pq_m != 20 && pq_m != 64) {
        throw std::invalid_argument(
            "segmented query-LUT/QUERY_LUT_BIAS scan supports pq_m=16, 20 or 64");
    }
    if (tile_schedule &&
        ((!tile_schedule->entry_tile_offsets && tile_schedule->work_count > 0) ||
         tile_schedule->entry_count != schedule.entry_count ||
         tile_schedule->work_count < 0 ||
         (tile_schedule->tile_rows != 256 && tile_schedule->tile_rows != 512))) {
        throw std::invalid_argument("invalid fixed-tile segmented scan schedule");
    }
    if (probe_begin < 0 || probe_begin > probe_count) {
        throw std::invalid_argument("invalid segmented probe_begin");
    }
    if (probe_end < 0) probe_end = probe_count;
    if (probe_end > probe_count || probe_begin > probe_end) {
        throw std::invalid_argument("invalid segmented probe_end");
    }
    if (probe_begin == probe_end || schedule.entry_count <= 0) return;
    const int work_count = tile_schedule
        ? tile_schedule->work_count
        : schedule.entry_count;
    if (work_count <= 0) return;
    if (pq_query_lut.kind == PQQueryLutKind::PlainL2) {
        if (tile_schedule || row_mode != PQAdcListScanRowMode::Serial ||
            lut_read_mode != PQAdcLutReadMode::Global ||
            probe_begin != 0 || probe_end != probe_count)
            throw std::invalid_argument("plain query-LUT currently requires serial entry/global full-probe scan");
        const auto launch_plain = [&](auto m_tag, auto u8_tag) {
            constexpr int M = decltype(m_tag)::value;
            constexpr bool U8 = decltype(u8_tag)::value;
            threshold_adc_pq_query_lut_segmented_kernel<
                M, false, false, false, kCompactLutStride, U8, false,
                false, false, false, false, true><<<
                work_count, kSegmentedScanThreads, 0, stream>>>(
                pq_query_lut.query_codeword_dot_lut, pq_query_lut.query_codeword_dot_lut_u8,
                pq_query_lut.query_lut_scale, pq_query_lut.query_lut_lo_sum,
                pq_query_lut.query_lut_inflation, nullptr, {},
                schedule.clusters, schedule.starts, schedule.counts, nullptr,
                schedule.queries, schedule.probe_ranks, codes, list_offsets, list_counts,
                thresholds, query_count, probe_count, schedule.entry_count,
                kSegmentedTileRows, output.capacity, output.indices, output.distances,
                output.queries, output.query_counts, output.total_count, output.status,
                output.accepted_counts, physical_code_rows, 0, probe_count, {});
        };
        const auto dispatch_m = [&](auto m_tag) {
            if (pq_query_lut.quantized()) launch_plain(m_tag, std::true_type{});
            else launch_plain(m_tag, std::false_type{});
        };
        if (pq_m == 16) dispatch_m(std::integral_constant<int, 16>{});
        else if (pq_m == 20) dispatch_m(std::integral_constant<int, 20>{});
        else dispatch_m(std::integral_constant<int, 64>{});
        CUDA_LAUNCH_CHECK("plain query-LUT segmented scan");
        return;
    }
    const bool compact_shared_lut =
        shared_lut_layout == PQAdcSharedLutLayout::Compact;
    if (lut_read_mode != PQAdcLutReadMode::Shared &&
        lut_read_mode != PQAdcLutReadMode::Global) {
        throw std::invalid_argument("invalid query LUT read mode");
    }
    if (shared_lut_layout != PQAdcSharedLutLayout::Padded &&
        shared_lut_layout != PQAdcSharedLutLayout::Compact) {
        throw std::invalid_argument("invalid shared LUT layout");
    }
    // Dispatch probe ranges once on the host; full scans omit range checks.
    const auto launch_segmented = [&](auto range_tag, auto metadata_tag) {
    constexpr bool kFilterProbeRange = decltype(range_tag)::value;
    constexpr bool kOptimizedMetadata = decltype(metadata_tag)::value;
#define LAUNCH_SEGMENTED(FIXED_TILE_WORK, INTERLEAVE_ROWS, SHARED_LUT, LUT_STRIDE, QUANTIZED, QUANTIZED_PC, OFFSETS, TQ) \
    do { \
    constexpr bool optimized_metadata = kOptimizedMetadata && !FIXED_TILE_WORK && \
        !INTERLEAVE_ROWS && !SHARED_LUT && QUANTIZED && QUANTIZED_PC && !TQ; \
        threshold_adc_pq_query_lut_segmented_kernel< \
        kSegmentedPqM, FIXED_TILE_WORK, INTERLEAVE_ROWS, SHARED_LUT, LUT_STRIDE, QUANTIZED, QUANTIZED_PC, kFilterProbeRange, TQ, \
        optimized_metadata, false><<< \
        work_count, kSegmentedScanThreads, (SHARED_LUT ? \
            (size_t)kSegmentedPqM * LUT_STRIDE * \
                (QUANTIZED ? sizeof(uint8_t) : sizeof(float)) : 0), stream>>>( \
        pq_query_lut.query_codeword_dot_lut, \
        pq_query_lut.query_codeword_dot_lut_u8, \
        pq_query_lut.query_lut_scale, \
        pq_query_lut.query_lut_lo_sum, \
        pq_query_lut.query_lut_inflation, \
        pq_query_lut.query_centroid_distances, \
        pq_query_lut.query_lut_bias, \
        schedule.clusters, \
        schedule.starts, \
        schedule.counts, \
        OFFSETS, \
        schedule.queries, \
        schedule.probe_ranks, \
        TQ ? reinterpret_cast<const uint8_t*>(pq_query_lut.sign_codec.codes) : codes, \
        list_offsets, \
        list_counts, \
        thresholds, \
        query_count, \
        probe_count, \
        schedule.entry_count, \
        (tile_schedule ? tile_schedule->tile_rows : kSegmentedTileRows), \
        output.capacity, \
        output.indices, \
        output.distances, \
        output.queries, \
        output.query_counts, \
        output.total_count, \
        output.status, \
        output.accepted_counts, \
        physical_code_rows, \
        probe_begin, probe_end, \
        pq_query_lut.sign_codec, nullptr, 0); \
    } while (false)
#define LAUNCH_SEGMENTED_PC(FIXED_TILE_WORK, INTERLEAVE_ROWS, SHARED_LUT, LUT_STRIDE, QUANTIZED, OFFSETS) \
    do { \
        if (quantized_query_lut_bias) { \
            LAUNCH_SEGMENTED(FIXED_TILE_WORK, INTERLEAVE_ROWS, SHARED_LUT, LUT_STRIDE, QUANTIZED, true, OFFSETS, false); \
        } else { \
            LAUNCH_SEGMENTED(FIXED_TILE_WORK, INTERLEAVE_ROWS, SHARED_LUT, LUT_STRIDE, QUANTIZED, false, OFFSETS, false); \
        } \
    } while (false)
    if (pq_query_lut.sign_codec.codes) {
        const char* scan = std::getenv("PARASELECT_TURBOQUANT_SCAN");
        if (scan && std::strcmp(scan, "direct") != 0 && std::strcmp(scan, "mma") != 0)
            throw std::invalid_argument("PARASELECT_TURBOQUANT_SCAN must be direct or mma");
        if (scan && std::strcmp(scan, "mma") == 0) {
            const char* width = std::getenv("PARASELECT_RPQ_PROBE_ENTRY_WIDTH");
            if (!width || std::strcmp(width, "32") != 0)
                throw std::invalid_argument("TurboQuant MMA requires PROBE_ENTRY_WIDTH=32; width 8 prevents tensor reuse");
            if (probe_begin != 0 || probe_end != probe_count)
                throw std::invalid_argument("experimental TurboQuant MMA requires a full-probe scan");
            const IVFProbeTileScheduleView tiles = tile_schedule ? *tile_schedule : IVFProbeTileScheduleView{};
            turboquant::mma_scan<<<work_count, 256, 0, stream>>>(pq_query_lut.sign_codec,
                schedule, tiles, list_offsets, list_counts, thresholds, output);
            CUDA_LAUNCH_CHECK("turboquant_mma_scan");
            return;
        }
        if (tile_schedule) {
            LAUNCH_SEGMENTED(true, false, true, kCompactLutStride, false, false,
                             tile_schedule->entry_tile_offsets, true);
        } else {
            LAUNCH_SEGMENTED(false, false, true, kCompactLutStride, false, false, nullptr, true);
        }
        CUDA_LAUNCH_CHECK("turboquant_shared_list_schedule");
        return;
    }
    if (row_mode != PQAdcListScanRowMode::Serial &&
        row_mode != PQAdcListScanRowMode::Interleaved &&
        row_mode != PQAdcListScanRowMode::QueryRowTile4) {
        throw std::invalid_argument("invalid segmented row scan mode");
    }
    if (row_mode == PQAdcListScanRowMode::QueryRowTile4) {
        if (tile_schedule || pq_m != 16 || !pq_query_lut.quantized() ||
            !pq_query_lut.quantized_query_lut_bias() ||
            lut_read_mode != PQAdcLutReadMode::Shared) {
            throw std::invalid_argument(
                "query_row_tile4 requires entry/M=16/shared/u8/QUERY_LUT_BIAS_U16");
        }
        constexpr size_t kTileSharedBytes =
            4u * (16u * kPQK + 4u) + 4u * 16u * sizeof(float) +
            128u * sizeof(uint4) + 128u * sizeof(uint16_t);
        threshold_adc_query_row_tile4_kernel<<<
            schedule.entry_count, 256, kTileSharedBytes, stream>>>(
            pq_query_lut.query_codeword_dot_lut_u8,
            pq_query_lut.query_lut_scale,
            pq_query_lut.query_lut_lo_sum,
            pq_query_lut.query_lut_inflation,
            pq_query_lut.query_centroid_distances,
            pq_query_lut.query_lut_bias,
            schedule.clusters, schedule.starts, schedule.counts,
            schedule.queries, schedule.probe_ranks, codes,
            list_offsets, list_counts, thresholds, query_count, probe_count,
            schedule.entry_count, output.capacity, output.indices,
            output.distances, output.queries, output.query_counts,
            output.total_count, output.status, output.accepted_counts,
            physical_code_rows);
        CUDA_LAUNCH_CHECK("threshold_adc_query_row_tile4");
        return;
    }
    const bool interleave_rows = row_mode == PQAdcListScanRowMode::Interleaved;
    const bool shared_lut = lut_read_mode == PQAdcLutReadMode::Shared;
    const bool quantized_lut = pq_query_lut.quantized();
    const bool quantized_query_lut_bias = pq_query_lut.quantized_query_lut_bias();
    if (pq_m == 20 || pq_m == 64) {
        // Packed widths beyond M16: SPACEV uses 20 code bytes, 1024D text
        // embeddings 64. Each is a separate static specialization: all ADC
        // terms and code words are compile-time constants in the scan loop.
        const auto launch_packed = [&](auto m_tag) {
        constexpr int M = decltype(m_tag)::value;
            if (interleave_rows || shared_lut) {
                throw std::invalid_argument(
                    "pq_m=20/64 list-QUERY_LUT_BIAS currently supports entry-or-tile/serial/global");
            }
            // The global-read scan handles both LUT encodings; FP32 serves the
            // quantization ablation, U8 the formal path.
            const auto launch = [&](auto tiled_tag, auto quantized_tag, auto bias_tag) {
                constexpr bool Tiled = decltype(tiled_tag)::value;
                threshold_adc_pq_query_lut_segmented_kernel<
                    M, Tiled, false, false, kCompactLutStride,
                    decltype(quantized_tag)::value, decltype(bias_tag)::value, kFilterProbeRange><<<
                    work_count, kSegmentedScanThreads, 0, stream>>>(
                    pq_query_lut.query_codeword_dot_lut, pq_query_lut.query_codeword_dot_lut_u8,
                    pq_query_lut.query_lut_scale, pq_query_lut.query_lut_lo_sum,
                    pq_query_lut.query_lut_inflation, pq_query_lut.query_centroid_distances,
                    pq_query_lut.query_lut_bias,
                    schedule.clusters, schedule.starts, schedule.counts,
                    Tiled ? tile_schedule->entry_tile_offsets : nullptr,
                    schedule.queries, schedule.probe_ranks,
                    codes, list_offsets, list_counts, thresholds,
                    query_count, probe_count, schedule.entry_count,
                    Tiled ? tile_schedule->tile_rows : kSegmentedTileRows,
                    output.capacity,
                    output.indices, output.distances, output.queries, output.query_counts,
                    output.total_count, output.status, output.accepted_counts, physical_code_rows,
                    probe_begin, probe_end, pq_query_lut.sign_codec);
            };
            const auto with_bias = [&](auto tiled_tag, auto quantized_tag) {
                if (quantized_query_lut_bias) launch(tiled_tag, quantized_tag, std::true_type{});
                else launch(tiled_tag, quantized_tag, std::false_type{});
            };
            const auto with_lut = [&](auto tiled_tag) {
                if (quantized_lut) with_bias(tiled_tag, std::true_type{});
                else with_bias(tiled_tag, std::false_type{});
            };
            if (tile_schedule) with_lut(std::true_type{});
            else with_lut(std::false_type{});
        };
        if (pq_m == 20) launch_packed(std::integral_constant<int, 20>{});
        else launch_packed(std::integral_constant<int, 64>{});
        CUDA_LAUNCH_CHECK("threshold_adc_pq_query_lut_segmented_packed_m");
        return;
    }
    if (quantized_lut) {
        if (tile_schedule && interleave_rows && shared_lut) {
            LAUNCH_SEGMENTED_PC(true, true, true, kCompactLutStride,
                true, tile_schedule->entry_tile_offsets);
        } else if (tile_schedule && shared_lut) {
            LAUNCH_SEGMENTED_PC(true, false, true, kCompactLutStride,
                true, tile_schedule->entry_tile_offsets);
        } else if (interleave_rows && shared_lut) {
            LAUNCH_SEGMENTED_PC(false, true, true, kCompactLutStride,
                true, nullptr);
        } else if (shared_lut) {
            LAUNCH_SEGMENTED_PC(false, false, true, kCompactLutStride,
                true, nullptr);
        } else if (tile_schedule && interleave_rows) {
            LAUNCH_SEGMENTED_PC(true, true, false, kCompactLutStride,
                true, tile_schedule->entry_tile_offsets);
        } else if (tile_schedule) {
            LAUNCH_SEGMENTED_PC(true, false, false, kCompactLutStride,
                true, tile_schedule->entry_tile_offsets);
        } else if (interleave_rows) {
            LAUNCH_SEGMENTED_PC(false, true, false, kCompactLutStride,
                true, nullptr);
        } else {
            LAUNCH_SEGMENTED_PC(false, false, false, kCompactLutStride,
                true, nullptr);
        }
        CUDA_LAUNCH_CHECK("threshold_adc_pq_query_lut_segmented_quantized");
        return;
    }
    if (tile_schedule && interleave_rows && shared_lut && compact_shared_lut) {
        LAUNCH_SEGMENTED_PC(true, true, true, kCompactLutStride,
            false, tile_schedule->entry_tile_offsets);
    } else if (tile_schedule && interleave_rows && shared_lut) {
        LAUNCH_SEGMENTED_PC(true, true, true, kLutStride,
            false, tile_schedule->entry_tile_offsets);
    } else if (tile_schedule && interleave_rows) {
        LAUNCH_SEGMENTED_PC(true, true, false, kCompactLutStride,
            false, tile_schedule->entry_tile_offsets);
    } else if (tile_schedule && shared_lut && compact_shared_lut) {
        LAUNCH_SEGMENTED_PC(true, false, true, kCompactLutStride,
            false, tile_schedule->entry_tile_offsets);
    } else if (tile_schedule && shared_lut) {
        LAUNCH_SEGMENTED_PC(true, false, true, kLutStride,
            false, tile_schedule->entry_tile_offsets);
    } else if (tile_schedule) {
        LAUNCH_SEGMENTED_PC(true, false, false, kCompactLutStride,
            false, tile_schedule->entry_tile_offsets);
    } else if (interleave_rows && shared_lut && compact_shared_lut) {
        LAUNCH_SEGMENTED_PC(false, true, true, kCompactLutStride, false, nullptr);
    } else if (interleave_rows && shared_lut) {
        LAUNCH_SEGMENTED_PC(false, true, true, kLutStride, false, nullptr);
    } else if (interleave_rows) {
        LAUNCH_SEGMENTED_PC(false, true, false, kCompactLutStride, false, nullptr);
    } else if (shared_lut && compact_shared_lut) {
        LAUNCH_SEGMENTED_PC(false, false, true, kCompactLutStride, false, nullptr);
    } else if (shared_lut) {
        LAUNCH_SEGMENTED_PC(false, false, true, kLutStride, false, nullptr);
    } else {
        LAUNCH_SEGMENTED_PC(false, false, false, kCompactLutStride, false, nullptr);
    }
#undef LAUNCH_SEGMENTED_PC
#undef LAUNCH_SEGMENTED
    };
    // Real replay and forward/reverse pipeline A/B determine this dispatch.
    // Keep the measured specialization outside the device hot loop.
    // The alignment check also preserves the public non-owning view contract.
    const bool optimized_metadata_supported = !tile_schedule &&
        row_mode == PQAdcListScanRowMode::Serial &&
        use_optimized_scan_metadata(pq_query_lut, pq_m, query_count, probe_count, lut_read_mode);
    const auto dispatch_scales = [&](auto range_tag) {
        // Retain only the large-workload specialization supported by full
        // pipeline A/B; all launches use the original 256-thread geometry.
        if (optimized_metadata_supported)
            launch_segmented(range_tag, std::true_type{});
        else launch_segmented(range_tag, std::false_type{});
    };
    if (probe_begin != 0 || probe_end != probe_count) dispatch_scales(std::true_type{});
    else dispatch_scales(std::false_type{});
}

std::size_t paraselect::search::pq_adc_threshold_workspace_required_bytes(
    int max_queries,
    int candidate_capacity,
    int seed_probes) {
    const int pool_capacity = resolve_global_pool_capacity();
    const std::size_t sort_temp_bytes =
        pq_candidate_cap_select_temp_bytes(pool_capacity, max_queries);
    return pq_adc_threshold_workspace_required_bytes(PQAdcThresholdWorkspaceShape{
        max_queries,
        candidate_capacity,
        seed_probes,
        pool_capacity,
        sort_temp_bytes});
}

void paraselect::search::pq_adc_threshold_workspace_reserve(
    PQAdcThresholdWorkspace* ws,
    int max_queries,
    int candidate_capacity,
    int seed_probes) {
    if (!ws || max_queries <= 0 || candidate_capacity <= 0 || seed_probes <= 0) {
        throw std::invalid_argument("pq ADC threshold workspace dimensions are invalid");
    }
    if (ws->max_queries >= max_queries && ws->candidate_capacity >= candidate_capacity &&
        ws->max_seed_probes >= seed_probes) {
        return;
    }
    pq_adc_threshold_workspace_release(ws);
    for (auto& event : ws->timing_events) CUDA_CHECK(cudaEventCreate(&event));
    CUDA_CHECK(cudaMallocHost(&ws->h_seed_scanned_codes, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&ws->d_seed_top_nprobe,
        (size_t)max_queries * seed_probes * sizeof(idx_t)));
    // Match ExternalBaseline: threshold candidates share one fixed batch pool. The
    // per-query candidate_capacity below is only the output stride.
    const size_t pool_capacity = static_cast<size_t>(resolve_global_pool_capacity());
    CUDA_CHECK(cudaMalloc(&ws->d_pool_idx, pool_capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws->d_pool_dist, pool_capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->d_pool_query, pool_capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws->d_global_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws->d_physical_code_rows, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&ws->d_scanned_codes, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMallocHost(&ws->h_scanned_codes, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&ws->d_candidate_idx,
        (size_t)max_queries * candidate_capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws->d_candidate_dist,
        (size_t)max_queries * candidate_capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->d_candidate_count, (size_t)max_queries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws->d_threshold, (size_t)max_queries * sizeof(float)));
    ws->histogram_bins = 1024;
    CUDA_CHECK(cudaMalloc(&ws->d_seed_histogram,
        (size_t)max_queries * ws->histogram_bins * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&ws->d_overflow, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws->d_sort_keys_in, pool_capacity * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&ws->d_sort_keys_out, pool_capacity * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&ws->d_sort_values_out, pool_capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws->d_offsets, (size_t)(max_queries + 1) * sizeof(int)));
    ws->sort_temp_bytes = pq_candidate_cap_select_temp_bytes(static_cast<int>(pool_capacity), max_queries);
    if (ws->sort_temp_bytes > 0) CUDA_CHECK(cudaMalloc(&ws->d_sort_temp, ws->sort_temp_bytes));
    ws->max_queries = max_queries;
    ws->max_seed_probes = seed_probes;
    ws->candidate_capacity = candidate_capacity;
    ws->pool_capacity = static_cast<int>(pool_capacity);
}

void paraselect::search::pq_adc_threshold_workspace_release(PQAdcThresholdWorkspace* ws) {
    if (!ws) return;
    for (auto event : ws->timing_events) if (event) cudaEventDestroy(event);
    if (ws->h_seed_scanned_codes) cudaFreeHost(ws->h_seed_scanned_codes);
    if (ws->d_accepted_count) cudaFree(ws->d_accepted_count);
    if (ws->d_seed_top_nprobe) cudaFree(ws->d_seed_top_nprobe);
    if (ws->d_pool_idx) cudaFree(ws->d_pool_idx);
    if (ws->d_pool_dist) cudaFree(ws->d_pool_dist);
    if (ws->d_pool_query) cudaFree(ws->d_pool_query);
    if (ws->d_global_count) cudaFree(ws->d_global_count);
    if (ws->d_physical_code_rows) cudaFree(ws->d_physical_code_rows);
    if (ws->d_scanned_codes) cudaFree(ws->d_scanned_codes);
    if (ws->h_scanned_codes) cudaFreeHost(ws->h_scanned_codes);
    if (ws->d_candidate_idx) cudaFree(ws->d_candidate_idx);
    if (ws->d_candidate_dist) cudaFree(ws->d_candidate_dist);
    if (ws->d_candidate_count) cudaFree(ws->d_candidate_count);
    if (ws->d_threshold) cudaFree(ws->d_threshold);
    if (ws->d_seed_histogram) cudaFree(ws->d_seed_histogram);
    if (ws->d_overflow) cudaFree(ws->d_overflow);
    if (ws->d_sort_keys_in) cudaFree(ws->d_sort_keys_in);
    if (ws->d_sort_keys_out) cudaFree(ws->d_sort_keys_out);
    if (ws->d_sort_values_out) cudaFree(ws->d_sort_values_out);
    if (ws->d_offsets) cudaFree(ws->d_offsets);
    if (ws->d_sort_temp) cudaFree(ws->d_sort_temp);
    *ws = PQAdcThresholdWorkspace{};
}

namespace {
// Shared policy, candidate pool, accounting and cap-select. The source-specific
// seed/scan operations capture only the inputs their kernels consume.
template<class Seed, class Histogram, class Scan>
void threshold_search(
    const idx_t* d_top_nprobe, const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets, const int* d_cluster_counts,
    int n_query, int n_total_clusters, int n_probes, int pq_m,
    const paraselect::search::PQAdcThresholdParams& params,
    const paraselect::search::PQQueryLutView* pq_query_lut,
    const paraselect::search::IVFProbeScheduleView* probe_schedule,
    const paraselect::search::IVFProbeTileScheduleView* probe_tile_schedule,
    cudaStream_t stream, paraselect::search::PQAdcThresholdWorkspace* threshold_workspace,
    ResidualPQGpuDebugTiming* timing, Seed seed, Histogram histogram, Scan scan) {
    if (n_query <= 0) return;
    if (!d_top_nprobe || !d_cluster_offsets || !d_cluster_counts || !threshold_workspace)
        throw std::invalid_argument("threshold search: missing probes, storage or workspace");
    if (pq_query_lut && !pq_query_lut->enabled()) {
        throw std::invalid_argument(
            "pq_adc_threshold_search_gpu: query-LUT/QUERY_LUT_BIAS view is incomplete");
    }
    const bool list_centric = params.scan_layout ==
        paraselect::search::PQAdcThresholdScanLayout::ListCentric;
    const bool fixed_tile_work = params.list_work_mode ==
        paraselect::search::PQAdcListScanWorkMode::FixedTile;
    const bool interleave_rows = params.list_row_mode ==
        paraselect::search::PQAdcListScanRowMode::Interleaved;
    const bool query_row_tile4 = params.list_row_mode ==
        paraselect::search::PQAdcListScanRowMode::QueryRowTile4;
    if (params.scan_layout !=
            paraselect::search::PQAdcThresholdScanLayout::PairCentric &&
        !list_centric) {
        throw std::invalid_argument("pq_adc_threshold_search_gpu: invalid scan layout");
    }
    if (list_centric && (!pq_query_lut || !probe_schedule ||
                         (pq_m != 16 && pq_m != 20 && pq_m != 64))) {
        throw std::invalid_argument(
            "pq_adc_threshold_search_gpu: list-centric scan requires query-LUT, "
            "a probe schedule, and pq_m=16, 20 or 64");
    }
    if (fixed_tile_work && (!list_centric || !probe_tile_schedule)) {
        throw std::invalid_argument(
            "pq_adc_threshold_search_gpu: fixed-tile work requires a list-centric tile schedule");
    }
    if (params.list_row_mode !=
            paraselect::search::PQAdcListScanRowMode::Serial &&
        !interleave_rows && !query_row_tile4) {
        throw std::invalid_argument("pq_adc_threshold_search_gpu: invalid list row mode");
    }
    if (query_row_tile4 && (!pq_query_lut || !list_centric || fixed_tile_work || pq_m != 16 ||
                            params.lut_read_mode !=
                                paraselect::search::PQAdcLutReadMode::Shared ||
                            !pq_query_lut->quantized() ||
                            !pq_query_lut->quantized_query_lut_bias())) {
        throw std::invalid_argument(
            "pq_adc_threshold_search_gpu: query_row_tile4 requires entry/M=16/shared/u8/QUERY_LUT_BIAS_U16");
    }
    const int seed_probes = std::min(params.seed_probes, n_probes);
    const int capacity = threshold_workspace->candidate_capacity;
    const bool use_histogram = params.policy ==
        paraselect::search::PQThresholdPolicy::DistanceHistogram;
    const int seed_rank = use_histogram ? params.n_rerank : params.seed_rank;
    const bool query_lut_seed = pq_query_lut && !pq_query_lut->sign_codec.codes;
    if (use_histogram && ((pq_query_lut && pq_query_lut->sign_codec.codes) ||
        params.histogram_bins <= 0 || params.histogram_bins > (query_lut_seed ? paraselect::search::PQDistanceHistogramMaxBins : 1024))) {
        throw std::invalid_argument("unsupported ResPQ ADC histogram size");
    }
    const int global_pool_capacity = threshold_workspace->pool_capacity;
    const int candidate_k = params.candidate_k;
    if (seed_probes <= 0 || seed_rank <= 0 || seed_rank > capacity ||
        candidate_k <= 0 || candidate_k > capacity ||
        seed_rank > params.candidate_capacity || candidate_k > params.candidate_capacity ||
        params.candidate_capacity < 512 || params.candidate_capacity > 8192 ||
        (params.candidate_capacity & (params.candidate_capacity - 1)) ||
        threshold_workspace->max_queries < n_query ||
        threshold_workspace->max_seed_probes < seed_probes ||
        global_pool_capacity <= 0) {
        throw std::invalid_argument("pq_adc_threshold_search_gpu: unsupported prepared spec");
    }
    if (!query_lut_seed && seed_rank > 512) {
        throw std::invalid_argument(
            "pq_adc_threshold_search_gpu: seed_rank > 512 requires query-LUT seed");
    }

    const auto& events = threshold_workspace->timing_events;
    const auto seed_start = events[0], seed_compact_stop = events[1], seed_stop = events[2];
    const auto policy_start = events[3], policy_stop = events[4];
    const auto scan_start = events[5], scan_setup_stop = events[6], scan_stop = events[7];
    const auto cap_select_start = events[8], cap_select_stop = events[9];
    if (timing) CUDA_CHECK(cudaEventRecord(seed_start, stream));

    const int seed_items = n_query * seed_probes;
    // When the seed covers the complete probe row, the source already has the
    // exact stride expected by the dense seed kernels. Avoid a redundant copy.
    const idx_t* seed_top_nprobe = threshold_workspace->d_seed_top_nprobe;
    if (seed_probes == n_probes) {
        seed_top_nprobe = d_top_nprobe;
    } else {
        compact_seed_probes_kernel<<<(seed_items + 255) / 256, 256, 0, stream>>>(
            d_top_nprobe, n_query, n_probes, seed_probes,
            threshold_workspace->d_seed_top_nprobe);
        CUDA_LAUNCH_CHECK("pq_adc_compact_seed_probes_kernel");
    }
    if (timing) CUDA_CHECK(cudaEventRecord(seed_compact_stop, stream));
    unsigned long long seed_scanned_codes = 0;
    CUDA_CHECK(cudaMemsetAsync(threshold_workspace->d_physical_code_rows, 0,
        sizeof(unsigned long long), stream));
    if (!use_histogram) seed_scanned_codes = seed(seed_top_nprobe, seed_probes, seed_rank);
    const int histogram_bins = params.histogram_bins;
    float histogram_log_step = 0.5f;
    if (use_histogram) {
        if (histogram_bins != 64 && histogram_bins != 128 &&
            histogram_bins != 256 && histogram_bins != 512 &&
            histogram_bins != 1024 &&
            !(query_lut_seed && (histogram_bins == 2048 ||
              histogram_bins == 4096 || histogram_bins == 8192))) {
            throw std::invalid_argument(
                "histogram bins must be powers of two from 64 to 1024 (8192 for query LUT)");
        }
        const float histogram_log_min = -8.0f;
        histogram_log_step = 32.0f / static_cast<float>(histogram_bins);
        if (histogram_bins > threshold_workspace->histogram_bins) {
            unsigned int* expanded = nullptr;
            CUDA_CHECK(cudaMalloc(&expanded, size_t(threshold_workspace->max_queries) *
                histogram_bins * sizeof(unsigned int)));
            CUDA_CHECK(cudaFree(threshold_workspace->d_seed_histogram));
            threshold_workspace->d_seed_histogram = expanded;
            threshold_workspace->histogram_bins = histogram_bins;
        }
        CUDA_CHECK(cudaMemsetAsync(
            threshold_workspace->d_seed_histogram, 0,
            static_cast<size_t>(n_query) * histogram_bins * sizeof(unsigned int),
            stream));
        histogram(seed_probes, histogram_bins, histogram_log_min, histogram_log_step);
        CUDA_LAUNCH_CHECK("pq_seed_distance_histogram");
    }
    if (timing) {
        CUDA_CHECK(cudaEventRecord(seed_stop, stream));
    }
    if (use_histogram) {
        paraselect::search::pq_count_scanned_codes_async(
            seed_top_nprobe, d_cluster_counts, n_query, seed_probes,
            stream, threshold_workspace->d_scanned_codes, threshold_workspace->h_seed_scanned_codes);
    }
    if (query_lut_seed && !use_histogram) {
        CUDA_CHECK(cudaMemcpyAsync(threshold_workspace->h_seed_scanned_codes,
            threshold_workspace->d_physical_code_rows,
            sizeof(seed_scanned_codes), cudaMemcpyDeviceToHost, stream));
    }

    // Diagnostic seed-count readback is not threshold-policy execution.
    if (timing) CUDA_CHECK(cudaEventRecord(policy_start, stream));
    CUDA_CHECK(cudaMemsetAsync(threshold_workspace->d_overflow, 0, sizeof(int), stream));
    if (use_histogram) {
        paraselect::search::pq_histogram_threshold_generate_gpu(
            threshold_workspace->d_seed_histogram, n_query,
            histogram_bins, seed_rank, -8.0f, histogram_log_step,
            threshold_workspace->d_threshold, threshold_workspace->d_overflow,
            stream);
    } else {
        paraselect::search::pq_threshold_generate_gpu(
            threshold_workspace->d_candidate_dist,
            threshold_workspace->d_candidate_idx,
            n_query, seed_rank, params,
            threshold_workspace->d_threshold,
            threshold_workspace->d_overflow, stream);
    }
    CUDA_LAUNCH_CHECK("pq_threshold_policy_kernel");
    if (timing) CUDA_CHECK(cudaEventRecord(policy_stop, stream));
    int status = 0;
    if (timing) CUDA_CHECK(cudaEventRecord(scan_start, stream));

    const bool audit = params.audit;
    if (audit) {
        if (params.trace_seed && !use_histogram) {
            paraselect::search::detail::audit_threshold_seed(
                seed_top_nprobe,
                threshold_workspace->d_candidate_idx, threshold_workspace->d_candidate_dist,
                n_query, seed_probes, seed_rank);
        }
        if (!threshold_workspace->d_accepted_count) {
            CUDA_CHECK(cudaMalloc(&threshold_workspace->d_accepted_count,
                (size_t)threshold_workspace->max_queries * sizeof(unsigned long long)));
        }
        CUDA_CHECK(cudaMemsetAsync(threshold_workspace->d_accepted_count, 0,
            (size_t)n_query * sizeof(unsigned long long), stream));
    }
    initialize_threshold_pool_kernel<<<(n_query + 255) / 256, 256, 0, stream>>>(
        n_query, threshold_workspace->d_candidate_count, threshold_workspace->d_global_count);
    CUDA_LAUNCH_CHECK("pq_adc_threshold_initialize_pool");
    paraselect::search::pq_count_scanned_codes_async(
        d_top_nprobe, d_cluster_counts, n_query, n_probes, stream, threshold_workspace->d_scanned_codes, threshold_workspace->h_scanned_codes);
    unsigned long long full_scan_physical_rows = 0;
    bool copy_physical_rows = false;
    if (timing) CUDA_CHECK(cudaEventRecord(scan_setup_stop, stream));
        const paraselect::search::PQAdcThresholdCandidatePoolView output{
            global_pool_capacity,
            threshold_workspace->d_pool_idx,
            threshold_workspace->d_pool_dist,
            threshold_workspace->d_pool_query,
            threshold_workspace->d_candidate_count,
            threshold_workspace->d_global_count,
            threshold_workspace->d_overflow,
            audit ? threshold_workspace->d_accepted_count : nullptr};
    scan(output, 0, n_probes);
    copy_physical_rows = list_centric;
    if (timing) CUDA_CHECK(cudaEventRecord(scan_stop, stream));
    CUDA_LAUNCH_CHECK("pq_adc_threshold_scan_kernel");
    if (copy_physical_rows) {
        CUDA_CHECK(cudaMemcpyAsync(
            &full_scan_physical_rows,
            threshold_workspace->d_physical_code_rows,
            sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(&status, threshold_workspace->d_overflow, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (audit) {
        std::vector<idx_t> host_probes((size_t)n_query * n_probes);
        std::vector<int> host_counts(n_total_clusters);
        CUDA_CHECK(cudaMemcpy(host_probes.data(), d_top_nprobe,
            host_probes.size() * sizeof(idx_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_counts.data(), d_cluster_counts,
            host_counts.size() * sizeof(int), cudaMemcpyDeviceToHost));
        std::fprintf(stderr, "[PQ_THRESHOLD_QUERY_ROWS]");
        const int sample[] = {0, 1, 2, 3, n_query - 10, n_query - 9,
                              n_query - 8, n_query - 7, n_query - 6, n_query - 5,
                              n_query - 4, n_query - 3, n_query - 2, n_query - 1};
        for (int q : sample) {
            if (q < 0 || q >= n_query) continue;
            unsigned long long rows = 0;
            for (int p = 0; p < n_probes; ++p) {
                const idx_t c = host_probes[(size_t)q * n_probes + p];
                if (c >= 0 && c < n_total_clusters) rows += host_counts[c];
            }
            std::fprintf(stderr, " q%d=%llu", q, rows);
        }
        std::fprintf(stderr, "\n");
    }
    int observed_global_count = 0;
    CUDA_CHECK(cudaMemcpy(&observed_global_count, threshold_workspace->d_global_count,
                          sizeof(int), cudaMemcpyDeviceToHost));
    if (status != 0) {
        std::fprintf(stderr,
            "[PQ_THRESHOLD_STATUS] status=%d global_count=%d global_capacity=%d n_query=%d nprobe=%d candidate_capacity=%d\n",
            status, observed_global_count, global_pool_capacity, n_query, n_probes, capacity);
        std::fflush(stderr);
    }
    if (query_lut_seed || use_histogram) seed_scanned_codes = *threshold_workspace->h_seed_scanned_codes;
    threshold_workspace->scanned_codes =
        seed_scanned_codes + *threshold_workspace->h_scanned_codes;
    if (!list_centric) {
        full_scan_physical_rows = *threshold_workspace->h_scanned_codes;
    }
    threshold_workspace->physical_code_rows =
        seed_scanned_codes + full_scan_physical_rows;
    int total_candidates = observed_global_count;
    if (audit) {
        std::vector<unsigned long long> accepted(n_query);
        std::vector<int> kept(n_query);
        CUDA_CHECK(cudaMemcpy(accepted.data(), threshold_workspace->d_accepted_count,
            (size_t)n_query * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kept.data(), threshold_workspace->d_candidate_count,
            (size_t)n_query * sizeof(int), cudaMemcpyDeviceToHost));
        unsigned long long total = 0, stored = 0;
        for (int q = 0; q < n_query; ++q) {
            if (status == 0 && (unsigned long long)kept[q] != accepted[q]) {
                throw std::runtime_error("PQ threshold audit count invariant failed");
            }
            total += accepted[q];
            stored += kept[q];
        }
        // Counts here precede cap-select; the public candidate output below is
        // capped and must not be used to diagnose threshold candidate growth.
        auto sorted_kept = kept;
        std::sort(sorted_kept.begin(), sorted_kept.end());
        std::fprintf(stderr,
            "[PQ_THRESHOLD_AUDIT] queries=%d output_stride=%d global_pool_capacity=%d accepted=%llu stored=%llu logical_code_rows=%llu physical_code_rows=%llu overflow=%d candidate_min=%d candidate_p50=%d candidate_p95=%d candidate_max=%d\n",
            n_query, candidate_k, global_pool_capacity, total, stored,
            threshold_workspace->scanned_codes,
            threshold_workspace->physical_code_rows, status,
            sorted_kept.front(), sorted_kept[n_query/2], sorted_kept[size_t(n_query-1)*95/100], sorted_kept.back());
        if (status != 0) {
            std::vector<std::pair<int, int>> largest;
            largest.reserve(n_query);
            for (int q = 0; q < n_query; ++q) largest.emplace_back(kept[q], q);
            const size_t begin = largest.size() > 10 ? largest.size() - 10 : 0;
            std::nth_element(largest.begin(), largest.begin() + begin, largest.end());
            std::sort(largest.begin() + begin, largest.end(), std::greater<>());
            std::fprintf(stderr, "[PQ_THRESHOLD_QUERY_IDS_V2]");
            for (size_t i = begin; i < largest.size(); ++i)
                std::fprintf(stderr, " q%d=%d", largest[i].second, largest[i].first);
            std::fprintf(stderr, "\n");
        }
        if (use_histogram) {
            std::vector<unsigned> histogram(size_t(n_query)*histogram_bins);
            CUDA_CHECK(cudaMemcpy(histogram.data(), threshold_workspace->d_seed_histogram,
                histogram.size()*sizeof(unsigned), cudaMemcpyDeviceToHost));
            std::vector<unsigned long long> seed_rows(n_query,0);
            unsigned long long seed_total = 0;
            int insufficient = 0;
            for (int q=0; q<n_query; ++q) {
                for (int bin=0; bin<histogram_bins; ++bin)
                    seed_rows[q] += histogram[size_t(q)*histogram_bins+bin];
                seed_total += seed_rows[q];
                insufficient += seed_rows[q] < static_cast<unsigned long long>(seed_rank);
            }
            auto sorted_seed_rows = seed_rows;
            std::sort(sorted_seed_rows.begin(),sorted_seed_rows.end());
            std::fprintf(stderr,
                "[PQ_SEED_HISTOGRAM_AUDIT] queries=%d probes=%d rank=%d rows=%llu row_min=%llu row_p50=%llu row_max=%llu below_rank=%d\n",
                n_query, seed_probes, seed_rank, seed_total,
                sorted_seed_rows.front(),sorted_seed_rows[n_query/2],sorted_seed_rows.back(),insufficient);
            std::vector<float> thresholds(n_query);
            CUDA_CHECK(cudaMemcpy(thresholds.data(), threshold_workspace->d_threshold,
                (size_t)n_query * sizeof(float), cudaMemcpyDeviceToHost));
            std::fprintf(stderr, "[PQ_SEED_HISTOGRAM_TAIL]");
            const int tail_begin = std::max(0, n_query - 16);
            for (int q = tail_begin; q < n_query; ++q) {
                const unsigned overflow_bin = histogram[(size_t)q * histogram_bins +
                    (histogram_bins - 1)];
                std::fprintf(stderr, " q%d:rows=%llu:overflow=%u:threshold=%g",
                    q, seed_rows[q], overflow_bin, thresholds[q]);
            }
            std::fprintf(stderr, "\n");
        }
    }
    if (total_candidates < 0 || total_candidates > global_pool_capacity) {
        throw std::runtime_error("pq ADC threshold global candidate pool count is invalid");
    }
    if (status != 0) throw_threshold_status(status);
    if (timing) CUDA_CHECK(cudaEventRecord(cap_select_start, stream));
    paraselect::search::pq_candidate_cap_select(
        threshold_workspace->d_pool_idx, threshold_workspace->d_pool_dist,
        threshold_workspace->d_pool_query, threshold_workspace->d_candidate_count,
        total_candidates, n_query, capacity, candidate_k,
        threshold_workspace->d_sort_keys_in, threshold_workspace->d_sort_keys_out,
        threshold_workspace->d_sort_values_out, threshold_workspace->d_offsets,
        threshold_workspace->d_candidate_idx, threshold_workspace->d_candidate_dist,
        threshold_workspace->d_candidate_count, threshold_workspace->d_sort_temp,
        threshold_workspace->sort_temp_bytes, stream);
    const char* fingerprint = std::getenv("PARASELECT_PQ_THRESHOLD_CANDIDATE_FINGERPRINT");
    if (fingerprint && std::strcmp(fingerprint, "1") == 0) {
        std::vector<int> counts(n_query);
        std::vector<int> indices((size_t)n_query * capacity);
        std::vector<float> distances((size_t)n_query * capacity);
        CUDA_CHECK(cudaMemcpyAsync(counts.data(), threshold_workspace->d_candidate_count,
            (size_t)n_query * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(indices.data(), threshold_workspace->d_candidate_idx,
            indices.size() * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(distances.data(), threshold_workspace->d_candidate_dist,
            distances.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int q = 0; q < n_query; ++q) {
            uint64_t digest = 14695981039346656037ULL;
            const int kept = counts[q];
            if (kept < 0 || kept > candidate_k) {
                throw std::runtime_error("invalid candidate count in fingerprint audit");
            }
            for (int rank = 0; rank < kept; ++rank) {
                const size_t pos = (size_t)q * capacity + rank;
                uint32_t distance_bits;
                std::memcpy(&distance_bits, &distances[pos], sizeof(distance_bits));
                digest = (digest ^ static_cast<uint32_t>(indices[pos])) * 1099511628211ULL;
                digest = (digest ^ distance_bits) * 1099511628211ULL;
            }
            std::fprintf(stderr, "[PQ_CANDIDATES] query=%d kept=%d hash=%016llx\n",
                q, kept, static_cast<unsigned long long>(digest));
        }
    }
    if (timing) {
        CUDA_CHECK(cudaEventRecord(cap_select_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(cap_select_stop));
        CUDA_CHECK(cudaEventElapsedTime(
            &timing->candidate_cap_select_ms, cap_select_start, cap_select_stop));
        CUDA_CHECK(cudaEventElapsedTime(
            &timing->seed_count_ms, seed_stop, policy_start));
        CUDA_CHECK(cudaEventElapsedTime(
            &timing->candidate_readback_ms, scan_stop, cap_select_start));
        CUDA_CHECK(cudaEventElapsedTime(
            &timing->candidate_reduce_ms, seed_start, seed_stop));
        CUDA_CHECK(cudaEventElapsedTime(
            &timing->candidate_desc_ms, policy_start, policy_stop));
        CUDA_CHECK(cudaEventElapsedTime(
            &timing->seed_compact_ms, seed_start, seed_compact_stop));
        CUDA_CHECK(cudaEventElapsedTime(
            &timing->seed_adc_ms, seed_compact_stop, seed_stop));
        CUDA_CHECK(cudaEventElapsedTime(
            &timing->threshold_policy_ms, policy_start, policy_stop));
        CUDA_CHECK(cudaEventElapsedTime(&timing->pq_scan_ms, scan_start, scan_stop));
        CUDA_CHECK(cudaEventElapsedTime(
            &timing->scan_setup_ms, scan_start, scan_setup_stop));
        CUDA_CHECK(cudaEventElapsedTime(
            &timing->scan_kernel_ms, scan_setup_stop, scan_stop));

    }

}

} // namespace
void pq_query_lut_threshold_search_gpu(
    const paraselect::search::PQQueryLutView& lut,
    const idx_t* d_top_nprobe, const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets, const int* d_cluster_counts,
    int n_query, int n_total_clusters, int n_probes, int pq_m,
    const paraselect::search::PQAdcThresholdParams& params,
    const paraselect::search::IVFProbeScheduleView* probe_schedule,
    const paraselect::search::IVFProbeTileScheduleView* probe_tile_schedule,
    cudaStream_t stream, paraselect::search::PQAdcThresholdWorkspace* threshold_workspace,
    ResidualPQGpuDebugTiming* timing,
    const float* raw_queries, const float* raw_codebook, int raw_dim) {
    if (n_query <= 0) return;
    if (!lut.enabled() || lut.sign_codec.codes || !d_pq_codes || !threshold_workspace)
        throw std::invalid_argument("query-LUT threshold requires a PQ LUT, codes and workspace");
    const bool list_centric = params.scan_layout == paraselect::search::PQAdcThresholdScanLayout::ListCentric;
    const bool hybrid=params.code_tile_consumer == paraselect::search::PQAdcCodeTileConsumer::Fp16Hybrid;
    const bool u8_hybrid=params.code_tile_consumer == paraselect::search::PQAdcCodeTileConsumer::U8Hybrid;
    if((hybrid || u8_hybrid) && (!list_centric || pq_m!=16 || (raw_dim!=128&&(!hybrid||raw_dim!=96)) || !raw_queries || !raw_codebook ||
                       !probe_schedule ||
                       lut.kind!=paraselect::search::PQQueryLutKind::ResidualDot))
        throw std::invalid_argument("codeword TC requires static list M16, D128 or FP16 D96 residual PQ");
    auto seed = [&](const idx_t*, int seed_probes, int seed_rank) {
        int* seed_ids = threshold_workspace->d_candidate_idx;
        float* seed_distances = threshold_workspace->d_candidate_dist;
        paraselect::search::pq_query_lut_seed_topk(
            lut, d_top_nprobe, d_pq_codes, d_cluster_offsets,
            d_cluster_counts, n_query, n_probes, seed_probes, pq_m,
            seed_rank, lut.quantized(),
            threshold_workspace->d_sort_keys_in,
            threshold_workspace->d_sort_keys_out,
            static_cast<size_t>(threshold_workspace->pool_capacity), seed_ids, seed_distances,
            threshold_workspace->d_overflow,
            threshold_workspace->d_physical_code_rows, stream);
        return 0ULL;
    };
    auto histogram = [&](int seed_probes, int bins, float log_min, float log_step) {
        paraselect::search::pq_query_lut_seed_histogram(
            lut, d_top_nprobe, d_pq_codes, d_cluster_offsets, d_cluster_counts,
            n_query, n_probes, seed_probes, n_total_clusters, pq_m,
            lut.quantized() && params.lut_read_mode == paraselect::search::PQAdcLutReadMode::Global &&
                params.list_row_mode == paraselect::search::PQAdcListScanRowMode::Serial,
            threshold_workspace->d_seed_histogram, bins, log_min, log_step, stream,
            params.seed_mean_list_rows);
    };
    auto scan = [&](const paraselect::search::PQAdcThresholdCandidatePoolView& output,
                    int probe_begin, int probe_end) {
        constexpr int skip_seed_probes = 0;
        if (list_centric) {
            CUDA_CHECK(cudaMemsetAsync(
                threshold_workspace->d_physical_code_rows, 0,
                sizeof(unsigned long long), stream));
            if(u8_hybrid){
                if(probe_tile_schedule || params.list_row_mode!=paraselect::search::PQAdcListScanRowMode::Serial || params.lut_read_mode!=paraselect::search::PQAdcLutReadMode::Global)
                    throw std::invalid_argument("U8 hybrid requires serial/global entry list scan");
                paraselect::search::pq_codeword_scan_u8_hybrid(raw_queries,raw_codebook,lut,*probe_schedule,
                    d_pq_codes,d_cluster_offsets,d_cluster_counts,threshold_workspace->d_threshold,
                    n_query,n_probes,output,stream,probe_begin,probe_end,threshold_workspace->d_physical_code_rows);
                return;
            }
            if(hybrid){
                if(probe_tile_schedule || params.list_row_mode!=paraselect::search::PQAdcListScanRowMode::Serial || params.lut_read_mode!=paraselect::search::PQAdcLutReadMode::Global)
                    throw std::invalid_argument("FP16 hybrid requires serial/global entry list scan");
                paraselect::search::pq_codeword_scan_hybrid(raw_queries,raw_codebook,lut,*probe_schedule,
                    d_pq_codes,d_cluster_offsets,d_cluster_counts,threshold_workspace->d_threshold,
                    n_query,n_probes,output,stream,probe_begin,probe_end,threshold_workspace->d_physical_code_rows,raw_dim);
                return;
            }
            paraselect::search::pq_adc_query_lut_segmented_scan(
                lut, *probe_schedule, probe_tile_schedule,
                d_pq_codes, d_cluster_offsets,
                d_cluster_counts, threshold_workspace->d_threshold, n_query,
                n_probes, pq_m, params.list_row_mode,
                params.lut_read_mode,
                threshold_workspace->d_physical_code_rows,
                output, stream, params.shared_lut_layout,
                probe_begin, probe_end);
        } else {
            if (probe_begin != 0 || probe_end != n_probes)
                throw std::invalid_argument("pair threshold scan does not support probe ranges");
            paraselect::search::pq_adc_query_lut_pair_scan(
                lut, d_top_nprobe, d_pq_codes, d_cluster_offsets,
                d_cluster_counts, threshold_workspace->d_threshold, n_query,
                n_total_clusters, n_probes, pq_m, output, stream,
                params.lut_read_mode, params.shared_lut_layout);
        }
    };
    threshold_search(d_top_nprobe,d_pq_codes,d_cluster_offsets,d_cluster_counts,
        n_query,n_total_clusters,n_probes,pq_m,params,&lut,
        probe_schedule,probe_tile_schedule,stream,threshold_workspace,timing,seed,histogram,scan);
}
void pq_adc_threshold_search_gpu(
    paraselect::search::PQAdcDistanceMode distance_mode,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    const paraselect::search::ResidualPQGpuSearchSpec& spec,
    const paraselect::search::IVFProbeScheduleView* probe_schedule,
    const paraselect::search::IVFProbeTileScheduleView* probe_tile_schedule,
    cudaStream_t stream,
    paraselect::search::PQAdcThresholdWorkspace* threshold_workspace,
    paraselect::search::ResidualPQGroupedWorkspace* grouped_workspace,
    ResidualPQGpuDebugTiming* timing) {
    if (n_query <= 0) return;
    if (!d_queries || !d_top_nprobe || !d_cluster_offsets ||
        !d_cluster_counts || !threshold_workspace || !grouped_workspace ||
        (!spec.sign_codec.codes && (!d_pq_codes || !d_codebook))) {
        throw std::invalid_argument("pq_adc_threshold_search_gpu: missing required input");
    }
    if (distance_mode == paraselect::search::PQAdcDistanceMode::Residual && !d_centroids) {
        throw std::invalid_argument("pq_adc_threshold_search_gpu: residual ADC requires centroids");
    }
    if (spec.pq_k != kPQK || dim != spec.pq_dim_padded ||
        (spec.pq_d_sub_padded != 2 && spec.pq_d_sub_padded != 4 && spec.pq_d_sub_padded != 8))
        throw std::invalid_argument("raw ADC threshold: unsupported codebook geometry");
    paraselect::search::PQQueryLutView sign_view;
    sign_view.sign_codec = spec.sign_codec;
    const auto* pq_query_lut = spec.sign_codec.codes ? &sign_view : nullptr;
    const auto& params = spec.threshold;
    const int pq_m = spec.pq_m;
    const bool list_centric = params.scan_layout == paraselect::search::PQAdcThresholdScanLayout::ListCentric;
    auto seed = [&](const idx_t* seed_top_nprobe, int seed_probes, int seed_rank) {
        auto seed_spec=spec;
        seed_spec.global_k=seed_rank;
        seed_spec.search_strategy=paraselect::search::PQAdcSearchStrategy::Dense;
        pq_adc_search_gpu(distance_mode,d_queries,d_centroids,seed_top_nprobe,d_pq_codes,
            d_cluster_offsets,d_cluster_counts,d_codebook,n_query,dim,n_total_clusters,
            seed_probes,seed_spec,nullptr,nullptr,threshold_workspace->d_candidate_idx,
            threshold_workspace->d_candidate_dist,stream,nullptr,grouped_workspace);
        return grouped_workspace->scanned_codes;
    };
    auto histogram = [&](int seed_probes, int bins, float log_min, float log_step) {
        if (spec.sign_codec.codes)
            throw std::invalid_argument("probe-residual histogram requires PQ codes");
        const size_t shared_bytes = (size_t)pq_m * kLutStride * sizeof(float) + bins * sizeof(unsigned int);
        auto launch = [&]<int DSub, bool Residual>() {
            threshold_adc_scan_kernel<DSub, Residual, false, true>
                <<<dim3(n_query, seed_probes), kScanThreads, shared_bytes, stream>>>(
                    d_queries, d_centroids, d_top_nprobe, d_pq_codes,
                    d_cluster_offsets, d_cluster_counts, d_codebook, nullptr,
                    n_query, dim, n_total_clusters, seed_probes, pq_m, 0,
                    nullptr, nullptr, nullptr, nullptr, nullptr, threshold_workspace->d_overflow,
                    nullptr, {}, 0, threshold_workspace->d_seed_histogram, bins, log_min, log_step, n_probes);
        };
        auto dispatch = [&]<int DSub>() {
            if (distance_mode == paraselect::search::PQAdcDistanceMode::Residual)
                launch.template operator()<DSub, true>();
            else launch.template operator()<DSub, false>();
        };
        switch (spec.pq_d_sub_padded) {
        case 2: dispatch.template operator()<2>(); break;
        case 4: dispatch.template operator()<4>(); break;
        case 8: dispatch.template operator()<8>(); break;
        }
    };
    auto scan = [&](const paraselect::search::PQAdcThresholdCandidatePoolView& output,
                    int probe_begin, int probe_end) {
        const int global_pool_capacity=threshold_workspace->pool_capacity;
        const bool audit=params.audit;
        constexpr int skip_seed_probes=0;
        if (spec.sign_codec.codes && !list_centric) {
            threshold_adc_scan_kernel<8, true, true>
                <<<dim3(n_query, n_probes), kScanThreads, (dim + 1) * sizeof(float), stream>>>(
                    d_queries, d_centroids, d_top_nprobe, d_pq_codes, d_cluster_offsets,
                    d_cluster_counts, d_codebook, threshold_workspace->d_threshold,
                    n_query, dim, n_total_clusters, n_probes, spec.pq_m, global_pool_capacity,
                    threshold_workspace->d_pool_idx, threshold_workspace->d_pool_dist,
                    threshold_workspace->d_pool_query, threshold_workspace->d_candidate_count,
                    threshold_workspace->d_global_count, threshold_workspace->d_overflow,
                    audit ? threshold_workspace->d_accepted_count : nullptr, spec.sign_codec,
                    skip_seed_probes);
        } else if (pq_query_lut) {
            CUDA_CHECK(cudaMemsetAsync(
                threshold_workspace->d_physical_code_rows, 0,
                sizeof(unsigned long long), stream));
            paraselect::search::pq_adc_query_lut_segmented_scan(
                *pq_query_lut, *probe_schedule, probe_tile_schedule,
                d_pq_codes, d_cluster_offsets,
                d_cluster_counts, threshold_workspace->d_threshold, n_query,
                n_probes, pq_m, params.list_row_mode,
                params.lut_read_mode,
                threshold_workspace->d_physical_code_rows,
                output, stream, params.shared_lut_layout,
                probe_begin, probe_end);
        } else {
            if (probe_begin != 0 || probe_end != n_probes)
                throw std::invalid_argument("plain pair threshold scan does not support probe ranges");
            auto launch = [&]<int DSub>() {
                launch_threshold_scan<DSub>(distance_mode, d_queries, d_centroids, d_top_nprobe,
                    d_pq_codes, d_cluster_offsets, d_cluster_counts, d_codebook,
                    threshold_workspace->d_threshold, n_query, dim, n_total_clusters, n_probes,
                    spec.pq_m, global_pool_capacity, threshold_workspace->d_pool_idx,
                    threshold_workspace->d_pool_dist, threshold_workspace->d_pool_query,
                    threshold_workspace->d_candidate_count, threshold_workspace->d_global_count,
                    threshold_workspace->d_overflow,
                    audit ? threshold_workspace->d_accepted_count : nullptr, stream,
                    probe_begin);
            };
            switch (spec.pq_d_sub_padded) {
            case 2: launch.template operator()<2>(); break;
            case 4: launch.template operator()<4>(); break;
            case 8: launch.template operator()<8>(); break;
            }
        }
    };
    auto resolved=params;
    if (resolved.candidate_k<=0) resolved.candidate_k=spec.global_k;
    threshold_search(d_top_nprobe,d_pq_codes,d_cluster_offsets,d_cluster_counts,
        n_query,n_total_clusters,n_probes,pq_m,resolved,pq_query_lut,
        probe_schedule,probe_tile_schedule,stream,threshold_workspace,timing,seed,histogram,scan);
}
