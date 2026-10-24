#include "search/ivf_common/ivf_probe_schedule.cuh"

#include "utils/check_cuda_error.cuh"

#include <cub/cub.cuh>
#include <cuda/functional>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace paraselect::search {
namespace {

constexpr int kThreads = 256;
constexpr int kMinEntryWidth = 8;

int probe_entry_width_from_env() {
    const char* value = std::getenv("PARASELECT_RPQ_PROBE_ENTRY_WIDTH");
    if (!value || !*value) return kMinEntryWidth;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' ||
        (parsed < kMinEntryWidth || parsed > INT_MAX)) {
        throw std::invalid_argument(
            "PARASELECT_RPQ_PROBE_ENTRY_WIDTH must be in [8, INT_MAX]");
    }
    return static_cast<int>(parsed);
}

int sparse_slot_count(int reference_count) {
    if (reference_count <= 0 || reference_count > INT_MAX / 2) {
        throw std::invalid_argument("invalid sparse IVF schedule capacity");
    }
    int slots = 1;
    const int required = reference_count * 2;
    while (slots < required) {
        if (slots > INT_MAX / 2) {
            throw std::overflow_error("sparse IVF schedule capacity overflow");
        }
        slots *= 2;
    }
    return slots;
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct ScheduleSummary {
    int entries;
    int references;
    int invalid;
    int max_list_queries;
    unsigned long long scanned;
};

__global__ void gather_summary_kernel(
    const int* entries, const int* references, const int* invalid,
    const unsigned long long* scanned, ScheduleSummary* summary) {
    if (threadIdx.x == 0) {
        summary->entries = *entries;
        summary->references = *references;
        summary->invalid = *invalid;
        summary->scanned = *scanned;
    }
}

template <bool AllowPadding>
__global__ void count_dense_refs_kernel(
    const idx_t* probes,
    int query_count,
    int probe_count,
    int probe_limit,
    int nlist,
    int* cluster_counts,
    int* invalid) {
    const int ref = blockIdx.x * blockDim.x + threadIdx.x;
    const int selected_count = query_count * probe_limit;
    if (ref >= selected_count) return;
    const int query = ref / probe_limit;
    const int rank = ref - query * probe_limit;
    const idx_t cluster64 = probes[(size_t)query * probe_count + rank];
    if (cluster64 < 0 || cluster64 >= nlist) {
        if constexpr (AllowPadding) {
            if (cluster64 == -1) return;
        }
        atomicExch(invalid, 1);
        return;
    }
    atomicAdd(cluster_counts + static_cast<int>(cluster64), 1);
}

__global__ void fill_dense_refs_kernel(
    const idx_t* probes,
    int query_count,
    int probe_count,
    int probe_limit,
    int nlist,
    int* write_positions,
    int* grouped_queries,
    int* grouped_probe_ranks) {
    const int ref = blockIdx.x * blockDim.x + threadIdx.x;
    const int selected_count = query_count * probe_limit;
    if (ref >= selected_count) return;
    const int query = ref / probe_limit;
    const int rank = ref - query * probe_limit;
    const idx_t cluster64 = probes[(size_t)query * probe_count + rank];
    if (cluster64 < 0 || cluster64 >= nlist) return;
    const int cluster = static_cast<int>(cluster64);
    const int output = atomicAdd(write_positions + cluster, 1);
    grouped_queries[output] = query;
    grouped_probe_ranks[output] = rank;
}

__global__ void count_dense_entries_kernel(
    const int* cluster_counts,
    const int* list_counts,
    const int* ref_offsets,
    int* write_positions,
    int nlist,
    int entry_width,
    int* entry_counts,
    unsigned long long* scanned_vectors, int* max_list_queries) {
    using Sum = cub::BlockReduce<unsigned long long, kThreads>;
    __shared__ typename Sum::TempStorage storage;
    using Max = cub::BlockReduce<int, kThreads>;
    __shared__ typename Max::TempStorage max_storage;
    const int cluster = blockIdx.x * blockDim.x + threadIdx.x;
    const int refs = cluster < nlist ? cluster_counts[cluster] : 0;
    unsigned long long vectors = 0;
    if (cluster < nlist) {
        write_positions[cluster] = ref_offsets[cluster];
        entry_counts[cluster] = refs / entry_width + (refs % entry_width != 0);
        if (refs != 0)
            vectors = static_cast<unsigned long long>(refs) * list_counts[cluster];
    }
    const auto total = Sum(storage).Sum(vectors);
    if (threadIdx.x == 0 && total != 0) atomicAdd(scanned_vectors, total);
    const int maximum = Max(max_storage).Reduce(refs, cuda::maximum<>{});
    if (threadIdx.x == 0 && maximum) atomicMax(max_list_queries, maximum);
}

__global__ void fill_dense_entries_kernel(
    const int* cluster_counts,
    const int* cluster_offsets,
    const int* entry_offsets,
    int nlist,
    int entry_width,
    int* entry_clusters,
    int* entry_starts,
    int* entry_counts) {
    const int cluster = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster >= nlist) return;
    const int refs = cluster_counts[cluster];
    const int first_ref = cluster_offsets[cluster];
    const int first_entry = entry_offsets[cluster];
    for (int offset = 0; offset < refs; offset += min(entry_width, refs - offset)) {
        const int entry = first_entry + offset / entry_width;
        entry_clusters[entry] = cluster;
        entry_starts[entry] = first_ref + offset;
        entry_counts[entry] = min(entry_width, refs - offset);
    }
}

__global__ void count_entry_tiles_kernel(
    const int* entry_clusters,
    int entry_count,
    int nlist,
    const int* list_counts,
    int tile_rows,
    int* entry_tile_counts,
    int* invalid) {
    const int entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry > entry_count) return;
    if (entry == entry_count) {
        entry_tile_counts[entry] = 0;
        return;
    }
    const int cluster = entry_clusters[entry];
    if (cluster < 0 || cluster >= nlist || list_counts[cluster] < 0) {
        entry_tile_counts[entry] = 0;
        atomicExch(invalid, 1);
        return;
    }
    entry_tile_counts[entry] = static_cast<int>(
        (static_cast<long long>(list_counts[cluster]) + tile_rows - 1) / tile_rows);
}

__device__ __forceinline__ unsigned int hash_cluster(unsigned int value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

template <bool AllowPadding>
__global__ void count_sparse_refs_kernel(
    const idx_t* probes,
    int query_count,
    int probe_count,
    int probe_limit,
    int nlist,
    int hash_mask,
    int* hash_keys,
    int* hash_counts,
    int* invalid,
    const int* list_counts,
    unsigned long long* scanned_vectors) {
    using Sum = cub::BlockReduce<unsigned long long, kThreads>;
    __shared__ typename Sum::TempStorage storage;
    const int ref = blockIdx.x * blockDim.x + threadIdx.x;
    const int selected_count = query_count * probe_limit;
    unsigned long long vectors = 0;
    if (ref < selected_count) {
        const int query = ref / probe_limit;
        const int rank = ref - query * probe_limit;
        const idx_t cluster64 = probes[(size_t)query * probe_count + rank];
        if (cluster64 < 0 || cluster64 >= nlist) {
            if (!(AllowPadding && cluster64 == -1)) atomicExch(invalid, 1);
        } else {
            const int cluster = static_cast<int>(cluster64);
            int slot = static_cast<int>(hash_cluster(cluster)) & hash_mask;
            while (true) {
                const int old = atomicCAS(hash_keys + slot, -1, cluster);
                if (old == -1 || old == cluster) break;
                slot = (slot + 1) & hash_mask;
            }
            atomicAdd(hash_counts + slot, 1);
            vectors = static_cast<unsigned long long>(list_counts[cluster]);
        }
    }
    // Padding and tail threads must also participate in the block collective.
    const auto total = Sum(storage).Sum(vectors);
    if (threadIdx.x == 0 && total != 0) atomicAdd(scanned_vectors, total);
}

__global__ void fill_sparse_refs_kernel(
    const idx_t* probes,
    int query_count,
    int probe_count,
    int probe_limit,
    int nlist,
    int hash_mask,
    const int* hash_keys,
    int* write_positions,
    int* queries,
    int* probe_ranks) {
    const int ref = blockIdx.x * blockDim.x + threadIdx.x;
    const int selected_count = query_count * probe_limit;
    if (ref >= selected_count) return;
    const int query = ref / probe_limit;
    const int rank = ref - query * probe_limit;
    const idx_t cluster64 = probes[(size_t)query * probe_count + rank];
    if (cluster64 < 0 || cluster64 >= nlist) return;
    const int cluster = static_cast<int>(cluster64);
    int slot = static_cast<int>(hash_cluster(cluster)) & hash_mask;
    while (hash_keys[slot] != cluster) slot = (slot + 1) & hash_mask;
    const int output = atomicAdd(write_positions + slot, 1);
    queries[output] = query;
    probe_ranks[output] = rank;
}

__global__ void count_sparse_entries_kernel(
    const int* hash_counts,
    const int* ref_offsets,
    int* write_positions,
    int slot_count,
    int entry_width,
    int* entry_counts, int* max_list_queries) {
    using Max = cub::BlockReduce<int, kThreads>;
    __shared__ typename Max::TempStorage storage;
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    const int refs = slot < slot_count ? hash_counts[slot] : 0;
    if (slot < slot_count) {
        write_positions[slot] = ref_offsets[slot];
        entry_counts[slot] =
            refs / entry_width + (refs % entry_width != 0);
    }
    const int maximum = Max(storage).Reduce(refs, cuda::maximum<>{});
    if (threadIdx.x == 0 && maximum) atomicMax(max_list_queries, maximum);
}

__global__ void fill_sparse_entries_kernel(
    const int* hash_keys,
    const int* hash_counts,
    const int* ref_offsets,
    const int* entry_offsets,
    int slot_count,
    int entry_width,
    int* entry_clusters,
    int* entry_starts,
    int* entry_counts) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slot_count) return;
    const int refs = hash_counts[slot];
    const int first_ref = ref_offsets[slot];
    const int first_entry = entry_offsets[slot];
    for (int offset = 0; offset < refs; offset += min(entry_width, refs - offset)) {
        const int entry = first_entry + offset / entry_width;
        entry_clusters[entry] = hash_keys[slot];
        entry_starts[entry] = first_ref + offset;
        entry_counts[entry] = min(entry_width, refs - offset);
    }
}

}  // namespace

IVFProbeScheduleStrategy choose_ivf_probe_schedule_strategy(
    IVFProbeScheduleStrategy requested,
    size_t probe_reference_count,
    int nlist) {
    if (nlist <= 0 || probe_reference_count > static_cast<size_t>(INT_MAX)) {
        throw std::invalid_argument("invalid IVF probe schedule shape");
    }
    if (requested == IVFProbeScheduleStrategy::Dense ||
        requested == IVFProbeScheduleStrategy::Sparse) {
        return requested;
    }
    if (requested != IVFProbeScheduleStrategy::Auto) {
        throw std::invalid_argument("unknown IVF probe schedule strategy");
    }
    // Dense scans nlist-sized arrays; sparse clears and scans hash slots sized
    // from the probe references. Keep auto conservative around launch costs.
    return probe_reference_count < static_cast<size_t>(nlist) / 32
        ? IVFProbeScheduleStrategy::Sparse
        : IVFProbeScheduleStrategy::Dense;
}

const char* ivf_probe_schedule_strategy_name(IVFProbeScheduleStrategy strategy) {
    switch (strategy) {
        case IVFProbeScheduleStrategy::Auto: return "auto";
        case IVFProbeScheduleStrategy::Dense: return "dense";
        case IVFProbeScheduleStrategy::Sparse: return "sparse";
    }
    throw std::invalid_argument("unknown IVF probe schedule strategy");
}

struct IVFProbeScheduleBuilder::Impl {
    Impl(
        int requested_nlist,
        int requested_max_refs,
        const int* requested_list_counts,
        IVFProbeScheduleStrategy requested_strategy)
        : nlist(requested_nlist),
          max_refs(requested_max_refs),
          list_counts(requested_list_counts),
          strategy(requested_strategy) {
        if (nlist <= 0 || max_refs <= 0 || !list_counts) {
            throw std::invalid_argument("invalid IVF probe schedule builder configuration");
        }
        if (strategy != IVFProbeScheduleStrategy::Auto &&
            strategy != IVFProbeScheduleStrategy::Dense &&
            strategy != IVFProbeScheduleStrategy::Sparse) {
            throw std::invalid_argument("unknown IVF probe schedule strategy");
        }
        // First entry of each occupied list costs one reference; every
        // additional entry costs at least kMinEntryWidth more references.
        max_entries = max_refs <= nlist ? max_refs
            : nlist + (max_refs - nlist) / kMinEntryWidth;
        try {
            allocate_workspace();
        } catch (...) {
            release();
            throw;
        }
    }

    ~Impl() { release(); }

    template <typename T>
    void allocate(T** output, size_t count, const char* operation) {
        check_cuda(cudaMalloc(output, count * sizeof(T)), operation);
    }

    void allocate_workspace() {
        allocate(&entry_clusters, max_entries, "allocate IVF schedule entry clusters");
        allocate(&entry_starts, max_entries, "allocate IVF schedule entry starts");
        allocate(&entry_counts, max_entries, "allocate IVF schedule entry counts");
        allocate(&entry_tile_counts, (size_t)max_entries + 1,
                 "allocate IVF schedule tile counts");
        allocate(&entry_tile_offsets, (size_t)max_entries + 1,
                 "allocate IVF schedule tile offsets");
        allocate(&device_summary, 1, "allocate IVF schedule summary");
        check_cuda(cudaMallocHost(&host_summary, sizeof(ScheduleSummary)),
                   "allocate pinned IVF schedule summary");
        allocate(&invalid, 1, "allocate IVF schedule validation flag");
        allocate(&scanned_vectors, 1, "allocate IVF schedule scanned count");

        const bool needs_dense = strategy != IVFProbeScheduleStrategy::Sparse;
        const bool needs_sparse = strategy != IVFProbeScheduleStrategy::Dense;
        if (needs_dense) {
            allocate(&dense_cluster_counts, (size_t)nlist + 1,
                     "allocate dense IVF schedule counts");
            allocate(&dense_cluster_offsets, (size_t)nlist + 1,
                     "allocate dense IVF schedule offsets");
            allocate(&dense_write_positions, nlist,
                     "allocate dense IVF schedule positions");
            allocate(&dense_queries, max_refs, "allocate dense IVF schedule queries");
            allocate(&dense_probe_ranks, max_refs,
                     "allocate dense IVF schedule probe ranks");
            allocate(&dense_entry_counts, (size_t)nlist + 1,
                     "allocate dense IVF schedule entry counts");
            allocate(&dense_entry_offsets, (size_t)nlist + 1,
                     "allocate dense IVF schedule entry offsets");
        }

        if (needs_sparse) {
            max_sparse_slots = sparse_slot_count(max_refs);
            allocate(&sparse_hash_keys, max_sparse_slots,
                     "allocate sparse IVF schedule hash keys");
            allocate(&sparse_hash_counts, (size_t)max_sparse_slots + 1,
                     "allocate sparse IVF schedule hash counts");
            allocate(&sparse_ref_offsets, (size_t)max_sparse_slots + 1,
                     "allocate sparse IVF schedule ref offsets");
            allocate(&sparse_write_positions, max_sparse_slots,
                     "allocate sparse IVF schedule write positions");
            allocate(&sparse_entry_counts, (size_t)max_sparse_slots + 1,
                     "allocate sparse IVF schedule entry counts");
            allocate(&sparse_entry_offsets, (size_t)max_sparse_slots + 1,
                     "allocate sparse IVF schedule entry offsets");
            allocate(&sparse_queries, max_refs, "allocate sparse IVF schedule queries");
            allocate(&sparse_probe_ranks, max_refs,
                     "allocate sparse IVF schedule probe ranks");
        }

        size_t dense_scan_bytes = 0;
        size_t sparse_scan_bytes = 0;
        size_t tile_scan_bytes = 0;
        if (needs_dense) {
            check_cuda(cub::DeviceScan::ExclusiveSum(
                           nullptr, dense_scan_bytes, dense_cluster_counts,
                           dense_cluster_offsets, nlist + 1),
                       "size dense IVF schedule scan workspace");
        }
        if (needs_sparse) {
            check_cuda(cub::DeviceScan::ExclusiveSum(
                           nullptr, sparse_scan_bytes, sparse_hash_counts,
                           sparse_ref_offsets, max_sparse_slots + 1),
                       "size sparse IVF schedule scan workspace");
        }
        check_cuda(cub::DeviceScan::ExclusiveSum(
                       nullptr, tile_scan_bytes, entry_tile_counts,
                       entry_tile_offsets, max_entries + 1),
                   "size IVF tile schedule scan workspace");
        cub_temp_bytes = std::max(
            std::max(dense_scan_bytes, sparse_scan_bytes), tile_scan_bytes);
        check_cuda(cudaMalloc(&cub_temp, cub_temp_bytes),
                   "allocate IVF schedule CUB workspace");
    }

    IVFProbeScheduleResult build(
        const IVFDeviceProbeView& probes,
        int probe_limit,
        cudaStream_t stream,
        bool allow_padding) {
        if (!probes.indices || probes.query_count <= 0 || probes.probe_count <= 0 ||
            probe_limit <= 0 || probe_limit > probes.probe_count) {
            throw std::invalid_argument("invalid IVF probe schedule request");
        }
        const size_t ref_count64 =
            static_cast<size_t>(probes.query_count) * probe_limit;
        if (ref_count64 > static_cast<size_t>(max_refs)) {
            throw std::invalid_argument("IVF probe schedule request exceeds workspace");
        }
        const int ref_count = static_cast<int>(ref_count64);
        check_cuda(cudaMemsetAsync(&device_summary->max_list_queries, 0, sizeof(int), stream),
                   "clear IVF list query maximum");
        const auto selected = choose_ivf_probe_schedule_strategy(
            strategy, ref_count64, nlist);
        return selected == IVFProbeScheduleStrategy::Sparse
            ? build_sparse(probes, probe_limit, ref_count, allow_padding, stream)
            : build_dense(probes, probe_limit, ref_count, allow_padding, stream);
    }

    IVFProbeTileScheduleResult build_tiles(
        const IVFProbeScheduleView& schedule,
        int tile_rows,
        cudaStream_t stream) {
        if (!schedule.clusters || schedule.entry_count < 0 ||
            schedule.entry_count > max_entries || tile_rows <= 0) {
            throw std::invalid_argument("invalid IVF probe tile schedule request");
        }
        check_cuda(cudaMemsetAsync(invalid, 0, sizeof(int), stream),
                   "clear IVF tile schedule validation flag");
        const int item_count = schedule.entry_count + 1;
        const int blocks = (item_count + kThreads - 1) / kThreads;
        count_entry_tiles_kernel<<<blocks, kThreads, 0, stream>>>(
            schedule.clusters, schedule.entry_count, nlist, list_counts,
            tile_rows, entry_tile_counts, invalid);
        CUDA_LAUNCH_CHECK("count IVF probe tile schedule work");
        check_cuda(cub::DeviceScan::ExclusiveSum(
                       cub_temp, cub_temp_bytes, entry_tile_counts,
                       entry_tile_offsets, item_count, stream),
                   "scan IVF probe tile schedule work");

        int host_work_count = 0;
        int host_invalid = 0;
        check_cuda(cudaMemcpyAsync(
                       &host_work_count, entry_tile_offsets + schedule.entry_count,
                       sizeof(int), cudaMemcpyDeviceToHost, stream),
                   "copy IVF probe tile work count");
        check_cuda(cudaMemcpyAsync(
                       &host_invalid, invalid, sizeof(int),
                       cudaMemcpyDeviceToHost, stream),
                   "copy IVF tile schedule validation flag");
        check_cuda(cudaStreamSynchronize(stream), "finish IVF probe tile schedule");
        if (host_invalid != 0) {
            throw std::out_of_range("IVF tile schedule contains an invalid list");
        }
        if (host_work_count < 0) {
            throw std::overflow_error("IVF probe tile schedule work count overflow");
        }
        return {{entry_tile_offsets, schedule.entry_count, host_work_count, tile_rows}};
    }

    IVFProbeScheduleResult build_dense(
        const IVFDeviceProbeView& probes,
        int probe_limit,
        int ref_count,
        bool allow_padding,
        cudaStream_t stream) {
        const int entry_width = probe_entry_width_from_env();
        check_cuda(cudaMemsetAsync(
                       dense_cluster_counts, 0, ((size_t)nlist + 1) * sizeof(int), stream),
                   "clear dense IVF schedule counts");
        check_cuda(cudaMemsetAsync(invalid, 0, sizeof(int), stream),
                   "clear IVF schedule validation flag");
        check_cuda(cudaMemsetAsync(scanned_vectors, 0, sizeof(unsigned long long), stream),
                   "clear IVF schedule scanned count");
        const int ref_blocks = (ref_count + kThreads - 1) / kThreads;
        if (allow_padding) {
            count_dense_refs_kernel<true><<<ref_blocks, kThreads, 0, stream>>>(
                probes.indices, probes.query_count, probes.probe_count, probe_limit,
                nlist, dense_cluster_counts, invalid);
        } else {
            count_dense_refs_kernel<false><<<ref_blocks, kThreads, 0, stream>>>(
                probes.indices, probes.query_count, probes.probe_count, probe_limit,
                nlist, dense_cluster_counts, invalid);
        }
        CUDA_LAUNCH_CHECK("count dense IVF probe references");
        check_cuda(cub::DeviceScan::ExclusiveSum(
                       cub_temp, cub_temp_bytes, dense_cluster_counts,
                       dense_cluster_offsets, nlist + 1, stream),
                   "scan dense IVF schedule counts");
        check_cuda(cudaMemsetAsync(
                       dense_entry_counts + nlist, 0, sizeof(int), stream),
                   "clear dense IVF schedule entry sentinel");
        const int list_blocks = (nlist + kThreads - 1) / kThreads;
        count_dense_entries_kernel<<<list_blocks, kThreads, 0, stream>>>(
            dense_cluster_counts, list_counts, dense_cluster_offsets,
            dense_write_positions, nlist, entry_width,
            dense_entry_counts, scanned_vectors, &device_summary->max_list_queries);
        CUDA_LAUNCH_CHECK("count dense IVF schedule entries");
        fill_dense_refs_kernel<<<ref_blocks, kThreads, 0, stream>>>(
            probes.indices, probes.query_count, probes.probe_count, probe_limit,
            nlist, dense_write_positions, dense_queries, dense_probe_ranks);
        CUDA_LAUNCH_CHECK("fill dense IVF probe references");
        check_cuda(cub::DeviceScan::ExclusiveSum(
                       cub_temp, cub_temp_bytes, dense_entry_counts,
                       dense_entry_offsets, nlist + 1, stream),
                   "scan dense IVF schedule entries");
        fill_dense_entries_kernel<<<list_blocks, kThreads, 0, stream>>>(
            dense_cluster_counts, dense_cluster_offsets, dense_entry_offsets,
            nlist, entry_width, entry_clusters, entry_starts, entry_counts);
        CUDA_LAUNCH_CHECK("fill dense IVF schedule entries");
        return finish(
            dense_entry_offsets + nlist, dense_cluster_offsets + nlist,
            dense_queries, dense_probe_ranks,
            IVFProbeScheduleStrategy::Dense, stream);
    }

    IVFProbeScheduleResult build_sparse(
        const IVFDeviceProbeView& probes,
        int probe_limit,
        int ref_count,
        bool allow_padding,
        cudaStream_t stream) {
        const int entry_width = probe_entry_width_from_env();
        check_cuda(cudaMemsetAsync(invalid, 0, sizeof(int), stream),
                   "clear IVF schedule validation flag");
        check_cuda(cudaMemsetAsync(scanned_vectors, 0, sizeof(unsigned long long), stream),
                   "clear IVF schedule scanned count");
        const int slot_count = sparse_slot_count(ref_count);
        const int hash_mask = slot_count - 1;
        check_cuda(cudaMemsetAsync(
                       sparse_hash_keys, 0xff, (size_t)slot_count * sizeof(int), stream),
                   "clear sparse IVF schedule hash keys");
        check_cuda(cudaMemsetAsync(
                       sparse_hash_counts, 0, ((size_t)slot_count + 1) * sizeof(int), stream),
                   "clear sparse IVF schedule hash counts");
        check_cuda(cudaMemsetAsync(
                       sparse_entry_counts + slot_count, 0, sizeof(int), stream),
                   "clear sparse IVF schedule entry counts");
        const int ref_blocks = (ref_count + kThreads - 1) / kThreads;
        if (allow_padding) {
            count_sparse_refs_kernel<true><<<ref_blocks, kThreads, 0, stream>>>(
                probes.indices, probes.query_count, probes.probe_count, probe_limit,
                nlist, hash_mask, sparse_hash_keys, sparse_hash_counts, invalid,
                list_counts, scanned_vectors);
        } else {
            count_sparse_refs_kernel<false><<<ref_blocks, kThreads, 0, stream>>>(
                probes.indices, probes.query_count, probes.probe_count, probe_limit,
                nlist, hash_mask, sparse_hash_keys, sparse_hash_counts, invalid,
                list_counts, scanned_vectors);
        }
        CUDA_LAUNCH_CHECK("count sparse IVF probe references");
        check_cuda(cub::DeviceScan::ExclusiveSum(
                       cub_temp, cub_temp_bytes, sparse_hash_counts,
                       sparse_ref_offsets, slot_count + 1, stream),
                   "scan sparse IVF schedule references");
        const int slot_blocks = (slot_count + kThreads - 1) / kThreads;
        count_sparse_entries_kernel<<<slot_blocks, kThreads, 0, stream>>>(
            sparse_hash_counts, sparse_ref_offsets, sparse_write_positions,
            slot_count, entry_width, sparse_entry_counts, &device_summary->max_list_queries);
        CUDA_LAUNCH_CHECK("count sparse IVF schedule entries");
        fill_sparse_refs_kernel<<<ref_blocks, kThreads, 0, stream>>>(
            probes.indices, probes.query_count, probes.probe_count, probe_limit,
            nlist, hash_mask, sparse_hash_keys, sparse_write_positions,
            sparse_queries, sparse_probe_ranks);
        CUDA_LAUNCH_CHECK("fill sparse IVF probe references");
        check_cuda(cub::DeviceScan::ExclusiveSum(
                       cub_temp, cub_temp_bytes, sparse_entry_counts,
                       sparse_entry_offsets, slot_count + 1, stream),
                   "scan sparse IVF schedule entries");
        fill_sparse_entries_kernel<<<slot_blocks, kThreads, 0, stream>>>(
            sparse_hash_keys, sparse_hash_counts, sparse_ref_offsets,
            sparse_entry_offsets, slot_count, entry_width,
            entry_clusters, entry_starts, entry_counts);
        CUDA_LAUNCH_CHECK("fill sparse IVF schedule entries");

        return finish(
            sparse_entry_offsets + slot_count, sparse_ref_offsets + slot_count,
            sparse_queries,
            sparse_probe_ranks, IVFProbeScheduleStrategy::Sparse, stream);
    }

    IVFProbeScheduleResult finish(
        const int* device_entry_count,
        const int* device_reference_count,
        const int* queries,
        const int* probe_ranks,
        IVFProbeScheduleStrategy selected,
        cudaStream_t stream) {
        gather_summary_kernel<<<1, 32, 0, stream>>>(
            device_entry_count, device_reference_count, invalid, scanned_vectors,
            device_summary);
        CUDA_LAUNCH_CHECK("gather IVF schedule summary");
        check_cuda(cudaMemcpyAsync(
                       host_summary, device_summary, sizeof(ScheduleSummary),
                       cudaMemcpyDeviceToHost, stream),
                   "copy IVF schedule summary");
        check_cuda(cudaStreamSynchronize(stream), "finish IVF probe schedule");
        const int host_entry_count = host_summary->entries;
        const int host_reference_count = host_summary->references;
        const int host_invalid = host_summary->invalid;
        const auto host_scanned = host_summary->scanned;
        if (host_invalid != 0) {
            throw std::out_of_range("IVF probe is outside list layout");
        }
        if (host_entry_count < 0 || host_entry_count > max_entries) {
            throw std::runtime_error("IVF probe schedule entry count is invalid");
        }
        if (host_reference_count < 0 || host_reference_count > max_refs) {
            throw std::runtime_error("IVF probe schedule reference count is invalid");
        }
        return {
            {entry_clusters, entry_starts, entry_counts, queries, probe_ranks,
             host_entry_count, host_summary->max_list_queries},
            selected,
            host_scanned,
            host_reference_count};
    }

    void release() noexcept {
#define RELEASE(ptr) do { if (ptr) cudaFree(ptr); ptr = nullptr; } while (0)
        RELEASE(entry_clusters); RELEASE(entry_starts); RELEASE(entry_counts);
        RELEASE(entry_tile_counts); RELEASE(entry_tile_offsets);
        RELEASE(invalid); RELEASE(scanned_vectors);
        RELEASE(device_summary);
        if (host_summary) cudaFreeHost(host_summary);
        host_summary = nullptr;
        RELEASE(dense_cluster_counts); RELEASE(dense_cluster_offsets);
        RELEASE(dense_write_positions); RELEASE(dense_queries);
        RELEASE(dense_probe_ranks); RELEASE(dense_entry_counts);
        RELEASE(dense_entry_offsets);
        RELEASE(sparse_hash_keys); RELEASE(sparse_hash_counts);
        RELEASE(sparse_ref_offsets); RELEASE(sparse_write_positions);
        RELEASE(sparse_entry_counts); RELEASE(sparse_entry_offsets);
        RELEASE(sparse_queries); RELEASE(sparse_probe_ranks);
        RELEASE(cub_temp);
#undef RELEASE
        cub_temp_bytes = 0;
    }

    int nlist = 0;
    int max_refs = 0;
    int max_entries = 0;
    int max_sparse_slots = 0;
    const int* list_counts = nullptr;
    IVFProbeScheduleStrategy strategy = IVFProbeScheduleStrategy::Auto;

    int* entry_clusters = nullptr;
    int* entry_starts = nullptr;
    int* entry_counts = nullptr;
    int* entry_tile_counts = nullptr;
    int* entry_tile_offsets = nullptr;
    ScheduleSummary* device_summary = nullptr;
    ScheduleSummary* host_summary = nullptr;
    int* invalid = nullptr;
    unsigned long long* scanned_vectors = nullptr;

    int* dense_cluster_counts = nullptr;
    int* dense_cluster_offsets = nullptr;
    int* dense_write_positions = nullptr;
    int* dense_queries = nullptr;
    int* dense_probe_ranks = nullptr;
    int* dense_entry_counts = nullptr;
    int* dense_entry_offsets = nullptr;

    int* sparse_hash_keys = nullptr;
    int* sparse_hash_counts = nullptr;
    int* sparse_ref_offsets = nullptr;
    int* sparse_write_positions = nullptr;
    int* sparse_entry_counts = nullptr;
    int* sparse_entry_offsets = nullptr;
    int* sparse_queries = nullptr;
    int* sparse_probe_ranks = nullptr;

    void* cub_temp = nullptr;
    size_t cub_temp_bytes = 0;
};

IVFProbeScheduleBuilder::IVFProbeScheduleBuilder(
    int nlist,
    int max_probe_references,
    const int* device_list_counts,
    IVFProbeScheduleStrategy strategy)
    : impl_(std::make_unique<Impl>(
          nlist, max_probe_references, device_list_counts, strategy)) {}

IVFProbeScheduleBuilder::~IVFProbeScheduleBuilder() = default;

IVFProbeScheduleResult IVFProbeScheduleBuilder::build(
    const IVFDeviceProbeView& probes,
    int probe_limit,
    cudaStream_t stream,
    bool allow_padding) {
    return impl_->build(probes, probe_limit, stream, allow_padding);
}

IVFProbeTileScheduleResult IVFProbeScheduleBuilder::build_tiles(
    const IVFProbeScheduleView& schedule,
    int tile_rows,
    cudaStream_t stream) {
    return impl_->build_tiles(schedule, tile_rows, stream);
}

}  // namespace paraselect::search
