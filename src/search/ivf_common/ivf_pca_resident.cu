#include "quantization/pca/pca_storage.h"
#include "search/ivf_common/ivf_pca_resident.h"

#include "layout/ivf_interleaved_layout.cuh"
#include "search/kernels/fine/gpu_fine/resident/resident_full_l2_topk.cuh"
#include "search/kernels/fine/gpu_fine/resident/resident_threshold_l2_topk.cuh"
#include "utils/check_cuda_error.cuh"
#include "utils/cuda_scan.cuh"
#include "utils/utils.cuh"

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <climits>
#include <stdexcept>
#include <string>
#include <vector>

namespace paraselect::search {
namespace {

constexpr int kThreads = 128;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <int StaticDim, bool PackDp4a>
__global__ void pack_pca_interleaved_kernel(
    const int8_t* __restrict__ rows,
    const int* __restrict__ padded_ids,
    const float* __restrict__ scales,
    int count,
    int runtime_dim,
    int8_t* __restrict__ interleaved,
    float* __restrict__ squared_norms) {
    const int row = blockIdx.x;
    if (row >= count) return;
    constexpr int kGroup = paraselect::kInterleavedGroupSize;
    const int dim = StaticDim > 0 ? StaticDim : runtime_dim;
    const int padded = padded_ids[row];
    const int block = padded / kGroup;
    const int lane = padded % kGroup;
    float local_norm = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        const int8_t code = rows[(size_t)row * dim + d];
        size_t output = (size_t)block * kGroup * dim;
        if constexpr (PackDp4a) {
            output += ((size_t)(d / 4) * kGroup + lane) * 4 + d % 4;
        } else {
            output += (size_t)d * kGroup + lane;
        }
        interleaved[output] = code;
        const float value = static_cast<float>(code) * scales[d];
        local_norm += value * value;
    }
    __shared__ float reduction[kThreads];
    reduction[threadIdx.x] = local_norm;
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width /= 2) {
        if (threadIdx.x < width) reduction[threadIdx.x] += reduction[threadIdx.x + width];
        __syncthreads();
    }
    if (threadIdx.x == 0) squared_norms[padded] = reduction[0];
}

template <int StaticDim>
__global__ void project_weighted_pca_queries_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ projection,
    const float* __restrict__ bias,
    const float* __restrict__ scales,
    int batch_n,
    int source_dim,
    int runtime_dim,
    float* __restrict__ weighted_queries,
    float* __restrict__ squared_norms) {
    const int query_id = blockIdx.x;
    if (query_id >= batch_n) return;
    const int dim = StaticDim > 0 ? StaticDim : runtime_dim;
    const float* query = queries + (size_t)query_id * source_dim;
    float local_norm = 0.0f;
    for (int output_dim = threadIdx.x; output_dim < dim; output_dim += blockDim.x) {
        const float* row = projection + (size_t)output_dim * source_dim;
        float value = bias[output_dim];
#pragma unroll 4
        for (int d = 0; d < source_dim; ++d) value += row[d] * query[d];
        weighted_queries[(size_t)query_id * dim + output_dim] = value * scales[output_dim];
        local_norm += value * value;
    }
    __shared__ float reduction[kThreads];
    reduction[threadIdx.x] = local_norm;
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width /= 2) {
        if (threadIdx.x < width) reduction[threadIdx.x] += reduction[threadIdx.x + width];
        __syncthreads();
    }
    if (threadIdx.x == 0) squared_norms[query_id] = reduction[0];
}

__global__ void quantize_weighted_pca_queries_kernel(
    const float* __restrict__ weighted_queries,
    int batch_n,
    int dim,
    int8_t* __restrict__ quantized_queries,
    float* __restrict__ dot_scales) {
    const int query_id = blockIdx.x;
    if (query_id >= batch_n) return;
    const float* input = weighted_queries + (size_t)query_id * dim;
    float local_max = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(input[d]));
    }
    __shared__ float reduction[kThreads];
    __shared__ float shared_scale;
    reduction[threadIdx.x] = local_max;
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width /= 2) {
        if (threadIdx.x < width) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x], reduction[threadIdx.x + width]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        shared_scale = reduction[0] > 0.0f ? reduction[0] / 127.0f : 1.0f;
        dot_scales[query_id] = shared_scale;
    }
    __syncthreads();
    int8_t* output = quantized_queries + (size_t)query_id * dim;
    const float inverse_scale = 1.0f / shared_scale;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        const int value = max(-127, min(127, __float2int_rn(input[d] * inverse_scale)));
        output[d] = static_cast<int8_t>(value);
    }
}

__global__ void init_probe_topk_kernel(int count, float* dist, idx_t* index) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    dist[i] = FLT_MAX;
    index[i] = -1;
}

__global__ void fill_counts_kernel(int count, int value, int* output) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = value;
}

struct ProbeRef {
    int cluster;
    int query;
    int probe;
};

struct EntryBatch {
    std::vector<int> cluster;
    std::vector<int> start;
    std::vector<int> count;
    std::vector<int> queries;
    std::vector<int> probes;
};

struct DeviceEntryView {
    const int* cluster = nullptr;
    const int* start = nullptr;
    const int* count = nullptr;
    const int* queries = nullptr;
    const int* probes = nullptr;
    int size = 0;
};

struct PreparedMetadata {
    DeviceEntryView all;
    DeviceEntryView seed;
    unsigned long long scanned_vectors = 0;
    IVFProbeScheduleStrategy strategy = IVFProbeScheduleStrategy::Dense;
};

DeviceEntryView as_device_entry_view(const IVFProbeScheduleView& view) {
    return {
        view.clusters, view.starts, view.counts,
        view.queries, view.probe_ranks, view.entry_count};
}

EntryBatch make_entries(const std::vector<ProbeRef>& sorted_refs, int max_probe) {
    EntryBatch out;
    out.queries.reserve(sorted_refs.size());
    out.probes.reserve(sorted_refs.size());
    size_t begin = 0;
    while (begin < sorted_refs.size()) {
        const int cluster = sorted_refs[begin].cluster;
        const size_t run_begin = out.queries.size();
        size_t end = begin;
        while (end < sorted_refs.size() && sorted_refs[end].cluster == cluster) {
            if (sorted_refs[end].probe < max_probe) {
                out.queries.push_back(sorted_refs[end].query);
                out.probes.push_back(sorted_refs[end].probe);
            }
            ++end;
        }
        const size_t selected = out.queries.size() - run_begin;
        for (size_t offset = 0; offset < selected; offset += 8) {
            out.cluster.push_back(cluster);
            out.start.push_back(static_cast<int>(run_begin + offset));
            out.count.push_back(static_cast<int>(std::min<size_t>(8, selected - offset)));
        }
        begin = end;
    }
    return out;
}

}  // namespace

struct IVFPCAResidentSearch::Impl {
    Impl(
        const paraselect::quantization::PCAStorageView& storage_view,
        const IVFListLayoutView& list_view,
        const IVFPCAResidentConfig& requested)
        : storage(storage_view), config(requested) {
        validate_ivf_list_layout(list_view);
        if (!storage.vectors || !storage.projection || !storage.bias || !storage.scales ||
            storage.count <= 0 || storage.count > INT_MAX ||
            storage.count != list_view.n_vectors || storage.source_dim <= 0 ||
            storage.dim <= 0 || storage.dim > storage.source_dim ||
            config.max_batch <= 0 || config.max_nprobe <= 0 ||
            config.max_nprobe > list_view.nlist || config.output_count <= 0 ||
            config.output_count > 256 || config.prepare_chunk_vectors <= 0) {
            throw std::invalid_argument("invalid resident PCA search configuration");
        }
        if (config.mode == IVFPCAResidentMode::Threshold &&
            (config.threshold_seed_probes <= 0 ||
             config.threshold_candidate_capacity < config.output_count)) {
            throw std::invalid_argument("invalid resident PCA threshold configuration");
        }
        if (config.scan == IVFPCAScanKind::I8Dp4a &&
            (storage.dim % 4 != 0 || storage.dim > 128)) {
            throw std::invalid_argument(
                "resident PCA dp4a requires a dimension divisible by 4 and at most 128");
        }
        if (config.dp4a_kernel != IVFPCADp4aKernelVersion::V1 &&
            config.dp4a_kernel != IVFPCADp4aKernelVersion::V2 &&
            config.dp4a_kernel != IVFPCADp4aKernelVersion::V3) {
            throw std::invalid_argument("unknown resident PCA dp4a kernel version");
        }
        if (config.probe_schedule != IVFProbeScheduleStrategy::Auto &&
            config.probe_schedule != IVFProbeScheduleStrategy::Dense &&
            config.probe_schedule != IVFProbeScheduleStrategy::Sparse) {
            throw std::invalid_argument("unknown resident PCA probe schedule strategy");
        }
        const size_t max_probe_references =
            static_cast<size_t>(config.max_batch) * config.max_nprobe;
        if (max_probe_references > static_cast<size_t>(INT_MAX) / 2) {
            throw std::invalid_argument(
                "resident PCA probe schedule exceeds int32 workspace limit");
        }
        nlist = list_view.nlist;
        counts.resize(nlist);
        compact_offsets.resize((size_t)nlist + 1);
        padded_offsets.resize((size_t)nlist + 1);
        host_cluster_ref_counts.resize(nlist);
        host_cluster_write.resize(nlist);
        host_sorted_refs.resize((size_t)config.max_batch * config.max_nprobe);
        long long padded_total = 0;
        for (int list = 0; list < nlist; ++list) {
            const idx_t compact = list_view.offsets[list];
            const idx_t count64 = list_view.offsets[list + 1] - compact;
            if (compact > INT_MAX || count64 > INT_MAX) {
                throw std::overflow_error("resident PCA list metadata exceeds int32 range");
            }
            compact_offsets[list] = static_cast<int>(compact);
            padded_offsets[list] = static_cast<int>(padded_total);
            counts[list] = static_cast<int>(count64);
            padded_total += ((count64 + 31) / 32) * 32;
            if (padded_total > INT_MAX) {
                throw std::overflow_error("resident PCA padded storage exceeds int32 range");
            }
        }
        compact_offsets[nlist] = static_cast<int>(storage.count);
        padded_offsets[nlist] = static_cast<int>(padded_total);
        padded_count = static_cast<int>(padded_total);

        try {
            allocate_static();
            prepare_vectors();
            allocate_workspace();
        } catch (...) {
            release();
            throw;
        }
    }

    ~Impl() { release(); }

    void search_prepared(
        const float* host_queries,
        int batch_n,
        int nprobe,
        int* host_output_compact_ids,
        IVFPCAResidentStats* stats,
        cudaStream_t stream,
        const PreparedMetadata& metadata,
        double metadata_ms);

    template <typename T>
    void allocate(T** ptr, size_t count, const char* operation) {
        check_cuda(cudaMalloc(ptr, count * sizeof(T)), operation);
    }

    void allocate_static() {
        allocate(&d_projection, (size_t)storage.dim * storage.source_dim,
                 "allocate resident PCA projection");
        allocate(&d_bias, storage.dim, "allocate resident PCA bias");
        allocate(&d_scales, storage.dim, "allocate resident PCA scales");
        allocate(&d_vectors, (size_t)padded_count * storage.dim,
                 "allocate resident PCA interleaved vectors");
        allocate(&d_vector_squared_norms, padded_count,
                 "allocate resident PCA vector norms");
        allocate(&d_counts, nlist, "allocate resident PCA list counts");
        allocate(&d_compact_offsets, (size_t)nlist + 1,
                 "allocate resident PCA compact offsets");
        allocate(&d_padded_offsets, (size_t)nlist + 1,
                 "allocate resident PCA padded offsets");
        check_cuda(cudaMemcpy(d_projection, storage.projection,
                              (size_t)storage.dim * storage.source_dim * sizeof(float),
                              cudaMemcpyHostToDevice), "copy resident PCA projection");
        check_cuda(cudaMemcpy(d_bias, storage.bias, (size_t)storage.dim * sizeof(float),
                              cudaMemcpyHostToDevice), "copy resident PCA bias");
        check_cuda(cudaMemcpy(d_scales, storage.scales, (size_t)storage.dim * sizeof(float),
                              cudaMemcpyHostToDevice), "copy resident PCA scales");
        check_cuda(cudaMemcpy(d_counts, counts.data(), (size_t)nlist * sizeof(int),
                              cudaMemcpyHostToDevice), "copy resident PCA list counts");
        check_cuda(cudaMemcpy(d_compact_offsets, compact_offsets.data(),
                              ((size_t)nlist + 1) * sizeof(int), cudaMemcpyHostToDevice),
                   "copy resident PCA compact offsets");
        check_cuda(cudaMemcpy(d_padded_offsets, padded_offsets.data(),
                              ((size_t)nlist + 1) * sizeof(int), cudaMemcpyHostToDevice),
                   "copy resident PCA padded offsets");
        check_cuda(cudaMemset(d_vectors, 0, (size_t)padded_count * storage.dim),
                   "clear resident PCA interleaved padding");
        check_cuda(cudaMemset(d_vector_squared_norms, 0, (size_t)padded_count * sizeof(float)),
                   "clear resident PCA norm padding");
    }

    void prepare_vectors() {
        const int chunk_capacity = std::min<int>(
            config.prepare_chunk_vectors, static_cast<int>(storage.count));
        int8_t* d_rows = nullptr;
        int* d_padded_ids = nullptr;
        allocate(&d_rows, (size_t)chunk_capacity * storage.dim,
                 "allocate resident PCA row staging");
        allocate(&d_padded_ids, chunk_capacity, "allocate resident PCA ID staging");
        std::vector<int> host_padded_ids((size_t)chunk_capacity);
        int list = 0;
        try {
            for (int begin = 0; begin < static_cast<int>(storage.count); begin += chunk_capacity) {
                const int count = std::min(chunk_capacity, static_cast<int>(storage.count) - begin);
                for (int row = 0; row < count; ++row) {
                    const int compact = begin + row;
                    while (list + 1 < nlist && compact >= compact_offsets[list + 1]) ++list;
                    host_padded_ids[row] = padded_offsets[list] + compact - compact_offsets[list];
                }
                check_cuda(cudaMemcpy(d_rows, storage.vectors + (size_t)begin * storage.dim,
                                      (size_t)count * storage.dim, cudaMemcpyHostToDevice),
                           "stage resident PCA rows");
                check_cuda(cudaMemcpy(d_padded_ids, host_padded_ids.data(),
                                      (size_t)count * sizeof(int), cudaMemcpyHostToDevice),
                           "stage resident PCA padded IDs");
                if (config.scan == IVFPCAScanKind::I8Dp4a &&
                    config.dp4a_kernel != IVFPCADp4aKernelVersion::V1) {
                    if (storage.dim == 64) {
                        pack_pca_interleaved_kernel<64, true><<<count, kThreads>>>(
                            d_rows, d_padded_ids, d_scales, count, storage.dim,
                            d_vectors, d_vector_squared_norms);
                    } else {
                        pack_pca_interleaved_kernel<0, true><<<count, kThreads>>>(
                            d_rows, d_padded_ids, d_scales, count, storage.dim,
                            d_vectors, d_vector_squared_norms);
                    }
                } else if (storage.dim == 64) {
                    pack_pca_interleaved_kernel<64, false><<<count, kThreads>>>(
                        d_rows, d_padded_ids, d_scales, count, storage.dim,
                        d_vectors, d_vector_squared_norms);
                } else {
                    pack_pca_interleaved_kernel<0, false><<<count, kThreads>>>(
                        d_rows, d_padded_ids, d_scales, count, storage.dim,
                        d_vectors, d_vector_squared_norms);
                }
                CUDA_LAUNCH_CHECK("pack resident PCA interleaved chunk");
                check_cuda(cudaDeviceSynchronize(), "finish resident PCA interleaved chunk");
            }
        } catch (...) {
            cudaFree(d_rows);
            cudaFree(d_padded_ids);
            throw;
        }
        check_cuda(cudaFree(d_rows), "release resident PCA row staging");
        check_cuda(cudaFree(d_padded_ids), "release resident PCA ID staging");
    }

    void allocate_workspace() {
        const size_t max_pairs = (size_t)config.max_batch * config.max_nprobe;
        const int max_probe_topk = config.mode == IVFPCAResidentMode::Threshold
            ? std::min(config.max_nprobe, config.threshold_seed_probes)
            : config.max_nprobe;
        const size_t probe_values =
            (size_t)config.max_batch * max_probe_topk * config.output_count;
        allocate(&d_queries, (size_t)config.max_batch * storage.source_dim,
                 "allocate resident PCA query input");
        allocate(&d_weighted_queries, (size_t)config.max_batch * storage.dim,
                 "allocate resident PCA weighted queries");
        allocate(&d_query_squared_norms, config.max_batch,
                 "allocate resident PCA query norms");
        if (config.scan == IVFPCAScanKind::I8Dp4a) {
            allocate(&d_quantized_queries, (size_t)config.max_batch * storage.dim,
                     "allocate resident PCA quantized queries");
            allocate(&d_query_dot_scales, config.max_batch,
                     "allocate resident PCA query dot scales");
        }
        allocate(&d_entry_cluster, max_pairs, "allocate resident PCA entry clusters");
        allocate(&d_entry_start, max_pairs, "allocate resident PCA entry starts");
        allocate(&d_entry_count, max_pairs, "allocate resident PCA entry counts");
        allocate(&d_entry_queries, max_pairs, "allocate resident PCA entry queries");
        allocate(&d_entry_probes, max_pairs, "allocate resident PCA entry probes");
        device_schedule = std::make_unique<IVFProbeScheduleBuilder>(
            nlist, static_cast<int>(max_pairs), d_counts,
            config.probe_schedule);
        allocate(&d_probe_dist, probe_values, "allocate resident PCA probe distances");
        allocate(&d_probe_index, probe_values, "allocate resident PCA probe indices");
        allocate(&d_candidate_counts, config.max_batch,
                 "allocate resident PCA candidate counts");
        allocate(&d_final_dist, (size_t)config.max_batch * config.output_count,
                 "allocate resident PCA output distances");
        allocate(&d_final_index, (size_t)config.max_batch * config.output_count,
                 "allocate resident PCA output indices");
        host_final_index.resize((size_t)config.max_batch * config.output_count);
        if (config.mode == IVFPCAResidentMode::Threshold) {
            allocate(&d_seed_entry_cluster, max_pairs,
                     "allocate resident PCA seed entry clusters");
            allocate(&d_seed_entry_start, max_pairs,
                     "allocate resident PCA seed entry starts");
            allocate(&d_seed_entry_count, max_pairs,
                     "allocate resident PCA seed entry sizes");
            allocate(&d_seed_entry_queries, max_pairs,
                     "allocate resident PCA seed entry queries");
            allocate(&d_seed_entry_probes, max_pairs,
                     "allocate resident PCA seed entry probes");
            const int max_seed_probes =
                std::min(config.max_nprobe, config.threshold_seed_probes);
            device_seed_schedule = std::make_unique<IVFProbeScheduleBuilder>(
                nlist, config.max_batch * max_seed_probes, d_counts,
                config.probe_schedule);
            allocate(&d_thresholds, config.max_batch, "allocate resident PCA thresholds");
            allocate(&d_overflow, 1, "allocate resident PCA overflow flag");
            allocate(&d_threshold_dist,
                     (size_t)config.max_batch * config.threshold_candidate_capacity,
                     "allocate resident PCA threshold distances");
            allocate(&d_threshold_index,
                     (size_t)config.max_batch * config.threshold_candidate_capacity,
                     "allocate resident PCA threshold indices");
        }
        check_cuda(cudaEventCreate(&transform_begin), "create resident PCA event");
        check_cuda(cudaEventCreate(&transform_end), "create resident PCA event");
        check_cuda(cudaEventCreate(&scan_end), "create resident PCA event");
        check_cuda(cudaEventCreate(&reduce_end), "create resident PCA event");
    }

    void copy_entries(
        const EntryBatch& entries,
        int* clusters,
        int* starts,
        int* counts_out,
        int* queries,
        int* probes,
        cudaStream_t stream) {
        const size_t n_entry = entries.cluster.size();
        const size_t n_ref = entries.queries.size();
        check_cuda(cudaMemcpyAsync(clusters, entries.cluster.data(), n_entry * sizeof(int),
                                   cudaMemcpyHostToDevice, stream), "copy resident PCA entry clusters");
        check_cuda(cudaMemcpyAsync(starts, entries.start.data(), n_entry * sizeof(int),
                                   cudaMemcpyHostToDevice, stream), "copy resident PCA entry starts");
        check_cuda(cudaMemcpyAsync(counts_out, entries.count.data(), n_entry * sizeof(int),
                                   cudaMemcpyHostToDevice, stream), "copy resident PCA entry counts");
        check_cuda(cudaMemcpyAsync(queries, entries.queries.data(), n_ref * sizeof(int),
                                   cudaMemcpyHostToDevice, stream), "copy resident PCA entry queries");
        check_cuda(cudaMemcpyAsync(probes, entries.probes.data(), n_ref * sizeof(int),
                                   cudaMemcpyHostToDevice, stream), "copy resident PCA entry probes");
    }

    DeviceEntryView all_entries(int size) const {
        return {d_entry_cluster, d_entry_start, d_entry_count,
                d_entry_queries, d_entry_probes, size};
    }

    DeviceEntryView seed_entries(int size) const {
        return {d_seed_entry_cluster, d_seed_entry_start, d_seed_entry_count,
                d_seed_entry_queries, d_seed_entry_probes, size};
    }

    PreparedMetadata prepare_host_metadata(
        const idx_t* host_top_probes,
        int batch_n,
        int nprobe,
        cudaStream_t stream) {
        std::vector<ProbeRef> refs;
        refs.reserve((size_t)batch_n * nprobe);
        unsigned long long scanned = 0;
        for (int query = 0; query < batch_n; ++query) {
            for (int probe = 0; probe < nprobe; ++probe) {
                const idx_t cluster64 =
                    host_top_probes[(size_t)query * nprobe + probe];
                if (cluster64 < 0 || cluster64 >= nlist) {
                    throw std::out_of_range(
                        "resident PCA probe is outside list layout");
                }
                const int cluster = static_cast<int>(cluster64);
                refs.push_back({cluster, query, probe});
                scanned += static_cast<unsigned long long>(counts[cluster]);
            }
        }
        std::fill(host_cluster_ref_counts.begin(),
                  host_cluster_ref_counts.end(), 0);
        for (const ProbeRef& ref : refs) ++host_cluster_ref_counts[ref.cluster];
        int write = 0;
        for (int cluster = 0; cluster < nlist; ++cluster) {
            host_cluster_write[cluster] = write;
            write += host_cluster_ref_counts[cluster];
        }
        host_sorted_refs.resize(refs.size());
        for (const ProbeRef& ref : refs) {
            host_sorted_refs[host_cluster_write[ref.cluster]++] = ref;
        }
        const EntryBatch all = make_entries(host_sorted_refs, nprobe);
        copy_entries(
            all, d_entry_cluster, d_entry_start, d_entry_count,
            d_entry_queries, d_entry_probes, stream);

        int seed_size = 0;
        if (config.mode == IVFPCAResidentMode::Threshold) {
            const EntryBatch seed = make_entries(
                host_sorted_refs,
                std::min(nprobe, config.threshold_seed_probes));
            seed_size = static_cast<int>(seed.cluster.size());
            copy_entries(
                seed, d_seed_entry_cluster, d_seed_entry_start,
                d_seed_entry_count, d_seed_entry_queries,
                d_seed_entry_probes, stream);
        }
        return {
            all_entries(static_cast<int>(all.cluster.size())),
            seed_entries(seed_size),
            scanned,
            IVFProbeScheduleStrategy::Dense};
    }

    PreparedMetadata prepare_device_metadata(
        const IVFDeviceProbeView& probe_view,
        cudaStream_t stream) {
        const int nprobe = probe_view.probe_count;
        const IVFProbeScheduleResult all =
            device_schedule->build(probe_view, nprobe, stream);
        IVFProbeScheduleResult seed;
        if (config.mode == IVFPCAResidentMode::Threshold) {
            const int seed_probes =
                std::min(nprobe, config.threshold_seed_probes);
            seed = device_seed_schedule->build(
                probe_view, seed_probes, stream);
        }
        return {
            as_device_entry_view(all.view),
            as_device_entry_view(seed.view),
            all.scanned_vectors,
            all.strategy};
    }

    void release() noexcept {
        device_seed_schedule.reset();
        device_schedule.reset();
#define RELEASE(ptr) do { if (ptr) cudaFree(ptr); ptr = nullptr; } while (0)
        RELEASE(d_projection); RELEASE(d_bias); RELEASE(d_scales); RELEASE(d_vectors);
        RELEASE(d_vector_squared_norms); RELEASE(d_counts); RELEASE(d_compact_offsets);
        RELEASE(d_padded_offsets); RELEASE(d_queries); RELEASE(d_weighted_queries);
        RELEASE(d_quantized_queries); RELEASE(d_query_dot_scales);
        RELEASE(d_query_squared_norms); RELEASE(d_entry_cluster); RELEASE(d_entry_start);
        RELEASE(d_entry_count); RELEASE(d_entry_queries); RELEASE(d_entry_probes);
        RELEASE(d_seed_entry_cluster); RELEASE(d_seed_entry_start);
        RELEASE(d_seed_entry_count); RELEASE(d_seed_entry_queries);
        RELEASE(d_seed_entry_probes);
        RELEASE(d_probe_dist); RELEASE(d_probe_index); RELEASE(d_candidate_counts);
        RELEASE(d_final_dist); RELEASE(d_final_index); RELEASE(d_thresholds);
        RELEASE(d_overflow); RELEASE(d_threshold_dist); RELEASE(d_threshold_index);
#undef RELEASE
        if (transform_begin) cudaEventDestroy(transform_begin);
        if (transform_end) cudaEventDestroy(transform_end);
        if (scan_end) cudaEventDestroy(scan_end);
        if (reduce_end) cudaEventDestroy(reduce_end);
        transform_begin = transform_end = scan_end = reduce_end = nullptr;
    }

    paraselect::quantization::PCAStorageView storage;
    IVFPCAResidentConfig config;
    int nlist = 0;
    int padded_count = 0;
    std::vector<int> counts;
    std::vector<int> compact_offsets;
    std::vector<int> padded_offsets;
    std::vector<int> host_cluster_ref_counts;
    std::vector<int> host_cluster_write;
    std::vector<ProbeRef> host_sorted_refs;
    std::vector<idx_t> host_final_index;
    float* d_projection = nullptr;
    float* d_bias = nullptr;
    float* d_scales = nullptr;
    int8_t* d_vectors = nullptr;
    float* d_vector_squared_norms = nullptr;
    int* d_counts = nullptr;
    int* d_compact_offsets = nullptr;
    int* d_padded_offsets = nullptr;
    float* d_queries = nullptr;
    float* d_weighted_queries = nullptr;
    int8_t* d_quantized_queries = nullptr;
    float* d_query_dot_scales = nullptr;
    float* d_query_squared_norms = nullptr;
    int* d_entry_cluster = nullptr;
    int* d_entry_start = nullptr;
    int* d_entry_count = nullptr;
    int* d_entry_queries = nullptr;
    int* d_entry_probes = nullptr;
    std::unique_ptr<IVFProbeScheduleBuilder> device_schedule;
    std::unique_ptr<IVFProbeScheduleBuilder> device_seed_schedule;
    int* d_seed_entry_cluster = nullptr;
    int* d_seed_entry_start = nullptr;
    int* d_seed_entry_count = nullptr;
    int* d_seed_entry_queries = nullptr;
    int* d_seed_entry_probes = nullptr;
    float* d_probe_dist = nullptr;
    idx_t* d_probe_index = nullptr;
    int* d_candidate_counts = nullptr;
    float* d_final_dist = nullptr;
    idx_t* d_final_index = nullptr;
    float* d_thresholds = nullptr;
    int* d_overflow = nullptr;
    float* d_threshold_dist = nullptr;
    idx_t* d_threshold_index = nullptr;
    cudaEvent_t transform_begin = nullptr;
    cudaEvent_t transform_end = nullptr;
    cudaEvent_t scan_end = nullptr;
    cudaEvent_t reduce_end = nullptr;
};

IVFPCAResidentSearch::IVFPCAResidentSearch(
    const paraselect::quantization::PCAStorageView& storage,
    const IVFListLayoutView& lists,
    const IVFPCAResidentConfig& config)
    : impl_(std::make_unique<Impl>(storage, lists, config)) {}

IVFPCAResidentSearch::~IVFPCAResidentSearch() = default;

void IVFPCAResidentSearch::Impl::search_prepared(
    const float* host_queries,
    int batch_n,
    int nprobe,
    int* host_output_compact_ids,
    IVFPCAResidentStats* stats,
    cudaStream_t stream,
    const PreparedMetadata& metadata,
    double metadata_ms) {
    Impl* impl_ = this;

    check_cuda(cudaMemcpyAsync(
        impl_->d_queries, host_queries,
        (size_t)batch_n * impl_->storage.source_dim * sizeof(float),
        cudaMemcpyHostToDevice, stream), "copy resident PCA queries");
    check_cuda(cudaEventRecord(impl_->transform_begin, stream),
               "record resident PCA transform start");
    if (impl_->storage.dim == 64) {
        project_weighted_pca_queries_kernel<64><<<batch_n, kThreads, 0, stream>>>(
            impl_->d_queries, impl_->d_projection, impl_->d_bias, impl_->d_scales,
            batch_n, impl_->storage.source_dim, impl_->storage.dim,
            impl_->d_weighted_queries, impl_->d_query_squared_norms);
    } else {
        project_weighted_pca_queries_kernel<0><<<batch_n, kThreads, 0, stream>>>(
            impl_->d_queries, impl_->d_projection, impl_->d_bias, impl_->d_scales,
            batch_n, impl_->storage.source_dim, impl_->storage.dim,
            impl_->d_weighted_queries, impl_->d_query_squared_norms);
    }
    CUDA_LAUNCH_CHECK("project resident PCA queries");
    if (impl_->config.scan == IVFPCAScanKind::I8Dp4a) {
        quantize_weighted_pca_queries_kernel<<<batch_n, kThreads, 0, stream>>>(
            impl_->d_weighted_queries, batch_n, impl_->storage.dim,
            impl_->d_quantized_queries, impl_->d_query_dot_scales);
        CUDA_LAUNCH_CHECK("quantize resident PCA queries");
    }
    check_cuda(cudaEventRecord(impl_->transform_end, stream),
               "record resident PCA transform end");

    const int probe_count = impl_->config.mode == IVFPCAResidentMode::Threshold
        ? std::min(nprobe, impl_->config.threshold_seed_probes)
        : nprobe;
    const int probe_values = batch_n * probe_count * impl_->config.output_count;
    init_probe_topk_kernel<<<(probe_values + 255) / 256, 256, 0, stream>>>(
        probe_values, impl_->d_probe_dist, impl_->d_probe_index);
    CUDA_LAUNCH_CHECK("initialize resident PCA probe top-k");

    int overflow = 0;
    int max_selected_per_query = 0;
    int selected = batch_n * nprobe * impl_->config.output_count;
    const auto launch_probe_topk = [&](
        const DeviceEntryView& entries, int output_probe_count) {
        if (impl_->config.scan == IVFPCAScanKind::I8Dp4a) {
            const auto launch =
                impl_->config.dp4a_kernel == IVFPCADp4aKernelVersion::V1
                ? launch_resident_pca_i8_l2_topk_dp4a_v1
                : launch_resident_pca_i8_l2_topk_dp4a_v2;
            launch(
                impl_->config.output_count, dim3(256), impl_->storage.dim,
                impl_->d_quantized_queries, impl_->d_query_dot_scales,
                impl_->d_vectors, impl_->d_padded_offsets,
                impl_->d_compact_offsets, impl_->d_counts,
                entries.cluster, entries.start, entries.count,
                entries.queries, entries.probes, impl_->d_query_squared_norms,
                impl_->d_vector_squared_norms,
                entries.size, output_probe_count,
                impl_->d_probe_dist, impl_->d_probe_index, stream);
        } else {
            launch_resident_pca_i8_l2_topk_full(
                impl_->config.output_count, dim3(256), impl_->storage.dim,
                impl_->d_weighted_queries, impl_->d_vectors,
                impl_->d_padded_offsets, impl_->d_compact_offsets,
                impl_->d_counts, entries.cluster,
                entries.start, entries.count,
                entries.queries, entries.probes,
                impl_->d_query_squared_norms, impl_->d_vector_squared_norms,
                entries.size, output_probe_count,
                impl_->d_probe_dist, impl_->d_probe_index, stream);
        }
    };
    if (impl_->config.mode == IVFPCAResidentMode::Full) {
        launch_probe_topk(metadata.all, nprobe);
        CUDA_LAUNCH_CHECK("resident PCA full scan");
        check_cuda(cudaEventRecord(impl_->scan_end, stream),
                   "record resident PCA scan end");
        fill_counts_kernel<<<(batch_n + 255) / 256, 256, 0, stream>>>(
            batch_n, nprobe * impl_->config.output_count, impl_->d_candidate_counts);
        launch_resident_threshold_reduce(
            impl_->config.output_count, batch_n,
            nprobe * impl_->config.output_count, static_cast<int>(impl_->storage.count),
            impl_->d_candidate_counts, impl_->d_probe_dist, impl_->d_probe_index,
            nullptr, impl_->d_final_dist, impl_->d_final_index, stream);
    } else {
        launch_probe_topk(metadata.seed, probe_count);
        launch_resident_threshold_from_probe_topk(
            impl_->config.output_count, batch_n, probe_count, probe_count,
            impl_->d_probe_dist, impl_->d_thresholds, stream);
        check_cuda(cudaMemsetAsync(impl_->d_candidate_counts, 0,
                                   (size_t)batch_n * sizeof(int), stream),
                   "clear resident PCA threshold counts");
        check_cuda(cudaMemsetAsync(impl_->d_overflow, 0, sizeof(int), stream),
                   "clear resident PCA threshold overflow");
        if (impl_->config.scan == IVFPCAScanKind::I8Dp4a) {
            const auto launch = impl_->config.dp4a_kernel ==
                                    IVFPCADp4aKernelVersion::V1
                ? launch_resident_pca_i8_dp4a_v1_threshold_grouped_scan
                : impl_->config.dp4a_kernel == IVFPCADp4aKernelVersion::V2
                    ? launch_resident_pca_i8_dp4a_v2_threshold_grouped_scan
                    : launch_resident_pca_i8_dp4a_v3_threshold_grouped_scan;
            launch(
                impl_->storage.dim, metadata.all.size,
                impl_->config.threshold_candidate_capacity,
                impl_->d_quantized_queries, impl_->d_query_dot_scales,
                impl_->d_vectors, impl_->d_padded_offsets,
                impl_->d_compact_offsets, impl_->d_counts,
                metadata.all.cluster, metadata.all.start,
                metadata.all.count, metadata.all.queries,
                impl_->d_query_squared_norms, impl_->d_vector_squared_norms,
                impl_->d_thresholds, impl_->d_candidate_counts,
                impl_->d_overflow, impl_->d_threshold_dist,
                impl_->d_threshold_index, stream);
        } else {
            launch_resident_pca_i8_threshold_grouped_scan(
                impl_->storage.dim, metadata.all.size,
                impl_->config.threshold_candidate_capacity,
                impl_->d_weighted_queries, impl_->d_vectors,
                impl_->d_padded_offsets, impl_->d_compact_offsets,
                impl_->d_counts, metadata.all.cluster,
                metadata.all.start, metadata.all.count,
                metadata.all.queries, impl_->d_query_squared_norms,
                impl_->d_vector_squared_norms, impl_->d_thresholds,
                impl_->d_candidate_counts, impl_->d_overflow,
                impl_->d_threshold_dist, impl_->d_threshold_index, stream);
        }
        CUDA_LAUNCH_CHECK("resident PCA threshold scan");
        check_cuda(cudaEventRecord(impl_->scan_end, stream),
                   "record resident PCA threshold scan end");
        launch_resident_threshold_reduce(
            impl_->config.output_count, batch_n,
            impl_->config.threshold_candidate_capacity,
            static_cast<int>(impl_->storage.count), impl_->d_candidate_counts,
            impl_->d_threshold_dist, impl_->d_threshold_index, nullptr,
            impl_->d_final_dist, impl_->d_final_index, stream);
        check_cuda(cudaMemcpyAsync(&overflow, impl_->d_overflow, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream),
                   "copy resident PCA threshold overflow");
        std::vector<int> candidate_counts((size_t)batch_n);
        check_cuda(cudaMemcpyAsync(candidate_counts.data(), impl_->d_candidate_counts,
                                   (size_t)batch_n * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream),
                   "copy resident PCA threshold counts");
        selected = 0;
        check_cuda(cudaStreamSynchronize(stream), "read resident PCA threshold counts");
        for (int count : candidate_counts) {
            selected += count;
            max_selected_per_query = std::max(max_selected_per_query, count);
        }
    }
    CUDA_LAUNCH_CHECK("reduce resident PCA candidates");
    check_cuda(cudaEventRecord(impl_->reduce_end, stream),
               "record resident PCA reduce end");
    check_cuda(cudaMemcpyAsync(
        impl_->host_final_index.data(), impl_->d_final_index,
        (size_t)batch_n * impl_->config.output_count * sizeof(idx_t),
        cudaMemcpyDeviceToHost, stream), "copy resident PCA compact IDs");
    check_cuda(cudaStreamSynchronize(stream),
               "finish resident PCA search");
    if (overflow) {
        throw std::runtime_error(
            "resident PCA threshold candidate capacity exceeded: required=" +
            std::to_string(max_selected_per_query) + " capacity=" +
            std::to_string(impl_->config.threshold_candidate_capacity));
    }
    const size_t output_values = (size_t)batch_n * impl_->config.output_count;
    for (size_t i = 0; i < output_values; ++i) {
        const idx_t value = impl_->host_final_index[i];
        host_output_compact_ids[i] = value >= 0 ? static_cast<int>(value) : -1;
    }

    if (stats) {
        float transform_ms = 0.0f;
        float scan_ms = 0.0f;
        float reduce_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&transform_ms, impl_->transform_begin,
                                        impl_->transform_end),
                   "measure resident PCA transform");
        check_cuda(cudaEventElapsedTime(&scan_ms, impl_->transform_end,
                                        impl_->scan_end),
                   "measure resident PCA scan");
        check_cuda(cudaEventElapsedTime(&reduce_ms, impl_->scan_end,
                                        impl_->reduce_end),
                   "measure resident PCA reduce");
        stats->metadata_ms = metadata_ms;
        stats->query_transform_ms = transform_ms;
        stats->scan_ms = scan_ms;
        stats->reduce_ms = reduce_ms;
        stats->avg_scanned_vectors =
            static_cast<double>(metadata.scanned_vectors) / batch_n;
        stats->avg_reduce_input_candidates = static_cast<double>(selected) / batch_n;
        stats->probe_schedule = metadata.strategy;
    }
}

void IVFPCAResidentSearch::search(
    const float* host_queries,
    const idx_t* host_top_probes,
    int batch_n,
    int nprobe,
    int* host_output_compact_ids,
    IVFPCAResidentStats* stats,
    cudaStream_t stream) {
    if (!host_queries || !host_top_probes || !host_output_compact_ids ||
        batch_n <= 0 || batch_n > impl_->config.max_batch || nprobe <= 0 ||
        nprobe > impl_->config.max_nprobe) {
        throw std::invalid_argument("invalid resident PCA search request");
    }
    const auto begin = std::chrono::steady_clock::now();
    const PreparedMetadata metadata = impl_->prepare_host_metadata(
        host_top_probes, batch_n, nprobe, stream);
    const double metadata_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    impl_->search_prepared(
        host_queries, batch_n, nprobe, host_output_compact_ids, stats, stream,
        metadata, metadata_ms);
}

void IVFPCAResidentSearch::search_device_probes(
    const float* host_queries,
    const IVFDeviceProbeView& probes,
    int* host_output_compact_ids,
    IVFPCAResidentStats* stats,
    cudaStream_t stream) {
    if (!host_queries || !probes.indices || !host_output_compact_ids ||
        probes.query_count <= 0 ||
        probes.query_count > impl_->config.max_batch ||
        probes.probe_count <= 0 ||
        probes.probe_count > impl_->config.max_nprobe) {
        throw std::invalid_argument(
            "invalid resident PCA device probe search request");
    }
    const auto begin = std::chrono::steady_clock::now();
    const PreparedMetadata metadata =
        impl_->prepare_device_metadata(probes, stream);
    const double metadata_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    impl_->search_prepared(
        host_queries, probes.query_count, probes.probe_count,
        host_output_compact_ids, stats, stream, metadata, metadata_ms);
}

}  // namespace paraselect::search
