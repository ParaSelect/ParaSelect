#ifndef PARASELECT_DATASET_ASSIGNMENT_MATERIALIZER_CUH
#define PARASELECT_DATASET_ASSIGNMENT_MATERIALIZER_CUH

#include "common/binary_format.h"
#include "common/dataset_source.h"
#include "common/temporary_mapping.h"
#include "cpu/utils/cpu_affinity.h"
#include "cpu/utils/persistent_thread_pool.h"
#include "cpu/utils/numa_cluster_owner.h"
#include "search/kernels/fine/cpu_fine/cpu_fine_layout.h"
#include "dataset/ownership_helper.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <malloc.h>
#include <mutex>
#include <sys/mman.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace paraselect {
namespace dataset {

template <typename T>
struct ReorderedMaterializeResult {
    T* data = nullptr;
    paraselect::ownership::BufferOwner owner = paraselect::ownership::BufferOwner::None;
};

using ReorderedFp32MaterializeResult = ReorderedMaterializeResult<float>;
using ReorderedU8MaterializeResult = ReorderedMaterializeResult<uint8_t>;

struct ReorderedAosoaMaterializeResult {
    float* data = nullptr;
    long long* offsets = nullptr;
    int tile = 0;
    paraselect::ownership::BufferOwner data_owner = paraselect::ownership::BufferOwner::None;
    paraselect::ownership::BufferOwner offsets_owner = paraselect::ownership::BufferOwner::None;
};

namespace detail {

template <typename T>
inline ReorderedMaterializeResult<T> materialize_reordered_sequential_impl(
    const DatasetSource& source,
    const idx_t* reordered_indices,
    int n_total_vectors,
    int vector_dim,
    VectorDType expected_dtype,
    const int* row_order = nullptr) {
    if (!reordered_indices || n_total_vectors <= 0 || vector_dim <= 0) {
        throw std::invalid_argument("invalid sequential reordered materialization shape");
    }
    if (source.spec().dtype != expected_dtype) {
        throw std::invalid_argument("sequential reordered materialization dtype mismatch");
    }
    if (static_cast<uint64_t>(n_total_vectors) > UINT32_MAX) {
        throw std::invalid_argument("sequential reordered materialization exceeds UINT32_MAX rows");
    }
    bool prefault_output = false;
    if (const char* value = std::getenv("PARASELECT_REORDER_PREFAULT")) {
        if (std::strcmp(value, "0") != 0 && std::strcmp(value, "1") != 0) {
            throw std::invalid_argument("PARASELECT_REORDER_PREFAULT must be 0 or 1");
        }
        prefault_output = std::strcmp(value, "1") == 0;
    }
    const char* temporary_directory = std::getenv("PARASELECT_TMPDIR");
    const std::string directory = temporary_directory && *temporary_directory
        ? temporary_directory : ".tmp";
    std::filesystem::create_directories(directory);
    // Reclaimable file pages avoid swapping this once-used multi-GB inverse map.
    TemporaryMapping<uint32_t> source_to_row_workspace(
        directory, "reorder-source-to-row", static_cast<size_t>(n_total_vectors));
    uint32_t* source_to_row = source_to_row_workspace.data();
    std::fill(source_to_row, source_to_row + n_total_vectors, UINT32_MAX);
    for (uint32_t row = 0; row < static_cast<uint32_t>(n_total_vectors); ++row) {
        const int source_row = row_order ? row_order[row] : static_cast<int>(row);
        if (source_row < 0 || source_row >= n_total_vectors)
            throw std::invalid_argument("materialize row order is out of range");
        const idx_t source_id = reordered_indices[source_row];
        if (source_id < 0 || source_id >= n_total_vectors) {
            throw std::invalid_argument("reordered id is out of range");
        }
        uint32_t& destination = source_to_row[static_cast<size_t>(source_id)];
        if (destination != UINT32_MAX) {
            throw std::invalid_argument("reordered ids are not a permutation");
        }
        destination = row;
    }

    const size_t elements = static_cast<size_t>(n_total_vectors) * vector_dim;
    const size_t output_bytes = elements * sizeof(T);
    // 2 MB alignment + MADV_HUGEPAGE. The rerank reads this buffer randomly
    // over >= 100 GB, so 4 KB pages thrash the TLB; transparent huge pages cut
    // the rerank stage by roughly a third on the measured SIFT1B pipeline.
    void* output_raw = memalign(2u << 20, output_bytes);
    if (output_raw) {
        madvise(output_raw, output_bytes, MADV_HUGEPAGE);
    }
    paraselect::ownership::BufferHandle output(
        output_raw, output_bytes, paraselect::ownership::BufferOwner::StdMalloc);
    if (!output.data) throw std::bad_alloc();
    T* destination = output.as<T>();
    if (prefault_output) {
        // Sequential first-touch keeps THP advice while avoiding concurrent
        // random faults during scatter. This is outside query timing.
        std::fprintf(stderr, "[MATERIALIZE_PREFAULT] begin bytes=%zu\n", output_bytes);
        const auto start = std::chrono::steady_clock::now();
        std::memset(destination, 0, output_bytes);
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        std::fprintf(stderr, "[MATERIALIZE_PREFAULT] complete seconds=%.3f\n", seconds);
    }
    constexpr size_t kChunkRows = 1U << 16;
    std::vector<T> source_rows(
        std::min(kChunkRows, static_cast<size_t>(n_total_vectors)) * vector_dim);
    // Sequential read + parallel scatter: the read stream stays single, the
    // random writes into the destination run on a persistent pool. This is the
    // dominant cost of the per-process exact-storage build.
    const int scatter_threads = std::max(
        1, std::min(paraselect::default_worker_thread_count(), 64));
    paraselect::PersistentThreadPool scatter_pool(scatter_threads);
    for (size_t begin = 0; begin < static_cast<size_t>(n_total_vectors); begin += kChunkRows) {
        const size_t rows = std::min(kChunkRows, static_cast<size_t>(n_total_vectors) - begin);
        source.read_base_range(begin, rows, source_rows.data());
        scatter_pool.run([&](int tid) {
            for (size_t local = static_cast<size_t>(tid); local < rows;
                 local += static_cast<size_t>(scatter_threads)) {
                const uint32_t row = source_to_row[begin + local];
                std::memcpy(
                    destination + static_cast<size_t>(row) * vector_dim,
                    source_rows.data() + local * static_cast<size_t>(vector_dim),
                    static_cast<size_t>(vector_dim) * sizeof(T));
            }
        });
    }
    return {static_cast<T*>(output.detach()), paraselect::ownership::BufferOwner::StdMalloc};
}

enum class MaterializeFileKind {
    Fbin,
    HeaderU8,
    Bvecs,
};

struct MaterializeFileLayout {
    MaterializeFileKind kind = MaterializeFileKind::Fbin;
    size_t n = 0;
    int dim = 0;
    size_t row_stride = 0;
    size_t payload_offset = 0;
};

inline MaterializeFileLayout resolve_materialize_layout(const std::string& path) {
    MaterializeFileLayout out;
    if (BinaryFormat::is_header_u8bin(path)) {
        const DatasetInfo info = BinaryFormat::read_header_ibin(path);
        out.kind = MaterializeFileKind::HeaderU8;
        out.n = info.n;
        out.dim = info.dim;
        out.row_stride = (size_t)info.dim;
        out.payload_offset = 8;
        return out;
    }

    const DatasetFormat fmt = BinaryFormat::detect_format(path);
    if (fmt == DatasetFormat::FBIN) {
        const DatasetInfo info = BinaryFormat::read_header_fbin(path);
        out.kind = MaterializeFileKind::Fbin;
        out.n = info.n;
        out.dim = info.dim;
        out.row_stride = (size_t)info.dim * sizeof(float);
        out.payload_offset = 8;
        return out;
    }
    if (fmt == DatasetFormat::BVECS) {
        const DatasetInfo info = BinaryFormat::get_file_info(path);
        out.kind = MaterializeFileKind::Bvecs;
        out.n = info.n;
        out.dim = info.dim;
        out.row_stride = 4 + (size_t)info.dim;
        out.payload_offset = 4;
        return out;
    }
    throw std::runtime_error("unsupported materialize source format: " + path);
}

inline void pread_exact(int fd, void* dst, size_t bytes, off_t offset) {
    char* p = static_cast<char*>(dst);
    size_t done = 0;
    while (done < bytes) {
        const ssize_t got = ::pread(fd, p + done, bytes - done, offset + (off_t)done);
        if (got < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("pread failed: ") + std::strerror(errno));
        }
        if (got == 0) throw std::runtime_error("short pread while materializing reordered data");
        done += (size_t)got;
    }
}

inline off_t vector_file_offset(const MaterializeFileLayout& layout, idx_t src_id) {
    if (src_id < 0 || (size_t)src_id >= layout.n) {
        throw std::runtime_error("reordered index out of file bounds");
    }
    return (off_t)(layout.payload_offset + (size_t)src_id * layout.row_stride);
}

inline void read_one_vector_as_float(
    int fd,
    const MaterializeFileLayout& layout,
    idx_t src_id,
    float* dst,
    std::vector<uint8_t>& scratch) {
    const off_t offset = vector_file_offset(layout, src_id);
    if (layout.kind == MaterializeFileKind::Fbin) {
        pread_exact(fd, dst, (size_t)layout.dim * sizeof(float), offset);
        return;
    }
    if (scratch.size() < (size_t)layout.dim) scratch.resize((size_t)layout.dim);
    pread_exact(fd, scratch.data(), (size_t)layout.dim, offset);
    for (int d = 0; d < layout.dim; ++d) {
        dst[(size_t)d] = static_cast<float>(scratch[(size_t)d]);
    }
}

inline void read_one_vector_as_u8(
    int fd,
    const MaterializeFileLayout& layout,
    idx_t src_id,
    uint8_t* dst,
    std::vector<float>& scratch) {
    const off_t offset = vector_file_offset(layout, src_id);
    if (layout.kind == MaterializeFileKind::HeaderU8 || layout.kind == MaterializeFileKind::Bvecs) {
        pread_exact(fd, dst, (size_t)layout.dim, offset);
        return;
    }
    if (scratch.size() < (size_t)layout.dim) scratch.resize((size_t)layout.dim);
    pread_exact(fd, scratch.data(), (size_t)layout.dim * sizeof(float), offset);
    for (int d = 0; d < layout.dim; ++d) {
        float v = scratch[(size_t)d];
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        dst[(size_t)d] = static_cast<uint8_t>(v + 0.5f);
    }
}

template <typename T, typename Scratch, typename Reader>
inline ReorderedMaterializeResult<T> materialize_reordered_from_file_impl(
    const std::string& raw_base_path,
    const idx_t* reordered_indices,
    const idx_t* cluster_offsets,
    const int* cluster_counts,
    int n_total_vectors,
    int vector_dim,
    int n_clusters,
    bool numa_first_touch,
    const char* storage_name,
    Reader reader)
{
    if (raw_base_path.empty()) {
        throw std::runtime_error(std::string(storage_name) + " materialization requires raw_base_path");
    }
    if (!reordered_indices || !cluster_offsets || !cluster_counts) {
        throw std::runtime_error(std::string(storage_name) + " materialization requires assignment metadata");
    }
    if (n_total_vectors <= 0 || vector_dim <= 0 || n_clusters <= 0) {
        throw std::runtime_error(std::string("invalid ") + storage_name + " materialization shape");
    }

    const MaterializeFileLayout layout = resolve_materialize_layout(raw_base_path);
    if (layout.dim != vector_dim) {
        throw std::runtime_error(std::string(storage_name) + " materialization dim mismatch");
    }
    if ((size_t)n_total_vectors > layout.n) {
        throw std::runtime_error(std::string(storage_name) + " materialization vector count exceeds source file");
    }

    const size_t bytes = (size_t)n_total_vectors * (size_t)vector_dim * sizeof(T);
    paraselect::ownership::BufferHandle out_handle(
        memalign(64, bytes), bytes, paraselect::ownership::BufferOwner::StdMalloc);
    if (!out_handle.data) throw std::bad_alloc();
    T* out = out_handle.as<T>();

    const int total_threads = std::max(1, std::min(paraselect::default_worker_thread_count(), n_clusters));
    const int hw_nodes = std::max(1, paraselect::get_numa_node_count());
    const int numa_nodes = numa_first_touch ? std::max(1, std::min(hw_nodes, total_threads)) : 1;
    const std::vector<int> owners =
        paraselect::numa::build_cluster_owner_by_counts(cluster_counts, n_clusters, numa_nodes);
    std::vector<std::vector<int>> clusters_by_node((size_t)numa_nodes);
    for (int c = 0; c < n_clusters; ++c) {
        const int node = std::max(0, std::min(owners[(size_t)c], numa_nodes - 1));
        clusters_by_node[(size_t)node].push_back(c);
    }

    std::atomic<bool> failed{false};
    std::string error_message;
    std::mutex error_mutex;
    std::vector<std::thread> node_workers;
    node_workers.reserve((size_t)numa_nodes);
    int remaining_threads = total_threads;
    for (int node = 0; node < numa_nodes; ++node) {
        const int nodes_left = numa_nodes - node;
        const int node_threads = std::max(1, remaining_threads / std::max(1, nodes_left));
        remaining_threads -= node_threads;
        node_workers.emplace_back([&, node, node_threads]() {
            const auto& clusters = clusters_by_node[(size_t)node];
            if (clusters.empty()) return;
            std::atomic<int> next_cluster{0};
            std::vector<std::thread> workers;
            workers.reserve((size_t)node_threads);
            for (int t = 0; t < node_threads; ++t) {
                workers.emplace_back([&, t]() {
                    if (numa_first_touch) paraselect::pin_thread_to_numa_core(node, t);
                    const int fd = ::open(raw_base_path.c_str(), O_RDONLY);
                    if (fd < 0) {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        failed.store(true, std::memory_order_relaxed);
                        error_message = "open failed during materialization: " + raw_base_path;
                        return;
                    }
                    Scratch scratch;
                    try {
                        while (!failed.load(std::memory_order_relaxed)) {
                            const int pos = next_cluster.fetch_add(1, std::memory_order_relaxed);
                            if (pos >= (int)clusters.size()) break;
                            const int c = clusters[(size_t)pos];
                            const idx_t begin = cluster_offsets[c];
                            const idx_t end = begin + (idx_t)std::max(0, cluster_counts[c]);
                            for (idx_t row = begin; row < end; ++row) {
                                const idx_t src_id = reordered_indices[row];
                                T* dst = out + (size_t)row * (size_t)vector_dim;
                                reader(fd, layout, src_id, dst, scratch);
                            }
                        }
                    } catch (const std::exception& e) {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        failed.store(true, std::memory_order_relaxed);
                        error_message = e.what();
                    }
                    ::close(fd);
                });
            }
            for (auto& worker : workers) worker.join();
        });
    }
    for (auto& worker : node_workers) worker.join();
    if (failed.load(std::memory_order_relaxed)) {
        throw std::runtime_error(error_message.empty() ? "reordered materialization failed" : error_message);
    }

    ReorderedMaterializeResult<T> result;
    result.owner = out_handle.owner;
    result.data = static_cast<T*>(out_handle.detach());
    return result;
}

} // namespace detail

inline ReorderedFp32MaterializeResult materialize_reordered_fp32_sequential(
    const DatasetSource& source,
    const idx_t* reordered_indices,
    int n_total_vectors,
    int vector_dim,
    const int* row_order = nullptr) {
    return detail::materialize_reordered_sequential_impl<float>(
        source, reordered_indices, n_total_vectors, vector_dim, VectorDType::F32, row_order);
}

inline ReorderedU8MaterializeResult materialize_reordered_u8_sequential(
    const DatasetSource& source,
    const idx_t* reordered_indices,
    int n_total_vectors,
    int vector_dim,
    const int* row_order = nullptr) {
    return detail::materialize_reordered_sequential_impl<uint8_t>(
        source, reordered_indices, n_total_vectors, vector_dim, VectorDType::U8, row_order);
}

inline ReorderedFp32MaterializeResult materialize_reordered_fp32(
    const DatasetSource& source,
    const idx_t* reordered_indices,
    const idx_t* cluster_offsets,
    const int* cluster_counts,
    int n_total_vectors,
    int vector_dim,
    int n_clusters,
    bool numa_first_touch)
{
    return detail::materialize_reordered_from_file_impl<float, std::vector<uint8_t>>(
        source.base_path(),
        reordered_indices,
        cluster_offsets,
        cluster_counts,
        n_total_vectors,
        vector_dim,
        n_clusters,
        numa_first_touch,
        "fp32",
        detail::read_one_vector_as_float);
}

inline ReorderedU8MaterializeResult materialize_reordered_u8(
    const DatasetSource& source,
    const idx_t* reordered_indices,
    const idx_t* cluster_offsets,
    const int* cluster_counts,
    int n_total_vectors,
    int vector_dim,
    int n_clusters,
    bool numa_first_touch)
{
    return detail::materialize_reordered_from_file_impl<uint8_t, std::vector<float>>(
        source.base_path(),
        reordered_indices,
        cluster_offsets,
        cluster_counts,
        n_total_vectors,
        vector_dim,
        n_clusters,
        numa_first_touch,
        "u8",
        detail::read_one_vector_as_u8);
}

inline ReorderedAosoaMaterializeResult materialize_reordered_aosoa(
    const DatasetSource& source,
    const idx_t* reordered_indices,
    const idx_t* cluster_offsets,
    const int* cluster_counts,
    int n_total_vectors,
    int vector_dim,
    int n_clusters,
    int tile,
    bool numa_first_touch)
{
    const std::string& raw_base_path = source.base_path();
    if (raw_base_path.empty()) throw std::runtime_error("aosoa materialization requires raw_base_path");
    if (!reordered_indices || !cluster_offsets || !cluster_counts) {
        throw std::runtime_error("aosoa materialization requires assignment metadata");
    }
    if (n_total_vectors <= 0 || vector_dim <= 0 || n_clusters <= 0 || tile <= 0) {
        throw std::runtime_error("invalid aosoa materialization shape");
    }
    const detail::MaterializeFileLayout layout = detail::resolve_materialize_layout(raw_base_path);
    if (layout.dim != vector_dim) throw std::runtime_error("aosoa materialization dim mismatch");
    if ((size_t)n_total_vectors > layout.n) {
        throw std::runtime_error("aosoa materialization vector count exceeds source file");
    }

    using Owner = paraselect::ownership::BufferOwner;
    using Handle = paraselect::ownership::BufferHandle;
    Handle offsets_buf(memalign(64, (size_t)(n_clusters + 1) * sizeof(long long)),
                       (size_t)(n_clusters + 1) * sizeof(long long), Owner::StdMalloc);
    if (!offsets_buf.data) throw std::bad_alloc();
    const long long total = paraselect::cpu_fine::plan_aosoa_layout(
        cluster_counts, n_clusters, vector_dim, tile, static_cast<long long*>(offsets_buf.data));
    Handle data_buf(memalign(64, (size_t)total * sizeof(float)),
                    (size_t)total * sizeof(float), Owner::StdMalloc);
    if (!data_buf.data) throw std::bad_alloc();
    float* out = static_cast<float*>(data_buf.data);
    std::memset(out, 0, (size_t)total * sizeof(float));
    const long long* aosoa_offsets = static_cast<const long long*>(offsets_buf.data);

    const int total_threads = std::max(1, std::min(paraselect::default_worker_thread_count(), n_clusters));
    const int hw_nodes = std::max(1, paraselect::get_numa_node_count());
    const int numa_nodes = numa_first_touch ? std::max(1, std::min(hw_nodes, total_threads)) : 1;
    const std::vector<int> owners = paraselect::numa::build_cluster_owner_by_counts(
        cluster_counts, n_clusters, numa_nodes);
    std::vector<std::vector<int>> clusters_by_node((size_t)numa_nodes);
    for (int c = 0; c < n_clusters; ++c) {
        const int node = std::max(0, std::min(owners[(size_t)c], numa_nodes - 1));
        clusters_by_node[(size_t)node].push_back(c);
    }

    std::atomic<bool> failed{false};
    std::string error_message;
    std::mutex error_mutex;
    std::vector<std::thread> node_workers;
    int remaining_threads = total_threads;
    for (int node = 0; node < numa_nodes; ++node) {
        const int nodes_left = numa_nodes - node;
        const int node_threads = std::max(1, remaining_threads / std::max(1, nodes_left));
        remaining_threads -= node_threads;
        node_workers.emplace_back([&, node, node_threads]() {
            const auto& clusters = clusters_by_node[(size_t)node];
            if (clusters.empty()) return;
            std::atomic<int> next_cluster{0};
            std::vector<std::thread> workers;
            for (int t = 0; t < node_threads; ++t) {
                workers.emplace_back([&, t]() {
                    if (numa_first_touch) paraselect::pin_thread_to_numa_core(node, t);
                    const int fd = ::open(raw_base_path.c_str(), O_RDONLY);
                    if (fd < 0) {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        failed.store(true, std::memory_order_relaxed);
                        error_message = "open failed during aosoa materialization: " + raw_base_path;
                        return;
                    }
                    std::vector<uint8_t> scratch;
                    std::vector<float> vec((size_t)vector_dim);
                    try {
                        while (!failed.load(std::memory_order_relaxed)) {
                            const int pos = next_cluster.fetch_add(1, std::memory_order_relaxed);
                            if (pos >= (int)clusters.size()) break;
                            const int c = clusters[(size_t)pos];
                            const int count = std::max(0, cluster_counts[c]);
                            const idx_t row_base = cluster_offsets[c];
                            float* dst_cluster = out + aosoa_offsets[c];
                            const int groups = (count + tile - 1) / tile;
                            for (int g = 0; g < groups; ++g) {
                                const int local_begin = g * tile;
                                const int local_end = std::min(local_begin + tile, count);
                                float* dst_group = dst_cluster + (long long)g * vector_dim * tile;
                                for (int lane_idx = local_begin; lane_idx < local_end; ++lane_idx) {
                                    const int lane = lane_idx - local_begin;
                                    const idx_t src_id = reordered_indices[row_base + lane_idx];
                                    detail::read_one_vector_as_float(fd, layout, src_id, vec.data(), scratch);
                                    for (int d = 0; d < vector_dim; ++d) {
                                        dst_group[(long long)d * tile + lane] = vec[(size_t)d];
                                    }
                                }
                            }
                        }
                    } catch (const std::exception& e) {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        failed.store(true, std::memory_order_relaxed);
                        error_message = e.what();
                    }
                    ::close(fd);
                });
            }
            for (auto& worker : workers) worker.join();
        });
    }
    for (auto& worker : node_workers) worker.join();
    if (failed.load(std::memory_order_relaxed)) {
        throw std::runtime_error(error_message.empty() ? "aosoa materialization failed" : error_message);
    }

    ReorderedAosoaMaterializeResult result;
    result.data_owner = data_buf.owner;
    result.offsets_owner = offsets_buf.owner;
    result.data = static_cast<float*>(data_buf.detach());
    result.offsets = static_cast<long long*>(offsets_buf.detach());
    result.tile = tile;
    return result;
}

inline ReorderedFp32MaterializeResult materialize_reordered_from_file_numa(
    const std::string& raw_base_path,
    const idx_t* reordered_indices,
    const idx_t* cluster_offsets,
    const int* cluster_counts,
    int n_total_vectors,
    int vector_dim,
    int n_clusters)
{
    return detail::materialize_reordered_from_file_impl<float, std::vector<uint8_t>>(
        raw_base_path, reordered_indices, cluster_offsets, cluster_counts,
        n_total_vectors, vector_dim, n_clusters, true, "fp32",
        detail::read_one_vector_as_float);
}

} // namespace dataset
} // namespace paraselect

#endif // PARASELECT_DATASET_ASSIGNMENT_MATERIALIZER_CUH
