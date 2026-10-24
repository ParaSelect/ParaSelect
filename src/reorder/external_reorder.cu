/**
 * @file external_reorder.cu
 * @brief (Radix Grouped + Persistent ThreadPool )
 */

#include "reorder/external_reorder.cuh"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

// POSIX IO / Memory
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef __linux__
#include <sys/mman.h>
#endif

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace paraselect {
namespace reorder {

// ============================================================================
//
// ============================================================================

inline void copy_floats(float* dst, const float* src, size_t dim) {
#if defined(__AVX2__) || defined(__AVX512F__)
    if ((reinterpret_cast<uintptr_t>(dst) % 32) == 0) {
        size_t i = 0;
        for (; i + 8 <= dim; i += 8) {
            __m256 v = _mm256_loadu_ps(src + i);
            _mm256_stream_ps(dst + i, v);
        }
        for (; i < dim; ++i) {
            dst[i] = src[i];
        }
        return;
    }
#endif
    std::memcpy(dst, src, dim * sizeof(float));
}

static inline uint32_t sortable_cluster_key(int cluster_id) {
    return static_cast<uint32_t>(cluster_id) ^ 0x80000000u;
}

// ============================================================================
// FileHandle
// ============================================================================

FileHandle::FileHandle(const std::string& path, const char* mode) {
    fp_ = std::fopen(path.c_str(), mode);
    if (!fp_) {
        throw std::runtime_error("Failed to open file: " + path + ", error: " + std::strerror(errno));
    }
}

FileHandle::~FileHandle() { close(); }

FileHandle::FileHandle(FileHandle&& other) noexcept : fp_(other.fp_) {
    other.fp_ = nullptr;
}

FileHandle& FileHandle::operator=(FileHandle&& other) noexcept {
    if (this != &other) {
        close();
        fp_ = other.fp_;
        other.fp_ = nullptr;
    }
    return *this;
}

void FileHandle::close() {
    if (fp_) {
        std::fclose(fp_);
        fp_ = nullptr;
    }
}

// ============================================================================
// IOBuffer
// ============================================================================

IOBuffer::IOBuffer(size_t n_vectors, int dim) : capacity_(n_vectors) {
    bytes_ = n_vectors * static_cast<size_t>(dim) * sizeof(float);
    if (posix_memalign(reinterpret_cast<void**>(&data_), 64, bytes_) != 0 || !data_) {
        throw std::bad_alloc();
    }
}

IOBuffer::~IOBuffer() {
    if (data_) {
        std::free(data_);
        data_ = nullptr;
    }
}

IOBuffer::IOBuffer(IOBuffer&& other) noexcept
    : data_(other.data_), capacity_(other.capacity_), bytes_(other.bytes_) {
    other.data_ = nullptr;
    other.capacity_ = 0;
    other.bytes_ = 0;
}

IOBuffer& IOBuffer::operator=(IOBuffer&& other) noexcept {
    if (this != &other) {
        if (data_) std::free(data_);
        data_ = other.data_;
        capacity_ = other.capacity_;
        bytes_ = other.bytes_;
        other.data_ = nullptr;
        other.capacity_ = 0;
        other.bytes_ = 0;
    }
    return *this;
}

// ============================================================================
//
// ============================================================================

struct ClusterRun {
    int cluster_id;
    size_t begin_idx;     //  sorted_indices
    size_t end_idx;       //  sorted_indices ()
    long long dst_offset; // (:vector)
};

static inline void radix_sort_indices_by_cluster(
    const int* cluster_ids_base,
    size_t count,
    std::vector<uint32_t>& indices,
    std::vector<uint32_t>& temp,
    std::vector<uint32_t>& hist0,
    std::vector<uint32_t>& hist1) {

    if (count == 0) return;

    std::fill(hist0.begin(), hist0.end(), 0u);
    std::fill(hist1.begin(), hist1.end(), 0u);

    for (size_t i = 0; i < count; ++i) {
        indices[i] = static_cast<uint32_t>(i);
        const uint32_t key = sortable_cluster_key(cluster_ids_base[i]);
        ++hist0[key & 0xFFFFu];
        ++hist1[(key >> 16) & 0xFFFFu];
    }

    uint32_t sum = 0;
    for (size_t i = 0; i < hist0.size(); ++i) {
        const uint32_t c = hist0[i];
        hist0[i] = sum;
        sum += c;
    }

    for (size_t i = 0; i < count; ++i) {
        const uint32_t idx = indices[i];
        const uint32_t key = sortable_cluster_key(cluster_ids_base[idx]);
        temp[hist0[key & 0xFFFFu]++] = idx;
    }

    sum = 0;
    for (size_t i = 0; i < hist1.size(); ++i) {
        const uint32_t c = hist1[i];
        hist1[i] = sum;
        sum += c;
    }

    for (size_t i = 0; i < count; ++i) {
        const uint32_t idx = temp[i];
        const uint32_t key = sortable_cluster_key(cluster_ids_base[idx]);
        indices[hist1[(key >> 16) & 0xFFFFu]++] = idx;
    }
}

static inline void build_cluster_runs(
    const int* cluster_ids_base,
    const std::vector<uint32_t>& sorted_indices,
    size_t count,
    const long long* offsets,
    int* counts,
    std::vector<ClusterRun>& runs) {

    runs.clear();
    if (count == 0) return;

    size_t i = 0;
    while (i < count) {
        const uint32_t first_idx = sorted_indices[i];
        const int cluster_id = cluster_ids_base[first_idx];

        size_t j = i + 1;
        while (j < count && cluster_ids_base[sorted_indices[j]] == cluster_id) {
            ++j;
        }

        const int run_len = static_cast<int>(j - i);
        const long long dst_offset = offsets[cluster_id] + counts[cluster_id];
        counts[cluster_id] += run_len;

        ClusterRun run;
        run.cluster_id = cluster_id;
        run.begin_idx = i;
        run.end_idx = j;
        run.dst_offset = dst_offset;
        runs.push_back(run);

        i = j;
    }
}

static inline std::vector<std::pair<size_t, size_t>> partition_runs_for_threads(
    const std::vector<ClusterRun>& runs,
    size_t num_threads) {

    std::vector<std::pair<size_t, size_t>> partitions;
    if (runs.empty() || num_threads == 0) return partitions;

    num_threads = std::min(num_threads, runs.size());
    partitions.reserve(num_threads);

    size_t total_vectors = 0;
    for (const auto& r : runs) {
        total_vectors += (r.end_idx - r.begin_idx);
    }

    const size_t target_per_thread = std::max<size_t>(1, (total_vectors + num_threads - 1) / num_threads);

    size_t run_begin = 0;
    size_t acc = 0;

    for (size_t i = 0; i < runs.size(); ++i) {
        acc += (runs[i].end_idx - runs[i].begin_idx);
        if (acc >= target_per_thread && partitions.size() + 1 < num_threads) {
            partitions.emplace_back(run_begin, i + 1);
            run_begin = i + 1;
            acc = 0;
        }
    }

    if (run_begin < runs.size()) {
        partitions.emplace_back(run_begin, runs.size());
    }

    return partitions;
}

// ============================================================================
//  Scatter
// ============================================================================

class ScatterThreadPool {
public:
    explicit ScatterThreadPool(size_t num_threads)
        : stop_(false),
          active_job_id_(0),
          completed_workers_(0),
          active_workers_(0),
          dim_(0),
          memory_ptr_(nullptr),
          src_base_(nullptr),
          sorted_indices_(nullptr),
          runs_(nullptr) {
        if (num_threads == 0) num_threads = 1;
        workers_.reserve(num_threads);
        ranges_.resize(num_threads, std::make_pair(size_t(0), size_t(0)));

        for (size_t tid = 0; tid < num_threads; ++tid) {
            workers_.emplace_back(&ScatterThreadPool::worker_loop, this, tid);
        }
    }

    ~ScatterThreadPool() {
        {
            std::unique_lock<std::mutex> lk(mu_);
            stop_ = true;
            cv_start_.notify_all();
        }
        for (auto& th : workers_) {
            if (th.joinable()) th.join();
        }
    }

    size_t size() const { return workers_.size(); }

    void run(
        float* memory_ptr,
        int dim,
        const float* src_base,
        const std::vector<uint32_t>& sorted_indices,
        const std::vector<ClusterRun>& runs,
        const std::vector<std::pair<size_t, size_t>>& partitions) {

        if (partitions.empty()) return;

        {
            std::unique_lock<std::mutex> lk(mu_);
            memory_ptr_ = memory_ptr;
            dim_ = dim;
            src_base_ = src_base;
            sorted_indices_ = &sorted_indices;
            runs_ = &runs;

            active_workers_ = partitions.size();
            completed_workers_ = 0;

            for (size_t i = 0; i < workers_.size(); ++i) {
                if (i < partitions.size()) {
                    ranges_[i] = partitions[i];
                } else {
                    ranges_[i] = std::make_pair(size_t(0), size_t(0));
                }
            }

            ++active_job_id_;
            cv_start_.notify_all();
        }

        std::unique_lock<std::mutex> lk(mu_);
        cv_done_.wait(lk, [&]() { return completed_workers_ >= active_workers_; });
    }

private:
    void worker_loop(size_t tid) {
        uint64_t seen_job_id = 0;

        while (true) {
            std::pair<size_t, size_t> my_range;
            float* local_memory_ptr = nullptr;
            const float* local_src_base = nullptr;
            const std::vector<uint32_t>* local_sorted_indices = nullptr;
            const std::vector<ClusterRun>* local_runs = nullptr;
            int local_dim = 0;

            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_start_.wait(lk, [&]() { return stop_ || active_job_id_ != seen_job_id; });

                if (stop_) return;

                seen_job_id = active_job_id_;

                if (tid >= active_workers_) {
                    continue;
                }

                my_range = ranges_[tid];
                local_memory_ptr = memory_ptr_;
                local_src_base = src_base_;
                local_sorted_indices = sorted_indices_;
                local_runs = runs_;
                local_dim = dim_;
            }

            for (size_t r = my_range.first; r < my_range.second; ++r) {
                const ClusterRun& run = (*local_runs)[r];
                float* dst = local_memory_ptr + run.dst_offset * static_cast<long long>(local_dim);

                for (size_t i = run.begin_idx; i < run.end_idx; ++i) {
                    const uint32_t original_idx = (*local_sorted_indices)[i];
                    const float* src = local_src_base + static_cast<size_t>(original_idx) * local_dim;
                    copy_floats(dst, src, static_cast<size_t>(local_dim));
                    dst += local_dim;
                }
            }

#if defined(__AVX2__) || defined(__AVX512F__)
            _mm_sfence();
#endif

            {
                std::unique_lock<std::mutex> lk(mu_);
                ++completed_workers_;
                if (completed_workers_ >= active_workers_) {
                    cv_done_.notify_one();
                }
            }
        }
    }

private:
    std::vector<std::thread> workers_;
    std::vector<std::pair<size_t, size_t>> ranges_;

    std::mutex mu_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;

    bool stop_;
    uint64_t active_job_id_;
    size_t completed_workers_;
    size_t active_workers_;

    int dim_;
    float* memory_ptr_;
    const float* src_base_;
    const std::vector<uint32_t>* sorted_indices_;
    const std::vector<ClusterRun>* runs_;
};

// ============================================================================
// Raw Binary
// ============================================================================

bool external_reorder_vectors(
    float* memory_ptr,
    size_t n,
    int dim,
    const int* cluster_ids,
    const long long* offsets,
    int* counts,
    const std::string& file_path,
    const ExternalReorderConfig& config,
    ExternalReorderStats* stats) {

    if (!memory_ptr || n == 0 || dim <= 0 || !cluster_ids || !offsets || !counts || file_path.empty()) {
        std::fprintf(stderr, "[external_reorder] ERROR: Invalid parameters\n");
        return false;
    }

    ExternalReorderStats local_stats;
    local_stats.total_vectors = n;
    local_stats.total_bytes = n * static_cast<size_t>(dim) * sizeof(float);

    auto start_time = std::chrono::high_resolution_clock::now();

    double total_read_time_ms = 0.0;
    double total_group_time_ms = 0.0;
    double total_run_time_ms = 0.0;
    double total_scatter_time_ms = 0.0;
    double total_sync_time_ms = 0.0;
    size_t chunk_count = 0;

#ifdef __linux__
    madvise(memory_ptr, local_stats.total_bytes, MADV_HUGEPAGE);
#endif

    try {
        int fd = open(file_path.c_str(), O_RDONLY);
        if (fd < 0) {
            std::fprintf(stderr, "[external_reorder] ERROR: Cannot open file: %s\n", std::strerror(errno));
            return false;
        }

#ifdef __linux__
        posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif

        IOBuffer bufferA(config.chunk_vectors, dim);
        IOBuffer bufferB(config.chunk_vectors, dim);
        IOBuffer* current_buffer = &bufferA;
        IOBuffer* next_buffer = &bufferB;

        const size_t bytes_per_vector = static_cast<size_t>(dim) * sizeof(float);
        size_t vectors_processed = 0;

        std::vector<uint32_t> sorted_indices(config.chunk_vectors);
        std::vector<uint32_t> temp_indices(config.chunk_vectors);
        std::vector<uint32_t> hist0(1u << 16, 0u);
        std::vector<uint32_t> hist1(1u << 16, 0u);
        std::vector<ClusterRun> runs;
        runs.reserve(std::min<size_t>(config.chunk_vectors, 1u << 16));

        size_t num_threads = config.num_threads > 0 ? config.num_threads : std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;

        ScatterThreadPool scatter_pool(num_threads);

        auto async_read_func = [](int file_fd, float* buf_data, size_t to_read_vectors, size_t bytes_per_vec) -> size_t {
            const size_t bytes_to_read = to_read_vectors * bytes_per_vec;
            size_t total_read = 0;
            char* ptr = reinterpret_cast<char*>(buf_data);

            while (total_read < bytes_to_read) {
                ssize_t bytes = read(file_fd, ptr + total_read, bytes_to_read - total_read);
                if (bytes < 0) {
                    if (errno == EINTR) continue;
                    break;
                }
                if (bytes == 0) break;
                total_read += static_cast<size_t>(bytes);
            }
            return total_read / bytes_per_vec;
        };

        const size_t first_read = std::min(bufferA.capacity(), n);
        size_t valid_vectors_in_current = async_read_func(fd, current_buffer->data(), first_read, bytes_per_vector);
        local_stats.io_reads++;
        local_stats.io_bytes_read += valid_vectors_in_current * bytes_per_vector;

        std::future<size_t> future_read;

        if (config.verbose) {
            std::fprintf(stderr,
                "[external_reorder] Radix + persistent-thread-pool engine started...\n"
                "  Threads: %zu, Dim: %d\n"
                "  Chunk Size: %zu vectors (%.2f MB)\n",
                num_threads, dim, bufferA.capacity(), bufferA.bytes() / (1024.0 * 1024.0));
        }

        while (vectors_processed < n && valid_vectors_in_current > 0) {
            const size_t next_read = std::min(
                next_buffer->capacity(),
                n - vectors_processed - valid_vectors_in_current);

            auto read_launch_start = std::chrono::high_resolution_clock::now();
            if (next_read > 0) {
                future_read = std::async(
                    std::launch::async,
                    async_read_func,
                    fd,
                    next_buffer->data(),
                    next_read,
                    bytes_per_vector);
            }
            auto read_launch_end = std::chrono::high_resolution_clock::now();
            total_read_time_ms += std::chrono::duration<double, std::milli>(read_launch_end - read_launch_start).count();

            const int* cluster_ids_base = cluster_ids + vectors_processed;

            auto group_start = std::chrono::high_resolution_clock::now();
            radix_sort_indices_by_cluster(
                cluster_ids_base,
                valid_vectors_in_current,
                sorted_indices,
                temp_indices,
                hist0,
                hist1);
            auto group_end = std::chrono::high_resolution_clock::now();
            total_group_time_ms += std::chrono::duration<double, std::milli>(group_end - group_start).count();

            auto run_start = std::chrono::high_resolution_clock::now();
            build_cluster_runs(
                cluster_ids_base,
                sorted_indices,
                valid_vectors_in_current,
                offsets,
                counts,
                runs);
            auto run_end = std::chrono::high_resolution_clock::now();
            total_run_time_ms += std::chrono::duration<double, std::milli>(run_end - run_start).count();

            auto scatter_start = std::chrono::high_resolution_clock::now();
            const auto partitions = partition_runs_for_threads(runs, scatter_pool.size());
            scatter_pool.run(
                memory_ptr,
                dim,
                current_buffer->data(),
                sorted_indices,
                runs,
                partitions);
            auto scatter_end = std::chrono::high_resolution_clock::now();
            total_scatter_time_ms += std::chrono::duration<double, std::milli>(scatter_end - scatter_start).count();

            local_stats.overwrite_ops += valid_vectors_in_current;
            vectors_processed += valid_vectors_in_current;
            ++chunk_count;

            auto sync_start = std::chrono::high_resolution_clock::now();
            if (next_read > 0) {
                valid_vectors_in_current = future_read.get();
                local_stats.io_reads++;
                local_stats.io_bytes_read += valid_vectors_in_current * bytes_per_vector;
                std::swap(current_buffer, next_buffer);
            } else {
                valid_vectors_in_current = 0;
            }
            auto sync_end = std::chrono::high_resolution_clock::now();
            total_sync_time_ms += std::chrono::duration<double, std::milli>(sync_end - sync_start).count();

            if (config.verbose && (vectors_processed % config.progress_interval == 0 || vectors_processed >= n)) {
                std::fprintf(stderr,
                    "[external_reorder] Progress: %.1f%% (%zu / %zu vectors)\n",
                    100.0 * vectors_processed / n,
                    vectors_processed,
                    n);
            }
        }

        close(fd);

        auto end_time = std::chrono::high_resolution_clock::now();
        local_stats.elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        if (local_stats.elapsed_ms > 0.0) {
            local_stats.throughput_gbps =
                local_stats.total_bytes / (1024.0 * 1024.0 * 1024.0) / (local_stats.elapsed_ms / 1000.0);
        }

        if (config.verbose) {
            std::fprintf(stderr,
                "[external_reorder] Completed!\n"
                "  Vectors: %zu, IO reads: %zu, Chunks: %zu\n"
                "  Elapsed: %.2f ms (%.2f s)\n"
                "  Speed: %.2f GB/s\n"
                "\n[Timing Breakdown]:\n"
                "  Read (async launch): %.2f ms (%.1f%%)\n"
                "  Radix group:         %.2f ms (%.1f%%)\n"
                "  Build runs/counts:   %.2f ms (%.1f%%)\n"
                "  Scatter write:       %.2f ms (%.1f%%)\n"
                "  Sync/IO wait:        %.2f ms (%.1f%%)\n",
                local_stats.total_vectors,
                local_stats.io_reads,
                chunk_count,
                local_stats.elapsed_ms,
                local_stats.elapsed_ms / 1000.0,
                local_stats.throughput_gbps,
                total_read_time_ms, 100.0 * total_read_time_ms / local_stats.elapsed_ms,
                total_group_time_ms, 100.0 * total_group_time_ms / local_stats.elapsed_ms,
                total_run_time_ms, 100.0 * total_run_time_ms / local_stats.elapsed_ms,
                total_scatter_time_ms, 100.0 * total_scatter_time_ms / local_stats.elapsed_ms,
                total_sync_time_ms, 100.0 * total_sync_time_ms / local_stats.elapsed_ms);
        }

        if (stats) *stats = local_stats;
        return true;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "[external_reorder] ERROR: %s\n", e.what());
        return false;
    }
}

} // namespace reorder
} // namespace paraselect

extern "C" {

bool ivf_external_reorder_vectors(
    float* memory_ptr,
    size_t n,
    int dim,
    const int* cluster_ids,
    const long long* offsets,
    int* counts,
    const char* file_path,
    size_t chunk_vectors,
    bool verbose) {

    paraselect::reorder::ExternalReorderConfig config;
    config.chunk_vectors = chunk_vectors;
    config.verbose = verbose;

    return paraselect::reorder::external_reorder_vectors(
        memory_ptr,
        n,
        dim,
        cluster_ids,
        offsets,
        counts,
        std::string(file_path),
        config,
        nullptr);
}

} // extern "C"
