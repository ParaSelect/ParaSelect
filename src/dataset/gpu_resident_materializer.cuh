#ifndef PARASELECT_DATASET_GPU_RESIDENT_MATERIALIZER_CUH
#define PARASELECT_DATASET_GPU_RESIDENT_MATERIALIZER_CUH

#include "dataset/assignment_materializer.cuh"
#include "dataset/gpu_resident_interleaved_storage.cuh"
#include "dataset/gpu_resident_storage_estimate.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace paraselect::dataset {

struct GpuResidentMaterializePlan {
    int n_padded = 0;
    int storage_dim = 0;
    bool use_u8 = false;
    bool packed_u8 = false;
    // Experimental: group 4 d4 words per lane contiguously so the scan can use
    // one 16B (uint4) load per group instead of four 4B loads. Same byte size
    // as packed_u8; only the permutation inside each 32-vector tile differs.
    bool vec4_u8 = false;
    GpuResidentStorageEstimate estimate;
};

inline GpuResidentMaterializePlan plan_gpu_resident_interleaved(
    const int* cluster_counts,
    int n_vectors,
    int dim,
    int nlist,
    bool use_u8,
    bool packed_u8,
    bool vec4_u8 = false) {
    if (!cluster_counts || n_vectors <= 0 || dim <= 0 || nlist <= 0) {
        throw std::invalid_argument("invalid GPU resident materialization shape");
    }
    if (packed_u8 && !use_u8) {
        throw std::invalid_argument("packed GPU resident layout requires u8 storage");
    }
    if (vec4_u8 && !packed_u8) {
        throw std::invalid_argument("vec4 GPU resident layout requires packed u8 storage");
    }
    if (vec4_u8 && (dim % 16) != 0) {
        throw std::invalid_argument("vec4 GPU resident layout requires dim % 16 == 0");
    }

    GpuResidentMaterializePlan plan;
    plan.use_u8 = use_u8;
    plan.packed_u8 = packed_u8;
    plan.vec4_u8 = vec4_u8;
    plan.storage_dim = packed_u8 ? ((dim + 3) / 4) * 4 : dim;
    int64_t total = 0;
    int64_t counted = 0;
    for (int cid = 0; cid < nlist; ++cid) {
        const int count = cluster_counts[cid];
        if (count < 0) throw std::invalid_argument("negative IVF cluster count");
        counted += count;
        total += ((int64_t)count + 31) / 32 * 32;
        if (total > INT_MAX) {
            throw std::overflow_error("GPU resident padded row count exceeds INT_MAX");
        }
    }
    if (counted != n_vectors) {
        throw std::invalid_argument("IVF cluster counts do not match vector count");
    }
    plan.n_padded = (int)total;
    plan.estimate = estimate_gpu_resident_storage(
        (size_t)plan.n_padded,
        (size_t)n_vectors,
        plan.storage_dim,
        use_u8 ? sizeof(uint8_t) : sizeof(float));
    return plan;
}

namespace detail {

struct ResidentBlockTask {
    idx_t compact_begin = 0;
    int valid_rows = 0;
};

inline void resident_cuda_check(cudaError_t err, const char* label) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(err));
    }
}

template <typename T>
inline void scatter_interleaved_row(
    const T* row,
    T* block,
    int lane,
    int dim,
    bool packed_u8,
    bool vec4_u8 = false) {
    if (packed_u8) {
        for (int d = 0; d < dim; ++d) {
            const int d4 = d / 4;
            const int component = d % 4;
            if (vec4_u8) {
                const int group = d4 / 4;
                const int within = d4 % 4;
                block[((size_t)group * 32 + (size_t)lane) * 16 +
                      (size_t)within * 4 + (size_t)component] = row[d];
            } else {
                block[((size_t)d4 * 32 + (size_t)lane) * 4 + (size_t)component] = row[d];
            }
        }
        return;
    }
    for (int d = 0; d < dim; ++d) {
        block[(size_t)d * 32 + (size_t)lane] = row[d];
    }
}

} // namespace detail

inline GpuResidentInterleavedStorage materialize_gpu_resident_interleaved(
    const DatasetSource& source,
    const idx_t* reordered_indices,
    const idx_t* cluster_offsets,
    const int* cluster_counts,
    int n_vectors,
    int dim,
    int nlist,
    const GpuResidentMaterializePlan& plan,
    size_t staging_bytes = 256ULL * 1024ULL * 1024ULL) {
    if (!reordered_indices || !cluster_offsets || !cluster_counts) {
        throw std::invalid_argument("GPU resident materialization requires IVF metadata");
    }
    if (plan.n_padded < n_vectors || plan.n_padded % 32 != 0 ||
        plan.storage_dim < dim) {
        throw std::invalid_argument("invalid GPU resident materialization plan");
    }

    const auto layout = detail::resolve_materialize_layout(source.base_path());
    const bool signed_byte_source = source.base_has_signed_byte_encoding();
    if (layout.dim != dim || (size_t)n_vectors > layout.n) {
        throw std::invalid_argument("GPU resident source shape mismatch");
    }

    const size_t element_bytes = plan.use_u8 ? sizeof(uint8_t) : sizeof(float);
    const size_t vector_block_bytes =
        (size_t)32 * (size_t)plan.storage_dim * element_bytes;
    const size_t norm_block_bytes = (size_t)32 * sizeof(float);
    const size_t bytes_per_block = vector_block_bytes + norm_block_bytes;
    const size_t total_blocks = (size_t)plan.n_padded / 32;
    const size_t blocks_per_chunk = std::min(
        total_blocks,
        std::max<size_t>(1, staging_bytes / bytes_per_block));
    const size_t chunk_vector_bytes = blocks_per_chunk * vector_block_bytes;
    const size_t chunk_norm_bytes = blocks_per_chunk * norm_block_bytes;

    void* h_vectors = nullptr;
    float* h_norms = nullptr;
    int fd = -1;
    GpuResidentInterleavedStorage storage;
    storage.n_vectors = n_vectors;
    storage.n_padded = plan.n_padded;
    storage.dim = dim;
    storage.nlist = nlist;
    storage.bytes = plan.estimate.total_bytes();

    try {
        if (plan.use_u8) {
            detail::resident_cuda_check(
                cudaMalloc(&storage.d_vectors_u8, plan.estimate.vector_bytes),
                "cudaMalloc resident u8 vectors failed");
        } else {
            detail::resident_cuda_check(
                cudaMalloc(&storage.d_vectors, plan.estimate.vector_bytes),
                "cudaMalloc resident fp32 vectors failed");
        }
        detail::resident_cuda_check(
            cudaMalloc(&storage.d_norms, plan.estimate.norm_bytes),
            "cudaMalloc resident norms failed");
        detail::resident_cuda_check(
            cudaMalloc(&storage.d_reordered_indices, plan.estimate.id_bytes),
            "cudaMalloc resident reordered ids failed");
        detail::resident_cuda_check(
            cudaHostAlloc(&h_vectors, chunk_vector_bytes, cudaHostAllocDefault),
            "cudaHostAlloc resident vector staging failed");
        detail::resident_cuda_check(
            cudaHostAlloc(&h_norms, chunk_norm_bytes, cudaHostAllocDefault),
            "cudaHostAlloc resident norm staging failed");

        fd = ::open(source.base_path().c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("open failed during GPU resident materialization: " +
                                     source.base_path());
        }

        std::vector<detail::ResidentBlockTask> tasks;
        tasks.reserve(blocks_per_chunk);
        size_t global_block = 0;
        int cid = 0;
        int cluster_block = 0;
        while (cid < nlist) {
            tasks.clear();
            while (cid < nlist && tasks.size() < blocks_per_chunk) {
                const int count = cluster_counts[cid];
                const int blocks = (count + 31) / 32;
                while (cluster_block < blocks && tasks.size() < blocks_per_chunk) {
                    const int local_row = cluster_block * 32;
                    tasks.push_back({
                        cluster_offsets[cid] + local_row,
                        std::min(32, count - local_row),
                    });
                    ++cluster_block;
                }
                if (cluster_block == blocks) {
                    ++cid;
                    cluster_block = 0;
                }
            }
            if (tasks.empty()) continue;

            const size_t vector_bytes = tasks.size() * vector_block_bytes;
            const size_t norm_bytes = tasks.size() * norm_block_bytes;
            std::memset(h_vectors, 0, vector_bytes);
            std::memset(h_norms, 0, norm_bytes);

            std::atomic<size_t> next_task{0};
            std::atomic<bool> failed{false};
            std::mutex error_mutex;
            std::string error_message;
            const int thread_count = std::max(
                1,
                std::min<int>(paraselect::default_worker_thread_count(), (int)tasks.size()));
            std::vector<std::thread> workers;
            workers.reserve((size_t)thread_count);
            for (int thread = 0; thread < thread_count; ++thread) {
                workers.emplace_back([&]() {
                    std::vector<float> fp32_row((size_t)dim);
                    std::vector<uint8_t> u8_row((size_t)dim);
                    std::vector<float> fp32_scratch;
                    std::vector<uint8_t> u8_scratch;
                    try {
                        while (!failed.load(std::memory_order_relaxed)) {
                            const size_t task_id = next_task.fetch_add(1, std::memory_order_relaxed);
                            if (task_id >= tasks.size()) break;
                            const auto task = tasks[task_id];
                            auto* vector_block = static_cast<uint8_t*>(h_vectors) +
                                                 task_id * vector_block_bytes;
                            float* norm_block = h_norms + task_id * 32;
                            for (int lane = 0; lane < task.valid_rows; ++lane) {
                                const idx_t compact_row = task.compact_begin + lane;
                                const idx_t source_id = reordered_indices[compact_row];
                                double squared_norm = 0.0;
                                if (plan.use_u8) {
                                    detail::read_one_vector_as_u8(
                                        fd, layout, source_id, u8_row.data(), fp32_scratch);
                                    if (signed_byte_source) {
                                        for (uint8_t& value : u8_row) value ^= 0x80;
                                    }
                                    detail::scatter_interleaved_row(
                                        u8_row.data(), vector_block, lane, dim, plan.packed_u8,
                                        plan.vec4_u8);
                                    for (int d = 0; d < dim; ++d) {
                                        const double value = u8_row[(size_t)d];
                                        squared_norm += value * value;
                                    }
                                } else {
                                    detail::read_one_vector_as_float(
                                        fd, layout, source_id, fp32_row.data(), u8_scratch);
                                    if (signed_byte_source) {
                                        for (float& value : fp32_row) {
                                            value = static_cast<float>(
                                                static_cast<uint8_t>(value) ^ 0x80);
                                        }
                                    }
                                    detail::scatter_interleaved_row(
                                        fp32_row.data(), reinterpret_cast<float*>(vector_block),
                                        lane, dim, false);
                                    for (int d = 0; d < dim; ++d) {
                                        const double value = fp32_row[(size_t)d];
                                        squared_norm += value * value;
                                    }
                                }
                                norm_block[lane] = (float)std::sqrt(squared_norm);
                            }
                        }
                    } catch (const std::exception& e) {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (!failed.exchange(true, std::memory_order_relaxed)) {
                            error_message = e.what();
                        }
                    }
                });
            }
            for (auto& worker : workers) worker.join();
            if (failed.load(std::memory_order_relaxed)) {
                throw std::runtime_error(
                    error_message.empty() ? "GPU resident staging failed" : error_message);
            }

            const size_t vector_offset = global_block * vector_block_bytes;
            const size_t norm_offset = global_block * norm_block_bytes;
            void* d_vectors = plan.use_u8
                ? static_cast<void*>(storage.d_vectors_u8 + vector_offset)
                : static_cast<void*>(reinterpret_cast<uint8_t*>(storage.d_vectors) + vector_offset);
            detail::resident_cuda_check(
                cudaMemcpy(d_vectors, h_vectors, vector_bytes, cudaMemcpyHostToDevice),
                "cudaMemcpy resident vector chunk failed");
            detail::resident_cuda_check(
                cudaMemcpy(reinterpret_cast<uint8_t*>(storage.d_norms) + norm_offset,
                           h_norms, norm_bytes, cudaMemcpyHostToDevice),
                "cudaMemcpy resident norm chunk failed");
            global_block += tasks.size();
        }

        if (global_block != total_blocks) {
            throw std::runtime_error("GPU resident block count mismatch");
        }
        detail::resident_cuda_check(
            cudaMemcpy(storage.d_reordered_indices, reordered_indices,
                       plan.estimate.id_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy resident reordered ids failed");
        ::close(fd);
        fd = -1;
        cudaFreeHost(h_vectors);
        h_vectors = nullptr;
        cudaFreeHost(h_norms);
        h_norms = nullptr;
        return storage;
    } catch (...) {
        if (fd >= 0) ::close(fd);
        if (h_vectors) cudaFreeHost(h_vectors);
        if (h_norms) cudaFreeHost(h_norms);
        storage.release();
        throw;
    }
}

} // namespace paraselect::dataset

#endif // PARASELECT_DATASET_GPU_RESIDENT_MATERIALIZER_CUH
