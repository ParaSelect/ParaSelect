/**
 * @file external_reorder.cuh
 * @brief  -  Reordering
 *
 * ## :
 *
 * :
 * -  80GB
 * -  80GB  →  160GB → OOM
 * - malloc  Soft Page Fault
 *
 * :
 * 1. ****: 80GB , malloc
 * 2. ** Page Fault**:,
 * 3. ****:,
 * 4. ** IO**:,
 * 5. ****:memcpy ,
 *
 * ##
 *
 * ```
 *  raw_vectors.bin (80GB)
 *        │
 *        ▼ ()
 * ┌─────────────┐
 * │ IO Buffer   │  ◀──  ( 10 = 51MB)
 * │ ()      │
 * └─────────────┘
 *        │
 *        ▼ (memcpy )
 * ┌─────────────┐
 * │ memory_ptr  │  ◀──  80GB  (Warm Memory)
 * │ ()   │      ,
 * └─────────────┘
 * ```
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

namespace paraselect {
namespace reorder {

/**
 * @brief
 */
struct ExternalReorderConfig {
    // IO ()
    //  100,000  = 100000 * 128 * 4 = 51.2MB
    size_t chunk_vectors = 100000;

    //
    bool verbose = true;

    // ()
    size_t progress_interval = 1000000;

    // Scatter (0 = )
    size_t num_threads = 0;
};

/**
 * @brief
 */
struct ExternalReorderStats {
    //
    size_t total_vectors = 0;

    //
    size_t total_bytes = 0;

    // IO
    size_t io_reads = 0;

    // IO
    size_t io_bytes_read = 0;

    //
    size_t overwrite_ops = 0;

    // ()
    double elapsed_ms = 0.0;

    // (GB/s)
    double throughput_gbps = 0.0;
};

/**
 * @brief
 *
 * :
 * 1. ()
 * 2. (, IO)
 * 3. ( Page Fault)
 *
 * @param memory_ptr        [in/out] ,
 * @param n                 [in]
 * @param dim               [in]
 * @param cluster_ids       [in]  ID, n
 * @param offsets           [in] (), k
 * @param counts            [in/out] ,, k
 *                           0,
 * @param file_path         [in] ( float32 )
 * @param config            [in]
 * @param stats             [out] ()
 *
 * @return true ,false
 *
 * @note :
 * 1. **?**
 *    - ,
 *    -  10-20%,
 *
 * 2. **?**
 *    - (Warm Memory)
 *    -  Soft Page Fault, malloc+ 10-100
 *
 * 3. ** counts  in/out?**
 *    -
 *    - ()
 */
bool external_reorder_vectors(
    float* memory_ptr,
    size_t n,
    int dim,
    const int* cluster_ids,
    const long long* offsets,
    int* counts,
    const std::string& file_path,
    const ExternalReorderConfig& config = ExternalReorderConfig{},
    ExternalReorderStats* stats = nullptr
);

/**
 * @brief RAII
 *
 *
 */
class FileHandle {
public:
    explicit FileHandle(const std::string& path, const char* mode);
    ~FileHandle();

    //
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;

    //
    FileHandle(FileHandle&& other) noexcept;
    FileHandle& operator=(FileHandle&& other) noexcept;

    //
    FILE* get() const { return fp_; }

    //
    bool is_open() const { return fp_ != nullptr; }

    //
    void close();

private:
    FILE* fp_ = nullptr;
};

/**
 * @brief RAII
 *
 *  IO
 */
class IOBuffer {
public:
    IOBuffer(size_t n_vectors, int dim);
    ~IOBuffer();

    //
    IOBuffer(const IOBuffer&) = delete;
    IOBuffer& operator=(const IOBuffer&) = delete;

    //
    IOBuffer(IOBuffer&& other) noexcept;
    IOBuffer& operator=(IOBuffer&& other) noexcept;

    //
    float* data() { return data_; }
    const float* data() const { return data_; }

    // ()
    size_t capacity() const { return capacity_; }

    //
    size_t bytes() const { return bytes_; }

private:
    float* data_ = nullptr;
    size_t capacity_ = 0;
    size_t bytes_ = 0;
};

} // namespace reorder
} // namespace paraselect

//  C
extern "C" {

/**
 * @brief C :
 */
bool ivf_external_reorder_vectors(
    float* memory_ptr,
    size_t n,
    int dim,
    const int* cluster_ids,
    const long long* offsets,
    int* counts,
    const char* file_path,
    size_t chunk_vectors,
    bool verbose
);

} // extern "C"
