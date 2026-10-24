#ifndef CPU_ARRAY_UTILS_H
#define CPU_ARRAY_UTILS_H

#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <algorithm>
#include <type_traits>
#include <cstring>
#include <cstdlib>
#include <cstddef>
#if defined(__linux__)
#include <sys/mman.h>
#endif
#include "layout/ivf_interleaved_layout.cuh"

/**
 * CPU
 *
 *
 * Interleaved:init_array_interleaved_multithreaded  Host ( row-major  pack);
 * Host  vec L2  cpu/l2norm_cpu.h  compute_l2_norm_cpu_interleaved.
 */

constexpr size_t kCpuArrayHugePageAlignment = 2ull * 1024ull * 1024ull;
constexpr size_t kCpuArrayOsPageBytes = 4096ull;

inline bool cpu_array_env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return false;
    }
    return std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0 &&
           std::strcmp(value, "off") != 0 &&
           std::strcmp(value, "OFF") != 0;
}

/** num_threads<=0  hardware_concurrency( 1) */
inline int cpu_array_resolve_num_threads(int num_threads) {
    if (num_threads > 0)
        return num_threads;
    unsigned hc = std::thread::hardware_concurrency();
    return hc ? (int)hc : 1;
}

inline void* allocate_hugepage_aligned_bytes(
    size_t bytes,
    bool advise_hugepage = true,
    size_t alignment = kCpuArrayHugePageAlignment)
{
    if (bytes == 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    const int rc = posix_memalign(&ptr, alignment, bytes);
    if (rc != 0) {
        return nullptr;
    }
#if defined(__linux__)
    if (advise_hugepage) {
        (void)madvise(ptr, bytes, MADV_HUGEPAGE);
    }
#else
    (void)advise_hugepage;
#endif
    return ptr;
}

inline void free_hugepage_aligned_bytes(void* ptr) {
    std::free(ptr);
}

template <typename T>
inline T* allocate_hugepage_aligned_array(
    size_t count,
    bool advise_hugepage = true,
    size_t alignment = kCpuArrayHugePageAlignment)
{
    static_assert(std::is_trivially_copyable<T>::value,
                  "large aligned array helper expects trivially copyable element type");
    if (count == 0) {
        return nullptr;
    }
    if (count > static_cast<size_t>(-1) / sizeof(T)) {
        return nullptr;
    }
    return static_cast<T*>(
        allocate_hugepage_aligned_bytes(count * sizeof(T), advise_hugepage, alignment));
}

/**
 *  first-touch: OS page  1 byte, NUMA .
 *
 * ,.
 * , parallel_fill_array .
 */
inline void parallel_first_touch_pages(
    void* ptr,
    size_t bytes,
    int num_threads = 0)
{
    if (!ptr || bytes == 0) {
        return;
    }
    const size_t pages = (bytes + kCpuArrayOsPageBytes - 1) / kCpuArrayOsPageBytes;
    num_threads = std::min(cpu_array_resolve_num_threads(num_threads), (int)pages);
    num_threads = std::max(1, num_threads);

    std::vector<std::thread> workers;
    workers.reserve((size_t)num_threads);
    char* base = static_cast<char*>(ptr);
    for (int tid = 0; tid < num_threads; ++tid) {
        const size_t begin = (pages * (size_t)tid) / (size_t)num_threads;
        const size_t end = (pages * (size_t)(tid + 1)) / (size_t)num_threads;
        workers.emplace_back([=]() {
            for (size_t page = begin; page < end; ++page) {
                volatile char* p = base + page * kCpuArrayOsPageBytes;
                *p = *p;
            }
        });
    }
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

template <typename T>
inline void parallel_fill_array(
    T* data,
    size_t count,
    T value = T{},
    int num_threads = 0)
{
    static_assert(std::is_trivially_copyable<T>::value,
                  "parallel_fill_array expects trivially copyable element type");
    if (!data || count == 0) {
        return;
    }
    num_threads = std::min(cpu_array_resolve_num_threads(num_threads), (int)count);
    num_threads = std::max(1, num_threads);
    const size_t chunk = (count + (size_t)num_threads - 1) / (size_t)num_threads;
    std::vector<std::thread> workers;
    workers.reserve((size_t)num_threads);
    for (int tid = 0; tid < num_threads; ++tid) {
        workers.emplace_back([=]() {
            const size_t start = (size_t)tid * chunk;
            if (start >= count) {
                return;
            }
            const size_t end = std::min(start + chunk, count);
            std::fill(data + start, data + end, value);
        });
    }
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

template <typename T>
inline T* allocate_hugepage_aligned_array_first_touch(
    size_t count,
    int num_threads = 0,
    bool advise_hugepage = true)
{
    T* data = allocate_hugepage_aligned_array<T>(count, advise_hugepage);
    if (data) {
        parallel_first_touch_pages(data, count * sizeof(T), num_threads);
    }
    return data;
}


 /**
  *
  *
  * @tparam T  ( int, float, double )
  * @param data
  * @param size
  * @param seed
  * @param min_val
  * @param max_val
  * @param num_threads
  */
 template <typename T>
 inline void init_array_multithreaded(
     T* data,
     size_t size,
     unsigned int seed = 1234,
     T min_val = static_cast<T>(-10),
     T max_val = static_cast<T>(10),
     int num_threads = 0
 ) {
     // : T (),
     static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type (int, float, double, etc.)");

     // 1.
     if (num_threads <= 0) {
         unsigned int hw_concurrency = std::thread::hardware_concurrency();
         num_threads = std::max(1, (int)hw_concurrency);
     }

     // 2.  T ,
     //  -> uniform_real_distribution
     //    -> uniform_int_distribution
     using DistributionType = typename std::conditional<
         std::is_floating_point<T>::value,
         std::uniform_real_distribution<T>,
         std::uniform_int_distribution<T>
     >::type;

     //
     const size_t chunk_size = (size + num_threads - 1) / num_threads;
     std::vector<std::thread> workers;
     workers.reserve(num_threads);

     // 3.
     for (int tid = 0; tid < num_threads; ++tid) {
         workers.emplace_back([=]() {
             const size_t start = tid * chunk_size;
             //
             if (start >= size) return;
             const size_t end = std::min(start + chunk_size, size);

             //  RNG,
             std::mt19937 rng(seed + tid);

             //  DistributionType
             DistributionType dist(min_val, max_val);

             //
             for (size_t i = start; i < end; ++i) {
                 data[i] = dist(rng);
             }
         });
     }

     // 4.
     for (auto& t : workers) {
         if (t.joinable()) {
             t.join();
         }
     }
 }

/**
 *  interleaved ([min_val, max_val]), memset  padding,
 *  pack_rowmajor_to_interleaved , row-major .
 *
 * @param interleaved  paraselect::interleaved_storage_floats(n_vec, n_dim)
 */
inline void init_array_interleaved_multithreaded(
    float* interleaved,
    int n_vec,
    int n_dim,
    unsigned seed,
    float min_val,
    float max_val,
    int num_threads) {
    if (!interleaved || n_vec <= 0 || n_dim <= 0)
        return;

    namespace ivt = paraselect;
    const int kG = ivt::kInterleavedGroupSize;
    const int num_blocks = (n_vec + kG - 1) / kG;
    const size_t wpb = (size_t)kG * (size_t)n_dim;
    std::memset(interleaved, 0, ivt::interleaved_storage_bytes(n_vec, n_dim));

    num_threads = cpu_array_resolve_num_threads(num_threads);
    const int nt = std::min(num_threads, num_blocks);
    std::vector<std::thread> workers;
    workers.reserve((size_t)nt);
    for (int t = 0; t < nt; ++t) {
        const int lo = (num_blocks * t) / nt;
        const int hi = (num_blocks * (t + 1)) / nt;
        workers.emplace_back([=]() {
            std::mt19937 rng(seed + (unsigned)t);
            std::uniform_real_distribution<float> dist(min_val, max_val);
            for (int b = lo; b < hi; ++b) {
                for (int lane = 0; lane < kG; ++lane) {
                    const int global_v = b * kG + lane;
                    if (global_v >= n_vec)
                        continue;
                    for (int j = 0; j < n_dim; ++j) {
                        const size_t dst = (size_t)b * wpb + (size_t)j * kG + lane;
                        interleaved[dst] = dist(rng);
                    }
                }
            }
        });
    }
    for (auto& w : workers) {
        if (w.joinable())
            w.join();
    }
}

#endif // CPU_ARRAY_UTILS_H
