#ifndef COMPARE_UTILS_H
#define COMPARE_UTILS_H
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <utility>
#include "output_macros.cuh"
#include "params_macros.cuh"
#include "metrics_collector.cuh"
#include "vector_nd.cuh"
#include "test_config.h"
/**
 * :(C++14 )
 */
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, bool>::type
compare_numbers_relative(T a, T b, float epsilon) {
    return std::abs(a - b) * 2.0 / (std::abs(a) + std::abs(b)) < epsilon;
}

template<typename T>
typename std::enable_if<!std::is_floating_point<T>::value, bool>::type
compare_numbers_relative(T a, T b, float epsilon) {
    (void)epsilon;  //
    return a == b;
}

/**
* :(C++14 )
*/
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, bool>::type
compare_numbers(T a, T b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

template<typename T>
typename std::enable_if<!std::is_floating_point<T>::value, bool>::type
compare_numbers(T a, T b, float epsilon = 1e-5f) {
    (void)epsilon;  //
    return a == b;
}

/*
*
*/
template<typename T>
bool compare_1D(T* a, T* b, int n, float epsilon = 1e-5f, bool use_relative = false) {
    std::atomic<int> err_happens(0);
    std::mutex debug_mutex;
    std::atomic<bool> first_error(true);

    const int num_threads = std::min(static_cast<int>(std::thread::hardware_concurrency()), n);
    const int chunk_size = (n + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, n);

            for(int i = start; i < end; ++i){
                bool match = use_relative ?
                    compare_numbers_relative(a[i], b[i], epsilon) :
                    compare_numbers(a[i], b[i], epsilon);
                if (!match) {
                    int err_count = err_happens.fetch_add(1) + 1;

                    if(DEBUG){
                        std::lock_guard<std::mutex> lock(debug_mutex);
                        if(err_count == 1){
                            COUT_ENDL("mismatch!");
                            COUT_TABLE("i", "a[i]", "b[i]", "diff");
                        }
                        COUT_TABLE(i, a[i], b[i], a[i] - b[i]);
                    }
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    int total_errors = err_happens.load();
    if(total_errors == 0){
        return true;
    }
    if(DEBUG)
        COUT_ENDL(total_errors);
    return false;
}

template<typename T>
bool compare_2D(T* a, T* b, int nx, int ny, float epsilon = 1e-5f, bool use_relative = false) {
    std::atomic<int> err_happens(0);
    std::mutex debug_mutex;

    const int num_threads = std::min(static_cast<int>(std::thread::hardware_concurrency()), nx);
    const int chunk_size = (nx + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, nx);

            for(int i = start; i < end; ++i){
                for (int j = 0; j < ny; j++) {
                    bool match = use_relative ?
                        compare_numbers_relative(a[i][j], b[i][j], epsilon) :
                        compare_numbers(a[i][j], b[i][j], epsilon);
                    if (!match) {
                        int err_count = err_happens.fetch_add(1) + 1;

                        if(DEBUG){
                            std::lock_guard<std::mutex> lock(debug_mutex);
                            if(err_count == 1){
                                COUT_ENDL("mismatch!");
                                COUT_TABLE("i", "j", "a[i][j]", "b[i][j]", "diff");
                            }
                            COUT_TABLE(i, j, a[i][j], b[i][j], a[i][j] - b[i][j]);
                        }
                    }
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    int total_errors = err_happens.load();
    if(total_errors == 0){
        return true;
    }
    if(DEBUG)
        COUT_ENDL(total_errors);
    return false;
}

template<typename T>
bool compare_1D_relative(T* a, T* b, int n, float epsilon = 1e-5f) {
    return compare_1D(a, b, n, epsilon, true);
}

template<typename T>
bool compare_2D_relative(T* a, T* b, int nx, int ny, float epsilon = 1e-5f) {
    return compare_2D(a, b, nx, ny, epsilon, true);
}

template<typename T>
bool compare_set_1D(T** a, T** b, int n, float epsilon = 1e-5f, bool use_relative = false) {
    std::atomic<int> err_happens(0);
    std::mutex debug_mutex;

    std::sort(a, a + n);
    std::sort(b, b + n);

    const int num_threads = std::min(static_cast<int>(std::thread::hardware_concurrency()), n);
    const int chunk_size = (n + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, n);

            for(int i = start; i < end; ++i){
                bool match = use_relative ?
                    compare_numbers_relative(a[i], b[i], epsilon) :
                    compare_numbers(a[i], b[i], epsilon);
                if (!match) {
                    int err_count = err_happens.fetch_add(1) + 1;

                    if(DEBUG){
                        std::lock_guard<std::mutex> lock(debug_mutex);
                        if(err_count == 1){
                            COUT_ENDL("mismatch!");
                            COUT_TABLE("i", "a[i]", "b[i]", "diff");
                        }
                        COUT_TABLE(i, a[i], b[i], a[i] - b[i]);
                    }
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    int total_errors = err_happens.load();
    if(total_errors == 0){
        return true;
    }
    if(DEBUG)
        COUT_ENDL(total_errors);
    return false;
}

template<typename T>
bool compare_set_2D(T** a, T** b, int nx, int ny, float epsilon = 1e-5f, bool use_relative = false) {
    std::atomic<int> err_happens(0);
    std::mutex debug_mutex;

    const int num_threads = std::min(static_cast<int>(std::thread::hardware_concurrency()), nx);
    const int chunk_size = (nx + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, nx);

            for(int i = start; i < end; ++i){
                std::sort(a[i], a[i] + ny);
                std::sort(b[i], b[i] + ny);

                for (int j = 0; j < ny; j++) {
                    bool match = use_relative ?
                        compare_numbers_relative(a[i][j], b[i][j], epsilon) :
                        compare_numbers(a[i][j], b[i][j], epsilon);
                    if (!match) {
                        int err_count = err_happens.fetch_add(1) + 1;

                        if(DEBUG){
                            std::lock_guard<std::mutex> lock(debug_mutex);
                            if(err_count == 1){
                                COUT_ENDL("mismatch!");
                                COUT_TABLE("i", "j", "a[i][j]", "b[i][j]", "diff");
                            }
                            COUT_TABLE(i, j, a[i][j], b[i][j], a[i][j] - b[i][j]);
                        }
                    }
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    int total_errors = err_happens.load();
    if(total_errors == 0){
        return true;
    }
    if(DEBUG)
        COUT_ENDL(total_errors);
    return false;
}

template<typename T>
bool compare_set_1D_relative(T** a, T** b, int n, float epsilon = 1e-5f) {
    return compare_set_1D(a, b, n, epsilon, true);
}

template<typename T>
bool compare_set_2D_relative(T** a, T** b, int nx, int ny, float epsilon = 1e-5f) {
    return compare_set_2D(a, b, nx, ny, epsilon, true);
}

/**
* ()(epsilon)
* a/b,
*
*/
template<typename T>
int count_equal_elements_set_1D(T** a, T** b, int n, float epsilon = 1e-5f, bool use_relative = false) {
    std::sort(a, a + n);
    std::sort(b, b + n);
    int i = 0, j = 0;
    int count_equal = 0;
    while (i < n && j < n) {
        bool match = use_relative ?
            compare_numbers_relative(a[i], b[j], epsilon) :
            compare_numbers(a[i], b[j], epsilon);
        if (match) {
            count_equal++;
            i++;
            j++;
        }
        else if (a[i] < b[j] - epsilon) {
            i++;
        }
        else {
            j++;
        }
    }
    if (DEBUG && count_equal != n)
        COUT_ENDL("Number of inequal elements:", n - count_equal, "out of", n);
    return count_equal == n;
}

/**
* ()(epsilon)
* a/b,
*
*/
template<typename T>
int count_equal_elements_set_2D(T** a, T** b, int nx, int ny, float epsilon = 1e-5f, bool use_relative = false) {
    std::atomic<int> count_equal(0);
    std::mutex debug_mutex;

    const int num_threads = std::min(static_cast<int>(std::thread::hardware_concurrency()), nx);
    const int chunk_size = (nx + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, nx);
            int local_count = 0;

            for(int i = start; i < end; ++i){
                std::sort(a[i], a[i] + ny);
                std::sort(b[i], b[i] + ny);

                int j = 0, k = 0;
                while (j < ny && k < ny) {
                    bool match = use_relative ?
                        compare_numbers_relative(a[i][j], b[i][k], epsilon) :
                        compare_numbers(a[i][j], b[i][k], epsilon);
                    if (match) {
                        local_count++;
                        j++;
                        k++;
                    }
                    else if (a[i][j] < b[i][k] - epsilon) {
                        j++;
                    }
                    else {
                        k++;
                    }
                }
            }

            count_equal.fetch_add(local_count);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    int total_count = count_equal.load();
    if (DEBUG && total_count != nx * ny) {
        std::lock_guard<std::mutex> lock(debug_mutex);
        COUT_ENDL("Number of inequal elements:", nx * ny - total_count, "out of", nx * ny);
    }
    return total_count == nx * ny;
}

template<typename T>
int count_equal_elements_set_1D_relative(T** a, T** b, int n, float epsilon = 1e-5f) {
    return count_equal_elements_set_1D(a, b, n, epsilon, true);
}

template<typename T>
int count_equal_elements_set_2D_relative(T** a, T** b, int nx, int ny, float epsilon = 1e-5f) {
    return count_equal_elements_set_2D(a, b, nx, ny, epsilon, true);
}

/**
 * :
 */
struct MismatchInfo {
    int query_idx;      //
    int pos;            //
    float cpu_dist;     // CPU
    float gpu_dist;     // GPU
    idx_t cpu_idx;      // CPU()
    idx_t gpu_idx;      // GPU()

    MismatchInfo(int qi, int p, float c_dist, float g_dist, idx_t c_idx = -1, idx_t g_idx = -1)
        : query_idx(qi), pos(p), cpu_dist(c_dist), gpu_dist(g_dist), cpu_idx(c_idx), gpu_idx(g_idx) {}
};

/**
 * :
 * +,CPUGPU
 *
 */
template<typename T>
std::vector<MismatchInfo> compare_set_2D_with_mismatches(
    T** cpu_dist, T** gpu_dist,
    idx_t** cpu_idx, idx_t** gpu_idx,
    int nx, int ny,
    float epsilon = 1e-5f,
    bool use_relative = false) {

    std::vector<MismatchInfo> mismatches;
    std::mutex mismatch_mutex;

    const int num_threads = std::min(static_cast<int>(std::thread::hardware_concurrency()), nx);
    const int chunk_size = (nx + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, nx);

            // ()
            std::vector<std::pair<T, idx_t>> cpu_sorted(ny);
            std::vector<std::pair<T, idx_t>> gpu_sorted(ny);

            for(int i = start; i < end; ++i){
                // ()
                for (int j = 0; j < ny; j++) {
                    cpu_sorted[j] = {cpu_dist[i][j], (cpu_idx ? cpu_idx[i][j] : static_cast<idx_t>(-1))};
                    gpu_sorted[j] = {gpu_dist[i][j], (gpu_idx ? gpu_idx[i][j] : static_cast<idx_t>(-1))};
                }

                // ()
                std::sort(cpu_sorted.begin(), cpu_sorted.end(),
                    [](const auto& a, const auto& b) {
                        if (a.second != b.second) return a.second < b.second;
                        return a.first < b.first;
                    });
                std::sort(gpu_sorted.begin(), gpu_sorted.end(),
                    [](const auto& a, const auto& b) {
                        if (a.second != b.second) return a.second < b.second;
                        return a.first < b.first;
                    });

                // CPUGPU
                int cpu_ptr = 0, gpu_ptr = 0;
                int cpu_pos = 0, gpu_pos = 0;  //

                while (cpu_ptr < ny || gpu_ptr < ny) {
                    if (cpu_ptr >= ny) {
                        // CPU,GPUGPU
                        while (gpu_ptr < ny) {
                            std::lock_guard<std::mutex> lock(mismatch_mutex);
                            mismatches.emplace_back(
                                i, gpu_pos,
                                std::numeric_limits<float>::max(),  // CPU
                                gpu_sorted[gpu_ptr].first,
                                -1,  // CPU
                                gpu_sorted[gpu_ptr].second
                            );
                            gpu_ptr++;
                            gpu_pos++;
                        }
                        break;
                    }

                    if (gpu_ptr >= ny) {
                        // GPU,CPUCPU
                        while (cpu_ptr < ny) {
                            std::lock_guard<std::mutex> lock(mismatch_mutex);
                            mismatches.emplace_back(
                                i, cpu_pos,
                                cpu_sorted[cpu_ptr].first,
                                std::numeric_limits<float>::max(),  // GPU
                                cpu_sorted[cpu_ptr].second,
                                -1  // GPU
                            );
                            cpu_ptr++;
                            cpu_pos++;
                        }
                        break;
                    }

                    //
                    idx_t cpu_idx_val = cpu_sorted[cpu_ptr].second;
                    idx_t gpu_idx_val = gpu_sorted[gpu_ptr].second;

                    if (cpu_idx_val == gpu_idx_val) {
                        // ,
                        bool dist_match = use_relative
                            ? compare_numbers_relative(cpu_sorted[cpu_ptr].first, gpu_sorted[gpu_ptr].first, epsilon)
                            : compare_numbers(cpu_sorted[cpu_ptr].first, gpu_sorted[gpu_ptr].first, epsilon);

                        if (dist_match) {
                            // ,
                            cpu_ptr++;
                            gpu_ptr++;
                            cpu_pos++;
                            gpu_pos++;
                        } else {
                            // ,
                            std::lock_guard<std::mutex> lock(mismatch_mutex);
                            mismatches.emplace_back(
                                i, cpu_pos,
                                cpu_sorted[cpu_ptr].first,
                                gpu_sorted[gpu_ptr].first,
                                cpu_sorted[cpu_ptr].second,
                                gpu_sorted[gpu_ptr].second
                            );
                            cpu_ptr++;
                            gpu_ptr++;
                            cpu_pos++;
                            gpu_pos++;
                        }
                    } else if (cpu_idx_val < gpu_idx_val) {
                        // CPU,CPU
                        std::lock_guard<std::mutex> lock(mismatch_mutex);
                        mismatches.emplace_back(
                            i, cpu_pos,
                            cpu_sorted[cpu_ptr].first,
                            std::numeric_limits<float>::max(),  // GPU
                            cpu_sorted[cpu_ptr].second,
                            -1  // GPU
                        );
                        cpu_ptr++;
                        cpu_pos++;
                    } else {
                        // GPU,GPU
                        std::lock_guard<std::mutex> lock(mismatch_mutex);
                        mismatches.emplace_back(
                            i, gpu_pos,
                            std::numeric_limits<float>::max(),  // CPU
                            gpu_sorted[gpu_ptr].first,
                            -1,  // CPU
                            gpu_sorted[gpu_ptr].second
                        );
                        gpu_ptr++;
                        gpu_pos++;
                    }
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return mismatches;
}

/**
 * ()
 */
template<typename T>
std::vector<MismatchInfo> compare_set_2D_with_mismatches_relative(
    T** cpu_dist, T** gpu_dist,
    idx_t** cpu_idx, idx_t** gpu_idx,
    int nx, int ny,
    float epsilon = 1e-5f) {
    return compare_set_2D_with_mismatches<T>(cpu_dist, gpu_dist, cpu_idx, gpu_idx, nx, ny, epsilon, true);
}

#endif
