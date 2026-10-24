// ============================================================================
// CPU Reference Implementations
// ============================================================================

#include <limits>
#include <thread>
#include <vector>
#include <algorithm>
#include <cfloat>
#include <queue>

// AVX2 SIMD
#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#define CPU_HAS_AVX2 1
#else
#define CPU_HAS_AVX2 0
#endif

/**
 * CPU :
 */
template<typename T, typename IdxT>
void cpu_sort(
    const T* input,
    T* output_vals,
    IdxT* output_idx,
    int n,
    bool ascending)
{
    std::vector<std::pair<T, IdxT>> pairs;
    pairs.reserve(n);

    for (int i = 0; i < n; i++) {
        pairs.push_back({input[i], static_cast<IdxT>(i)});
    }

    if (ascending) {
        std::sort(pairs.begin(), pairs.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
    } else {
        std::sort(pairs.begin(), pairs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    for (int i = 0; i < n; i++) {
        output_vals[i] = pairs[i].first;
        output_idx[i] = pairs[i].second;
    }
}

 /**
 * CPU : top-k
 */
void cpu_select_k(
    const float** input,
    int batch_size,
    int len,
    int k,
    float** output_vals,
    int** output_idx,
    bool select_min
) {
    //  batch
    auto process_batch = [&](int b) {
        if (len < k) {
            for (int i = 0; i < len; ++i) {
                output_vals[b][i] = input[b][i];
                output_idx[b][i] = i;
            }
            const float pad = select_min ? std::numeric_limits<float>::infinity()
                                         : -std::numeric_limits<float>::infinity();
            for (int i = len; i < k; ++i) {
                output_vals[b][i] = pad;
                output_idx[b][i] = -1;
            }
            return;
        }

        //  partial_sort : k
        //  O(n log n)  O(n log k)
        //  thread_local
        thread_local std::vector<std::pair<float, int>> candidates;
        candidates.clear();
        candidates.reserve(len);
        for (int i = 0; i < len; ++i) {
            float val = input[b][i];
            if (val < FLT_MAX && val == val) {  //  INF  NaN
                candidates.push_back({val, i});
            }
        }

        int n_select = std::min(k, (int)candidates.size());

        if (select_min) {
            //  partial_sort  k
            std::partial_sort(candidates.begin(),
                              candidates.begin() + n_select,
                              candidates.end());
        } else {
            //  partial_sort  k
            std::partial_sort(candidates.begin(),
                              candidates.begin() + n_select,
                              candidates.end(),
                              std::greater<std::pair<float, int>>());
        }

        for (int i = 0; i < n_select; ++i) {
            output_vals[b][i] = candidates[i].first;
            output_idx[b][i] = candidates[i].second;
        }
        const float pad = select_min ? std::numeric_limits<float>::infinity()
                                     : -std::numeric_limits<float>::infinity();
        for (int i = n_select; i < k; ++i) {
            output_vals[b][i] = pad;
            output_idx[b][i] = -1;
        }
    };

    //  batch
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    //  batch_size ,
    if (batch_size <= 4 || num_threads == 1) {
        for (int b = 0; b < batch_size; ++b) {
            process_batch(b);
        }
        return;
    }

    //  batch
    int batches_per_thread = (batch_size + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (unsigned int t = 0; t < num_threads; ++t) {
        int start_batch = t * batches_per_thread;
        int end_batch = std::min(start_batch + batches_per_thread, batch_size);

        if (start_batch >= batch_size) break;

        threads.emplace_back([start_batch, end_batch, &process_batch]() {
            for (int b = start_batch; b < end_batch; ++b) {
                process_batch(b);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}

// ============================================================================
// SIMD  Top-K ( AVX2)
// ============================================================================
#if CPU_HAS_AVX2

/**
 * AVX2 : 8  float
 *  8  mask,
 */
inline int avx2_compare_mask(__m256 a, __m256 b, bool less_than) {
    if (less_than) {
        __m256 cmp = _mm256_cmp_ps(a, b, _CMP_LT_OQ);
        return _mm256_movemask_ps(cmp);
    } else {
        __m256 cmp = _mm256_cmp_ps(a, b, _CMP_GT_OQ);
        return _mm256_movemask_ps(cmp);
    }
}

/**
 * AVX2  Top-K
 *  nth_element (O(n)) + SIMD
 */
void cpu_select_k_avx2(
    const float** input,
    int batch_size,
    int len,
    int k,
    float** output_vals,
    int** output_idx,
    bool select_min
) {
    auto process_batch = [&](int b) {
        if (len < k) {
            for (int i = 0; i < len; ++i) {
                output_vals[b][i] = input[b][i];
                output_idx[b][i] = i;
            }
            const float pad = select_min ? std::numeric_limits<float>::infinity()
                                         : -std::numeric_limits<float>::infinity();
            for (int i = len; i < k; ++i) {
                output_vals[b][i] = pad;
                output_idx[b][i] = -1;
            }
            return;
        }

        // SIMD : AVX2  candidates
        //  8  float
        const int simd_width = 8;

        thread_local std::vector<std::pair<float, int>> candidates;
        candidates.clear();
        candidates.reserve(len);

        int i = 0;
        //  8
        for (; i < len && (reinterpret_cast<uintptr_t>(&input[b][i]) % 32 != 0); ++i) {
            float val = input[b][i];
            if (val < FLT_MAX && val == val) {
                candidates.push_back({val, i});
            }
        }

        // SIMD : 8
        for (; i + simd_width <= len; i += simd_width) {
            __m256 vals = _mm256_load_ps(&input[b][i]);  //

            //  8
            alignas(32) float temp[8];
            _mm256_store_ps(temp, vals);

            for (int j = 0; j < 8; ++j) {
                float val = temp[j];
                if (val < FLT_MAX && val == val) {
                    candidates.push_back({val, i + j});
                }
            }
        }

        //
        for (; i < len; ++i) {
            float val = input[b][i];
            if (val < FLT_MAX && val == val) {
                candidates.push_back({val, i});
            }
        }

        int n_select = std::min(k, (int)candidates.size());

        //  nth_element  k  (O(n))
        if (select_min) {
            std::nth_element(candidates.begin(),
                            candidates.begin() + n_select,
                            candidates.end());
        } else {
            std::nth_element(candidates.begin(),
                            candidates.begin() + n_select,
                            candidates.end(),
                            std::greater<std::pair<float, int>>());
        }

        //  k ()
        for (int i = 0; i < n_select; ++i) {
            output_vals[b][i] = candidates[i].first;
            output_idx[b][i] = candidates[i].second;
        }

        const float pad = select_min ? std::numeric_limits<float>::infinity()
                                     : -std::numeric_limits<float>::infinity();
        for (int i = n_select; i < k; ++i) {
            output_vals[b][i] = pad;
            output_idx[b][i] = -1;
        }
    };

    //
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    if (batch_size <= 4 || num_threads == 1) {
        for (int b = 0; b < batch_size; ++b) process_batch(b);
        return;
    }

    int batches_per_thread = (batch_size + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (unsigned int t = 0; t < num_threads; ++t) {
        int start = t * batches_per_thread;
        int end = std::min(start + batches_per_thread, batch_size);
        if (start >= batch_size) break;
        threads.emplace_back([start, end, &process_batch]() {
            for (int b = start; b < end; ++b) process_batch(b);
        });
    }

    for (auto& t : threads) t.join();
}

#endif // CPU_HAS_AVX2

// ============================================================================
// nth_element  (O(n)  Top-K)
// ============================================================================

void cpu_select_k_nth(
    const float** input,
    int batch_size,
    int len,
    int k,
    float** output_vals,
    int** output_idx,
    bool select_min
) {
    auto process_batch = [&](int b) {
        if (len < k) {
            for (int i = 0; i < len; ++i) {
                output_vals[b][i] = input[b][i];
                output_idx[b][i] = i;
            }
            const float pad = select_min ? std::numeric_limits<float>::infinity()
                                         : -std::numeric_limits<float>::infinity();
            for (int i = len; i < k; ++i) {
                output_vals[b][i] = pad;
                output_idx[b][i] = -1;
            }
            return;
        }

        thread_local std::vector<std::pair<float, int>> candidates;
        candidates.clear();
        candidates.reserve(len);

        for (int i = 0; i < len; ++i) {
            float val = input[b][i];
            if (val < FLT_MAX && val == val) {
                candidates.push_back({val, i});
            }
        }

        int n_select = std::min(k, (int)candidates.size());

        //  nth_element  k
        //  O(n)
        if (select_min) {
            std::nth_element(candidates.begin(),
                            candidates.begin() + n_select,
                            candidates.end());
        } else {
            std::nth_element(candidates.begin(),
                            candidates.begin() + n_select,
                            candidates.end(),
                            std::greater<std::pair<float, int>>());
        }

        //  k ()
        for (int i = 0; i < n_select; ++i) {
            output_vals[b][i] = candidates[i].first;
            output_idx[b][i] = candidates[i].second;
        }

        const float pad = select_min ? std::numeric_limits<float>::infinity()
                                     : -std::numeric_limits<float>::infinity();
        for (int i = n_select; i < k; ++i) {
            output_vals[b][i] = pad;
            output_idx[b][i] = -1;
        }
    };

    //
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    if (batch_size <= 4 || num_threads == 1) {
        for (int b = 0; b < batch_size; ++b) process_batch(b);
        return;
    }

    int batches_per_thread = (batch_size + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (unsigned int t = 0; t < num_threads; ++t) {
        int start = t * batches_per_thread;
        int end = std::min(start + batches_per_thread, batch_size);
        if (start >= batch_size) break;
        threads.emplace_back([start, end, &process_batch]() {
            for (int b = start; b < end; ++b) process_batch(b);
        });
    }

    for (auto& t : threads) t.join();
}
