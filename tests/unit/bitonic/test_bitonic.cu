/**
 * Unit tests for Bitonic Sort class
 *
 *  bitonic.cu  Bitonic
 * :
 * 1. Bitonic::sort() -  bitonic
 * 2. Bitonic::merge() - bitonic
 * 3.  Size (1, 2, 4, 8)
 * 4.
 */

#include "utils/pch.h"
#include "common/test_utils.cuh"
#include "cpu/cpu_utils.h"
#include <algorithm>
#include <vector>
#include "src/search/kernels/topk/warpsort/bitonic.cuh"
#include "src/utils/utils.cuh"
// #include "src/warpsort_utils.cuh"
#define EPSILON 1e-5f

using namespace paraselect::bitonic;

// ============================================================================
// Test Kernels
// ============================================================================

/**
 *  Bitonic::sort()
 *
 * @tparam Size
 * @tparam Ascending
 */
template<int Size, bool Ascending, typename T, typename IdxT>
__global__ void test_bitonic_sort_kernel(
    const T* __restrict__ input,
    T* __restrict__ output_vals,
    IdxT* __restrict__ output_idx,
    int total_size)
{
    const int lane = laneId();
    const int warp_width = kWarpSize;

    /*  Size  */
    T keys[Size];
    IdxT indices[Size];

    /*  */
    #pragma unroll
    for (int i = 0; i < Size; i++) {
        int idx = i * warp_width + lane;
        if (idx < total_size) {
            keys[i] = input[idx];
            indices[i] = static_cast<IdxT>(idx);
        } else {
            /*  */
            keys[i] = Ascending ? upper_bound<T>() : lower_bound<T>();
            indices[i] = static_cast<IdxT>(idx);
        }
    }

    /*  bitonic sort */
    Bitonic<Size> sorter(Ascending, warp_width);
    sorter.sort(keys, indices);

    /*  */
    #pragma unroll
    for (int i = 0; i < Size; i++) {
        int idx = i * warp_width + lane;
        if (idx < total_size) {
            output_vals[idx] = keys[i];
            output_idx[idx] = indices[i];
        }
    }
}

/**
 *  Bitonic::merge()
 *
 * (ascending ,descending )
 *
 * @tparam Size
 * @tparam Ascending
 */
template<int Size, bool Ascending, typename T, typename IdxT>
__global__ void test_bitonic_merge_kernel(
    const T* __restrict__ input,
    T* __restrict__ output_vals,
    IdxT* __restrict__ output_idx,
    int total_size)
{
    const int lane = laneId();
    const int warp_width = kWarpSize;

    /*  Size  */
    T keys[Size];
    IdxT indices[Size];

    /*  */
    #pragma unroll
    for (int i = 0; i < Size; i++) {
        int idx = i * warp_width + lane;
        if (idx < total_size) {
            keys[i] = input[idx];
            indices[i] = static_cast<IdxT>(idx);
        } else {
            /*  */
            keys[i] = Ascending ? upper_bound<T>() : lower_bound<T>();
            indices[i] = static_cast<IdxT>(idx);
        }
    }

    /*  bitonic merge */
    Bitonic<Size> merger(Ascending, warp_width);
    merger.merge(keys, indices);

    /*  */
    #pragma unroll
    for (int i = 0; i < Size; i++) {
        int idx = i * warp_width + lane;
        if (idx < total_size) {
            output_vals[idx] = keys[i];
            output_idx[idx] = indices[i];
        }
    }
}

// ============================================================================
// Test Functions
// ============================================================================

/**
 *  Bitonic::sort()
 */
template<int Size, bool Ascending>
bool test_bitonic_sort_impl(int n_elements)
{
    using T = float;
    using IdxT = int;

    const int total_size = Size * kWarpSize;
    COUT_ENDL(": Size=", Size, "Ascending=", Ascending,
              "n_elements=", n_elements, "total_size=", total_size);

    /*  */
    T* h_input = (T*)malloc(total_size * sizeof(T));
    T* h_gpu_vals = (T*)malloc(total_size * sizeof(T));
    T* h_cpu_vals = (T*)malloc(total_size * sizeof(T));
    IdxT* h_gpu_idx = (IdxT*)malloc(total_size * sizeof(IdxT));
    IdxT* h_cpu_idx = (IdxT*)malloc(total_size * sizeof(IdxT));

    /*  */
    srand(42);
    for (int i = 0; i < n_elements; i++) {
        h_input[i] = static_cast<T>(rand() % 1000) / 10.0f;
    }
    /*  */
    for (int i = n_elements; i < total_size; i++) {
        h_input[i] = Ascending ? upper_bound<T>() : lower_bound<T>();
    }

    // print_1D("h_input", h_input, std::min(total_size, 64));

    /*  */
    T *d_input, *d_output_vals;
    IdxT *d_output_idx;
    cudaMalloc(&d_input, total_size * sizeof(T));
    cudaMalloc(&d_output_vals, total_size * sizeof(T));
    cudaMalloc(&d_output_idx, total_size * sizeof(IdxT));
    CHECK_CUDA_ERRORS;

    /*  */
    cudaMemcpy(d_input, h_input, total_size * sizeof(T), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS;

    /*  GPU kernel ( warp) */
    dim3 block(32);  /*  warp */
    dim3 grid(1);
    test_bitonic_sort_kernel<Size, Ascending, T, IdxT><<<grid, block>>>(
        d_input, d_output_vals, d_output_idx, total_size);
    CHECK_CUDA_ERRORS;

    /*  */
    cudaMemcpy(h_gpu_vals, d_output_vals, total_size * sizeof(T),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gpu_idx, d_output_idx, total_size * sizeof(IdxT),
               cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERRORS;

    /* CPU  */
    cpu_sort<T, IdxT>(h_input, h_cpu_vals, h_cpu_idx, total_size, Ascending);

    /*  */
    bool vals_match = compare_1D(h_gpu_vals, h_cpu_vals, total_size, EPSILON);
    // bool idx_match = compare_1D(h_gpu_idx, h_cpu_idx, total_size, 0);
    bool pass = vals_match;
    // bool pass = vals_match && idx_match;

    if (!pass) {
        COUT_ENDL("❌ !");
        /*  */
        // if (!vals_match) {
        //     COUT_ENDL("");
        //     print_1D("GPU vals", h_gpu_vals, std::min(total_size, 64));
        //     print_1D("CPU vals", h_cpu_vals, std::min(total_size, 64));
        // }
        // if (!idx_match) {
        //     COUT_ENDL("");
        //     print_1D("GPU idx", h_gpu_idx, std::min(total_size, 64));
        //     print_1D("CPU idx", h_cpu_idx, std::min(total_size, 64));
        // }
    } else {
        COUT_ENDL("✅ !");
    }

    /*  */
    free(h_input);
    free(h_gpu_vals);
    free(h_cpu_vals);
    free(h_gpu_idx);
    free(h_cpu_idx);
    cudaFree(d_input);
    cudaFree(d_output_vals);
    cudaFree(d_output_idx);

    return pass;
}

/**
 *  Bitonic::merge()
 *
 *  merge,
 *
 * :Bitonic sort ,,
 * ,
 */
template<int Size, bool Ascending>
bool test_bitonic_merge_impl(int n_elements)
{
    using T = float;
    using IdxT = int;

    const int total_size = Size * kWarpSize;
    COUT_ENDL(": Size=", Size, "Ascending=", Ascending,
              "n_elements=", n_elements, "total_size=", total_size);

    /*  */
    T* h_input = (T*)malloc(total_size * sizeof(T));
    T* h_gpu_vals = (T*)malloc(total_size * sizeof(T));
    T* h_cpu_vals = (T*)malloc(total_size * sizeof(T));
    IdxT* h_gpu_idx = (IdxT*)malloc(total_size * sizeof(IdxT));
    IdxT* h_cpu_idx = (IdxT*)malloc(total_size * sizeof(IdxT));

    /* () */
    srand(42);
    std::vector<std::pair<T, IdxT>> pairs;
    for (int i = 0; i < n_elements; i++) {
        /* :, */
        T val = static_cast<T>(i);
        pairs.push_back({val, static_cast<IdxT>(i)});
    }

    int mid = n_elements / 2;

    /* : */
    if (Ascending) {
        std::sort(pairs.begin(), pairs.begin() + mid,
            [](const auto& a, const auto& b) { return a.first < b.first; });
    } else {
        std::sort(pairs.begin(), pairs.begin() + mid,
            [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    /* : */
    if (Ascending) {
        std::sort(pairs.begin() + mid, pairs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
    } else {
        std::sort(pairs.begin() + mid, pairs.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
    }

    /*  h_input */
    for (int i = 0; i < n_elements; i++) {
        h_input[i] = pairs[i].first;
    }
    /*  */
    for (int i = n_elements; i < total_size; i++) {
        h_input[i] = Ascending ? upper_bound<T>() : lower_bound<T>();
    }

    /*  */
    T *d_input, *d_output_vals;
    IdxT *d_output_idx;
    cudaMalloc(&d_input, total_size * sizeof(T));
    cudaMalloc(&d_output_vals, total_size * sizeof(T));
    cudaMalloc(&d_output_idx, total_size * sizeof(IdxT));
    CHECK_CUDA_ERRORS;

    /*  */
    cudaMemcpy(d_input, h_input, total_size * sizeof(T), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS;

    /*  GPU kernel ( warp) */
    dim3 block(32);
    dim3 grid(1);
    test_bitonic_merge_kernel<Size, Ascending, T, IdxT><<<grid, block>>>(
        d_input, d_output_vals, d_output_idx, total_size);
    CHECK_CUDA_ERRORS;

    /*  */
    cudaMemcpy(h_gpu_vals, d_output_vals, total_size * sizeof(T),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gpu_idx, d_output_idx, total_size * sizeof(IdxT),
               cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERRORS;

    /* CPU :merge  */
    cpu_sort<T, IdxT>(h_input, h_cpu_vals, h_cpu_idx, total_size, Ascending);

    /*  */
    bool vals_match = compare_1D(h_gpu_vals, h_cpu_vals, total_size, EPSILON);
    bool idx_match = compare_1D(h_gpu_idx, h_cpu_idx, total_size, 0);
    bool pass = vals_match && idx_match;

    if (!pass) {
        COUT_ENDL("❌ FAIL");
        /*  */
        if (!vals_match) {
            COUT_ENDL("");
            print_1D("Input", h_input, std::min(total_size, 64));
            print_1D("GPU vals", h_gpu_vals, std::min(total_size, 64));
            print_1D("CPU vals", h_cpu_vals, std::min(total_size, 64));
        }
        if (!idx_match) {
            COUT_ENDL("");
            print_1D("GPU idx", h_gpu_idx, std::min(total_size, 64));
            print_1D("CPU idx", h_cpu_idx, std::min(total_size, 64));
        }
    } else {
        COUT_ENDL("✅ PASS");
    }

    /*  */
    free(h_input);
    free(h_gpu_vals);
    free(h_cpu_vals);
    free(h_gpu_idx);
    free(h_cpu_idx);
    cudaFree(d_input);
    cudaFree(d_output_vals);
    cudaFree(d_output_idx);

    return pass;
}

// ============================================================================
// Main Test Runners
// ============================================================================

/**
 *  Bitonic::sort()
 */
bool test_bitonic_sort()
{
    COUT_ENDL("\n===  Bitonic::sort() ===");

    bool all_pass = true;

    COUT_ENDL("\n--- Test 1: Size=1, Ascending=true ---");
    all_pass &= test_bitonic_sort_impl<1, true>(32);

    COUT_ENDL("\n--- Test 2: Size=1, Ascending=false ---");
    all_pass &= test_bitonic_sort_impl<1, false>(32);

    COUT_ENDL("\n--- Test 3: Size=2, Ascending=true ---");
    all_pass &= test_bitonic_sort_impl<2, true>(64);

    COUT_ENDL("\n--- Test 4: Size=2, Ascending=false ---");
    all_pass &= test_bitonic_sort_impl<2, false>(64);


    COUT_ENDL("\n--- Test 5: Size=4, Ascending=true ---");
    all_pass &= test_bitonic_sort_impl<4, true>(128);

    COUT_ENDL("\n--- Test 6: Size=4, Ascending=false ---");
    all_pass &= test_bitonic_sort_impl<4, false>(128);

    COUT_ENDL("\n--- Test 7: Size=8, Ascending=true ---");
    all_pass &= test_bitonic_sort_impl<8, true>(256);

    COUT_ENDL("\n--- Test 8: Size=8, Ascending=false ---");
    all_pass &= test_bitonic_sort_impl<8, false>(256);

    COUT_ENDL("\n--- Test 9: Size=16, Ascending=true ---");
    all_pass &= test_bitonic_merge_impl<16, true>(512);

    COUT_ENDL("\n--- Test 10: Size=16, Ascending=false ---");
    all_pass &= test_bitonic_merge_impl<16, false>(512);
    return all_pass;
    return all_pass;
}

/**
 *  Bitonic::merge()
 */
bool test_bitonic_merge()
{
    COUT_ENDL("\n===  Bitonic::merge() ===");

    bool all_pass = true;

    COUT_ENDL("\n--- Test 1: Size=1, Ascending=true ---");
    all_pass &= test_bitonic_merge_impl<1, true>(32);

    /* Test 2: Size=1,  */
    COUT_ENDL("\n--- Test 2: Size=1, Ascending=false ---");
    all_pass &= test_bitonic_merge_impl<1, false>(32);

    COUT_ENDL("\n--- Test 3: Size=2, Ascending=true ---");
    all_pass &= test_bitonic_merge_impl<2, true>(64);

    COUT_ENDL("\n--- Test 4: Size=2, Ascending=false ---");
    all_pass &= test_bitonic_merge_impl<2, false>(64);

    COUT_ENDL("\n--- Test 5: Size=4, Ascending=true ---");
    all_pass &= test_bitonic_merge_impl<4, true>(128);

    COUT_ENDL("\n--- Test 6: Size=4, Ascending=false ---");
    all_pass &= test_bitonic_merge_impl<4, false>(128);

    COUT_ENDL("\n--- Test 7: Size=8, Ascending=true ---");
    all_pass &= test_bitonic_merge_impl<8, true>(256);

    COUT_ENDL("\n--- Test 8: Size=8, Ascending=false ---");
    all_pass &= test_bitonic_merge_impl<8, false>(256);

    COUT_ENDL("\n--- Test 9: Size=16, Ascending=true ---");
    all_pass &= test_bitonic_merge_impl<16, true>(512);

    COUT_ENDL("\n--- Test 10: Size=16, Ascending=false ---");
    all_pass &= test_bitonic_merge_impl<16, false>(512);
    return all_pass;
}

// ============================================================================
// Main
// ============================================================================

int main()
{
    COUT_ENDL("  Bitonic ");

    /*  */
    bool test1 = test_bitonic_sort();
    // bool test2 = test_bitonic_merge();

    return 0;
}
