# test_clusteringCMake

cmake_minimum_required(VERSION 3.18)
project(test_clustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 70,,
set(CMAKE_CUDA_ARCHITECTURES ${PARASELECT_CUDA_ARCHITECTURES})

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/tests/common)

add_executable(test_kmeans_spec
    test_kmeans_spec.cpp
)
target_include_directories(test_kmeans_spec PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)
add_test(NAME kmeans_spec_test COMMAND test_kmeans_spec)
set_tests_properties(kmeans_spec_test PROPERTIES
    LABELS "unit;clustering;kmeans;spec"
)

# 1. test_kmeans (kmeans )
add_executable(test_kmeans
    test_kmeans.cu
    ${CMAKE_SOURCE_DIR}/tests/common/test_utils.cu
)
target_link_libraries(test_kmeans
    kmeans_lib
    utils_lib
    l2norm_lib
    paraselect_cuda
)
add_dependencies(test_kmeans kmeans_lib utils_lib l2norm_lib)
set_target_properties(test_kmeans PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(test_kmeans PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcompiler -std=c++17
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -std=c++17
    >
)
enable_testing()
add_test(NAME kmeans_test COMMAND test_kmeans)
set_tests_properties(kmeans_test PROPERTIES
    LABELS "unit;cuda;clustering;kmeans"
)

add_executable(test_flash_kmeans_assign
    test_flash_kmeans_assign.cu
    ${CMAKE_SOURCE_DIR}/tests/common/test_utils.cu
)
target_link_libraries(test_flash_kmeans_assign
    kmeans_lib
    utils_lib
    l2norm_lib
    paraselect_cuda
    CUDA::cublas
)
add_dependencies(test_flash_kmeans_assign kmeans_lib utils_lib l2norm_lib)
set_target_properties(test_flash_kmeans_assign PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(test_flash_kmeans_assign PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcompiler -std=c++17
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -std=c++17
    >
)
add_test(NAME flash_kmeans_assign_test COMMAND test_flash_kmeans_assign)
set_tests_properties(flash_kmeans_assign_test PROPERTIES
    LABELS "unit;cuda;clustering;kmeans;flash"
)

add_executable(test_kmeans_progressive_prefix
    test_kmeans_progressive_prefix.cu
    ${CMAKE_SOURCE_DIR}/tests/common/test_utils.cu
)
target_link_libraries(test_kmeans_progressive_prefix
    kmeans_lib
    utils_lib
    l2norm_lib
    paraselect_cuda
    CUDA::cublas
)
add_dependencies(test_kmeans_progressive_prefix kmeans_lib utils_lib l2norm_lib)
set_target_properties(test_kmeans_progressive_prefix PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(test_kmeans_progressive_prefix PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcompiler -std=c++17
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -std=c++17
    >
)
add_test(NAME kmeans_progressive_prefix_test COMMAND test_kmeans_progressive_prefix)
set_tests_properties(kmeans_progressive_prefix_test PROPERTIES
    LABELS "unit;cuda;clustering;kmeans;progressive-prefix"
)
add_executable(test_kmeans_fast_ivf
    test_kmeans_fast_ivf.cu
    ${CMAKE_SOURCE_DIR}/tests/common/test_utils.cu
)
target_link_libraries(test_kmeans_fast_ivf
    kmeans_lib
    utils_lib
    l2norm_lib
    paraselect_cuda
    CUDA::cublas
)
add_dependencies(test_kmeans_fast_ivf kmeans_lib utils_lib l2norm_lib)
set_target_properties(test_kmeans_fast_ivf PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(test_kmeans_fast_ivf PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcompiler -std=c++17
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -std=c++17
    >
)
add_test(NAME kmeans_fast_ivf_test COMMAND test_kmeans_fast_ivf)
set_tests_properties(kmeans_fast_ivf_test PROPERTIES
    LABELS "unit;cuda;clustering;kmeans;fast-ivf"
)

if(PARASELECT_ENABLE_CUVS)
    add_executable(test_kmeans_cagra_fp32
        test_kmeans_cagra_fp32.cu
    )
    target_link_libraries(test_kmeans_cagra_fp32
        kmeans_lib
        utils_lib
        l2norm_lib
        paraselect_cuda
    )
    set_target_properties(test_kmeans_cagra_fp32 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    add_test(NAME kmeans_cagra_fp32_test COMMAND test_kmeans_cagra_fp32)
    set_tests_properties(kmeans_cagra_fp32_test PROPERTIES
        LABELS "unit;cuda;clustering;kmeans;cagra"
    )
endif()


add_executable(test_wmma_gemm_vs_cublas
    test_wmma_gemm_vs_cublas.cu
    ${CMAKE_SOURCE_DIR}/tests/common/test_utils.cu
)
target_link_libraries(test_wmma_gemm_vs_cublas
    paraselect_cuda
    CUDA::cublas
)
set_target_properties(test_wmma_gemm_vs_cublas PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(test_wmma_gemm_vs_cublas PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcompiler -std=c++17
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -std=c++17
    >
)
add_test(NAME wmma_gemm_vs_cublas_test COMMAND test_wmma_gemm_vs_cublas)
set_tests_properties(wmma_gemm_vs_cublas_test PROPERTIES
    LABELS "benchmark;cuda;clustering;wmma"
)

add_executable(test_cutlass_gemm_vs_cublas
    test_cutlass_gemm_vs_cublas.cu
    ${CMAKE_SOURCE_DIR}/tests/common/test_utils.cu
)
target_include_directories(test_cutlass_gemm_vs_cublas PRIVATE
    ${PARASELECT_CUTLASS_INCLUDE_DIR}
    ${PARASELECT_CUTLASS_ROOT}/tools/util/include
)
target_link_libraries(test_cutlass_gemm_vs_cublas
    paraselect_cuda
    CUDA::cublas
)
set_target_properties(test_cutlass_gemm_vs_cublas PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(test_cutlass_gemm_vs_cublas PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcompiler -std=c++17
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -std=c++17
    >
)
add_test(NAME cutlass_gemm_vs_cublas_test COMMAND test_cutlass_gemm_vs_cublas)
set_tests_properties(cutlass_gemm_vs_cublas_test PROPERTIES
    LABELS "benchmark;cuda;clustering;cutlass"
)

# 1b. test_kmeans_aosoa: AoSoA
add_executable(test_kmeans_aosoa
    test_kmeans_aosoa.cu
    ${CMAKE_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/cpu_fine_layout.cpp
    ${CMAKE_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/cpu_fine_layout_aosoa.cpp
)
target_link_libraries(test_kmeans_aosoa
    paraselect_cuda
)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_kmeans_aosoa OpenMP::OpenMP_CXX)
endif()
set_target_properties(test_kmeans_aosoa PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(test_kmeans_aosoa PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcompiler -std=c++17
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -std=c++17
    >
)
add_test(NAME kmeans_aosoa_test COMMAND test_kmeans_aosoa)
set_tests_properties(kmeans_aosoa_test PROPERTIES
    LABELS "unit;cpu_fine;clustering;aosoa"
)

# 2. test_hierarchical (hierarchical )
add_executable(test_hierarchical
    test_hierarchical.cu
    ${CMAKE_SOURCE_DIR}/tests/common/test_utils.cu
)
target_link_libraries(test_hierarchical
    kmeans_lib
    utils_lib
    l2norm_lib
    paraselect_cuda
)
add_dependencies(test_hierarchical kmeans_lib utils_lib l2norm_lib)
set_target_properties(test_hierarchical PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(test_hierarchical PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcompiler -std=c++17
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -std=c++17
    >
)
add_test(NAME hierarchical_test COMMAND test_hierarchical)
set_tests_properties(hierarchical_test PROPERTIES
    LABELS "unit;cuda;clustering;hierarchical"
)

# 3. test_kmeans_fp16 (fp16 ) - ,
# add_executable(test_kmeans_fp16
#     test_kmeans_fp16.cu
#     ${CMAKE_SOURCE_DIR}/tests/common/test_utils.cu
# )
# target_link_libraries(test_kmeans_fp16
#     utils_lib
#     l2norm_lib
#     kmeans_lib
#     quantization_lib
#     interleaved_layout_lib
#     ${CUDA_LIBRARIES}
# )
# add_dependencies(test_kmeans_fp16 utils_lib l2norm_lib kmeans_lib quantization_lib interleaved_layout_lib)
# set_target_properties(test_kmeans_fp16 PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_RESOLVE_DEVICE_SYMBOLS ON
# )
# target_compile_options(test_kmeans_fp16 PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:
#         --expt-relaxed-constexpr
#         --expt-extended-lambda
#         -Xcompiler -std=c++17
#     >
#     $<$<COMPILE_LANGUAGE:CXX>:
#         -std=c++17
#     >
# )
# add_test(NAME kmeans_fp16_test COMMAND test_kmeans_fp16)

# 4. test_basic_hierarchical (basic hierarchical )
add_executable(test_basic_hierarchical
    test_basic_hierarchical.cu
    ${CMAKE_SOURCE_DIR}/tests/common/test_utils.cu
)
target_link_libraries(test_basic_hierarchical
    kmeans_lib
    utils_lib
    l2norm_lib
    paraselect_cuda
)
add_dependencies(test_basic_hierarchical kmeans_lib utils_lib l2norm_lib)
set_target_properties(test_basic_hierarchical PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(test_basic_hierarchical PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcompiler -std=c++17
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -std=c++17
    >
)
add_test(NAME basic_hierarchical_test COMMAND test_basic_hierarchical)
set_tests_properties(basic_hierarchical_test PROPERTIES
    LABELS "unit;cuda;clustering;hierarchical"
)
