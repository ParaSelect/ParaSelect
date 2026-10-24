cmake_minimum_required(VERSION 3.18)
project(test_ivfpq_resident_no_rerank LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES ${PARASELECT_CUDA_ARCHITECTURES})

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/tests/common)

add_executable(test_ivfpq_resident_no_rerank
    test_ivfpq_resident_no_rerank.cu
)
target_link_libraries(test_ivfpq_resident_no_rerank
    PRIVATE
    utils_lib
    l2norm_lib
    paraselect_cuda
    CUDA::cublas
    gemv_lib
    ivfpq_search_lib
    fusion_dist_topk_lib
)
set_target_properties(test_ivfpq_resident_no_rerank PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
add_test(NAME test_ivfpq_resident_no_rerank COMMAND test_ivfpq_resident_no_rerank)
set_tests_properties(test_ivfpq_resident_no_rerank PROPERTIES
    LABELS "integration;cuda;ivfpq;synthetic"
)
