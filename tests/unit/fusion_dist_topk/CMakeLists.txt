# test_fusion_dist_topkCMake

cmake_minimum_required(VERSION 3.18)
project(test_fusion_dist_topk_interleaved LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 70,,
set(CMAKE_CUDA_ARCHITECTURES ${PARASELECT_CUDA_ARCHITECTURES})

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/tests/common)

enable_testing()

# test_fusion_dist_topk_interleaved (interleaved)
set(SOURCES_INTERLEAVED
    test_fusion_dist_topk_interleaved.cu
    ${CMAKE_SOURCE_DIR}/tests/common/test_utils.cu
)
add_executable(test_fusion_dist_topk_interleaved ${SOURCES_INTERLEAVED})
target_link_libraries(test_fusion_dist_topk_interleaved
    gemv_lib
    utils_lib
    paraselect_cuda
)
set_target_properties(test_fusion_dist_topk_interleaved PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
add_test(NAME fusion_dist_topk_interleaved_test COMMAND test_fusion_dist_topk_interleaved)
set_tests_properties(fusion_dist_topk_interleaved_test PROPERTIES
    LABELS "unit;cuda;topk;fusion;interleaved"
)
