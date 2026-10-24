# =============================================================================
#
# =============================================================================

add_library(utils_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/utils.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/cuda_fill.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/cuda_scan.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/check_cuda_error.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark/search_benchmark.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark/recall_evaluator.cpp
)
target_include_directories(utils_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/common
)

target_link_libraries(utils_lib PUBLIC
    Threads::Threads
    cluster_cache
    dataset_source
)
set_target_properties(utils_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${PARASELECT_CUDA_ARCHITECTURES}
    POSITION_INDEPENDENT_CODE ON
)
