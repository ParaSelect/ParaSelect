# =============================================================================
# Reorder
# =============================================================================

add_library(reorder_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reorder/standard_reorder.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reorder/numa_reorder.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reorder/inplace_reorder.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reorder/cluster_reorder.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reorder/external_reorder.cu
)
target_include_directories(reorder_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/common
)
set_target_properties(reorder_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${PARASELECT_CUDA_ARCHITECTURES}
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(reorder_lib PUBLIC
    paraselect_cuda
    utils_lib
    l2norm_lib
    Threads::Threads
)
