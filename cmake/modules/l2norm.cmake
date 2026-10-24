# =============================================================================
# L2 Norm
# =============================================================================

add_library(l2norm_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/l2norm/l2norm.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/l2norm/l2norm_squared.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/l2norm/l2norm_interleaved.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/l2norm/l2norm_interleaved_fp16.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/utils/l2norm_cpu.cpp
)
target_include_directories(l2norm_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/common
)
target_link_libraries(l2norm_lib PUBLIC paraselect_cuda)
set_target_properties(l2norm_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${PARASELECT_CUDA_ARCHITECTURES}
    POSITION_INDEPENDENT_CODE ON
)
