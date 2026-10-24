# =============================================================================
# GEMV
# =============================================================================

add_library(gemv_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/row_major/gemv.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/row_major/gemv_cos.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/row_major/gemv_l2.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/gemv_interleaved_cos.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/gemv_interleaved_l2.cu
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/gemv_interleaved_cos_fp16.cu
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/gemv_interleaved_l2_fp16.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/chunk/gemv_interleaved_cos.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/chunk/gemv_interleaved_l2.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/chunk/gemv_interleaved_cos_fp16.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/chunk/gemv_interleaved_l2_fp16.cu
)
target_include_directories(gemv_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/common
)
set_target_properties(gemv_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${PARASELECT_CUDA_ARCHITECTURES}
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(gemv_lib
    PUBLIC
    utils_lib
    interleaved_layout_lib
)
