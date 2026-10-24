# =============================================================================
# Quantization
# FP16  + PQ (GPU + CPU)
# =============================================================================

add_library(quantization_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/linear/linear_fp16.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/linear/linear_fp16_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/linear/prepare_fp16_host.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/pq/pq_kmeans.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/pq/pq_trainer.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/pq/pq_encoder.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/pq/pq_quantizer_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/pq/pq_quantizer_gpu.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/pq/residual_pq_multigpu_trainer.cu
)
target_include_directories(quantization_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/common
)
set_target_properties(quantization_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${PARASELECT_CUDA_ARCHITECTURES}
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(quantization_lib PUBLIC
    paraselect_cuda
    kmeans_lib
    utils_lib
    CUDA::cublas
)
