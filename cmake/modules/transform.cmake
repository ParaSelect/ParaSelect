# =============================================================================
# Transform library
# =============================================================================

add_library(transform_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/transform/orthogonal_projection_transform.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/transform/pca_transform.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/transform/opq_transform.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/transform/opq_trainer.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/transform/opq_cache.cu
)
target_include_directories(transform_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/common
)
set_target_properties(transform_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${PARASELECT_CUDA_ARCHITECTURES}
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(transform_lib PUBLIC
    paraselect_cuda
    CUDA::cublas
    CUDA::cusolver
    quantization_lib
    kmeans_lib
    utils_lib
)
