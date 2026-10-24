add_library(ivf_cache STATIC
    ${CMAKE_SOURCE_DIR}/src/index/ivf_cache/kmeans.cu
    ${CMAKE_SOURCE_DIR}/src/index/ivf_cache/pca.cu
    ${CMAKE_SOURCE_DIR}/src/index/ivf_cache/plain_pq.cu
    ${CMAKE_SOURCE_DIR}/src/index/ivf_cache/residual_pq.cu
    ${CMAKE_SOURCE_DIR}/src/index/row_correction/row_correction.cpp
    ${CMAKE_SOURCE_DIR}/src/index/ivf_cache/source_vectors.cpp
)

target_include_directories(ivf_cache PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(ivf_cache PUBLIC
    dataset_source
    cluster_cache
    residual_pq_cache
    pca_cache
    kmeans_lib
    quantization_lib
    transform_lib
    paraselect_cuda
    CUDA::cublas
)

target_compile_features(ivf_cache PUBLIC cxx_std_17)

set_target_properties(ivf_cache PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${PARASELECT_CUDA_ARCHITECTURES}
)
