# =============================================================================
# KMeans
# =============================================================================

add_library(kmeans_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/utils.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/flash_kmeans/flash_kmeans_assign.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/flash_kmeans/flash_kmeans_assign_v8.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/kmeans/kmeans.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/kmeans/kmeans_minibatch.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/kmeans/kmeans_progressive_prefix.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/kmeans/kmeans_fast_ivf.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/kmeans/kmeans_cluster_only.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/kmeans/kmeans_cagra_train.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/kmeans/kmeans_final_assign.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/kmeans/kmeans_multigpu_assign.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/kmeans/kmeans_multigpu_train.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/kmeans/kmeans_utils.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/kmeans/kmeans_reorder.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/kmeans/kmeans_reorder_inplace.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/kmeans/kmeans_clustering.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/kmeans/kmeans_clustering_inplace.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/kmeans/kmeans_stats.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/hierarchical/hierarchical.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/hierarchical/basic_hierarchical.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/hierarchical/naive_hierarchical.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering/hierarchical/hierarchical_parallel_l2_kernels.cu
)

target_include_directories(kmeans_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/common
)
set(PARASELECT_CUTLASS_ROOT "" CACHE PATH "CUTLASS source or installation prefix")
if(NOT PARASELECT_CUTLASS_ROOT AND DEFINED ENV{PARASELECT_CUTLASS_ROOT})
    set(PARASELECT_CUTLASS_ROOT "$ENV{PARASELECT_CUTLASS_ROOT}")
endif()
find_path(PARASELECT_CUTLASS_INCLUDE_DIR
    NAMES cutlass/cutlass.h
    HINTS
        "${PARASELECT_CUTLASS_ROOT}"
        "$ENV{CONDA_PREFIX}"
        "$ENV{PARASELECT_CONDA_ENV}"
    PATH_SUFFIXES include
    REQUIRED)
target_include_directories(kmeans_lib PUBLIC "${PARASELECT_CUTLASS_INCLUDE_DIR}")
if(EXISTS "${PARASELECT_CUTLASS_ROOT}/tools/util/include")
    target_include_directories(kmeans_lib PUBLIC
        "${PARASELECT_CUTLASS_ROOT}/tools/util/include")
endif()
set_target_properties(kmeans_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${PARASELECT_CUDA_ARCHITECTURES}
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(kmeans_lib PUBLIC
    paraselect_cuda
    utils_lib
    l2norm_lib
    reorder_lib
    Threads::Threads
    CUDA::cublas
)
if(PARASELECT_ENABLE_CUVS)
    target_link_libraries(kmeans_lib PUBLIC paraselect_rapids)
endif()
