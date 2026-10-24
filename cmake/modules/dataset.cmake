# Dataset Reader Modules

message(STATUS "Configuring Dataset Reader modules")

# ============================================================================
# Common vector-file I/O and binary format reader
# ============================================================================
message(STATUS "  - Vector File IO")

add_library(vector_file_io STATIC
    ${CMAKE_SOURCE_DIR}/src/dataset/common/vector_file_io.cpp
)

set_target_properties(vector_file_io PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(vector_file_io PUBLIC
    ${CMAKE_SOURCE_DIR}/src/dataset/common
    ${CMAKE_SOURCE_DIR}/src
)

message(STATUS "  - Binary Format Reader")

add_library(binary_format STATIC
    ${CMAKE_SOURCE_DIR}/src/dataset/common/binary_format.cpp
)

set_target_properties(binary_format PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(binary_format PUBLIC
    ${CMAKE_SOURCE_DIR}/src/dataset/common
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(binary_format PUBLIC
    vector_file_io
    paraselect_cuda
)

# ============================================================================
# Cluster Cache Module
# ============================================================================
message(STATUS "  - Cluster Cache")

add_library(cluster_cache STATIC
    ${CMAKE_SOURCE_DIR}/src/dataset/cache/cluster_cache.cpp
    ${CMAKE_SOURCE_DIR}/src/dataset/common/npy_file.cpp
)

set_target_properties(cluster_cache PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(cluster_cache PUBLIC
    ${CMAKE_SOURCE_DIR}/src/dataset/common
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(cluster_cache PUBLIC
    binary_format
)


# ============================================================================
# Residual PQ Cache Module
# ============================================================================
message(STATUS "  - Residual PQ Cache")

add_library(residual_pq_cache STATIC
    ${CMAKE_SOURCE_DIR}/src/dataset/cache/residual_pq_cache.cpp
)

set_target_properties(residual_pq_cache PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(residual_pq_cache PUBLIC
    ${CMAKE_SOURCE_DIR}/src/dataset/common
    ${CMAKE_SOURCE_DIR}/src
)

add_library(turboquant_cache STATIC
    ${CMAKE_SOURCE_DIR}/src/dataset/cache/turboquant_cache.cpp
)

set_target_properties(turboquant_cache PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(turboquant_cache PUBLIC
    ${CMAKE_SOURCE_DIR}/src/dataset/common
    ${CMAKE_SOURCE_DIR}/src
)

message(STATUS "  - PCA Cache")

add_library(pca_cache STATIC
    ${CMAKE_SOURCE_DIR}/src/dataset/cache/pca_cache.cpp
)

set_target_properties(pca_cache PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(pca_cache PUBLIC
    ${CMAKE_SOURCE_DIR}/src/dataset/common
    ${CMAKE_SOURCE_DIR}/src
)

add_library(dataset_source STATIC
    ${CMAKE_SOURCE_DIR}/src/dataset/common/dataset_source.cpp
    ${CMAKE_SOURCE_DIR}/src/dataset/temporary_exact_store.cpp
)
target_include_directories(dataset_source
    PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/dataset/common)
target_link_libraries(dataset_source PUBLIC binary_format CUDA::cudart)

message(STATUS "Dataset Reader modules configured")


# Structured cache manifests use the vendored, header-only JSON parser.
target_include_directories(cluster_cache SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/third_party)
target_include_directories(residual_pq_cache SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/third_party)
target_include_directories(turboquant_cache SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/third_party)
target_include_directories(pca_cache SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/third_party)
