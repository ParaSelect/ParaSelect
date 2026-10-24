add_executable(test_dataset_spec test_dataset_spec.cpp)

target_link_libraries(test_dataset_spec PRIVATE dataset_source cluster_cache residual_pq_cache)

target_include_directories(test_dataset_spec PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

add_test(NAME test_dataset_spec COMMAND test_dataset_spec)
set_tests_properties(test_dataset_spec PROPERTIES
    LABELS "unit;dataset;kmeans;cache"
)

add_executable(test_pca_cache test_pca_cache.cpp)
target_include_directories(test_pca_cache PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_pca_cache PRIVATE pca_cache)
add_test(NAME test_pca_cache COMMAND test_pca_cache)
set_tests_properties(test_pca_cache PROPERTIES LABELS "unit;dataset;cache;pca")

add_executable(test_temporary_mapping test_temporary_mapping.cpp)
target_include_directories(test_temporary_mapping PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME test_temporary_mapping COMMAND test_temporary_mapping)
set_tests_properties(test_temporary_mapping PROPERTIES LABELS "unit;dataset")

add_executable(test_sequential_materializer test_sequential_materializer.cpp)
target_include_directories(test_sequential_materializer PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_sequential_materializer PRIVATE dataset_source)
add_test(NAME test_sequential_materializer COMMAND test_sequential_materializer)
set_tests_properties(test_sequential_materializer PROPERTIES LABELS "unit;dataset")

add_executable(test_manifest_cache test_manifest_cache.cpp)
target_include_directories(test_manifest_cache PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_manifest_cache PRIVATE residual_pq_cache)
add_test(NAME test_manifest_cache COMMAND test_manifest_cache)
set_tests_properties(test_manifest_cache PROPERTIES LABELS "unit;dataset;cache")

add_executable(test_row_correction_manifest test_row_correction_manifest.cpp
    ${CMAKE_SOURCE_DIR}/src/index/row_correction/row_correction.cpp)
target_include_directories(test_row_correction_manifest PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party)
add_test(NAME test_row_correction_manifest COMMAND test_row_correction_manifest)
set_tests_properties(test_row_correction_manifest PROPERTIES LABELS "unit;dataset;cache")

add_executable(test_kmeans_cache_policy test_kmeans_cache_policy.cu)
target_include_directories(test_kmeans_cache_policy PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_kmeans_cache_policy PRIVATE ivf_cache)
add_test(NAME test_kmeans_cache_policy COMMAND test_kmeans_cache_policy)
set_tests_properties(test_kmeans_cache_policy PROPERTIES LABELS "unit;dataset;kmeans;cache")

add_executable(test_cluster_manifest_publish test_cluster_manifest_publish.cpp)
target_include_directories(test_cluster_manifest_publish PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(test_cluster_manifest_publish PRIVATE cluster_cache)
add_test(NAME test_cluster_manifest_publish COMMAND test_cluster_manifest_publish)
set_tests_properties(test_cluster_manifest_publish PROPERTIES LABELS "unit;dataset;cache")

add_executable(verify_migrated_cache_assets verify_migrated_cache_assets.cpp
    ${CMAKE_SOURCE_DIR}/src/index/row_correction/row_correction.cpp)
target_include_directories(verify_migrated_cache_assets PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(verify_migrated_cache_assets PRIVATE cluster_cache residual_pq_cache turboquant_cache pca_cache)

add_executable(test_plain_pq_mapped test_plain_pq_mapped.cu)
target_include_directories(test_plain_pq_mapped PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_plain_pq_mapped PRIVATE ivfpq_search_lib ivf_cache)
add_test(NAME test_plain_pq_mapped COMMAND test_plain_pq_mapped)
