# =============================================================================
# Stats
# =============================================================================

# (,)
add_executable(test_kmeans_stats_core
    test_kmeans_stats_core.cu
)
target_link_libraries(test_kmeans_stats_core
    PRIVATE
    kmeans_lib
)
set_target_properties(test_kmeans_stats_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
add_dependencies(test_kmeans_stats_core kmeans_lib)

# ( SIFT/DEEP/TEXT )
add_executable(test_kmeans_stats
    test_kmeans_stats.cu
)
target_link_libraries(test_kmeans_stats
    PRIVATE
    kmeans_lib
    dataset_source
)
set_target_properties(test_kmeans_stats PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
add_dependencies(test_kmeans_stats kmeans_lib)

add_test(NAME test_kmeans_stats_core COMMAND test_kmeans_stats_core)
set_tests_properties(test_kmeans_stats_core PROPERTIES
    LABELS "unit;stats;clustering"
)

add_executable(test_search_csv_precision test_search_csv_precision.cu)
target_link_libraries(test_search_csv_precision PRIVATE utils_lib)
add_test(NAME test_search_csv_precision COMMAND test_search_csv_precision)
set_tests_properties(test_search_csv_precision PROPERTIES LABELS "unit;stats;host")
