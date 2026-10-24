# CUDA test modules
add_subdirectory(tests/smoke/print_cuda)
add_subdirectory(tests/smoke/config)
add_subdirectory(tests/smoke/numa)
if(PARASELECT_ENABLE_CUVS)
  add_subdirectory(tests/smoke/raft)
endif()

add_subdirectory(tests/unit/bitonic)
add_subdirectory(tests/unit/clustering)
add_subdirectory(tests/unit/dataset)
add_subdirectory(tests/unit/fusion_dist_topk)
add_subdirectory(tests/unit/gemv_interleaved)
add_subdirectory(tests/unit/ivf_common)
add_subdirectory(tests/unit/ivfrespq)
add_subdirectory(tests/unit/turboquant)
add_subdirectory(tests/unit/quantization)
add_subdirectory(tests/unit/stats)
add_subdirectory(tests/unit/transform)
add_subdirectory(tests/unit/warpsort)
add_subdirectory(tests/unit/resident_merge)

add_subdirectory(tests/integration/dataset)
add_subdirectory(tests/integration/ivfflat_search)

# Selected unit-test target
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS
        test_gemv_interleaved
        test_gemv_interleaved_fp16
        test_gemv_interleaved_pipeline
        test_gemv_interleaved_pipeline_chunk
        test_fusion_dist_topk_interleaved
        test_warpsort
        test_bitonic
        test_kmeans
    COMMENT "Build the selected CUDA unit tests"
)
