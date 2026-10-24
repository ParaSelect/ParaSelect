# test_warp_sortCMake

#
set(SOURCES
    test_warpsort.cu
    ${CMAKE_SOURCE_DIR}/tests/common/test_utils.cu
)

#
add_executable(test_warpsort ${SOURCES})

#
add_dependencies(test_warpsort utils_lib)

#
target_include_directories(test_warpsort PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests/common
)

# CUDA
set_target_properties(test_warpsort PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# CUDAutils
target_link_libraries(test_warpsort
    utils_lib
    fusion_dist_topk_lib
    paraselect_cuda
)

#
enable_testing()
add_test(NAME warpsort_test COMMAND test_warpsort)
set_tests_properties(warpsort_test PROPERTIES
    LABELS "unit;cuda;topk;sort"
)

