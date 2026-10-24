# test_bitonicCMake

#
set(SOURCES
    test_bitonic.cu
    ${CMAKE_SOURCE_DIR}/tests/common/test_utils.cu
)

#
add_executable(test_bitonic ${SOURCES})

#
add_dependencies(test_bitonic utils_lib)

#
target_include_directories(test_bitonic PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests/common
)

# CUDA
set_target_properties(test_bitonic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# CUDAutils
target_link_libraries(test_bitonic
    utils_lib
    paraselect_cuda
)

#
enable_testing()
add_test(NAME bitonic_test COMMAND test_bitonic)
set_tests_properties(bitonic_test PROPERTIES
    LABELS "unit;cuda;topk;sort"
)
