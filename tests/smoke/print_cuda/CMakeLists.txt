# test_print_cudaCMake

#
set(SOURCES
    test_print_cuda.cu
    ${CMAKE_SOURCE_DIR}/tests/common/test_utils.cu
)

#
add_executable(test_print_cuda ${SOURCES})
add_dependencies(test_print_cuda utils_lib)

#
target_include_directories(test_print_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/tests/common
)

# CUDA
set_target_properties(test_print_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_print_cuda PRIVATE
    utils_lib
    paraselect_cuda
)

#
enable_testing()
add_test(NAME print_cuda_test COMMAND test_print_cuda)
set_tests_properties(print_cuda_test PROPERTIES
    LABELS "smoke;unit;cuda"
)
