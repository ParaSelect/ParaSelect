# SIFT Dataset Integration Test

add_executable(test_sift test_sift.cu)

target_link_libraries(test_sift
    dataset_source
    binary_format
)

target_include_directories(test_sift PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src
)

set_target_properties(test_sift PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

#
add_test(NAME test_sift COMMAND test_sift)
set_tests_properties(test_sift PROPERTIES
    LABELS "integration;dataset;io"
)
