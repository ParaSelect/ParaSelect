# test_linear_fp16:fp16  encode/decode
cmake_minimum_required(VERSION 3.18)
project(test_linear_fp16 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES ${PARASELECT_CUDA_ARCHITECTURES})

set(SOURCES test_linear_fp16.cu)
add_executable(test_linear_fp16 ${SOURCES})
add_dependencies(test_linear_fp16 quantization_lib)

target_include_directories(test_linear_fp16 PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/common
)
target_link_libraries(test_linear_fp16
    quantization_lib
    paraselect_cuda
)

set_target_properties(test_linear_fp16 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

enable_testing()
add_test(NAME linear_fp16_test COMMAND test_linear_fp16)
set_tests_properties(linear_fp16_test PROPERTIES
    LABELS "unit;cuda;quantization"
)

add_test(NAME linear_fp16_perf_test COMMAND test_linear_fp16 --perf)
set_tests_properties(linear_fp16_perf_test PROPERTIES
    LABELS "benchmark;unit_perf;cuda;quantization"
    DISABLED TRUE
)
