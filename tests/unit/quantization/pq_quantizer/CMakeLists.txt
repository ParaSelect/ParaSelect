# test_pq_quantizer:PQ  encode/decode , test_linear_fp16(--cpu/--gpu)
cmake_minimum_required(VERSION 3.18)
project(test_pq_quantizer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES ${PARASELECT_CUDA_ARCHITECTURES})

set(SOURCES test_pq_quantizer.cu)
add_executable(test_pq_quantizer ${SOURCES})
add_dependencies(test_pq_quantizer quantization_lib)

target_include_directories(test_pq_quantizer PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/common
)
target_link_libraries(test_pq_quantizer
    quantization_lib
    paraselect_cuda
)

set_target_properties(test_pq_quantizer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

enable_testing()
add_test(NAME pq_quantizer_test COMMAND test_pq_quantizer)
set_tests_properties(pq_quantizer_test PROPERTIES
    LABELS "unit;cuda;quantization"
)

add_test(NAME pq_quantizer_perf_test COMMAND test_pq_quantizer --perf)
set_tests_properties(pq_quantizer_perf_test PROPERTIES
    LABELS "benchmark;unit_perf;cuda;quantization"
    DISABLED TRUE
)
