cmake_minimum_required(VERSION 3.18)
project(test_residual_pq_trainer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES ${PARASELECT_CUDA_ARCHITECTURES})

add_executable(test_residual_pq_trainer test_residual_pq_trainer.cu)
add_dependencies(test_residual_pq_trainer quantization_lib)

target_include_directories(test_residual_pq_trainer PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/common
)

target_link_libraries(test_residual_pq_trainer
    quantization_lib
    paraselect_cuda
)

set_target_properties(test_residual_pq_trainer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

enable_testing()
add_test(NAME residual_pq_trainer_test COMMAND test_residual_pq_trainer)
set_tests_properties(residual_pq_trainer_test PROPERTIES
    LABELS "unit;quantization;pq"
)
