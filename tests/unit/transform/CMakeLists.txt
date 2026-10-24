cmake_minimum_required(VERSION 3.18)
project(test_transform LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES ${PARASELECT_CUDA_ARCHITECTURES})

add_executable(test_orthogonal_projection_transform
    test_orthogonal_projection_transform.cu
    ${CMAKE_SOURCE_DIR}/tests/common/test_utils.cu
)
target_include_directories(test_orthogonal_projection_transform PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests/common
)
target_link_libraries(test_orthogonal_projection_transform PRIVATE
    transform_lib
    utils_lib
    paraselect_cuda
    CUDA::cublas
)
add_dependencies(test_orthogonal_projection_transform transform_lib utils_lib)
set_target_properties(test_orthogonal_projection_transform PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_test(NAME test_orthogonal_projection_transform COMMAND test_orthogonal_projection_transform)
set_tests_properties(test_orthogonal_projection_transform PROPERTIES
    LABELS "unit;cuda;transform;ivfjl"
)

add_executable(test_pca_transform test_pca_transform.cu)
target_include_directories(test_pca_transform PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_pca_transform PRIVATE
    transform_lib
    paraselect_cuda
    CUDA::cublas
    CUDA::cusolver
)
set_target_properties(test_pca_transform PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
add_test(NAME test_pca_transform COMMAND test_pca_transform)
set_tests_properties(test_pca_transform PROPERTIES LABELS "unit;cuda;transform;pca")


add_executable(test_opq_transform
    test_opq_transform.cu
    ${CMAKE_SOURCE_DIR}/tests/common/test_utils.cu
)
target_include_directories(test_opq_transform PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests/common
)
target_link_libraries(test_opq_transform PRIVATE
    transform_lib
    kmeans_lib
    utils_lib
    paraselect_cuda
    CUDA::cublas
    CUDA::cusolver
)
add_dependencies(test_opq_transform transform_lib kmeans_lib utils_lib)
set_target_properties(test_opq_transform PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_test(NAME test_opq_transform COMMAND test_opq_transform)
set_tests_properties(test_opq_transform PROPERTIES
    LABELS "unit;cuda;transform;opq"
)
