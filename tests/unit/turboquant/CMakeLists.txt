add_executable(
    test_turboquant_1bit_format
    test_turboquant_1bit_format.cpp)
target_include_directories(test_turboquant_1bit_format PRIVATE
    ${CMAKE_SOURCE_DIR}/src)
target_compile_options(test_turboquant_1bit_format PRIVATE -UNDEBUG)
add_test(
    NAME test_turboquant_1bit_format
    COMMAND test_turboquant_1bit_format)
set_tests_properties(
    test_turboquant_1bit_format
    PROPERTIES LABELS "unit;turboquant")

add_executable(
    test_turboquant_1bit_scan
    test_turboquant_1bit_scan.cu)
target_include_directories(test_turboquant_1bit_scan PRIVATE
    ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_turboquant_1bit_scan PRIVATE
    turboquant_search_lib CUDA::cudart)
target_compile_options(test_turboquant_1bit_scan PRIVATE -UNDEBUG)
set_target_properties(test_turboquant_1bit_scan PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
add_test(
    NAME test_turboquant_1bit_scan
    COMMAND test_turboquant_1bit_scan)
set_tests_properties(
    test_turboquant_1bit_scan
    PROPERTIES LABELS "unit;turboquant;gpu")

add_executable(
    test_turboquant_1bit_encode
    test_turboquant_1bit_encode.cu)
target_include_directories(test_turboquant_1bit_encode PRIVATE
    ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_turboquant_1bit_encode PRIVATE
    turboquant_search_lib CUDA::cudart)
target_compile_options(test_turboquant_1bit_encode PRIVATE -UNDEBUG)
set_target_properties(test_turboquant_1bit_encode PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
add_test(
    NAME test_turboquant_1bit_encode
    COMMAND test_turboquant_1bit_encode)
set_tests_properties(
    test_turboquant_1bit_encode
    PROPERTIES LABELS "unit;turboquant;gpu")

add_executable(test_turboquant_shared_threshold test_turboquant_shared_threshold.cu)
target_include_directories(test_turboquant_shared_threshold PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_turboquant_shared_threshold PRIVATE residual_codec_lib CUDA::cudart)
set_target_properties(test_turboquant_shared_threshold PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME test_turboquant_shared_threshold COMMAND test_turboquant_shared_threshold)
set_tests_properties(
    test_turboquant_shared_threshold
    PROPERTIES LABELS "unit;turboquant;gpu")
