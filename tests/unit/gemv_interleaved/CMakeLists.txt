# test_gemv_interleaved

# 1. test_gemv_interleaved (FP32 )
set(SOURCES
    test_gemv_interleaved.cu
    ${CMAKE_SOURCE_DIR}/tests/common/test_utils.cu
)
add_executable(test_gemv_interleaved ${SOURCES})
target_include_directories(test_gemv_interleaved PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests/common
)
target_link_libraries(test_gemv_interleaved
    PRIVATE
    gemv_lib
    utils_lib
    paraselect_cuda
)
set_target_properties(test_gemv_interleaved PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
enable_testing()
add_test(NAME test_gemv_interleaved COMMAND test_gemv_interleaved)
set_tests_properties(test_gemv_interleaved PROPERTIES
    LABELS "unit;cuda;gemv;interleaved"
)

# 2. test_gemv_interleaved_fp16 (FP16 )
set(SOURCES_FP16
    test_gemv_interleaved_fp16.cu
    ${CMAKE_SOURCE_DIR}/tests/common/test_utils.cu
)
add_executable(test_gemv_interleaved_fp16 ${SOURCES_FP16})
target_include_directories(test_gemv_interleaved_fp16 PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests/common
)
target_link_libraries(test_gemv_interleaved_fp16
    PRIVATE
    gemv_lib
    utils_lib
    paraselect_cuda
)
set_target_properties(test_gemv_interleaved_fp16 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
add_test(NAME test_gemv_interleaved_fp16 COMMAND test_gemv_interleaved_fp16)
set_tests_properties(test_gemv_interleaved_fp16 PROPERTIES
    LABELS "unit;cuda;gemv;fp16;interleaved"
)

# 3. test_gemv_interleaved_pipeline ()
set(SOURCES_PIPELINE
    test_gemv_interleaved_pipeline.cu
    ${CMAKE_SOURCE_DIR}/tests/common/test_utils.cu
)
add_executable(test_gemv_interleaved_pipeline ${SOURCES_PIPELINE})
target_include_directories(test_gemv_interleaved_pipeline PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests/common
)
target_link_libraries(test_gemv_interleaved_pipeline
    PRIVATE
    gemv_lib
    l2norm_lib
    utils_lib
    paraselect_cuda
    pthread
)
if(PARASELECT_USE_NVTX)
    target_compile_definitions(test_gemv_interleaved_pipeline PRIVATE USE_NVTX=1)
    target_link_libraries(test_gemv_interleaved_pipeline PRIVATE paraselect_nvtx)
else()
    target_compile_definitions(test_gemv_interleaved_pipeline PRIVATE USE_NVTX=0)
endif()
set_target_properties(test_gemv_interleaved_pipeline PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
add_test(NAME test_gemv_interleaved_pipeline COMMAND test_gemv_interleaved_pipeline)
set_tests_properties(test_gemv_interleaved_pipeline PROPERTIES
    LABELS "integration;cuda;gemv;pipeline"
)

# 4. test_gemv_interleaved_pipeline_chunk (chunk )
set(SOURCES_PIPELINE_CHUNK
    test_gemv_interleaved_pipeline_chunk.cu
    ${CMAKE_SOURCE_DIR}/tests/common/test_utils.cu
)
add_executable(test_gemv_interleaved_pipeline_chunk ${SOURCES_PIPELINE_CHUNK})
target_include_directories(test_gemv_interleaved_pipeline_chunk PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests/common
)
target_link_libraries(test_gemv_interleaved_pipeline_chunk
    PRIVATE
    gemv_lib
    l2norm_lib
    utils_lib
    paraselect_cuda
    pthread
)
if(PARASELECT_USE_NVTX)
    target_compile_definitions(test_gemv_interleaved_pipeline_chunk PRIVATE USE_NVTX=1)
    target_link_libraries(test_gemv_interleaved_pipeline_chunk PRIVATE paraselect_nvtx)
else()
    target_compile_definitions(test_gemv_interleaved_pipeline_chunk PRIVATE USE_NVTX=0)
endif()
set_target_properties(test_gemv_interleaved_pipeline_chunk PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
add_test(NAME test_gemv_interleaved_pipeline_chunk COMMAND test_gemv_interleaved_pipeline_chunk)
set_tests_properties(test_gemv_interleaved_pipeline_chunk PROPERTIES
    LABELS "integration;cuda;gemv;pipeline"
)
