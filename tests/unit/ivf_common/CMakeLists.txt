add_executable(test_ivf_list_layout test_ivf_list_layout.cpp)

target_include_directories(test_ivf_list_layout PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

add_test(NAME test_ivf_list_layout COMMAND test_ivf_list_layout)
set_tests_properties(test_ivf_list_layout PROPERTIES LABELS "unit;ivf")

add_executable(test_ivf_coarse_device test_ivf_coarse_device.cu)
target_include_directories(test_ivf_coarse_device PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_ivf_coarse_device PRIVATE
    ivf_common_search_lib CUDA::cudart)
set_target_properties(test_ivf_coarse_device PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME test_ivf_coarse_device COMMAND test_ivf_coarse_device)
set_tests_properties(test_ivf_coarse_device PROPERTIES LABELS "unit;ivf;cuda")

add_executable(test_ivf_probe_schedule test_ivf_probe_schedule.cu)
target_include_directories(test_ivf_probe_schedule PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_ivf_probe_schedule PRIVATE
    ivf_common_search_lib CUDA::cudart)
set_target_properties(test_ivf_probe_schedule PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME test_ivf_probe_schedule COMMAND test_ivf_probe_schedule)
set_tests_properties(test_ivf_probe_schedule PROPERTIES LABELS "unit;ivf;cuda;metadata")

add_executable(test_persistent_thread_pool_lifetime
    test_persistent_thread_pool_lifetime.cpp)
target_include_directories(test_persistent_thread_pool_lifetime PRIVATE
    ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_persistent_thread_pool_lifetime PRIVATE pthread)
add_test(NAME test_persistent_thread_pool_lifetime
    COMMAND test_persistent_thread_pool_lifetime)
set_tests_properties(test_persistent_thread_pool_lifetime PROPERTIES
    LABELS "unit;cpu;thread-pool")

add_executable(test_ivf_host_rerank test_ivf_host_rerank.cpp)
target_include_directories(test_ivf_host_rerank PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_ivf_host_rerank PRIVATE ivf_cpu_search_lib)
add_test(NAME test_ivf_host_rerank COMMAND test_ivf_host_rerank)
set_tests_properties(test_ivf_host_rerank PROPERTIES LABELS "unit;ivf;host")

add_executable(test_ivfpq_rerank_adapter test_ivfpq_rerank_adapter.cpp
    ${CMAKE_SOURCE_DIR}/src/search/ivfpq/stage/ivfpq_rerank_cpu.cpp)
target_include_directories(test_ivfpq_rerank_adapter PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_ivfpq_rerank_adapter PRIVATE
    ivf_common_search_lib ivf_cpu_search_lib CUDA::cudart CUDA::cublas)
add_test(NAME test_ivfpq_rerank_adapter COMMAND test_ivfpq_rerank_adapter)

add_executable(test_pq_query_lut_modes test_pq_query_lut_modes.cu
    ${CMAKE_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/pq_adc/pq_query_lut.cu)
target_include_directories(test_pq_query_lut_modes PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_pq_query_lut_modes PRIVATE CUDA::cudart)
add_test(NAME test_pq_query_lut_modes COMMAND test_pq_query_lut_modes)

add_executable(test_plain_pq_pipeline test_plain_pq_pipeline.cu)
target_include_directories(test_plain_pq_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_plain_pq_pipeline PRIVATE ivfrespq_search_lib)
add_test(NAME test_plain_pq_pipeline COMMAND test_plain_pq_pipeline)

set_tests_properties(test_plain_pq_pipeline PROPERTIES TIMEOUT 120)

add_executable(test_residual_lut_histogram test_residual_lut_histogram.cu)
target_include_directories(test_residual_lut_histogram PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_residual_lut_histogram PRIVATE ivf_common_search_lib ivf_pq_scan_kernels_lib CUDA::cudart)
set_target_properties(test_residual_lut_histogram PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME test_residual_lut_histogram COMMAND test_residual_lut_histogram)
