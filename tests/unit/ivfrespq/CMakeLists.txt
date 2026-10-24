add_executable(test_ivfrespq_timing test_ivfrespq_timing.cpp)

target_include_directories(test_ivfrespq_timing PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

add_test(NAME test_ivfrespq_timing COMMAND test_ivfrespq_timing)
set_tests_properties(test_ivfrespq_timing PROPERTIES LABELS "unit;ivfrespq")

add_executable(
    test_residual_pq_grouped_planner
    test_residual_pq_grouped_planner.cpp)
target_include_directories(test_residual_pq_grouped_planner PRIVATE
    ${CMAKE_SOURCE_DIR}/src)
add_test(
    NAME test_residual_pq_grouped_planner
    COMMAND test_residual_pq_grouped_planner)
set_tests_properties(
    test_residual_pq_grouped_planner
    PROPERTIES LABELS "unit;ivfrespq")

add_executable(
    test_residual_pq_workspace_budget
    test_residual_pq_workspace_budget.cpp)
target_include_directories(test_residual_pq_workspace_budget PRIVATE
    ${CMAKE_SOURCE_DIR}/src)
add_test(
    NAME test_residual_pq_workspace_budget
    COMMAND test_residual_pq_workspace_budget)
set_tests_properties(
    test_residual_pq_workspace_budget
    PROPERTIES LABELS "unit;ivfrespq")

add_executable(
    test_pq_query_lut_row_correction_segmented
    test_pq_query_lut_row_correction_segmented.cu)
target_include_directories(test_pq_query_lut_row_correction_segmented PRIVATE
    ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_pq_query_lut_row_correction_segmented PRIVATE
    ivf_cpu_search_lib
    ivf_common_search_lib ivf_pq_scan_kernels_lib CUDA::cudart)
set_target_properties(test_pq_query_lut_row_correction_segmented PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
add_test(
    NAME test_pq_query_lut_row_correction_segmented
    COMMAND test_pq_query_lut_row_correction_segmented)
set_tests_properties(
    test_pq_query_lut_row_correction_segmented
    PROPERTIES LABELS "unit;ivfrespq;gpu")

add_executable(
    test_pq_candidate_cap_select
    test_pq_candidate_cap_select.cu)
target_include_directories(test_pq_candidate_cap_select PRIVATE
    ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_pq_candidate_cap_select PRIVATE
    ivf_pq_scan_kernels_lib CUDA::cudart)
set_target_properties(test_pq_candidate_cap_select PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
add_test(
    NAME test_pq_candidate_cap_select
    COMMAND test_pq_candidate_cap_select)
set_tests_properties(
    test_pq_candidate_cap_select
    PROPERTIES LABELS "unit;ivfrespq;gpu")

add_executable(test_ivf_file_block_policy test_ivf_file_block_policy.cpp)
target_include_directories(test_ivf_file_block_policy PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests/common
)
add_test(NAME test_ivf_file_block_policy COMMAND test_ivf_file_block_policy)
set_tests_properties(test_ivf_file_block_policy PROPERTIES LABELS "unit;ivfrespq;file")

add_executable(test_ivf_file_rerank test_ivf_file_rerank.cpp)
target_include_directories(test_ivf_file_rerank PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_ivf_file_rerank PRIVATE
    ivf_file_rerank_lib dataset_source)
add_test(NAME test_ivf_file_rerank COMMAND test_ivf_file_rerank)
set_tests_properties(test_ivf_file_rerank PROPERTIES LABELS "unit;ivfrespq;file")

add_executable(test_res_row_correction test_res_row_correction.cpp)
target_include_directories(test_res_row_correction PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME test_res_row_correction COMMAND test_res_row_correction)
set_tests_properties(test_res_row_correction PROPERTIES LABELS "unit;res;row_correction")

add_executable(test_ivfrespq_host_rerank test_ivfrespq_host_rerank.cpp)
target_include_directories(test_ivfrespq_host_rerank PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_ivfrespq_host_rerank PRIVATE ivf_cpu_search_lib)
add_test(NAME test_ivfrespq_host_rerank COMMAND test_ivfrespq_host_rerank)
set_tests_properties(test_ivfrespq_host_rerank PROPERTIES LABELS "unit;ivfrespq;host")

add_executable(test_ivfrespq_prepare_host_dtype test_ivfrespq_prepare_host_dtype.cpp)
target_include_directories(test_ivfrespq_prepare_host_dtype PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_ivfrespq_prepare_host_dtype PRIVATE ivfrespq_search_lib)
add_test(NAME test_ivfrespq_prepare_host_dtype COMMAND test_ivfrespq_prepare_host_dtype)
set_tests_properties(test_ivfrespq_prepare_host_dtype PROPERTIES LABELS "unit;ivfrespq;gpu")

add_executable(test_ivf_pca_shortlist test_ivf_pca_shortlist.cpp)
target_include_directories(test_ivf_pca_shortlist PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_ivf_pca_shortlist PRIVATE ivf_pca_search_lib)
add_test(NAME test_ivf_pca_shortlist COMMAND test_ivf_pca_shortlist)
set_tests_properties(test_ivf_pca_shortlist PROPERTIES LABELS "unit;ivfrespq;pca")

add_executable(test_pq_resident_rerank test_pq_resident_rerank.cu)
target_include_directories(test_pq_resident_rerank PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_pq_resident_rerank PRIVATE
    ivf_pq_rerank_kernels_lib CUDA::cudart)
set_target_properties(test_pq_resident_rerank PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME test_pq_resident_rerank COMMAND test_pq_resident_rerank)
set_tests_properties(test_pq_resident_rerank PROPERTIES LABELS "unit;ivfrespq;resident")

add_executable(test_bruteforce_respq test_bruteforce_respq.cu)
target_include_directories(test_bruteforce_respq PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_bruteforce_respq PRIVATE ivfrespq_search_lib CUDA::cudart)
add_test(NAME test_bruteforce_respq COMMAND test_bruteforce_respq)

add_executable(test_query_lut_workspace test_query_lut_workspace.cu)
target_include_directories(test_query_lut_workspace PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_query_lut_workspace PRIVATE ivfrespq_search_lib)
add_test(NAME test_query_lut_workspace COMMAND test_query_lut_workspace)
set_tests_properties(test_query_lut_workspace PROPERTIES LABELS "unit;ivfrespq;gpu")

add_executable(test_row_correction_u16 ../quantization/test_row_correction_u16.cpp)
target_include_directories(test_row_correction_u16 PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME test_row_correction_u16 COMMAND test_row_correction_u16)
set_tests_properties(test_row_correction_u16 PROPERTIES LABELS "unit;quantization;row_correction")
