add_executable(test_raft_include
    test_raft_include.cu
)

target_include_directories(test_raft_include PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(test_raft_include PRIVATE paraselect_rapids)

set_target_properties(test_raft_include PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${PARASELECT_CUDA_ARCHITECTURES}
)
add_test(NAME test_raft_include COMMAND test_raft_include)
set_tests_properties(test_raft_include PROPERTIES
    LABELS "smoke;unit;rapids"
)

add_executable(test_raft_stream
    test_raft_stream.cu
)

target_include_directories(test_raft_stream PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(test_raft_stream PRIVATE paraselect_rapids)

set_target_properties(test_raft_stream PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${PARASELECT_CUDA_ARCHITECTURES}
)
add_test(NAME test_raft_stream COMMAND test_raft_stream)
set_tests_properties(test_raft_stream PROPERTIES
    LABELS "smoke;unit;rapids;cuda"
)
