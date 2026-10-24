set(SOURCES
    test_resident_merge.cu
    ${CMAKE_SOURCE_DIR}/tests/common/test_utils.cu
)

add_executable(test_resident_merge ${SOURCES})
add_dependencies(test_resident_merge utils_lib)
target_include_directories(test_resident_merge PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests/common
)
target_link_libraries(test_resident_merge
    utils_lib
    paraselect_cuda
)
set_target_properties(test_resident_merge PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

enable_testing()
add_test(NAME test_resident_merge COMMAND test_resident_merge)
set_tests_properties(test_resident_merge PROPERTIES
    LABELS "unit;cuda;resident;merge"
)
