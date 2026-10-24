add_executable(test_numa_affinity test_numa_affinity.cpp)

target_include_directories(test_numa_affinity PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

add_test(NAME test_numa_affinity COMMAND test_numa_affinity)
set_tests_properties(test_numa_affinity PROPERTIES
    LABELS "smoke;unit;numa"
)
