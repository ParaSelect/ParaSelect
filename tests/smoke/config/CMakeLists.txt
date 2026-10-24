add_executable(test_smoke_config test_smoke_config.cpp)

target_include_directories(test_smoke_config PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_smoke_config PRIVATE dataset_source paraselect_cuda)

add_test(NAME test_smoke_config COMMAND test_smoke_config)
set_tests_properties(test_smoke_config PROPERTIES
    LABELS "smoke;unit;config"
)
