add_library(paraselect_exact_recall_helper SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark/exact_recall_c_api.cpp
)

target_include_directories(paraselect_exact_recall_helper PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(paraselect_exact_recall_helper PUBLIC
    paraselect_cuda
    Threads::Threads
)

set_target_properties(paraselect_exact_recall_helper PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "paraselect_exact_recall_helper"
)
