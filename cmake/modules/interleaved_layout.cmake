# =============================================================================
# Interleaved Layout
# row-major <-> interleaved (CPU)
# =============================================================================

add_library(interleaved_layout_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/layout/ivf_interleaved_layout.cpp
)
target_include_directories(interleaved_layout_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(interleaved_layout_lib PUBLIC Threads::Threads paraselect_cuda)
