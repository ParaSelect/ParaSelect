# =============================================================================
#
# =============================================================================

# Guard:
if(DEFINED _IVF_SYSTEM_DEPS_DETECTED)
    return()
endif()
set(_IVF_SYSTEM_DEPS_DETECTED TRUE)

# CMake chooses C/C++ and CUDA host compilers before this module is included.
# Do not rewrite them here: callers can use CC, CXX, CUDACXX, or standard
# CMake cache variables during their first configure.
find_package(OpenMP QUIET)
find_package(Threads QUIET)

message(STATUS "========================================")
message(STATUS "System dependencies configured")
message(STATUS "========================================")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")
message(STATUS "========================================")
