# =============================================================================
#
# :,CUDA ,
# =============================================================================

# -----------------------------------------------------------------------------
#  -  detect_system_deps.cmake
# -----------------------------------------------------------------------------

# CMake chooses C/C++ and CUDA host compilers during its first configure.
# Respect CC, CXX, CUDACXX, and the corresponding CMake cache variables.
#
#  compile_commands.json, Cursor/VSCode
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PARASELECT_ENABLE_NATIVE_CPU_OPTIMIZATION "Enable -march=native for local Release builds" OFF)
option(PARASELECT_ENABLE_CPU_LTO "Enable CMake IPO/LTO for Release builds" OFF)
option(PARASELECT_ENABLE_FAST_MATH "Enable CUDA --use_fast_math" OFF)
option(PARASELECT_ENABLE_CUDA_LINEINFO "Include CUDA line information" OFF)
option(PARASELECT_SUPPRESS_CUDA_WARNINGS "Suppress CUDA compiler warnings" OFF)

add_library(paraselect_cuda_options INTERFACE)

if(CMAKE_BUILD_TYPE STREQUAL "Release" AND PARASELECT_ENABLE_NATIVE_CPU_OPTIMIZATION)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()

if(PARASELECT_ENABLE_CPU_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _paraselect_ipo_supported OUTPUT _paraselect_ipo_error)
    if(_paraselect_ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    else()
        message(WARNING "PARASELECT_ENABLE_CPU_LTO requested but unavailable: ${_paraselect_ipo_error}")
    endif()
endif()

if(PARASELECT_ENABLE_FAST_MATH)
    target_compile_options(paraselect_cuda_options INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()
if(PARASELECT_ENABLE_CUDA_LINEINFO)
    target_compile_options(paraselect_cuda_options INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()
if(PARASELECT_SUPPRESS_CUDA_WARNINGS)
    target_compile_options(paraselect_cuda_options INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:-w>
        $<$<COMPILE_LANGUAGE:CUDA>:--disable-warnings>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>)
endif()
target_compile_options(paraselect_cuda_options INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# CUDA architecture selection. "native" is portable for local builds; set a numeric
# value such as 120 explicitly for reproducible or cross-compiled artifacts.
if(NOT DEFINED CACHE{PARASELECT_CUDA_ARCHITECTURES})
    if(DEFINED ENV{PARASELECT_CUDA_ARCHITECTURES} AND NOT "$ENV{PARASELECT_CUDA_ARCHITECTURES}" STREQUAL "")
        set(PARASELECT_CUDA_ARCHITECTURES "$ENV{PARASELECT_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures (native or semicolon-separated numeric list)")
    else()
        set(PARASELECT_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures (native or semicolon-separated numeric list)")
    endif()
endif()
set(CMAKE_CUDA_ARCHITECTURES "${PARASELECT_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures" FORCE)
#  nvcc (nvcc  --threads ).,
#  PARASELECT_NVCC_THREADS=<N> cmake -S . -B build
#  -DPARASELECT_NVCC_THREADS=<N>.
if(DEFINED ENV{PARASELECT_NVCC_THREADS} AND NOT "$ENV{PARASELECT_NVCC_THREADS}" STREQUAL "")
    set(PARASELECT_NVCC_THREADS "$ENV{PARASELECT_NVCC_THREADS}" CACHE STRING "nvcc internal compiler threads" FORCE)
endif()
if(DEFINED PARASELECT_NVCC_THREADS AND NOT "${PARASELECT_NVCC_THREADS}" STREQUAL "")
    target_compile_options(paraselect_cuda_options INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--threads=${PARASELECT_NVCC_THREADS}>)
endif()

# :, ../../
# :
#   #include "cpu/cpu_utils.h"
#   #include "src/search/ivfflat/ivfflat_search.cuh"
#   #include "common/test_utils.cuh"  ( tests )
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
)

#
set(COMMON_HEADERS "-include ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/pch.h")
