find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_VERSION VERSION_LESS "12.0")
    message(FATAL_ERROR "ParaSelect requires CUDA Toolkit >= 12.0; found ${CUDAToolkit_VERSION}")
endif()

string(REGEX MATCH "^[0-9]+" PARASELECT_CUDA_MAJOR "${CMAKE_CUDA_COMPILER_VERSION}")
string(REGEX MATCH "^[0-9]+" _paraselect_toolkit_major "${CUDAToolkit_VERSION}")
if(NOT PARASELECT_CUDA_MAJOR STREQUAL _paraselect_toolkit_major)
    message(FATAL_ERROR
        "CUDA compiler ${CMAKE_CUDA_COMPILER_VERSION} and toolkit ${CUDAToolkit_VERSION} are from different major releases")
endif()

add_library(paraselect_cuda INTERFACE)
target_link_libraries(paraselect_cuda INTERFACE
    paraselect_cuda_options
    CUDA::cudart
)

find_package(nvtx3 CONFIG QUIET)
add_library(paraselect_nvtx INTERFACE)
if(TARGET nvtx3::nvtx3-c)
    target_link_libraries(paraselect_nvtx INTERFACE nvtx3::nvtx3-c)
elseif(TARGET CUDA::nvtx3)
    target_link_libraries(paraselect_nvtx INTERFACE CUDA::nvtx3)
elseif(TARGET CUDA::nvToolsExt)
    target_link_libraries(paraselect_nvtx INTERFACE CUDA::nvToolsExt)
else()
    message(FATAL_ERROR "The selected CUDA toolkit does not provide NVTX")
endif()

option(PARASELECT_ENABLE_CUVS "Build optional cuVS/RAFT backends" OFF)
add_library(paraselect_rapids INTERFACE)
target_link_libraries(paraselect_rapids INTERFACE paraselect_cuda)

set(PARASELECT_RAPIDS_PYTHON "" CACHE FILEPATH "Python environment providing optional RAPIDS wheels")
if(PARASELECT_ENABLE_CUVS)
  if(PARASELECT_RAPIDS_PYTHON)
    include(${CMAKE_CURRENT_LIST_DIR}/rapids_wheel.cmake)
  else()
    find_package(raft CONFIG REQUIRED)
    find_package(rmm CONFIG REQUIRED)
    find_package(cuvs CONFIG REQUIRED)
    get_target_property(PARASELECT_CUVS_LIBRARY cuvs::cuvs IMPORTED_LOCATION_RELEASE)
    if(NOT PARASELECT_CUVS_LIBRARY)
        get_target_property(PARASELECT_CUVS_LIBRARY cuvs::cuvs IMPORTED_LOCATION)
    endif()
    if(NOT PARASELECT_CUVS_LIBRARY)
        message(FATAL_ERROR "cuvs::cuvs does not expose its shared-library location")
    endif()

  endif()
    find_program(PARASELECT_READELF NAMES readelf llvm-readelf REQUIRED)
    execute_process(
        COMMAND "${PARASELECT_READELF}" -d "${PARASELECT_CUVS_LIBRARY}"
        OUTPUT_VARIABLE _paraselect_cuvs_dynamic
        RESULT_VARIABLE _paraselect_readelf_result)
    if(NOT _paraselect_readelf_result EQUAL 0)
        message(FATAL_ERROR "Could not inspect cuVS runtime dependencies: ${PARASELECT_CUVS_LIBRARY}")
    endif()
    string(REGEX MATCH "libcudart\\.so\\.([0-9]+)" _paraselect_cudart_needed "${_paraselect_cuvs_dynamic}")
    set(_paraselect_cuvs_cuda_major "${CMAKE_MATCH_1}")
    if(NOT _paraselect_cuvs_cuda_major)
        # Wheels may statically link cudart; cuBLAS still exposes the CUDA major.
        string(REGEX MATCH "libcublas\\.so\\.([0-9]+)" _paraselect_cublas_needed "${_paraselect_cuvs_dynamic}")
        set(_paraselect_cuvs_cuda_major "${CMAKE_MATCH_1}")
    endif()
    if(NOT _paraselect_cuvs_cuda_major)
        message(FATAL_ERROR "Could not determine the CUDA major required by ${PARASELECT_CUVS_LIBRARY}")
    endif()
    if(NOT _paraselect_cuvs_cuda_major STREQUAL PARASELECT_CUDA_MAJOR)
        message(FATAL_ERROR
            "cuVS requires CUDA ${_paraselect_cuvs_cuda_major}, but this build selected CUDA ${PARASELECT_CUDA_MAJOR}")
    endif()

  if(NOT PARASELECT_RAPIDS_PYTHON)
    target_link_libraries(paraselect_rapids INTERFACE
        raft::raft
        rmm::rmm
        cuvs::cuvs
    )
  endif()
    target_compile_definitions(paraselect_cuda INTERFACE PARASELECT_ENABLE_CUVS=1)
else()
    target_compile_definitions(paraselect_cuda INTERFACE PARASELECT_ENABLE_CUVS=0)
endif()

message(STATUS "ParaSelect CUDA stack: toolkit=${CUDAToolkit_VERSION}, cuVS=${PARASELECT_ENABLE_CUVS}")
