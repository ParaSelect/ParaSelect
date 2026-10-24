# Optional RAPIDS Python wheels expose the same C++ libraries and headers.
# Keep this dependency discovery in CMake, not in archive/link-command rewriting.
execute_process(COMMAND "${PARASELECT_RAPIDS_PYTHON}" -c
    "import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec('libcuvs').origin).parent.parent)"
    OUTPUT_VARIABLE _rapids_site OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
foreach(_package libcuvs libraft librmm rapids_logger)
    if(NOT EXISTS "${_rapids_site}/${_package}/include")
        message(FATAL_ERROR "Missing RAPIDS wheel headers: ${_package}")
    endif()
    target_include_directories(paraselect_rapids SYSTEM INTERFACE "${_rapids_site}/${_package}/include")
endforeach()
target_include_directories(paraselect_rapids BEFORE INTERFACE
    "${_rapids_site}/librmm/include/rapids")
target_compile_features(paraselect_rapids INTERFACE cxx_std_20 cuda_std_20)
include(TestBigEndian)
test_big_endian(_rapids_big_endian)
if(_rapids_big_endian)
    set(_rapids_little_endian 0)
else()
    set(_rapids_little_endian 1)
endif()
target_compile_definitions(paraselect_rapids INTERFACE
    LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
    RAFT_SYSTEM_LITTLE_ENDIAN=${_rapids_little_endian})
foreach(_pair "cuvs;libcuvs" "rmm;librmm" "rapids_logger;rapids_logger")
    list(GET _pair 0 _lib)
    list(GET _pair 1 _package)
    find_library(_rapids_${_lib} NAMES ${_lib} PATHS "${_rapids_site}/${_package}/lib64" NO_DEFAULT_PATH REQUIRED)
    add_library(paraselect_wheel_${_lib} SHARED IMPORTED)
    set_target_properties(paraselect_wheel_${_lib} PROPERTIES IMPORTED_LOCATION "${_rapids_${_lib}}")
    target_link_libraries(paraselect_rapids INTERFACE paraselect_wheel_${_lib})
endforeach()
set(PARASELECT_CUVS_LIBRARY "${_rapids_cuvs}")
