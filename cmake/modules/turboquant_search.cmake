add_library(turboquant_encoding_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/turboquant/turboquant_1bit_encode.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/turboquant/residual_encoding.cu)
configure_ivf_cuda_library(turboquant_encoding_lib)
target_link_libraries(turboquant_encoding_lib PUBLIC paraselect_cuda utils_lib)

add_library(residual_codec_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivf_common/quantized/residual_codec_factory.cpp)
target_include_directories(residual_codec_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(residual_codec_lib PUBLIC turboquant_encoding_lib)

# The TurboQuant ivf-cache builder needs both ivf_cache and the encoder, and
# ivf_cache.cmake is configured before this module, so the library is defined
# here and linked back into the ResPQ search stack.
add_library(turboquant_ivf_cache_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/index/ivf_cache/turboquant.cu)
configure_ivf_cuda_library(turboquant_ivf_cache_lib)
target_link_libraries(turboquant_ivf_cache_lib PUBLIC
    ivf_cache
    turboquant_cache
    turboquant_encoding_lib
    paraselect_cuda
    utils_lib
)

add_library(turboquant_search_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/turboquant/turboquant_1bit_grouped_scan.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/turboquant/turboquant_1bit_scan.cu
)
configure_ivf_cuda_library(turboquant_search_lib)
target_link_libraries(turboquant_search_lib PRIVATE
    paraselect_cuda
    utils_lib
    turboquant_encoding_lib
)

add_library(ivf_res_turboquant_search_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivf_res_turboquant/ivf_res_turboquant_search.cu
)
configure_ivf_cuda_library(ivf_res_turboquant_search_lib)
target_link_libraries(ivf_res_turboquant_search_lib PUBLIC
    turboquant_search_lib
    paraselect_cuda
    utils_lib
)

target_link_libraries(ivfrespq_search_lib PUBLIC residual_codec_lib turboquant_ivf_cache_lib)
