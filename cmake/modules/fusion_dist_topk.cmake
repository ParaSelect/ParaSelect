# =============================================================================
# Fusion Distance Top-K
#  cosine  L2  warpsort
#  k
# =============================================================================

add_library(fusion_dist_topk_lib STATIC
    # High-level row-wise/candidate top-k reducer.
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/topk/reducer/topk_reducer.cu
    #  -
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/topk/fusion/fusion_dist_topk_launcher.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/topk/fusion/fusion_dist_topk_launcher_64_128_256.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/topk/fusion/fusion_dist_topk_launcher_512_1024.cu
    # FP16
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/topk/fusion/fusion_cos_topk_warpsort_fp16.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/topk/fusion/fusion_l2_topk_warpsort_fp16.cu
    # cuVS
    $<$<BOOL:${PARASELECT_ENABLE_CUVS}>:${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/topk/fusion/fusion_dist_topk_cuvs.cu>
)

target_include_directories(fusion_dist_topk_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/common
    ${PARASELECT_RAPIDS_INCLUDE_DIR}
)

set_target_properties(fusion_dist_topk_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${PARASELECT_CUDA_ARCHITECTURES}
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

target_link_libraries(fusion_dist_topk_lib
    PUBLIC
    paraselect_cuda
    CUDA::cublas
    l2norm_lib
    utils_lib
)
if(PARASELECT_ENABLE_CUVS)
    target_link_libraries(fusion_dist_topk_lib PUBLIC paraselect_rapids)
endif()

option(PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK "Enable experimental Adaptive Threshold TopK SM120 FP32 coarse selection" OFF)
set(PARASELECT_CUTLASS_ROOT "" CACHE PATH "CUTLASS checkout used by Adaptive Threshold TopK")
if(PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK)
    add_compile_definitions(PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK=1)
    if(NOT EXISTS "${PARASELECT_CUTLASS_ROOT}/include/cutlass/cutlass.h")
        message(FATAL_ERROR "Set PARASELECT_CUTLASS_ROOT to the pinned CUTLASS checkout")
    endif()
    add_library(coarse_adaptive_threshold_topk_lib STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/coarse/topk/adaptive_threshold_topk/adaptive_threshold_topk_fp32.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/coarse/topk/adaptive_threshold_topk/fused/fused.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/coarse/topk/adaptive_threshold_topk/fused_gemm/fused_gemm_topk.cu)
    target_include_directories(coarse_adaptive_threshold_topk_lib PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/deepselect/csrc
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/deepselect/kerutils/include
        ${PARASELECT_CUTLASS_ROOT}/include)
    target_compile_options(coarse_adaptive_threshold_topk_lib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda;--ftz=false>)
    set_target_properties(coarse_adaptive_threshold_topk_lib PROPERTIES
        CUDA_ARCHITECTURES 120a CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(coarse_adaptive_threshold_topk_lib PUBLIC CUDA::cudart CUDA::cuda_driver)
    target_link_libraries(fusion_dist_topk_lib PUBLIC coarse_adaptive_threshold_topk_lib)
    target_compile_definitions(fusion_dist_topk_lib PUBLIC PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK=1)
endif()
