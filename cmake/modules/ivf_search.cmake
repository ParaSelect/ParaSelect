# =============================================================================
# IVF Search
# =============================================================================

# NVTX ranges for the Nsight Systems stage timeline
option(PARASELECT_USE_NVTX "Enable NVTX ranges for Nsight Systems" ON)
find_path(PARASELECT_URING_INCLUDE_DIR NAMES liburing.h REQUIRED)
find_library(PARASELECT_URING_LIBRARY NAMES uring REQUIRED)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" PARASELECT_SYSTEM_PROCESSOR)
set(PARASELECT_TARGET_X86 OFF)
if(PARASELECT_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|i[3-6]86)$")
    set(PARASELECT_TARGET_X86 ON)
endif()
option(PARASELECT_ENABLE_X86_SIMD "Build x86 AVX2/AVX-512 CPU fine backends" ${PARASELECT_TARGET_X86})
if(PARASELECT_ENABLE_X86_SIMD AND NOT PARASELECT_TARGET_X86)
    message(FATAL_ERROR "PARASELECT_ENABLE_X86_SIMD requires an x86 target")
endif()
set(PARASELECT_CPU_DISTANCE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/distance/distance_dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/distance/distance_dispatch_plain.cpp
)
if(PARASELECT_ENABLE_X86_SIMD)
    list(APPEND PARASELECT_CPU_DISTANCE_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/distance/distance_dispatch_avx2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/distance/distance_dispatch_avx512.cpp
    )
endif()

add_library(cpu_distance_lib STATIC
    ${PARASELECT_CPU_DISTANCE_SOURCES}
)
target_compile_definitions(cpu_distance_lib PUBLIC PARASELECT_ENABLE_X86_SIMD=$<BOOL:${PARASELECT_ENABLE_X86_SIMD}>)
target_include_directories(cpu_distance_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
set_target_properties(cpu_distance_lib PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# CPU fine core and the portable backend are always available.
set(PARASELECT_CPU_FINE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivf_common/ivf_host_rerank.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfflat/stage/ivfflat_fine_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfflat/stage/ivfflat_cpu_fine_stage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/executor/cpu_fine_executor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/fp32_rowmajor_runners.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/fp32_paged_runners.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/fp32_rowmajor_numa_runners.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/fp32_runners_plain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/u8_rowmajor_runners.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/u8_paged_runners.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/u8_rowmajor_numa_runners.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/u8_runners_plain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/cpu_fine_layout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/cpu_fine_layout_aosoa.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq/stage/ivfrespq_cpu_rerank.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/utils/bandwidth_profiler.cpp
)
if(PARASELECT_ENABLE_X86_SIMD)
    list(APPEND PARASELECT_CPU_FINE_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/fp32_aosoa_runners.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/fp32_runners_avx2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/fp32_runners_avx512.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/u8_runners_avx2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/u8_runners_avx512.cpp
    )
endif()

add_library(ivf_cpu_search_lib STATIC ${PARASELECT_CPU_FINE_SOURCES})
target_compile_definitions(ivf_cpu_search_lib PUBLIC
    PARASELECT_ENABLE_X86_SIMD=$<BOOL:${PARASELECT_ENABLE_X86_SIMD}>
)
target_include_directories(ivf_cpu_search_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
set_target_properties(ivf_cpu_search_lib PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Optional OpenMP support for the CPU fine-search backend
find_package(OpenMP)
target_link_libraries(ivf_cpu_search_lib PRIVATE
    paraselect_cuda
    Threads::Threads
    cpu_distance_lib
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ivf_cpu_search_lib PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(ivf_cpu_search_lib PRIVATE -fopenmp)
endif()

# Per-source optimization flags keep portable and x86 backends isolated.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/distance/distance_dispatch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/distance/distance_dispatch_plain.cpp
        PROPERTIES COMPILE_OPTIONS "-O3;-ffast-math;-funroll-loops"
    )
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/fp32_paged_runners.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/fp32_rowmajor_runners.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/fp32_rowmajor_numa_runners.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/fp32_runners_plain.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/u8_rowmajor_runners.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/u8_paged_runners.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/u8_rowmajor_numa_runners.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/u8_runners_plain.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/cpu_fine_layout.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/cpu_fine_layout_aosoa.cpp
        PROPERTIES COMPILE_OPTIONS "-O3;-ffast-math;-funroll-loops;-fopenmp"
    )
    if(PARASELECT_ENABLE_X86_SIMD)
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/distance/distance_dispatch_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-O3;-ffast-math;-funroll-loops;-mavx2;-mfma"
        )
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/distance/distance_dispatch_avx512.cpp
            PROPERTIES COMPILE_OPTIONS "-O3;-ffast-math;-funroll-loops;-mavx512f;-mavx512bw;-mfma"
        )
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/fp32_runners_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-O3;-ffast-math;-funroll-loops;-fopenmp;-mavx2;-mfma"
        )
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/fp32_runners_avx512.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/fp32_aosoa_runners.cpp
            PROPERTIES COMPILE_OPTIONS "-O3;-ffast-math;-funroll-loops;-fopenmp;-mavx512f;-mfma"
        )
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/u8_runners_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-O3;-ffast-math;-funroll-loops;-fopenmp;-mavx2"
        )
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/cpu_fine/runners/u8_runners_avx512.cpp
            PROPERTIES COMPILE_OPTIONS "-O3;-ffast-math;-funroll-loops;-fopenmp;-mavx512f;-mavx512bw"
        )
    endif()
endif()

# NVTX
if(PARASELECT_USE_NVTX)
    target_compile_definitions(ivf_cpu_search_lib PUBLIC USE_NVTX=1)
        target_link_libraries(ivf_cpu_search_lib PRIVATE paraselect_nvtx)
else()
    target_compile_definitions(ivf_cpu_search_lib PUBLIC USE_NVTX=0)
endif()

# Optional NUMA support for CPU memory placement
find_library(NUMA_LIBRARY numa)
if(NUMA_LIBRARY)
    target_link_libraries(ivf_cpu_search_lib PRIVATE ${NUMA_LIBRARY})
    target_compile_definitions(ivf_cpu_search_lib PUBLIC USE_NUMA=1)
else()
    message(WARNING "NUMA library not found. NUMA-aware optimization disabled.")
    target_compile_definitions(ivf_cpu_search_lib PUBLIC USE_NUMA=0)
endif()

function(configure_ivf_library target)
    target_include_directories(${target} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(PARASELECT_USE_NVTX)
        target_compile_definitions(${target} PUBLIC USE_NVTX=1)
        target_link_libraries(${target} PRIVATE paraselect_nvtx)
    else()
        target_compile_definitions(${target} PUBLIC USE_NVTX=0)
    endif()
    if(PARASELECT_ENABLE_CUVS)
        target_link_libraries(${target} PRIVATE paraselect_rapids)
    endif()
endfunction()

function(configure_ivf_cuda_library target)
    configure_ivf_library(${target})
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES ${PARASELECT_CUDA_ARCHITECTURES}
    )
endfunction()

add_library(ivf_common_search_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/coarse/topk/coarse_histogram_topk.cu
    # Retained-ID ordering is shared by plain and residual PQ.
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq/stage/ivfrespq_candidate_order.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivf_common/ivf_coarse_ctx.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivf_common/ivf_coarse_assign.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivf_common/ivf_probe_schedule.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/coarse/stage/coarse_stage.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/coarse/stage/coarse_fp16.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/coarse/stage/coarse_distance_histogram.cu
)
configure_ivf_cuda_library(ivf_common_search_lib)
add_library(coarse_runtime_objects OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/coarse/runtime/coarse_runtime.cu)
target_link_libraries(coarse_runtime_objects PRIVATE paraselect_cuda fusion_dist_topk_lib CUDA::cublas)
target_include_directories(coarse_runtime_objects PRIVATE ${CMAKE_SOURCE_DIR}/src)
set_target_properties(coarse_runtime_objects PROPERTIES
    CUDA_ARCHITECTURES ${PARASELECT_CUDA_ARCHITECTURES}
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
if(PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK)
    target_include_directories(coarse_runtime_objects PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/deepselect/csrc
        ${CMAKE_SOURCE_DIR}/third_party/deepselect/kerutils/include
        ${PARASELECT_CUTLASS_ROOT}/include)
    target_compile_options(coarse_runtime_objects PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda;--ftz=false;--diag-error=1143>)
    set_target_properties(coarse_runtime_objects PROPERTIES CUDA_ARCHITECTURES 120a)
endif()
target_sources(ivf_common_search_lib PRIVATE $<TARGET_OBJECTS:coarse_runtime_objects>)

target_link_libraries(ivf_common_search_lib PRIVATE
    paraselect_cuda
    CUDA::cublas
    l2norm_lib
    utils_lib
    gemv_lib
    fusion_dist_topk_lib
    kmeans_lib
    quantization_lib
)

add_library(ivf_resident_kernels_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/resident/resident_full_l2_topk.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/resident/resident_full_u8_l2_topk.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/resident/resident_full_u8_dp4a_l2_topk.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/resident/resident_threshold_l2_topk.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/resident/resident_chunked_l2_topk.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/resident/resident_chunked_u8_l2_topk.cu
)
configure_ivf_cuda_library(ivf_resident_kernels_lib)
target_link_libraries(ivf_resident_kernels_lib PRIVATE
    paraselect_cuda
    utils_lib
    fusion_dist_topk_lib
)

add_library(ivf_pca_search_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivf_common/ivf_pca_refine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivf_common/ivf_pca_refine.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivf_common/ivf_pca_resident.cu
)
configure_ivf_cuda_library(ivf_pca_search_lib)
target_link_libraries(ivf_pca_search_lib PUBLIC CUDA::cudart)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ivf_pca_search_lib PRIVATE OpenMP::OpenMP_CXX)
endif()
target_link_libraries(ivf_pca_search_lib PRIVATE
    paraselect_cuda
    utils_lib
    ivf_common_search_lib
    ivf_resident_kernels_lib
)

add_library(ivf_pq_rerank_kernels_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/resident/pq_resident_rerank.cu
)
configure_ivf_cuda_library(ivf_pq_rerank_kernels_lib)
target_link_libraries(ivf_pq_rerank_kernels_lib PRIVATE
    paraselect_cuda
    utils_lib
    fusion_dist_topk_lib
)

add_library(ivf_pq_scan_kernels_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/pq_adc/pq_candidate_cap_select.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/pq_adc/pq_query_lut.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/pq_adc/pq_codeword_scan.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/pq_adc/pq_codeword_fp16.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/pq_adc/pq_query_lut_seed.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/pq_adc/pq_threshold_policy.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/quantized_scan/threshold_search.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq/stage/residual_pq_gpu.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/fine/gpu_fine/quantized_scan/grouped_search.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq/stage/residual_pq_gpu_grouped_top512.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq/stage/residual_pq_gpu_grouped_top64.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq/stage/residual_pq_gpu_reduce.cu
)
configure_ivf_cuda_library(ivf_pq_scan_kernels_lib)
target_link_libraries(ivf_pq_scan_kernels_lib PRIVATE
    paraselect_cuda
    utils_lib
    fusion_dist_topk_lib
)

add_library(ivf_file_rerank_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivf_common/file/ivf_file_reader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivf_common/file/ivf_file_rerank.cpp
)
configure_ivf_library(ivf_file_rerank_lib)
target_include_directories(ivf_file_rerank_lib PRIVATE ${PARASELECT_URING_INCLUDE_DIR})
target_link_libraries(ivf_file_rerank_lib PRIVATE
    Threads::Threads
    cpu_distance_lib
    utils_lib
    ${PARASELECT_URING_LIBRARY}
)

add_library(ivfflat_search_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfflat/ivfflat_build.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfflat/ivfflat_index.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfflat/ivfflat_search.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfflat/pipeline/ivfflat_pipeline.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfflat/pipeline/ivfflat_cpu_overlap.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfflat/pipeline/ivfflat_probe_metadata.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfflat/stage/ivfflat_prepare.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfflat/stage/ivfflat_coarse.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfflat/stage/ivfflat_metadata.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfflat/stage/ivfflat_fine_gpu_resident.cu
)
configure_ivf_cuda_library(ivfflat_search_lib)
target_link_libraries(ivfflat_search_lib PRIVATE
    paraselect_cuda
    CUDA::cublas
    ivf_common_search_lib
    ivf_resident_kernels_lib
    ivf_cpu_search_lib
    kmeans_lib
    binary_format
    dataset_source
    ivf_cache
)

add_library(ivfpq_search_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfpq/ivfpq_build.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfpq/ivfpq_index.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfpq/pipeline/ivfpq_pipeline.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfpq/stage/ivfpq_prepare.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfpq/stage/ivfpq_coarse.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfpq/stage/ivfpq_fine_cpu_lut.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfpq/stage/ivfpq_resident_codes.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfpq/stage/ivfpq_rerank_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfpq/stage/ivfpq_map_ids.cpp
)
configure_ivf_cuda_library(ivfpq_search_lib)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ivfpq_search_lib PRIVATE OpenMP::OpenMP_CXX)
endif()
target_link_libraries(ivfpq_search_lib PRIVATE
    ivfrespq_search_lib
    paraselect_cuda
    CUDA::cublas
    ivf_common_search_lib
    ivf_pq_rerank_kernels_lib
    ivf_pq_scan_kernels_lib
    ivf_cpu_search_lib
    kmeans_lib
    binary_format
    dataset_source
    ivf_cache
    quantization_lib
    transform_lib
)

add_library(ivfrespq_search_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/bruteforce_respq/bruteforce_respq_search.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq/ivfrespq_search.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq_file/ivfrespq_file_build.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq_file/ivfrespq_file_index.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq/scheduler/ivfrespq_scheduler.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq/pipeline/ivfrespq_pipeline_cpu.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq/stage/ivfrespq_prepare.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq/stage/ivfrespq_release.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq/stage/ivfrespq_prepare_output.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq/stage/ivfrespq_prepare_validation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq/stage/ivfrespq_prepare_coarse.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq/stage/ivfrespq_prepare_workspace.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq/stage/ivfrespq_coarse.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq/stage/ivfrespq_gpu_lut.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq/stage/ivfrespq_fine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq/stage/ivfrespq_fine_cpu_lut.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq/stage/ivfrespq_fine_gpu_pqscan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfrespq/stage/ivfrespq_map_ids.cpp
)
configure_ivf_cuda_library(ivfrespq_search_lib)
target_link_libraries(ivfrespq_search_lib PRIVATE
    paraselect_cuda
    CUDA::cublas
    ivf_common_search_lib
    ivf_file_rerank_lib
    ivf_pca_search_lib
    ivf_pq_rerank_kernels_lib
    ivf_pq_scan_kernels_lib
    ivf_cpu_search_lib
    kmeans_lib
    binary_format
    dataset_source
    ivf_cache
    quantization_lib
    transform_lib
)

add_library(ivfpca_file_search_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfpca_file/ivfpca_file_build.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfpca_file/ivfpca_file_index.cu
)
configure_ivf_cuda_library(ivfpca_file_search_lib)
target_link_libraries(ivfpca_file_search_lib PRIVATE
    paraselect_cuda
    CUDA::cublas
    ivf_common_search_lib
    ivf_file_rerank_lib
    ivf_pca_search_lib
    kmeans_lib
    binary_format
    dataset_source
    ivf_cache
    quantization_lib
    transform_lib
)

add_library(ivfjl_search_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfjl/ivfjl_search.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfjl/ivfjl_prepare.cu
)
configure_ivf_cuda_library(ivfjl_search_lib)
target_link_libraries(ivfjl_search_lib PRIVATE
    paraselect_cuda
    CUDA::cublas
    ivf_common_search_lib
    ivfflat_search_lib
    transform_lib
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/ivfpq/stage/ivfpq_rerank_cpu.cpp
        PROPERTIES COMPILE_OPTIONS "-O3;-ffast-math;-funroll-loops"
    )
endif()

# Brute Force Search
add_library(brute_force_search_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/search/kernels/brute_force/brute_force_scan.cu
)
target_include_directories(brute_force_search_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
set_target_properties(brute_force_search_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${PARASELECT_CUDA_ARCHITECTURES}
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(brute_force_search_lib
    PUBLIC
    paraselect_cuda
    CUDA::cublas
    utils_lib
    l2norm_lib
    fusion_dist_topk_lib
)
