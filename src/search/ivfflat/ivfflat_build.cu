#include "search/ivfflat/ivfflat_build.h"

#include "dataset/assignment_materializer.cuh"
#include "dataset/gpu_resident_materializer.cuh"
#include "index/ivf_cache.h"
#include "search/kernels/fine/cpu_fine/dispatch/cpu_ivf_kernel_combos.h"
#include "search/kernels/fine/gpu_fine/resident/resident_l2_topk_combos.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <limits>
#include <stdexcept>

namespace paraselect::ivfflat {
using namespace paraselect::search;
namespace {

using Clock = std::chrono::steady_clock;

void initialize_metadata(
    ClusterDataset* dataset,
    const dataset::ClusterCache::CacheResult& cache,
    const IVFFlatBuildConfig& config) {
    dataset->init_from_cluster_metadata(
        cache.reordered_indices.data(),
        cache.centroids.data(),
        cache.cluster_offsets.data(),
        cache.cluster_counts.data(),
        static_cast<int>(config.n_vectors),
        config.source.spec().dim,
        config.nlist);
}

void build_resident_storage(
    ClusterDataset* dataset,
    const dataset::ClusterCache::CacheResult& cache,
    const IVFFlatBuildConfig& config) {
    initialize_metadata(dataset, cache, config);
    const auto* combo = resident_fine::resolve_resident_l2_topk_combo_ptr_or_throw(
        config.resident_combo.c_str());
    bool use_u8 = config.resident_dtype == IVFFlatResidentDType::U8;
    if (config.resident_dtype == IVFFlatResidentDType::Auto) {
        use_u8 = config.source.spec().dtype == dataset::VectorDType::U8;
    }
    if (use_u8 && config.source.spec().dtype != dataset::VectorDType::U8) {
        throw std::invalid_argument(
            "IVFFlat U8 resident storage requires a U8 dataset");
    }
    const bool packed_u8 = use_u8 &&
        (combo->kind == resident_fine::ResidentL2TopKKernelKind::Dp4a ||
         resident_fine::is_threshold_kind(combo->kind));
    // Experimental opt-in layout: only the dedicated vec4 combos request it, and
    // only for dim % 16 == 0. All other combos keep the historical interleave.
    const bool vec4_u8 = packed_u8 &&
        (combo->kind == resident_fine::ResidentL2TopKKernelKind::ThresholdGroupedVec4 ||
         combo->kind == resident_fine::ResidentL2TopKKernelKind::ThresholdGroupedVec4Shq) &&
        (config.source.spec().dim % 16 == 0);
    const auto plan = dataset::plan_gpu_resident_interleaved(
        dataset->cluster_info.counts,
        static_cast<int>(config.n_vectors),
        config.source.spec().dim,
        config.nlist,
        use_u8,
        packed_u8,
        vec4_u8);
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        throw std::runtime_error("failed to query GPU memory for IVFFlat storage");
    }
    if (plan.estimate.total_bytes() > free_bytes) {
        throw std::runtime_error("IVFFlat resident storage exceeds free GPU memory");
    }
    auto storage = dataset::materialize_gpu_resident_interleaved(
        config.source,
        dataset->reordered_indices,
        dataset->cluster_info.offsets,
        dataset->cluster_info.counts,
        static_cast<int>(config.n_vectors),
        config.source.spec().dim,
        config.nlist,
        plan);
    dataset->set_gpu_resident_interleaved_owned(storage);
}

void build_cpu_storage(
    ClusterDataset* dataset,
    const dataset::ClusterCache::CacheResult& cache,
    const IVFFlatBuildConfig& config) {
    initialize_metadata(dataset, cache, config);
    const std::string dataset_name = dataset::benchmark_dataset_name(
        config.source.spec());
    const auto* combo = cpu_fine::resolve_cpu_fine_combo_ptr_or_throw(
        config.cpu_fine_combo.c_str(), dataset_name.c_str());
    switch (combo->layout) {
        case CPU_FINE_LAYOUT_U8_ROW_MAJOR: {
            auto storage = dataset::materialize_reordered_u8(
                config.source,
                dataset->reordered_indices,
                dataset->cluster_info.offsets,
                dataset->cluster_info.counts,
                static_cast<int>(config.n_vectors),
                config.source.spec().dim,
                config.nlist,
                config.numa_first_touch);
            dataset->set_reordered_u8_owned(
                storage.data, storage.owner, config.numa_first_touch);
            storage.data = nullptr;
            break;
        }
        case CPU_FINE_LAYOUT_FP32_AOSOA: {
            auto storage = dataset::materialize_reordered_aosoa(
                config.source,
                dataset->reordered_indices,
                dataset->cluster_info.offsets,
                dataset->cluster_info.counts,
                static_cast<int>(config.n_vectors),
                config.source.spec().dim,
                config.nlist,
                16,
                config.numa_first_touch);
            dataset->set_aosoa_owned(
                storage.data,
                storage.data_owner,
                storage.offsets,
                storage.offsets_owner,
                storage.tile,
                config.numa_first_touch);
            storage.data = nullptr;
            storage.offsets = nullptr;
            break;
        }
        case CPU_FINE_LAYOUT_FP32_ROW_MAJOR: {
            auto storage = dataset::materialize_reordered_fp32(
                config.source,
                dataset->reordered_indices,
                dataset->cluster_info.offsets,
                dataset->cluster_info.counts,
                static_cast<int>(config.n_vectors),
                config.source.spec().dim,
                config.nlist,
                config.numa_first_touch);
            dataset->set_reordered_fp32_owned(
                storage.data,
                storage.owner,
                config.numa_first_touch,
                config.numa_first_touch);
            storage.data = nullptr;
            break;
        }
        default:
            throw std::invalid_argument("unknown IVFFlat CPU storage layout");
    }
}

}  // namespace

BuildTiming build_storage(
    const IVFFlatBuildConfig& config, ClusterDataset* dataset) {
    const auto base = config.source.base_info();
    if (!dataset || config.n_vectors == 0 || config.n_vectors > base.n ||
        config.n_vectors > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        config.nlist <= 0) {
        throw std::invalid_argument("invalid IVFFlat build configuration");
    }
    const auto kmeans_start = Clock::now();
    const auto cache = ivf_cache::ensure_kmeans(
        config.source, config.n_vectors, config.nlist, config.kmeans);
    BuildTiming timing;
    timing.kmeans_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - kmeans_start).count();
    if (config.fine_kind == IVFFlatFineKind::Resident) {
        build_resident_storage(dataset, cache, config);
    } else if (config.fine_kind == IVFFlatFineKind::Cpu) {
        build_cpu_storage(dataset, cache, config);
    } else {
        throw std::invalid_argument("unknown IVFFlat fine backend");
    }
    return timing;
}

}  // namespace paraselect::ivfflat
