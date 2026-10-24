#ifndef PARASELECT_CPU_IVF_KERNEL_COMBOS_H
#define PARASELECT_CPU_IVF_KERNEL_COMBOS_H

#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>
#include <string>

#include "search/kernels/fine/cpu_fine/types/cpu_fine_run.h"
#include "search/kernels/fine/cpu_fine/dispatch/cpu_isa.h"
#include "search/kernels/fine/cpu_fine/executor/cpu_fine_batch_view.h"
#include "search/kernels/fine/cpu_fine/runners/fp32_aosoa_runners.h"
#include "search/kernels/fine/cpu_fine/runners/fp32_rowmajor_runners.h"
#include "search/kernels/fine/cpu_fine/runners/fp32_rowmajor_numa_runners.h"
#include "search/kernels/fine/cpu_fine/runners/u8_rowmajor_runners.h"
#include "search/kernels/fine/cpu_fine/runners/u8_paged_runners.h"
#include "search/kernels/fine/cpu_fine/runners/fp32_paged_runners.h"
#include "search/kernels/fine/cpu_fine/runners/u8_rowmajor_numa_runners.h"

namespace paraselect {
namespace cpu_fine {

struct CpuFineKernelCombo {
    CpuFineVectorLayout layout;
    CpuFineKernelKind kernel;
    CpuFineSchedulePolicy schedule_policy;
    const char* nvtx_tag;
    bool require_avx512f;
    bool require_avx512bw;
    CpuFineClusterDynamicFn cluster_scratch;
    CpuFineClusterDynamicFn cluster_metadata;
    CpuFineClusterDynamicFn cluster_compact_metadata;
};

struct CpuFineComboPreset {
    const char* name;
    int combo_id;
    CpuFineVectorLayout layout;
    CpuFineKernelKind kernel;
    CpuFineSchedulePolicy schedule_policy;
};

inline bool cpu_fine_combo_supported_by_host(const CpuFineKernelCombo& combo) {
    if (combo.require_avx512f && !host_supports_avx512f()) return false;
    if (combo.require_avx512bw && !host_supports_avx512bw()) return false;
    return true;
}

inline const char* cpu_fine_combo_name_from_preset(
    CpuFineVectorLayout layout,
    CpuFineKernelKind kernel,
    CpuFineSchedulePolicy schedule_policy)
{
    if (layout == CPU_FINE_LAYOUT_FP32_ROW_MAJOR &&
        kernel == CPU_FINE_KERNEL_FP32_AUTO &&
        schedule_policy == CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC) {
        return "fp32_cluster_dynamic";
    }
    if (layout == CPU_FINE_LAYOUT_FP32_AOSOA &&
        kernel == CPU_FINE_KERNEL_FP32_AOSOA_V5 &&
        schedule_policy == CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC) {
        return "fp32_aosoa_cluster_dynamic";
    }
    if (layout == CPU_FINE_LAYOUT_FP32_AOSOA &&
        kernel == CPU_FINE_KERNEL_FP32_AOSOA_V6 &&
        schedule_policy == CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC) {
        return "fp32_aosoa_v1_cluster_dynamic";
    }
    if (layout == CPU_FINE_LAYOUT_U8_ROW_MAJOR &&
        kernel == CPU_FINE_KERNEL_U8 &&
        schedule_policy == CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC) {
        return "u8_cluster_dynamic";
    }
    if (layout == CPU_FINE_LAYOUT_FP32_PAGED &&
        kernel == CPU_FINE_KERNEL_FP32_AUTO &&
        schedule_policy == CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC) {
        return "fp32_paged_cluster_dynamic";
    }
    if (layout == CPU_FINE_LAYOUT_U8_PAGED &&
        kernel == CPU_FINE_KERNEL_U8 &&
        schedule_policy == CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC) {
        return "u8_paged_cluster_dynamic";
    }
    if (layout == CPU_FINE_LAYOUT_FP32_ROW_MAJOR &&
        kernel == CPU_FINE_KERNEL_FP32_AUTO &&
        schedule_policy == CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC_NUMA) {
        return "fp32_cluster_dynamic_numa";
    }
    if (layout == CPU_FINE_LAYOUT_U8_ROW_MAJOR &&
        kernel == CPU_FINE_KERNEL_U8 &&
        schedule_policy == CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC_NUMA) {
        return "u8_cluster_dynamic_numa";
    }
    return nullptr;
}

#define PARASELECT_CPU_FINE_COMBO_CLUSTER(LAYOUT, KERNEL, SCHEDULE, TAG, NEED_AVX512F, NEED_AVX512BW, FN_SCRATCH, FN_METADATA, FN_COMPACT) \
    { LAYOUT, KERNEL, SCHEDULE, TAG, NEED_AVX512F, NEED_AVX512BW, FN_SCRATCH, FN_METADATA, FN_COMPACT }

inline const CpuFineKernelCombo* lookup_cpu_fine_kernel_combo_ptr(
    CpuFineVectorLayout layout,
    CpuFineKernelKind kernel,
    CpuFineSchedulePolicy schedule_policy)
{
    static constexpr CpuFineKernelCombo kCombos[] = {
        PARASELECT_CPU_FINE_COMBO_CLUSTER(CPU_FINE_LAYOUT_FP32_ROW_MAJOR, CPU_FINE_KERNEL_FP32_AUTO,
            CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC, "Fp32ClusterDynamic_Fine", false, false,
            fp32_rowmajor_cluster_dynamic_scratch,
            fp32_rowmajor_cluster_dynamic_metadata,
            fp32_rowmajor_cluster_dynamic_compact_metadata),
        PARASELECT_CPU_FINE_COMBO_CLUSTER(CPU_FINE_LAYOUT_FP32_ROW_MAJOR, CPU_FINE_KERNEL_FP32_AUTO,
            CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC_NUMA, "Fp32ClusterDynamicNUMA_Fine", false, false,
            fp32_rowmajor_cluster_dynamic_numa_scratch,
            fp32_rowmajor_cluster_dynamic_numa_metadata,
            fp32_rowmajor_cluster_dynamic_numa_compact_metadata),
#if PARASELECT_ENABLE_X86_SIMD
        PARASELECT_CPU_FINE_COMBO_CLUSTER(CPU_FINE_LAYOUT_FP32_AOSOA, CPU_FINE_KERNEL_FP32_AOSOA_V5,
            CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC, "Fp32AosoaV0ClusterDynamic_Fine", true, false,
            fp32_aosoa_v0_cluster_dynamic_scratch,
            fp32_aosoa_v0_cluster_dynamic_metadata,
            fp32_aosoa_v0_cluster_dynamic_compact_metadata),
        PARASELECT_CPU_FINE_COMBO_CLUSTER(CPU_FINE_LAYOUT_FP32_AOSOA, CPU_FINE_KERNEL_FP32_AOSOA_V6,
            CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC, "Fp32AosoaV1ClusterDynamic_Fine", true, false,
            fp32_aosoa_v1_cluster_dynamic_scratch,
            fp32_aosoa_v1_cluster_dynamic_metadata,
            fp32_aosoa_v1_cluster_dynamic_compact_metadata),
#endif
        PARASELECT_CPU_FINE_COMBO_CLUSTER(CPU_FINE_LAYOUT_U8_ROW_MAJOR, CPU_FINE_KERNEL_U8,
            CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC, "U8ClusterDynamic_Fine", false, false,
            u8_rowmajor_cluster_dynamic_scratch,
            u8_rowmajor_cluster_dynamic_metadata,
            u8_rowmajor_cluster_dynamic_compact_metadata),
        PARASELECT_CPU_FINE_COMBO_CLUSTER(CPU_FINE_LAYOUT_FP32_PAGED, CPU_FINE_KERNEL_FP32_AUTO,
            CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC, "Fp32PagedClusterDynamic_Fine", false, false,
            fp32_paged_cluster_dynamic_scratch,
            fp32_paged_cluster_dynamic_metadata,
            fp32_paged_cluster_dynamic_compact_metadata),
        PARASELECT_CPU_FINE_COMBO_CLUSTER(CPU_FINE_LAYOUT_U8_PAGED, CPU_FINE_KERNEL_U8,
            CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC, "U8PagedClusterDynamic_Fine", false, false,
            u8_paged_cluster_dynamic_scratch,
            u8_paged_cluster_dynamic_metadata,
            u8_paged_cluster_dynamic_compact_metadata),
        PARASELECT_CPU_FINE_COMBO_CLUSTER(CPU_FINE_LAYOUT_U8_ROW_MAJOR, CPU_FINE_KERNEL_U8,
            CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC_NUMA, "U8ClusterDynamicNUMA_Fine", false, false,
            u8_rowmajor_cluster_dynamic_numa_scratch,
            u8_rowmajor_cluster_dynamic_numa_metadata,
            u8_rowmajor_cluster_dynamic_numa_compact_metadata),
    };
    for (const auto& combo : kCombos) {
        if (combo.layout == layout && combo.kernel == kernel && combo.schedule_policy == schedule_policy) {
            return &combo;
        }
    }
    return nullptr;
}

inline bool lookup_cpu_fine_combo_preset(const char* name, CpuFineComboPreset* out)
{
    if (!name || !out) return false;
    static constexpr CpuFineComboPreset kPresets[] = {
        {"fp32_cluster_dynamic", 1, CPU_FINE_LAYOUT_FP32_ROW_MAJOR, CPU_FINE_KERNEL_FP32_AUTO, CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC},
        {"fp32_aosoa_cluster_dynamic", 4, CPU_FINE_LAYOUT_FP32_AOSOA, CPU_FINE_KERNEL_FP32_AOSOA_V5, CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC},
        {"fp32_aosoa_v1_cluster_dynamic", 5, CPU_FINE_LAYOUT_FP32_AOSOA, CPU_FINE_KERNEL_FP32_AOSOA_V6, CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC},
        {"u8_cluster_dynamic", 6, CPU_FINE_LAYOUT_U8_ROW_MAJOR, CPU_FINE_KERNEL_U8, CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC},
        {"fp32_paged_cluster_dynamic", 10, CPU_FINE_LAYOUT_FP32_PAGED, CPU_FINE_KERNEL_FP32_AUTO, CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC},
        {"u8_paged_cluster_dynamic", 9, CPU_FINE_LAYOUT_U8_PAGED, CPU_FINE_KERNEL_U8, CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC},
        {"fp32_cluster_dynamic_numa", 7, CPU_FINE_LAYOUT_FP32_ROW_MAJOR, CPU_FINE_KERNEL_FP32_AUTO, CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC_NUMA},
        {"u8_cluster_dynamic_numa", 8, CPU_FINE_LAYOUT_U8_ROW_MAJOR, CPU_FINE_KERNEL_U8, CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC_NUMA},
    };
    for (const auto& preset : kPresets) {
        if (std::strcmp(name, preset.name) == 0) {
            *out = preset;
            return true;
        }
    }
    return false;
}

inline const char* resolve_default_cpu_fine_combo_name(const char* dataset_name) {
    if (!dataset_name || dataset_name[0] == '\0') return nullptr;
    std::string name(dataset_name);
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (name.find("SIFT") != std::string::npos) return "u8_cluster_dynamic";
    if (name.find("DEEP") != std::string::npos || name.find("TEXT") != std::string::npos) {
        return "fp32_cluster_dynamic";
    }
    return nullptr;
}

inline const CpuFineKernelCombo* resolve_cpu_fine_combo_ptr_or_throw(
    const char* cpu_fine_combo_name,
    const char* dataset_name = nullptr)
{
    const char* resolved_combo_name = cpu_fine_combo_name;
    bool use_dataset_default = (!resolved_combo_name || resolved_combo_name[0] == '\0');
    if (!use_dataset_default && std::strcmp(resolved_combo_name, "default") == 0) {
        use_dataset_default = true;
    }
    if (use_dataset_default) resolved_combo_name = resolve_default_cpu_fine_combo_name(dataset_name);
    if (!resolved_combo_name || resolved_combo_name[0] == '\0') {
        throw std::invalid_argument("cpu_fine_combo_name is required (no dataset default combo matched)");
    }
    CpuFineComboPreset preset{};
    if (!lookup_cpu_fine_combo_preset(resolved_combo_name, &preset)) {
        throw std::invalid_argument(std::string("invalid cpu_fine_combo_name: ") + resolved_combo_name);
    }
    const CpuFineKernelCombo* combo = lookup_cpu_fine_kernel_combo_ptr(
        preset.layout, preset.kernel, preset.schedule_policy);
    if (!combo) {
        throw std::invalid_argument(
            std::string("cpu_fine_combo_name mapped to unregistered combo: ") + resolved_combo_name);
    }
    return combo;
}

#undef PARASELECT_CPU_FINE_COMBO_CLUSTER

}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_CPU_IVF_KERNEL_COMBOS_H
