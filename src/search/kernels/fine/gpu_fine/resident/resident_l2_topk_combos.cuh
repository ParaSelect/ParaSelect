#ifndef PARASELECT_RESIDENT_L2_TOPK_COMBOS_CUH
#define PARASELECT_RESIDENT_L2_TOPK_COMBOS_CUH

#include <cstring>
#include <stdexcept>
#include <string>

namespace paraselect::resident_fine {

enum class ResidentL2TopKKernelKind {
    Full,
    Chunked,
    Dp4a,
    ThresholdQuery,
    ThresholdGrouped,
    ThresholdGroupedPair,
    ThresholdGroupedShq,
    ThresholdGroupedVec4,
    ThresholdGroupedVec4Shq,
    ThresholdGroupedMlp2,
    ThresholdGroupedCpasync,
};

struct ResidentL2TopKCombo {
    const char* name;
    ResidentL2TopKKernelKind kind;
    int default_chunk_blocks;
};

inline const ResidentL2TopKCombo* lookup_resident_l2_topk_combo(const char* name) {
    static constexpr ResidentL2TopKCombo kCombos[] = {
        {"chunked", ResidentL2TopKKernelKind::Chunked, 8},
        {"full", ResidentL2TopKKernelKind::Full, 0},
        {"dp4a", ResidentL2TopKKernelKind::Dp4a, 0},
        {"threshold_query", ResidentL2TopKKernelKind::ThresholdQuery, 8},
        {"threshold_grouped", ResidentL2TopKKernelKind::ThresholdGrouped, 0},
        {"threshold_grouped_pair", ResidentL2TopKKernelKind::ThresholdGroupedPair, 0},
        {"threshold_grouped_shq", ResidentL2TopKKernelKind::ThresholdGroupedShq, 0},
        {"threshold_grouped_vec4", ResidentL2TopKKernelKind::ThresholdGroupedVec4, 0},
        {"threshold_grouped_vec4_shq", ResidentL2TopKKernelKind::ThresholdGroupedVec4Shq, 0},
        {"threshold_grouped_mlp2", ResidentL2TopKKernelKind::ThresholdGroupedMlp2, 0},
        {"threshold_grouped_cpasync", ResidentL2TopKKernelKind::ThresholdGroupedCpasync, 0},
    };
    if (!name || !name[0]) name = "chunked";
    for (const auto& combo : kCombos) {
        if (std::strcmp(combo.name, name) == 0) return &combo;
    }
    return nullptr;
}

inline bool is_threshold_kind(ResidentL2TopKKernelKind kind) {
    return kind == ResidentL2TopKKernelKind::ThresholdQuery ||
           kind == ResidentL2TopKKernelKind::ThresholdGrouped ||
           kind == ResidentL2TopKKernelKind::ThresholdGroupedPair ||
           kind == ResidentL2TopKKernelKind::ThresholdGroupedShq ||
           kind == ResidentL2TopKKernelKind::ThresholdGroupedVec4 ||
           kind == ResidentL2TopKKernelKind::ThresholdGroupedVec4Shq ||
           kind == ResidentL2TopKKernelKind::ThresholdGroupedMlp2 ||
           kind == ResidentL2TopKKernelKind::ThresholdGroupedCpasync;
}

inline const ResidentL2TopKCombo* resolve_resident_l2_topk_combo_ptr_or_throw(const char* name) {
    const ResidentL2TopKCombo* combo = lookup_resident_l2_topk_combo(name);
    if (!combo) {
        throw std::invalid_argument(
            std::string("unknown resident IVFFlat combo: ") + (name ? name : "") +
            " (expected chunked|full|dp4a|threshold_query|threshold_grouped|threshold_grouped_pair|threshold_grouped_shq|threshold_grouped_vec4|threshold_grouped_vec4_shq|threshold_grouped_mlp2|threshold_grouped_cpasync)");
    }
    return combo;
}

} // namespace paraselect::resident_fine

#endif // PARASELECT_RESIDENT_L2_TOPK_COMBOS_CUH
