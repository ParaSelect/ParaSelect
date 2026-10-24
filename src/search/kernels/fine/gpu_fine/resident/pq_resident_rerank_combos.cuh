#ifndef PARASELECT_PQ_RESIDENT_RERANK_COMBOS_CUH
#define PARASELECT_PQ_RESIDENT_RERANK_COMBOS_CUH

#include <cstring>
#include <stdexcept>
#include <string>

namespace paraselect::pq_resident_fine {

enum class PQResidentRerankKernelKind {
    Full = 0,
};

struct PQResidentRerankCombo {
    const char* name;
    PQResidentRerankKernelKind kind;
};

inline const PQResidentRerankCombo* lookup_pq_resident_rerank_combo(const char* name) {
    static constexpr PQResidentRerankCombo kCombos[] = {
        {"full", PQResidentRerankKernelKind::Full},
    };
    if (!name || name[0] == '\0') return nullptr;
    for (const auto& combo : kCombos) {
        if (std::strcmp(name, combo.name) == 0) return &combo;
    }
    return nullptr;
}

inline const PQResidentRerankCombo* resolve_pq_resident_rerank_combo_ptr_or_throw(const char* name) {
    const char* resolved = (name && name[0] != '\0') ? name : "full";
    const PQResidentRerankCombo* combo = lookup_pq_resident_rerank_combo(resolved);
    if (!combo) {
        throw std::invalid_argument(
            std::string("unknown resident IVFPQ combo: ") + resolved + " (valid: full)");
    }
    return combo;
}

}  // namespace paraselect::pq_resident_fine

#endif  // PARASELECT_PQ_RESIDENT_RERANK_COMBOS_CUH
