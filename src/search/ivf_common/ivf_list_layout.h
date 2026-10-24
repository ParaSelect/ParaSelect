#ifndef PARASELECT_SEARCH_IVF_COMMON_IVF_LIST_LAYOUT_H
#define PARASELECT_SEARCH_IVF_COMMON_IVF_LIST_LAYOUT_H

#include "utils/types.h"

#include <climits>
#include <stdexcept>
#include <vector>

namespace paraselect::search {

struct IVFListLayoutView {
    const idx_t* offsets = nullptr;       // [nlist + 1]
    const idx_t* reordered_ids = nullptr; // [n_vectors]
    int nlist = 0;
    idx_t n_vectors = 0;
};

inline void validate_ivf_list_layout(const IVFListLayoutView& lists) {
    if (!lists.offsets) {
        throw std::invalid_argument("IVF list offsets are null");
    }
    if (lists.nlist <= 0 || lists.n_vectors < 0) {
        throw std::invalid_argument("invalid IVF list layout shape");
    }
    if (lists.n_vectors > 0 && !lists.reordered_ids) {
        throw std::invalid_argument("IVF reordered IDs are null");
    }
    if (lists.offsets[0] != 0) {
        throw std::invalid_argument("IVF list offsets must start at zero");
    }
    for (int list = 0; list < lists.nlist; ++list) {
        const idx_t begin = lists.offsets[list];
        const idx_t end = lists.offsets[list + 1];
        if (end < begin) {
            throw std::invalid_argument("IVF list offsets must be monotonic");
        }
        if (end - begin > INT_MAX) {
            throw std::invalid_argument("IVF list size exceeds int32 backend limit");
        }
    }
    if (lists.offsets[lists.nlist] != lists.n_vectors) {
        throw std::invalid_argument("IVF terminal offset does not match n_vectors");
    }
}

inline std::vector<int> derive_ivf_list_counts(const IVFListLayoutView& lists) {
    validate_ivf_list_layout(lists);
    std::vector<int> counts((size_t)lists.nlist);
    for (int list = 0; list < lists.nlist; ++list) {
        counts[(size_t)list] = static_cast<int>(lists.offsets[list + 1] - lists.offsets[list]);
    }
    return counts;
}

}  // namespace paraselect::search

#endif  // PARASELECT_SEARCH_IVF_COMMON_IVF_LIST_LAYOUT_H
