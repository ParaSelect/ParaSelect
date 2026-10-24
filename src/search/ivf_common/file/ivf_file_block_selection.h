#ifndef PARASELECT_SEARCH_IVF_COMMON_FILE_IVF_FILE_BLOCK_SELECTION_H
#define PARASELECT_SEARCH_IVF_COMMON_FILE_IVF_FILE_BLOCK_SELECTION_H

#include <cstddef>

namespace paraselect::search {

struct IVFFileBlockSelection {
    int sparse_ref_threshold = 0;
    int protected_candidate_rank = 0;

    bool enabled() const {
        return sparse_ref_threshold > 0 && protected_candidate_rank > 0;
    }
};

inline bool ivf_file_block_selected(
    const IVFFileBlockSelection& selection,
    bool cached,
    size_t candidate_refs,
    int best_pq_rank) {
    return !selection.enabled() || cached ||
        candidate_refs > static_cast<size_t>(selection.sparse_ref_threshold) ||
        best_pq_rank < selection.protected_candidate_rank;
}

}  // namespace paraselect::search

#endif  // PARASELECT_SEARCH_IVF_COMMON_FILE_IVF_FILE_BLOCK_SELECTION_H
