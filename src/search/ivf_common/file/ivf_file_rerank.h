#ifndef PARASELECT_SEARCH_IVF_COMMON_FILE_IVF_FILE_RERANK_H
#define PARASELECT_SEARCH_IVF_COMMON_FILE_IVF_FILE_RERANK_H

#include "search/ivf_common/file/ivf_file_block_selection.h"
#include "search/ivf_common/file/ivf_file_reader.h"
#include "search/ivf_common/ivf_rerank_stats.h"
#include "search/ivf_common/ivf_storage.h"

namespace paraselect::search {

void ivf_file_rerank(
    IVFFileReader& reader,
    const IVFFileStorageView& storage,
    const uint8_t* queries_u8,
    const float* queries_fp32,
    const int* candidates,
    int batch_n,
    int candidate_count,
    int topk,
    int* output_ids,
    float* output_distances,
    const IVFFileBlockSelection& block_selection,
    IVFRerankStats* stats);

void ivf_file_rerank_prewarm(const IVFFileStorageView& storage);

}  // namespace paraselect::search

#endif  // PARASELECT_SEARCH_IVF_COMMON_FILE_IVF_FILE_RERANK_H
