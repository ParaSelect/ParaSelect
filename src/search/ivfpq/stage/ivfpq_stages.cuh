#ifndef PARASELECT_IVFPQ_STAGES_H
#define PARASELECT_IVFPQ_STAGES_H

#include "search/ivfpq/ivfpq_search_context.h"

void ivfpq_prepare_index(IVFPQSearchContext* ctx);
void ivfpq_prepare(IVFPQSearchContext* ctx);
void ivfpq_release(IVFPQSearchContext* ctx);
void ivfpq_coarse(IVFPQSearchContext* ctx, const float* h_query, int n_query, idx_t* h_top_nprobe_out);
void ivfpq_coarse_to_device(IVFPQSearchContext* ctx, const float* h_query, int n_query);
void ivfpq_fine_cpu_lut(IVFPQSearchContext* ctx, int batch_start, int batch_n);
void ivfpq_fine_resident_codes_gpu(IVFPQSearchContext* ctx, int batch_start, int batch_n);
void ivfpq_rerank_cpu(IVFPQSearchContext* ctx, int batch_start, int batch_n);
void ivfpq_map_ids(IVFPQSearchContext* ctx, int batch_start, int batch_n);

#endif  // PARASELECT_IVFPQ_STAGES_H
