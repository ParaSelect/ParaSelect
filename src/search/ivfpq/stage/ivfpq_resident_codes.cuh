#ifndef PARASELECT_IVFPQ_RESIDENT_CODES_CUH
#define PARASELECT_IVFPQ_RESIDENT_CODES_CUH

#include "search/ivfpq/ivfpq_search_context.h"

void ivfpq_resident_codes_prepare_storage(IVFPQSearchContext* ctx);
void ivfpq_resident_codes_prepare_workspace(IVFPQSearchContext* ctx);
void ivfpq_resident_codes_release(IVFPQSearchContext* ctx);

#endif  // PARASELECT_IVFPQ_RESIDENT_CODES_CUH
