#ifndef PARASELECT_SEARCH_IVF_COMMON_IVF_PROBE_VIEW_H
#define PARASELECT_SEARCH_IVF_COMMON_IVF_PROBE_VIEW_H

#include "utils/pch.h"

namespace paraselect::search {

// Non-owning view of query-major IVF probe IDs resident on a CUDA device.
struct IVFDeviceProbeView {
    const idx_t* indices = nullptr;
    int query_count = 0;
    int probe_count = 0;
};

}  // namespace paraselect::search

#endif  // PARASELECT_SEARCH_IVF_COMMON_IVF_PROBE_VIEW_H
