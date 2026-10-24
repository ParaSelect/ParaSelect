#ifndef PARASELECT_CPU_FINE_QUERY_PREPARE_H
#define PARASELECT_CPU_FINE_QUERY_PREPARE_H

#include "search/kernels/fine/cpu_fine/executor/cpu_fine_batch_view.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace paraselect {
namespace cpu_fine {

inline uint8_t quantize_query_u8_sat(float x) {
    if (!(x > 0.0f)) return 0;
    if (x >= 255.0f) return static_cast<uint8_t>(255);
    return static_cast<uint8_t>(x + 0.5f);
}

inline const float* prepare_fp32_query_alias(
    const CpuFineBatchView* view,
    int,
    int)
{
    return view ? view->h_queries : nullptr;
}

inline const float* prepare_u8_query_alias(
    const CpuFineBatchView* view,
    int current_n_querys,
    int dim)
{
    if (!view) return nullptr;
    if (view->h_queries_u8) {
        return reinterpret_cast<const float*>(view->h_queries_u8);
    }
    if (!view->h_queries) return nullptr;
    uint8_t* query_u8_mut = view->h_query_u8_tmp;
    if (!query_u8_mut) {
        throw std::runtime_error("Stage3-CPU: missing h_query_u8_tmp scratch; prepare was not completed");
    }
    const size_t query_elems = static_cast<size_t>(current_n_querys) * static_cast<size_t>(dim);
    for (size_t i = 0; i < query_elems; ++i) {
        query_u8_mut[i] = quantize_query_u8_sat(view->h_queries[i]);
    }
    return reinterpret_cast<const float*>(query_u8_mut);
}

inline CpuFineQueryPrepareFn query_prepare_for_layout(CpuFineVectorLayout layout) {
    return (layout == CPU_FINE_LAYOUT_U8_ROW_MAJOR)
        ? prepare_u8_query_alias
        : prepare_fp32_query_alias;
}

}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_CPU_FINE_QUERY_PREPARE_H
