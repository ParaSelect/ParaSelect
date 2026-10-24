#ifndef PARASELECT_RESIDUAL_PQ_GPU_COMBOS_H
#define PARASELECT_RESIDUAL_PQ_GPU_COMBOS_H

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "search/kernels/fine/fine_search_strategy.h"

namespace paraselect {
namespace search {

enum class ResidualPQGpuCodeLayout {
    kRowMajor = 0,
};

struct ResidualPQGpuCombo {
    const char* name;
    ResidualPQGpuImpl impl;
    int thread_k;
    int local_k;
    int global_k;
};

inline int residual_pq_gpu_probe_block_threads(int local_k) {
    if (local_k <= 10) return 8 * 32;
    if (local_k <= 16) return 4 * 32;
    return 2 * 32;
}

inline bool residual_pq_gpu_impl_uses_probe_chunks(ResidualPQGpuImpl impl) {
    return impl == ResidualPQGpuImpl::ProbeChunk;
}

inline bool residual_pq_gpu_impl_uses_vector_chunks(ResidualPQGpuImpl impl) {
    return impl == ResidualPQGpuImpl::VectorChunk;
}

inline bool residual_pq_gpu_impl_uses_grouped_vector_chunks(ResidualPQGpuImpl impl) {
    return impl == ResidualPQGpuImpl::GroupedVectorChunk;
}

inline int residual_pq_gpu_default_probes_per_chunk(ResidualPQGpuImpl impl) {
    return residual_pq_gpu_impl_uses_probe_chunks(impl) ? 4 : 1;
}

inline int residual_pq_gpu_default_vector_chunk_size(ResidualPQGpuImpl impl) {
    return (residual_pq_gpu_impl_uses_vector_chunks(impl) ||
            residual_pq_gpu_impl_uses_grouped_vector_chunks(impl)) ? 2048 : 0;
}

inline int residual_pq_gpu_default_group_chunks(ResidualPQGpuImpl impl) {
    return residual_pq_gpu_impl_uses_grouped_vector_chunks(impl) ? 8 : 1;
}

inline int residual_pq_gpu_default_fine_query_tile(ResidualPQGpuImpl impl) {
    return residual_pq_gpu_impl_uses_grouped_vector_chunks(impl) ? 512 : 0;
}

inline int normalize_residual_pq_thread_k(int thread_k) {
    if (thread_k <= 4) return 4;
    if (thread_k <= 8) return 8;
    return 16;
}

inline int normalize_residual_pq_local_k(int requested, int topk) {
    if (requested <= topk) return topk;
    if (requested <= 16) return 16;
    if (requested <= 32) return 32;
    return 256;
}

inline const ResidualPQGpuCombo* lookup_residual_pq_gpu_combo_ptr(const char* name) {
    static constexpr ResidualPQGpuCombo kCombos[] = {
        {
            "auto",
            ResidualPQGpuImpl::GroupedVectorChunk,
            8,
            256,
            512,
        },
        {
            "dense_baseline",
            ResidualPQGpuImpl::Dense,
            8,
            32,
            128,
        },
        {
            "candidate64",
            ResidualPQGpuImpl::GroupedVectorChunk,
            8,
            256,
            64,
        },
        {
            "candidate10",
            ResidualPQGpuImpl::Dense,
            8,
            10,
            10,
        },
        {
            "target_chunk",
            ResidualPQGpuImpl::GroupedVectorChunk,
            8,
            256,
            512,
        },
        {
            "vector_chunk_baseline",
            ResidualPQGpuImpl::VectorChunk,
            8,
            256,
            512,
        },
    };
    if (!name || name[0] == '\0') return nullptr;
    for (const auto& combo : kCombos) {
        if (std::strcmp(name, combo.name) == 0) return &combo;
    }
    return nullptr;
}

inline const char* resolve_default_residual_pq_gpu_combo_name(const char* dataset_name) {
    (void)dataset_name;
    return "auto";
}

inline const char* resolve_residual_pq_gpu_combo_name(
    const char* combo_name,
    const char* dataset_name = nullptr)
{
    if (!combo_name || combo_name[0] == '\0' || std::strcmp(combo_name, "default") == 0) {
        return resolve_default_residual_pq_gpu_combo_name(dataset_name);
    }
    return combo_name;
}

inline const ResidualPQGpuCombo* resolve_residual_pq_gpu_combo_ptr_or_throw(
    const char* combo_name,
    const char* dataset_name = nullptr)
{
    const char* resolved = resolve_residual_pq_gpu_combo_name(combo_name, dataset_name);
    const ResidualPQGpuCombo* combo = lookup_residual_pq_gpu_combo_ptr(resolved);
    if (!combo) {
        throw std::invalid_argument(
            std::string("invalid rpq_gpu_combo_name: ") + (resolved ? resolved : "<null>"));
    }
    return combo;
}

inline ResidualPQGpuSearchSpec make_residual_pq_gpu_spec_from_combo(
    const ResidualPQGpuCombo& combo,
    int topk,
    int pq_m,
    int pq_k)
{
    ResidualPQGpuSearchSpec spec;
    spec.topk = topk;
    spec.local_k = combo.local_k;
    spec.thread_k = combo.thread_k;
    spec.global_k = combo.global_k;
    spec.impl = combo.impl;
    spec.pq_m = pq_m;
    spec.pq_k = pq_k;
    spec.probes_per_chunk = residual_pq_gpu_default_probes_per_chunk(combo.impl);
    spec.vector_chunk_size = residual_pq_gpu_default_vector_chunk_size(combo.impl);
    spec.group_chunks = residual_pq_gpu_default_group_chunks(combo.impl);
    spec.fine_query_tile = residual_pq_gpu_default_fine_query_tile(combo.impl);
    return spec;
}

inline ResidualPQGpuSearchSpec make_residual_pq_gpu_spec_from_combo_with_overrides(
    const ResidualPQGpuCombo& combo,
    int topk,
    int pq_m,
    int pq_k,
    bool allow_override,
    int override_thread_k = -1,
    int override_local_k = -1,
    int override_global_k = -1,
    int override_max_splits = -1,
    int override_split_threshold = -1)
{
    ResidualPQGpuSearchSpec spec =
        make_residual_pq_gpu_spec_from_combo(combo, topk, pq_m, pq_k);
    if (!allow_override) return spec;
    if (override_thread_k > 0) spec.thread_k = normalize_residual_pq_thread_k(override_thread_k);
    if (override_local_k > 0) spec.local_k = normalize_residual_pq_local_k(override_local_k, topk);
    if (override_global_k > 0) spec.global_k = override_global_k;
    return spec;
}

inline bool residual_pq_gpu_combo_uses_rowmajor(const ResidualPQGpuCombo& combo) {
    (void)combo;
    return true;
}

inline const char* residual_pq_gpu_code_layout_name(const ResidualPQGpuCombo& combo) {
    (void)combo;
    return "gpu_rowmajor";
}

}  // namespace search
}  // namespace paraselect

#endif  // PARASELECT_RESIDUAL_PQ_GPU_COMBOS_H
