#pragma once
#include "search/kernels/coarse/coarse_policy.h"
#include <cstdlib>
#include "utils/pch.h"
#include "runtime/coarse_runtime.h"
namespace paraselect::coarse {
inline int read_coarse_reducer_request() {
    return parse_coarse_reducer_request(std::getenv("PARASELECT_COARSE_STRATEGY"));
}
// All callers supply geometry and capabilities, never a concrete backend.
inline current::Plan select_coarse_plan(const Request& a,int request,
                                       bool asynchronous_only,bool sm120) {
    if(a.distance_mode!=L2_DISTANCE && a.distance_mode!=COSINE_DISTANCE)
        throw std::invalid_argument("unsupported coarse distance mode");
#ifdef PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK
    // The prepared FP16 dispatcher is the only production entry.  The old
    // materialized implementation remains available to explicit ablation
    // benchmarks, but it must never be selected as a compatibility fallback.
    if(sm120 && request==-1 && a.fp16_coarse && a.distance_mode==L2_DISTANCE &&
       (a.dim==96 || a.dim==100 || a.dim==128) && current::supported(a.n_query,a.nlist,a.nprobe)) {
        auto p=current::make_plan(a.n_query,a.nlist,a.nprobe);
        p.logical_dim=a.dim;
        if(asynchronous_only && p.path==current::Path::ProducerConsumer) {
            p.path=p.k<=2048 ? current::Path::GemmAdaptive : current::Path::Histogram;
            p.parts=1;
        }
        return p;
    }
    if(sm120 && request==-1 && a.fp16_coarse && a.distance_mode==L2_DISTANCE &&
       a.dim>128 && a.dim%8==0 && current::supported(a.n_query,a.nlist,a.nprobe))
        return current::make_wide_plan(a.n_query,a.nlist,a.nprobe,a.dim);
#endif
    throw std::invalid_argument(
        "no formal coarse dispatcher plan for this request; "
        "use the explicit materialized ablation entry");
}
inline std::unique_ptr<Workspace> prepare_coarse(
    bool fp16, int distance, int batch, int dim, int nlist, int k) {
    Request a;
    a.fp16_coarse = fp16;
    a.distance_mode = distance;
    a.n_query = batch;
    a.dim = dim;
    a.nlist = nlist;
    a.nprobe = k;
    return std::make_unique<Workspace>(a, -1, false);
}
}
