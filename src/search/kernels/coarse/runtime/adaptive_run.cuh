#pragma once
#include "kernels/adaptive.cuh"
namespace paraselect::coarse::current {
// Same selector/finish sequence as the frozen route, without variant parsing.
inline void run_adaptive(const detail::Execution& a,Path path,int parts,
                         float* scratch,float* scratch2,idx_t* ids0,idx_t* ids1) {
    using namespace kernels;
    if(path==Path::SimdFused) {simd_fused::paired(a,scratch,scratch2,ids0,ids1);return;}
    if(path==Path::SimdAdaptive) {
        if(a.n_query==2)simt_dot_pair<__half,16,2><<<dim3(a.nlist/16,1),256,0,a.stream>>>(a.d_query_fp16,a.d_cluster_centers_fp16,a.nlist,a.d_inner_product);
        else simt_dot_pair<__half,16,1><<<dim3(a.nlist/16,a.n_query),256,0,a.stream>>>(a.d_query_fp16,a.d_cluster_centers_fp16,a.nlist,a.d_inner_product);
    } else materialized_dot(a);
    const bool finish=path!=Path::SimdAdaptive;
    float* values=parts==1?a.d_top_nprobe_dist:scratch;
    idx_t* ids=parts==1?a.d_top_nprobe_index:ids0;
    auto select=[&]<int P>() {
        if(a.nprobe>2048) {
            if constexpr(P==1)fused_partition_select<1,true,128,true>(a,values,ids,false);
            else throw std::invalid_argument("wide adaptive fallback requires one partition");
        } else if(a.nprobe<=512)fused_partition_select<P,true,256>(a,values,ids,finish && P==1);
        else fused_partition_select<P,true,128>(a,values,ids,finish && P==1);
    };
    switch(parts) {
    case 1: select.template operator()<1>();break;
    case 4: select.template operator()<4>();break;
    case 8: select.template operator()<8>();break;
    case 16: select.template operator()<16>();break;
    case 32: select.template operator()<32>();break;
    default: throw std::invalid_argument("unsupported adaptive partitions");
    }
    if(parts==1) {
        if(!finish || a.nprobe>2048)restore_fused_values<<<(a.n_query*a.nprobe+255)/256,256,0,a.stream>>>(a.d_top_nprobe_dist,a.n_query*a.nprobe);
    } else if(finish)owned_merge_and_finish(a,parts,scratch,ids0);
    else {
        adaptive_threshold_topk_fp32(scratch,a.n_query,parts*a.nprobe,parts*a.nprobe,a.nprobe,a.d_top_nprobe_dist,a.d_top_nprobe_index,a.stream);
        finish_adaptive_split<<<(a.n_query*a.nprobe+255)/256,256,0,a.stream>>>(a.d_top_nprobe_dist,a.d_top_nprobe_index,ids0,a.nprobe,parts,a.nlist/parts,a.n_query*a.nprobe);
    }
}
}
