#include "topk_select.cuh"
#include "../validation.h"
#include "../small_topk.cuh"
#include "../adaptive_threshold_topk_fp32.h"
#include "utils/pch.h"
#include <stdexcept>
namespace paraselect::coarse {
__global__ void restore_l2(float*,size_t);
template<int K> void fused_launch(const topk_select_fused::FusedArgs& args) {
    using Config = TopkSelectConfig<float,int64_t,true,false,true,K,128,1,2048,2048,2>;
    topk_select_fused::run_topk_select_kernel<Config>(args);
}
void adaptive_threshold_topk_fused_scores(float* inner,const float* qn,const float* cn,int batch,int n,int k,
                         float* values,int64_t* ids,cudaStream_t stream) {
    detail::validate_l2_input(inner, qn, cn, batch, n, k, values, ids);
    if (k > 2048) throw std::invalid_argument("adaptive_fused supports k <= 2048; select another reducer in prepare");
    // This kernel's threshold loop requires at least one non-retained item.
    if (n == k) throw std::invalid_argument("fused L2 requires length > k");
    if (small_topk::launch_l2_if_supported(inner, qn, cn, batch, n, k,
                                           values, ids, stream)) {
        CUDA_LAUNCH_CHECK("adaptive fused small-K L2 selector");
        return;
    }
    int device = 0; CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    topk_select_fused::FusedArgs a{};
    a.batch_size=batch; a.vocab_size=n; a.topk=k;
    a.input=inner; a.output_value=values; a.output_index=ids;
    a.stride_input_batch=n; a.stride_output_value_batch=k; a.stride_output_index_batch=k;
    a.sorted_value=true; a.return_value=true; a.idx_oob_fill_value=-1;
    a.value_oob_fill_value=-INFINITY; a.abort_when_nan_found=true;
    a.shared_memory_size_per_sm=prop.sharedMemPerBlockOptin; a.stream=stream;
    a.query_norm=qn; a.centroid_norm=cn;
    if (k <= 128) fused_launch<128>(a);
    else if (k <= 256) fused_launch<256>(a);
    else if (k <= 512) fused_launch<512>(a);
    else if (k <= 1024) fused_launch<1024>(a);
    else fused_launch<2048>(a);
}
void adaptive_threshold_topk_fused(float* inner,const float* qn,const float* cn,int batch,int n,int k,
                         float* values,int64_t* ids,cudaStream_t stream) {
    adaptive_threshold_topk_fused_scores(inner,qn,cn,batch,n,k,values,ids,stream);
    restore_l2<<<(size_t(batch)*k+255)/256,256,0,stream>>>(values,size_t(batch)*k);
}
}
