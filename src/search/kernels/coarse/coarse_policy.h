#pragma once
#include <climits>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <initializer_list>
// Public requests name a family (`adaptive` or `histogram`). The concrete
// entries below are internal execution targets used by the prepared plan and
// by the comparison benchmark.
enum IVFCoarseL2TopKPolicy {
 IVF_COARSE_REDUCER_ADAPTIVE_DIRECT=3,
 IVF_COARSE_REDUCER_ADAPTIVE_FUSED=4,
 IVF_COARSE_REDUCER_HISTOGRAM_SORT=7, IVF_COARSE_REDUCER_HISTOGRAM_PARTITION_SMALLK=8,
 IVF_COARSE_REDUCER_ADAPTIVE=9, IVF_COARSE_REDUCER_HISTOGRAM=10
};
namespace paraselect::coarse {
inline constexpr int coarse_reducers[] = {
 IVF_COARSE_REDUCER_ADAPTIVE_DIRECT, IVF_COARSE_REDUCER_ADAPTIVE_FUSED,
 IVF_COARSE_REDUCER_HISTOGRAM_SORT, IVF_COARSE_REDUCER_HISTOGRAM_PARTITION_SMALLK
};
inline const char* l2_topk_policy_name(int id) {
 switch(id) {
 case -1: return "auto";
 case IVF_COARSE_REDUCER_ADAPTIVE: return "adaptive";
 case IVF_COARSE_REDUCER_HISTOGRAM: return "histogram";
 case IVF_COARSE_REDUCER_ADAPTIVE_DIRECT: return "adaptive_direct";
 case IVF_COARSE_REDUCER_ADAPTIVE_FUSED: return "adaptive_fused";
 case IVF_COARSE_REDUCER_HISTOGRAM_SORT: return "histogram_sort";
 case IVF_COARSE_REDUCER_HISTOGRAM_PARTITION_SMALLK: return "histogram_partition_smallk";
 default: throw std::invalid_argument("unknown concrete coarse reducer");
 }
}
// -1 denotes automatic selection; all other values are existing reducer IDs.
inline int parse_coarse_reducer_request(const char* name) {
 if(!name || !*name || std::strcmp(name,"auto")==0) return -1;
 if(std::strcmp(name,"adaptive")==0) return IVF_COARSE_REDUCER_ADAPTIVE;
 if(std::strcmp(name,"histogram")==0) return IVF_COARSE_REDUCER_HISTOGRAM;
 for(int id:coarse_reducers) if(std::strcmp(name,l2_topk_policy_name(id))==0) return id;
 throw std::invalid_argument(std::string("unknown/retired coarse reducer: ")+name);
}

inline bool coarse_reducer_supported(int id,int batch,int n,int k,bool fp16) {
 if(batch<=0 || n<=0 || k<=0 || k>n) return false;
#ifndef PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK
 if(id==IVF_COARSE_REDUCER_ADAPTIVE || id==IVF_COARSE_REDUCER_ADAPTIVE_DIRECT || id==IVF_COARSE_REDUCER_ADAPTIVE_FUSED) return false;
#endif
 const bool adaptive=n<(1<<23) && n%256==0 && k>=8 && k<=16384 && k%8==0;
 switch(id) {
 case IVF_COARSE_REDUCER_ADAPTIVE:return adaptive;
 case IVF_COARSE_REDUCER_HISTOGRAM:return batch<=65535;
 case IVF_COARSE_REDUCER_ADAPTIVE_DIRECT:return adaptive;
 case IVF_COARSE_REDUCER_ADAPTIVE_FUSED:return adaptive && k<=2048 && n>k;
 case IVF_COARSE_REDUCER_HISTOGRAM_SORT:return batch<=65535;
 case IVF_COARSE_REDUCER_HISTOGRAM_PARTITION_SMALLK:return fp16 && batch<=65535 && (k==8||k==16||k==32) && size_t((n+4095)/4096)*k<=4096;
 default:return false;
 }
}
inline void require_coarse_reducer(int id,int batch,int n,int k,bool fp16) {
 if(!coarse_reducer_supported(id,batch,n,k,fp16))
  throw std::invalid_argument(std::string("unsupported shape for coarse reducer ")+l2_topk_policy_name(id));
}
inline bool coarse_reducer_is_adaptive(int id) {
 return id==IVF_COARSE_REDUCER_ADAPTIVE || id==IVF_COARSE_REDUCER_ADAPTIVE_DIRECT ||
        id==IVF_COARSE_REDUCER_ADAPTIVE_FUSED;
}
inline bool coarse_reducer_is_histogram(int id) {
 return id==IVF_COARSE_REDUCER_HISTOGRAM || id==IVF_COARSE_REDUCER_HISTOGRAM_SORT ||
        id==IVF_COARSE_REDUCER_HISTOGRAM_PARTITION_SMALLK;
}
inline bool coarse_reducer_needs_histogram(int id) { return coarse_reducer_is_histogram(id); }
inline int coarse_reducer_score_copies(int id,int k) {
 return 1+(id==IVF_COARSE_REDUCER_ADAPTIVE_DIRECT && k>2048 ? 1:0);
}
// Select one concrete reducer valid for both full centroid tiles and the tail.
inline int select_coarse_reducer_plan(int request,int batch,int n,int k,
                                     int tile,bool histogram_prepared,bool fp16) {
 if(batch<=0 || n<=0 || k<=0 || k>n) throw std::invalid_argument("invalid coarse plan shape");
 const int width=fp16 || tile<=0 || tile>n ? n:tile;
 const int tail=n%width;
 auto supports=[&](int id) {
  return (!coarse_reducer_needs_histogram(id) || histogram_prepared) &&
   coarse_reducer_supported(id,batch,width,k<width?k:width,fp16) &&
   (!tail || coarse_reducer_supported(id,batch,tail,k<tail?k:tail,fp16));
 };
 if(request != -1) {
  if(request==IVF_COARSE_REDUCER_ADAPTIVE) {
   if(batch>=256 && supports(IVF_COARSE_REDUCER_ADAPTIVE_FUSED)) return IVF_COARSE_REDUCER_ADAPTIVE_FUSED;
   if(supports(IVF_COARSE_REDUCER_ADAPTIVE_DIRECT)) return IVF_COARSE_REDUCER_ADAPTIVE_DIRECT;
   throw std::invalid_argument("adaptive family has no supported internal path");
  }
  if(request==IVF_COARSE_REDUCER_HISTOGRAM) {
   if(fp16 && supports(IVF_COARSE_REDUCER_HISTOGRAM_PARTITION_SMALLK))
    return IVF_COARSE_REDUCER_HISTOGRAM_PARTITION_SMALLK;
   if(supports(IVF_COARSE_REDUCER_HISTOGRAM_SORT)) return IVF_COARSE_REDUCER_HISTOGRAM_SORT;
   throw std::invalid_argument("histogram family has no supported internal path");
  }
  if(!supports(request)) throw std::invalid_argument("explicit reducer cannot serve all prepared centroid tiles");
  return request;
 }
 if(k>2048 && supports(IVF_COARSE_REDUCER_HISTOGRAM_SORT)) return IVF_COARSE_REDUCER_HISTOGRAM_SORT;
 if(batch>=256 && supports(IVF_COARSE_REDUCER_ADAPTIVE_FUSED)) return IVF_COARSE_REDUCER_ADAPTIVE_FUSED;
 for(int id:{IVF_COARSE_REDUCER_ADAPTIVE_DIRECT,IVF_COARSE_REDUCER_HISTOGRAM_SORT}) if(supports(id)) return id;
 throw std::invalid_argument("no reducer supports all prepared centroid tiles");
}
inline int select_coarse_reducer(int request,int batch,int n,int k,bool histogram_prepared,bool fp16) {
 return select_coarse_reducer_plan(request,batch,n,k,n,histogram_prepared,fp16);
}


inline bool fused_gemm_shape_supported(int batch,int dim,int n,int k) {
#ifndef PARASELECT_ENABLE_ADAPTIVE_THRESHOLD_TOPK
    return false;
#else
    return batch>0 && dim==128 && n>=2048 && (n&(n-1))==0 &&
        (k==1 || k==8 || k==16 || k==32 || k==64 || k==128 || k==512);
#endif
}
inline bool select_fused_gemm_backend(int reducer,bool fp16,int batch,int dim,int n,int k) {
    const bool adaptive_family = reducer == -1 || reducer == IVF_COARSE_REDUCER_ADAPTIVE;
    return adaptive_family && fp16 && batch>=256 && (k==8 || k==16 || k==32) &&
        fused_gemm_shape_supported(batch,dim,n,k);
}

}
