#pragma once
#include <algorithm>
#include "../coarse_policy.h"
#include <cstdint>
#include <stdexcept>
namespace paraselect::coarse::current {
enum class Family { Adaptive, Histogram };
// Formal dispatcher paths only. The materialized GEMM baseline is exposed
// through the explicit coarse::detail ablation benchmark entry point.
enum class Path { SimdFused, SimdAdaptive, GemmAdaptive, ProducerConsumer, Histogram, HistogramPartition };
enum class Consumer { Small, Count512, Adaptive, WideSort, SegmentedSort };
enum class Layout { Plain, Swizzle, SwizzleL1 };
// One prepared value: no environment, variant spelling, or historical switches.
struct Plan {
    int batch, nlist, k;
    int logical_k=0;
    int dim=128, logical_dim=128, distance_mode=0;
    bool fp16=true, fused=false;
    Path path=Path::Histogram;
    int parts=1, capacity=0, splits=1;
    Consumer consumer=Consumer::Adaptive;
    Layout layout=Layout::Plain;
    int sort_run=8192;
    double oversample=4;
    bool reject_empty=true, count_cache=false;
    bool register_producer=false; // retained wide-sort producer, distinct from warp pipeline
    Family family() const {
        return path==Path::Histogram || path==Path::HistogramPartition ||
            (path==Path::ProducerConsumer && (consumer==Consumer::Count512 || consumer==Consumer::WideSort || consumer==Consumer::SegmentedSort))
            ? Family::Histogram : Family::Adaptive;
    }
};
inline bool supported(int b,int n,int k) {
    if(b<=0 || b>10000 || n<=0)return false;
    return k>=1 && k<=16384;
}
inline const char* path_name(Path p) {
    switch(p){case Path::SimdFused:return "simd_fused";case Path::SimdAdaptive:return "simd_adaptive";
    case Path::GemmAdaptive:return "gemm_adaptive";case Path::ProducerConsumer:return "producer_consumer";
    case Path::Histogram:return "histogram_sort";case Path::HistogramPartition:return "histogram_partition";}
    throw std::invalid_argument("invalid coarse path");
}
inline int wide_capacity(bool large_nlist, int kernel_k) {
    if (large_nlist) return 2048;
    return kernel_k <= 8192 ? 1024 : 1536;
}
inline int aligned_adaptive_parts(int parts,int n) {
    while(parts>1 && (parts==2 || n%parts || (n/parts)%512))parts/=2;
    return parts;
}
inline void validate(const Plan& p) {
    if(!supported(p.batch,p.nlist,p.k))throw std::invalid_argument("current coarse requires dim128 FP16 supported nlist/k, batch1..10000");
    if(p.path==Path::ProducerConsumer) {
        if(p.parts<=0 || p.splits<=0 || p.capacity<=0 || p.nlist%(p.parts*p.splits) ||
           (p.nlist/p.parts/p.splits)%128 || p.parts*p.capacity<=p.k || (p.parts*p.capacity)%256)
            throw std::invalid_argument("invalid coarse partition/candidate geometry");
        if(p.consumer==Consumer::Count512 && (p.k!=512 || p.parts!=256 || p.capacity!=32))
            throw std::invalid_argument("counted consumer requires k512 P256 C32");
    }
}
inline Plan make_plan(int b,int n,int k) {
    if(!supported(b,n,k))throw std::invalid_argument("unsupported current coarse shape");
    const int logical_k=k;
    // Public probe counts are continuous. Reuse the next supported kernel
    // size and let the runtime trim sorted output to logical_k.
    int kernel_k=8; while(kernel_k<k) kernel_k<<=1;
    if(kernel_k==4096)kernel_k=8192;
    Plan p{b,n,k}; p.logical_k=logical_k; p.k=kernel_k;
    if(b<4) {p.path=p.k<=64?Path::SimdFused:p.k<=512?Path::SimdAdaptive:Path::Histogram;p.parts=p.path==Path::SimdAdaptive?aligned_adaptive_parts(32,n):1;return p;}
    p.path=std::int64_t(b)*n>= (std::int64_t(1)<<24)?Path::ProducerConsumer:p.k<=512?Path::GemmAdaptive:Path::Histogram;
    // Measured crossover regions.  Keep policy selection continuous: exact
    // benchmark coordinates must not create hidden one-point routes.
    const bool large_nlist = n > 131072;
    if(large_nlist && b <= 8 && p.k <= 8)p.path=Path::HistogramPartition;
    if((!large_nlist && b >= 128 && b < 256 && p.k <= 32) ||
       (!large_nlist && b >= 128 && b < 256 && p.k >= 2048) ||
       (large_nlist && b >= 32 && b < 64 && p.k >= 128 && p.k < 512))
        p.path=Path::GemmAdaptive;
    if(p.k >= 16384 && ((!large_nlist && b >= 128 && b < 256) ||
                      (large_nlist && b >= 32 && b < 64)))
        p.path=Path::Histogram;
    // Count512 owns 256 partitions, each with at least one 128-center tile.
    if(p.path==Path::ProducerConsumer && p.k==512 && n<256*128)
        p.path=Path::GemmAdaptive;
    if(p.path==Path::GemmAdaptive) {
        p.parts=b<=8?(!large_nlist && p.k<=8?32:16):b<=32?(p.k<=16?1:(p.k>=32 && p.k<64)?8:4):1;
    } else if(p.path==Path::ProducerConsumer) {
        const bool counted_512 = p.k>511 && p.k<513;
        p.parts=counted_512?256:16;p.capacity=p.k<=128?128:counted_512?32:2048;
        p.consumer=p.k<=128?Consumer::Small:counted_512?Consumer::Count512:p.k<=2048?Consumer::Adaptive:Consumer::WideSort;
        p.layout=b<=128 && p.k<=2048?Layout::Swizzle:Layout::SwizzleL1;
        if(!large_nlist)p.splits=p.k<=128?(b<=512?16:b<=1024?8:b<=8192?4:1):(b<=512?16:1);
        else p.splits=(b>=128 && b<256)?8:b<=1024?16:b<=8192?8:p.k<=128?4:1;
        if(counted_512) {
            p.splits=large_nlist && b>=128 && b<256?4:1;
            p.layout=!large_nlist && b>=2048 && b<4096?Layout::SwizzleL1:Layout::Swizzle;
            p.reject_empty=large_nlist;p.count_cache=large_nlist;
        }
        if(p.k>=8192)p.oversample=!large_nlist?1.25:1.5;
        // Retained capacity/sort presets. These are whole regions, not a point-minimum table.
        if((!large_nlist && p.k>=8192) || (large_nlist && p.k>=16384 && b<=4096)) {
            p.register_producer=true;p.layout=Layout::Plain;p.splits=16;
            p.capacity=wide_capacity(large_nlist,p.k);
            if(large_nlist && b>=128 && b<256)p.consumer=Consumer::SegmentedSort;
            if(large_nlist && b>=256 && b<512)p.sort_run=2048;
        }
    }
    // Partitioned admission should shrink the input. On a small index its
    // reserved score/ID pool can exceed the entire GEMM distance matrix.
    if(p.path==Path::ProducerConsumer && p.parts*p.capacity>=n) {
        p.path=Path::GemmAdaptive;
        p.parts=1;
    }
    // Vector norm loading requires complete 512-center segments per partition.
    // Smaller indexes (including query tails) must use fewer partitions.
    if(p.path==Path::GemmAdaptive)
        p.parts=aligned_adaptive_parts(p.parts,n);
    validate(p);return p;
}
// Above 128 dimensions only the materialized family applies: SIMD and the PC
// producers hold a 128-wide center tile in registers/shared memory, whereas the
// cuBLAS GEMM and the adaptive/histogram selectors are dimension-independent.
// Those routes become GEMM + adaptive selection (histogram_sort beyond 2048).
inline Plan make_wide_plan(int b,int n,int k,int dim) {
    if(dim<=128 || dim%8)throw std::invalid_argument("wide coarse plan requires dim > 128 and dim % 8 == 0");
    Plan p=make_plan(b,n,k);
    p.dim=p.logical_dim=dim;
    if(p.path==Path::SimdFused || p.path==Path::SimdAdaptive || p.path==Path::ProducerConsumer) {
        p.consumer=Consumer::Adaptive;p.capacity=0;p.splits=1;p.layout=Layout::Plain;
        p.register_producer=false;p.count_cache=false;
        if(p.k>2048) {p.path=Path::Histogram;p.parts=1;}
        else {
            p.path=Path::GemmAdaptive;
            p.parts=b<=8?16:b<=32?(p.k<=16?1:(p.k>=32 && p.k<64)?8:4):1;
            p.parts=aligned_adaptive_parts(p.parts,n);
        }
    }
    validate(p);return p;
}
} // namespace paraselect::coarse::current
