#pragma once
// Validated SM120 center layout. Tests compare every fragment element against WMMA loads.
// Vector-sized XOR preserves each 16-byte copy's alignment and element order.
__device__ __forceinline__ int swizzled_center_offset(int row,int col) {
    return row*32+(col^(((row>>1)&3)*8));
}

template<bool CenterL1=false>
__device__ __forceinline__ void prefetch_swizzled_centers_warp(
    __half* destination,const __half* centers,int base,int dim) {
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    #pragma unroll
    for(int pass=0;pass<2;++pass) {
        const int row=warp*16+lane/4+pass*8,col=(lane%4)*8;
        const unsigned target=static_cast<unsigned>(__cvta_generic_to_shared(
            destination+swizzled_center_offset(row,col)));
        const __half* source=centers+size_t(base+row)*128+dim+col;
        if constexpr(CenterL1)
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                :: "r"(target),"l"(source) : "memory");
        else asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
            :: "r"(target),"l"(source) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template<class Fragment>
__device__ __forceinline__ void load_swizzled_center_fragment(
    Fragment& fragment,const __half* source,int first_row,int first_col) {
    static_assert(Fragment::num_elements==16);
    const int lane=threadIdx.x%32;
    const int row=first_row+lane%16,col=first_col+(lane/16)*8;
    const unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(
        source+swizzled_center_offset(row,col)));
    unsigned raw[4];
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(raw[0]),"=r"(raw[1]),"=r"(raw[2]),"=r"(raw[3]) : "r"(address));
    // WMMA's FP16 B fragment duplicates eight half elements on SM120 and
    // orders the middle two register pairs differently from native ldmatrix.
    constexpr int mapping[8]={0,1,4,5,2,3,6,7};
    #pragma unroll
    for(int i=0;i<16;++i)fragment.x[i]=reinterpret_cast<const __half*>(raw)[mapping[i&7]];
}
