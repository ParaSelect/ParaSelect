#pragma once
#include "pc_register.cuh"
#include "center_layout.cuh"
// Shared WMMA arithmetic; admission and ownership
// match the tiled producer. Only the center-load/WMMA schedule changes.
template<int KStage, int QueryRows=Rows, bool WarpEpilogue=false, bool RegisterEpilogue=false, bool PackedCenters=false>
struct alignas(32) WmmaPipelineStorage {
    static constexpr int CenterStride=KStage+(PackedCenters?0:8);
    static constexpr int DotRows = QueryRows == 32 ? 32 : 16;
    __half query[QueryRows * 136];
    float dots[RegisterEpilogue ? 1 : WarpEpilogue ? (Threads / 32) * 256 : DotRows * 136];
    float norms[Cols];
    alignas(32) __half center[2][Cols * CenterStride];
};

template<int KStage, bool WarpLocal=false, bool BankOrder=false, bool PackedCenters=false, bool SwizzledCenters=false, bool CenterL1=false>
__device__ __forceinline__ void prefetch_wmma_centers(
    __half* dst, const __half* centers, int base, int dim) {
    if constexpr(SwizzledCenters) {
        static_assert(KStage==32 && WarpLocal && PackedCenters && !BankOrder);
        prefetch_swizzled_centers_warp<CenterL1>(dst,centers,base,dim);
        return;
    }
    // Each thread copies aligned 16-byte vectors. Padding avoids a power-of-two
    // shared-memory row stride without changing the global center layout.
    constexpr int LoadRows=WarpLocal?16:Cols, LoadThreads=WarpLocal?32:Threads;
    const int load_thread=WarpLocal?threadIdx.x%32:threadIdx.x;
    const int first_row=WarpLocal?(threadIdx.x/32)*16:0;
    for (int v = load_thread; v < LoadRows * (KStage / 8); v += LoadThreads) {
        int row = first_row+v / (KStage / 8), col = (v % (KStage / 8)) * 8;
        if constexpr(BankOrder) {
            static_assert(WarpLocal && KStage==32);
            // Preserve each warp copy instruction's global address set;
            // distribute its eight-lane shared write groups across all banks.
            row=first_row+(v/32)*8+v%8;
            col=((v%32)/8)*8;
        }
        unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst + row * (KStage + (PackedCenters?0:8)) + col));
        const __half* global = centers + size_t(base + row) * 128 + dim + col;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(shared), "l"(global) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<int KStage, bool CacheQuery, int QueryRows, bool WarpEpilogue, bool RegisterEpilogue=false, bool CacheMeta=false, bool StageOverlap=false, bool CtaOwned=false, bool RejectEmpty=false, bool CtaInit=false, bool WarpPipeline=false, bool BankOrder=false, bool PackedCenters=false, bool SwizzledCenters=false, bool QueryLookahead=false, bool CenterL1=false, bool InterleaveAdmission=false>
__device__ __forceinline__ void wmma_pipeline_body(
    const __half* queries, const __half* centers, const float* qnorm, const float* cnorm,
    int batch, int nlist, int parts, int capacity, const float* thresholds,
    float* values, int* ids, int* counts, int center_tile) {
    static_assert(KStage == 16 || KStage == 32 || KStage == 64);
    static_assert(QueryRows == 32 || QueryRows == 64);
    static_assert(!CtaInit || CtaOwned);
    static_assert(!InterleaveAdmission || (SwizzledCenters && !QueryLookahead && WarpPipeline && QueryRows==64));
    static_assert(!QueryLookahead || (SwizzledCenters && !CacheQuery && QueryRows==64));
    static_assert(!WarpPipeline || (RegisterEpilogue && CacheMeta && StageOverlap && (QueryRows==64 || QueryRows==32) && KStage==32));
    constexpr int QueryGroups=QueryRows/16;
    constexpr int DotRows=WmmaPipelineStorage<KStage,QueryRows,WarpEpilogue,RegisterEpilogue>::DotRows;
    constexpr int DotGroups=DotRows/16;
    extern __shared__ __align__(32) unsigned char storage[];
    auto& s = *reinterpret_cast<WmmaPipelineStorage<KStage,QueryRows,WarpEpilogue,RegisterEpilogue,PackedCenters>*>(storage);
    // Immutable metadata is loaded once for all center tiles owned by this CTA.
    float* shared_qnorm=reinterpret_cast<float*>(&s+1);
    float* shared_threshold=shared_qnorm+QueryRows;
    int* local_counts=reinterpret_cast<int*>(reinterpret_cast<unsigned char*>(&s+1)+(CacheMeta?2*QueryRows*sizeof(float):0));
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    int qb = blockIdx.x * QueryRows, begin = blockIdx.y * center_tile;
    int end = begin + center_tile, part = begin / (nlist / parts);
    if constexpr(CtaOwned)if(tid<QueryRows)local_counts[tid]=0;
    if constexpr (CacheMeta) {
        if(tid<QueryRows) {
            shared_qnorm[tid]=qb+tid<batch?qnorm[qb+tid]:0.f;
            shared_threshold[tid]=qb+tid<batch?thresholds[qb+tid]:-INFINITY;
        }
    }
    for (int i = tid; i < QueryRows * 136; i += Threads) {
        int q = qb + i / 136, d = i % 136;
        s.query[i] = q < batch && d < 128 ? queries[size_t(q) * 128 + d] : __float2half(0.f);
    }
    prefetch_wmma_centers<KStage,WarpPipeline,BankOrder,PackedCenters,SwizzledCenters,CenterL1>(s.center[0], centers, begin, 0);
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    using QFrag = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,__half,nvcuda::wmma::row_major>;
    using CFrag = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,__half,nvcuda::wmma::col_major>;
    using AFrag = nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float>;
    QFrag cached[CacheQuery ? QueryGroups*8 : 1];
    if constexpr (CacheQuery) {
#pragma unroll
        for (int g = 0; g < QueryGroups; ++g) {
#pragma unroll
            for (int d = 0; d < 8; ++d)
                nvcuda::wmma::load_matrix_sync(cached[g * 8 + d], s.query + g * 16 * 136 + d * 16, 136);
        }
    }
    for (int base = begin; base < end; base += Cols) {
        if constexpr(WarpPipeline) {
            if(lane<16)s.norms[warp*16+lane]=cnorm[base+warp*16+lane];
        } else if (tid < Cols) s.norms[tid] = cnorm[base + tid];
        AFrag acc[QueryGroups];
#pragma unroll
        for(int g=0;g<QueryGroups;++g)nvcuda::wmma::fill_fragment(acc[g],0.f);
#pragma unroll
        for (int d = 0; d < 128; d += KStage) {
            constexpr int stride = KStage + (PackedCenters?0:8);
            int current = (d / KStage) & 1;
            int next_d = d + KStage == 128 ? 0 : d + KStage;
            int next_base = d + KStage == 128 ? base + Cols : base;
            if (next_base < end)
                prefetch_wmma_centers<KStage,WarpPipeline,BankOrder,PackedCenters,SwizzledCenters,CenterL1>(s.center[current ^ 1], centers, next_base, next_d);
#pragma unroll
            for (int kd = 0; kd < KStage; kd += 16) {
                CFrag cf;
                if constexpr(SwizzledCenters)load_swizzled_center_fragment(cf,s.center[current],warp*16,kd);
                else nvcuda::wmma::load_matrix_sync(cf, s.center[current] + warp * 16 * stride + kd, stride);
                if constexpr(QueryLookahead) {
                    // Keep only two A fragments live. Start the next group's
                    // shared load before issuing the current group's MMA.
                    QFrag query_fragment[2];
                    nvcuda::wmma::load_matrix_sync(query_fragment[0],s.query+d+kd,136);
                    #pragma unroll
                    for(int g=0;g<QueryGroups;++g) {
                        if(g+1<QueryGroups)nvcuda::wmma::load_matrix_sync(
                            query_fragment[(g+1)&1],s.query+(g+1)*16*136+d+kd,136);
                        nvcuda::wmma::mma_sync(acc[g],query_fragment[g&1],cf,acc[g]);
                    }
                } else {
                #pragma unroll
                for (int g = 0; g < QueryGroups; ++g) {
                    if constexpr (CacheQuery)
                        nvcuda::wmma::mma_sync(acc[g], cached[g * 8 + (d + kd) / 16], cf, acc[g]);
                    else {
                        QFrag qf;
                        nvcuda::wmma::load_matrix_sync(qf, s.query + g * 16 * 136 + d + kd, 136);
                        nvcuda::wmma::mma_sync(acc[g], qf, cf, acc[g]);
                    }
                    if constexpr(InterleaveAdmission)if(d+kd==112) {
                        // This query group's dot products are complete. Mix its
                        // admission with the remaining groups' final MMA work.
                        admit_wmma_registers<CtaOwned,RejectEmpty>(acc,s.norms,shared_qnorm,
                            shared_threshold,qb,batch,base,parts,part,capacity,values,ids,counts,
                            qb,local_counts,-1,g);
                    }
                }
                }
            }
            // Interior stages still publish the next buffer before use.
            // The last prefetched buffer belongs to the next center tile;
            // register admission reads only accumulators and current norms,
            // so that wait can share the post-admission reuse barrier.
            if constexpr (!StageOverlap) {
                asm volatile("cp.async.wait_group 0;\n" ::: "memory");
                __syncthreads();
            } else if (d+KStage<128) {
                asm volatile("cp.async.wait_group 0;\n" ::: "memory");
                if constexpr(WarpPipeline)__syncwarp();
                else __syncthreads();
            }
        }
        if constexpr (RegisterEpilogue) {
            if constexpr(!InterleaveAdmission)admit_wmma_registers<CtaOwned,RejectEmpty>(acc,s.norms,CacheMeta?shared_qnorm:qnorm,
                CacheMeta?shared_threshold:thresholds,qb,batch,base,parts,part,capacity,values,ids,counts,CacheMeta?qb:0,local_counts);
            if constexpr(StageOverlap)asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            if constexpr(WarpPipeline)__syncwarp();
            else __syncthreads();
        } else if constexpr (WarpEpilogue) {
            // Each warp consumes the same 16-center fragment it produced.
            // Two independent half-warps admit one query row each.
            float* tile = s.dots + warp * 256;
            const int col = lane & 15, half = lane & 16;
            const float cn = s.norms[warp * 16 + col];
#pragma unroll
            for (int g = 0; g < QueryGroups; ++g) {
                nvcuda::wmma::store_matrix_sync(tile, acc[g], 16, nvcuda::wmma::mem_row_major);
                __syncwarp();
#pragma unroll
                for (int r = lane / 16; r < 16; r += 2) {
                    const int q = qb + g * 16 + r;
                    float distance = 0.f;
                    if (q < batch) {
                        const float qn = qnorm[q];
                        distance = qn * qn + cn * cn - 2.f * tile[r * 16 + col];
                    }
                    const bool hit = q < batch && distance <= thresholds[q];
                    const unsigned mask = (__ballot_sync(0xffffffff, hit) >> half) & 0xffffu;
                    int offset = col == 0 && mask ? atomicAdd(counts + size_t(q) * parts + part, __popc(mask)) : 0;
                    offset = __shfl_sync(0xffffffff, offset, half);
                    const int slot = offset + __popc(mask & ((1u << col) - 1));
                    if (hit && slot < capacity) {
                        const size_t dst = (size_t(q) * parts + part) * capacity + slot;
                        values[dst] = -distance;
                        ids[dst] = base + warp * 16 + col;
                    }
                }
                __syncwarp();
            }
            // Protect shared norms from the next center tile's writers.
            __syncthreads();
        } else {
#pragma unroll
        for (int first=0;first<QueryGroups;first+=DotGroups) {
#pragma unroll
            for (int g = 0; g < DotGroups; ++g)
                nvcuda::wmma::store_matrix_sync(s.dots + g * 16 * 136 + warp * 16, acc[first+g], 136, nvcuda::wmma::mem_row_major);
            __syncthreads();
#pragma unroll
            for (int r = warp; r < DotRows; r += 8) {
                int q = qb + first*16 + r;
                if (q < batch) {
                    float qn = qnorm[q];
#pragma unroll
                    for (int col = lane; col < Cols; col += 32) {
                        float cn = s.norms[col];
                        float distance = qn * qn + cn * cn - 2.f * s.dots[r * 136 + col];
                        bool hit = distance <= thresholds[q];
                        unsigned mask = __ballot_sync(0xffffffff, hit);
                        int offset = lane == 0 && mask ? atomicAdd(counts + size_t(q) * parts + part, __popc(mask)) : 0;
                        offset = __shfl_sync(0xffffffff, offset, 0);
                        int slot = offset + __popc(mask & ((1u << lane) - 1));
                        if (hit && slot < capacity) {
                            size_t dst = (size_t(q) * parts + part) * capacity + slot;
                            values[dst] = -distance;
                            ids[dst] = base + col;
                        }
                    }
                }
            }
            __syncthreads();
        }
        }
    }
    // Exactly one CTA owns each (query, partition); publish the full count,
    // including overflow, after every bounded candidate write has completed.
    // Warp-local center buffers/norms need no inter-warp barrier in the loop;
    // the shared counters and final padding still require all warps to finish.
    if constexpr(WarpPipeline)__syncthreads();
    if constexpr(CtaOwned)if(tid<QueryRows && qb+tid<batch)
        counts[size_t(qb+tid)*parts+part]=local_counts[tid];
    if constexpr(CtaInit) {
        // The last iteration's CTA barrier completes all candidate writes.
        // This owner fills only unused slots; overflow counts remain intact.
        for(int r=warp;r<QueryRows;r+=Threads/32)if(qb+r<batch) {
            const size_t dst=(size_t(qb+r)*parts+part)*capacity;
            for(int slot=local_counts[r]+lane;slot<capacity;slot+=32) {
                values[dst+slot]=-INFINITY;
                ids[dst+slot]=-1;
            }
        }
    }
}

