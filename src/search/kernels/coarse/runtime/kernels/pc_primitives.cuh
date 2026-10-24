#pragma once
constexpr int Samples = 4096;
constexpr int Rows = 32;
constexpr int Cols = 128;
constexpr int Threads = 256;
constexpr int FallbackBatch = 32;

struct alignas(32) TileStorage {
    __half query[Rows * 136];
    float dots[Rows * 136];
    float norms[Cols];
    int count[Rows];
};

template<bool Sample, bool CacheQuery=false, bool Tiled=false>
__global__ void produce(const __half* queries, const __half* centers,
    const float* qnorm, const float* cnorm, int batch, int nlist, int parts,
    int capacity, const float* thresholds, float* values, int* ids, int* counts, int center_tile=0) {
    __shared__ TileStorage s;
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    int qb = blockIdx.x * Rows;
    const int part=Tiled ? int(blockIdx.y)*center_tile/(nlist/parts) : int(blockIdx.y);
    for (int i = tid; i < Rows * 136; i += Threads) {
        int q = qb + i / 136, d = i % 136;
        s.query[i] = q < batch && d < 128 ? queries[size_t(q) * 128 + d] : __float2half(0.f);
    }
    if (tid < Rows) s.count[tid] = 0;
    __syncthreads();
    int begin = Sample ? int(blockIdx.y) * (nlist / Samples) * Cols : Tiled ? int(blockIdx.y)*center_tile : part * (nlist / parts);
    int end = begin + (Sample ? Cols : Tiled ? center_tile : nlist / parts);
    using QFrag = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,__half,nvcuda::wmma::row_major>;
    using CFrag = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,__half,nvcuda::wmma::col_major>;
    using AFrag = nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float>;
    QFrag cached[CacheQuery?16:1];
    if constexpr(CacheQuery) {
#pragma unroll
        for(int g=0;g<2;++g) {
#pragma unroll
            for(int d=0;d<8;++d)
                nvcuda::wmma::load_matrix_sync(cached[g*8+d],s.query+g*16*136+d*16,136);
        }
    }
    for (int base = begin; base < end; base += Cols) {
        if (tid < Cols) s.norms[tid] = cnorm[base + tid];
        AFrag acc[2];
        nvcuda::wmma::fill_fragment(acc[0], 0.f);
        nvcuda::wmma::fill_fragment(acc[1], 0.f);
        QFrag qf; CFrag cf;
#pragma unroll
        for (int d = 0; d < 128; d += 16) {
            nvcuda::wmma::load_matrix_sync(cf, centers + size_t(base + warp * 16) * 128 + d, 128);
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                if constexpr(CacheQuery) nvcuda::wmma::mma_sync(acc[g],cached[g*8+d/16],cf,acc[g]);
                else {
                    nvcuda::wmma::load_matrix_sync(qf, s.query + g * 16 * 136 + d, 136);
                    nvcuda::wmma::mma_sync(acc[g], qf, cf, acc[g]);
                }
            }
        }
#pragma unroll
        for (int g = 0; g < 2; ++g)
            nvcuda::wmma::store_matrix_sync(s.dots + g * 16 * 136 + warp * 16, acc[g], 136, nvcuda::wmma::mem_row_major);
        __syncthreads();
#pragma unroll
        for (int r = warp; r < Rows; r += 8) {
            int q = qb + r;
            if (q < batch) {
                float qn = qnorm[q];
                int count = s.count[r];
#pragma unroll
                for (int col = lane; col < Cols; col += 32) {
                    float cn = s.norms[col];
                    float distance = qn * qn + cn * cn - 2.f * s.dots[r * 136 + col];
                    if constexpr (Sample) {
                        values[size_t(q) * Samples + blockIdx.y * Cols + col] = distance;
                    } else {
                        bool hit = distance <= thresholds[q];
                        unsigned mask = __ballot_sync(0xffffffff, hit);
                        int offset=count;
                        if constexpr(Tiled) {
                            offset=lane==0 && mask ? atomicAdd(counts+size_t(q)*parts+part,__popc(mask)) : 0;
                            offset=__shfl_sync(0xffffffff,offset,0);
                        }
                        int slot = offset + __popc(mask & ((1u << lane) - 1));
                        if (hit && slot < capacity) {
                            size_t dst = (size_t(q) * parts + part) * capacity + slot;
                            values[dst] = -distance;
                            ids[dst] = base + col;
                        }
                        if constexpr(!Tiled) count += __popc(mask);
                    }
                }
                if (lane == 0) s.count[r] = count;
            }
        }
        __syncthreads();
    }
    if constexpr (!Sample && !Tiled) {
        for (int r = warp; r < Rows; r += 8) {
            int q = qb + r;
            if (q < batch) {
                int count = s.count[r];
                size_t dst = (size_t(q) * parts + blockIdx.y) * capacity;
                for (int j = count + lane; j < capacity; j += 32) {
                    values[dst + j] = -INFINITY;
                    ids[dst + j] = -1;
                }
                if (lane == 0) counts[size_t(q) * parts + blockIdx.y] = count;
            }
        }
    }
}

__global__ void sample_threshold(const float* samples, float* thresholds, int rank, int force) {
    using Sort = cub::BlockRadixSort<float,128,32>;
    __shared__ typename Sort::TempStorage temp;
    float keys[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) keys[i] = samples[size_t(blockIdx.x) * Samples + threadIdx.x * 32 + i];
    Sort(temp).Sort(keys);
    if (threadIdx.x == rank / 32) {
        const int row_force = force == 2 ? (int(blockIdx.x) % 3 == 0 ? -1 : int(blockIdx.x) % 3 == 1 ? 1 : 0) : force;
        thresholds[blockIdx.x] = row_force < 0 ? -INFINITY : row_force > 0 ? INFINITY : keys[rank % 32];
    }
}

// Conservative sample-rank bound: retain the complete selected histogram bin.
// The final top-k is still exact; underfill/overflow use the existing fallback.
__global__ void sample_threshold_histogram(const float* samples, float* thresholds, int rank, int force) {
    __shared__ unsigned buckets[256], warp_min[4], warp_max[4], lower, upper;
    __shared__ int shift;
    unsigned keys[32], local_min=~0u, local_max=0;
    const int tid=threadIdx.x, lane=tid%32, warp=tid/32;
    #pragma unroll
    for(int i=0;i<32;++i) {
        const unsigned bits=__float_as_uint(samples[size_t(blockIdx.x)*Samples+tid+i*128]);
        keys[i]=(bits&0x80000000u)?~bits:(bits^0x80000000u);
        local_min=min(local_min,keys[i]);local_max=max(local_max,keys[i]);
    }
    #pragma unroll
    for(int offset=16;offset;offset/=2) {
        local_min=min(local_min,__shfl_down_sync(0xffffffff,local_min,offset));
        local_max=max(local_max,__shfl_down_sync(0xffffffff,local_max,offset));
    }
    if(lane==0){warp_min[warp]=local_min;warp_max[warp]=local_max;}
    for(int b=tid;b<256;b+=128)buckets[b]=0;
    __syncthreads();
    if(tid==0) {
        lower=min(min(warp_min[0],warp_min[1]),min(warp_min[2],warp_min[3]));
        upper=max(max(warp_max[0],warp_max[1]),max(warp_max[2],warp_max[3]));
        shift=max(0,24-__clz(upper-lower));
    }
    __syncthreads();
    if(lower!=upper) {
        #pragma unroll
        for(int i=0;i<32;++i)atomicAdd(buckets+((keys[i]-lower)>>shift),1u);
        __syncthreads();
    }
    if(tid==0) {
        unsigned bound=upper;
        if(lower!=upper) {
            unsigned count=0;
            for(unsigned b=0;b<256;++b) {
                count+=buckets[b];
                if(count>unsigned(rank)) {
                    const unsigned long long end=static_cast<unsigned long long>(lower)+((static_cast<unsigned long long>(b)+1)<<shift)-1;
                    bound=end<upper?unsigned(end):upper;
                    break;
                }
            }
        }
        const float threshold=__uint_as_float((bound&0x80000000u)?(bound^0x80000000u):~bound);
        const int row_force=force==2?(int(blockIdx.x)%3==0?-1:int(blockIdx.x)%3==1?1:0):force;
        thresholds[blockIdx.x]=row_force<0?-INFINITY:row_force>0?INFINITY:threshold;
    }
}

__global__ void validate_admission(const int* counts, int batch, int parts, int capacity, int k, int* bad, int* bad_count) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= batch) return;
    int total = 0; bool overflow = false;
    for (int p = 0; p < parts; ++p) {
        int count = counts[size_t(q) * parts + p];
        total += count; overflow |= count > capacity;
    }
    // Never select from a truncated pool or a pool containing fewer than k.
    if (overflow || total < k) bad[atomicAdd(bad_count, 1)] = q;
}

__global__ void finish_ids(const int* pool_ids, idx_t* ids, int k, int batch, int stride) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(batch) * k) {
        idx_t slot = ids[i];
        ids[i] = slot >= 0 && slot < stride ? pool_ids[(i / k) * stride + slot] : -1;
    }
}

__global__ void validate_admission_warp(const int* counts, int batch, int parts, int capacity, int k, int* bad, int* bad_count) {
    const int q=(blockIdx.x*blockDim.x+threadIdx.x)/32, lane=threadIdx.x%32;
    if(q>=batch)return;
    int total=0;bool overflow=false;
    for(int p=lane;p<parts;p+=32) {
        const int count=counts[size_t(q)*parts+p];
        total+=count;overflow|=count>capacity;
    }
    const bool invalid=__any_sync(0xffffffff,overflow);
    #pragma unroll
    for(int offset=16;offset;offset/=2)total+=__shfl_down_sync(0xffffffff,total,offset);
    if(lane==0 && (invalid || total<k))bad[atomicAdd(bad_count,1)]=q;
}

__global__ void pack_fallback(const __half* query, const float* norms, const int* bad, int batch, __half* packed, float* packed_norms) {
    int q = blockIdx.x;
    if (q < batch) {
        int source = bad[q];
        packed[size_t(q) * 128 + threadIdx.x] = query[size_t(source) * 128 + threadIdx.x];
        if (threadIdx.x == 0) packed_norms[q] = norms[source];
    }
}
__global__ void scatter_fallback(const float* values, const idx_t* ids, const int* bad, int batch, int k, float* out, idx_t* out_ids) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(batch) * k) {
        size_t dst = size_t(bad[i / k]) * k + i % k;
        out[dst] = values[i]; out_ids[dst] = ids[i];
    }
}
__global__ void segment_offsets(int* offsets, int batch, int stride) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if(q<=batch) offsets[q]=q*stride;
}
__global__ void finish_sorted(const float* values,const int* ids,int batch,int k,int stride,float* out,idx_t* out_ids) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<size_t(batch)*k) {
        size_t src=(i/k)*stride+i%k;out[i]=-values[src];out_ids[i]=ids[src];
    }
}

// Workspace-owned descriptors: output pointers are not part of a tensor map.
// Geometry changes, including a smaller final batch, rebuild only that map.
struct TopkLaunchCache {
    struct MapSlot {
        CUtensorMap value{};
        const void* input=nullptr;
        int batch=0, vocab=0, segment=0;
        int64_t stride=0;
    } pool, full;
    bool enabled=false, configured=false;
    size_t builds=0;
    template<class Kernel>
    typename Kernel::TmaParams get(const TopkSelectArgs& args, bool dense=false) {
        auto& slot=dense?full:pool;
        if(!enabled || slot.input!=args.input || slot.batch!=args.batch_size ||
            slot.vocab!=args.vocab_size || slot.stride!=args.stride_input_batch ||
            slot.segment!=Kernel::NUM_ELEMS_PER_SEG) {
            slot.value=Kernel::make_topk_tensor_map(args);
            slot.input=args.input;slot.batch=args.batch_size;slot.vocab=args.vocab_size;
            slot.stride=args.stride_input_batch;slot.segment=Kernel::NUM_ELEMS_PER_SEG;
            ++builds;
        }
        return typename Kernel::TmaParams{slot.value};
    }
};

template<class Kernel>
__launch_bounds__(Kernel::NUM_THREADS,Kernel::TARGET_OCCUPANCY,1)
__global__ void select_admitted_kernel(__grid_constant__ const TopkSelectArgs args,
    __grid_constant__ const typename Kernel::TmaParams map, const int* pool_ids, int stride) {
    Kernel::topk_select_kernel_devfunc(args,map);
    __syncthreads();
    for(int j=threadIdx.x;j<args.topk;j+=blockDim.x) {
        size_t out=size_t(blockIdx.x)*args.stride_output_index_batch+j;
        int64_t* ids=static_cast<int64_t*>(args.output_index);
        float* values=static_cast<float*>(args.output_value);
        const auto slot=ids[out];
        ids[out]=slot>=0 && slot<stride ? pool_ids[size_t(blockIdx.x)*stride+slot] : -1;
        values[out]=-values[out];
    }
}
template<int Capacity>
inline void select_admitted(const paraselect::coarse::detail::Execution& a,
    float* pool,const int* pool_ids,int stride,TopkLaunchCache& cache,int* lengths=nullptr) {
    using Config=TopkSelectConfig<float,int64_t,true,false,true,Capacity,128,1,2048,2048,2>;
    using Kernel=topk_select_fp32::TopkSelectKernelFP32<Config>;
    constexpr size_t smem=sizeof(typename Kernel::SharedMemoryPlanFP32);
    TopkSelectArgs args{};
    args.batch_size=a.n_query;args.vocab_size=stride;args.topk=a.nprobe;
    args.end_ptr=lengths;args.input=pool;args.output_value=a.d_top_nprobe_dist;args.output_index=a.d_top_nprobe_index;
    args.stride_input_batch=stride;args.stride_output_value_batch=a.nprobe;args.stride_output_index_batch=a.nprobe;
    args.sorted_value=true;args.return_value=true;args.idx_oob_fill_value=-1;
    args.value_oob_fill_value=-INFINITY;args.abort_when_nan_found=true;args.stream=a.stream;
    args.shared_memory_size_per_sm=99*1024;
    if(!cache.enabled || !cache.configured) {
        CUDA_CHECK(cudaFuncSetAttribute(select_admitted_kernel<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
        cache.configured=true;
    }
    auto map=cache.get<Kernel>(args);
    select_admitted_kernel<Kernel><<<a.n_query,Kernel::NUM_THREADS,smem,a.stream>>>(args,map,pool_ids,stride);
}

__global__ void initialize_candidate_pool(float* values, int* ids, int* counts,
    size_t elements, size_t partitions) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<elements){values[i]=-INFINITY;ids[i]=-1;}
    if(i<partitions)counts[i]=0;
}

