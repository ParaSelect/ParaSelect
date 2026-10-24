#include "search/kernels/fine/gpu_fine/pq_adc/pq_query_lut.cuh"

#include "utils/check_cuda_error.cuh"

#include <cfloat>
#include <stdexcept>
#include <type_traits>

namespace {

constexpr int kPqK = paraselect::search::kPqQueryLutGlobalStride;

template <int DSub, bool PlainL2, typename Book>
__global__ void build_query_codeword_dot_lut_kernel(
    const float* __restrict__ queries,
    const Book* __restrict__ codebook,
    int pq_m,
    float* __restrict__ lut) {
    const int q = blockIdx.x;
    for (int entry = threadIdx.x; entry < pq_m * kPqK; entry += blockDim.x) {
        const int m = entry / kPqK;
        const int code = entry - m * kPqK;
        const float* query = queries + ((size_t)q * pq_m + m) * DSub;
        const Book* codeword = codebook + ((size_t)m * kPqK + code) * DSub;
        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < DSub; ++d) {
            if constexpr (PlainL2) {
                const float delta = query[d] - float(codeword[d]);
                dot = fmaf(delta, delta, dot);
            } else dot = fmaf(query[d], float(codeword[d]), dot);
        }
        lut[((size_t)q * pq_m + m) * kPqK + code] = dot;
    }
}

__global__ void quantize_query_codeword_dot_lut_u8_kernel(
    const float* __restrict__ lut,
    int pq_m,
    uint8_t* __restrict__ lut_u8,
    float* __restrict__ scale,
    float* __restrict__ lo_sum,
    float* __restrict__ inflation,
    float distance_multiplier) {
    const int q = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float lo_parts[32];
    __shared__ float inflation_parts[32];
    float local_lo_sum = 0.0f;
    float local_inflation = 0.0f;
    for (int m = warp; m < pq_m; m += (blockDim.x >> 5)) {
        const size_t base = ((size_t)q * pq_m + m) * kPqK;
        float lo = FLT_MAX;
        float hi = -FLT_MAX;
        for (int code = lane; code < kPqK; code += 32) {
            const float value = lut[base + code];
            lo = fminf(lo, value);
            hi = fmaxf(hi, value);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            lo = fminf(lo, __shfl_down_sync(0xffffffffu, lo, offset));
            hi = fmaxf(hi, __shfl_down_sync(0xffffffffu, hi, offset));
        }
        lo = __shfl_sync(0xffffffffu, lo, 0);
        hi = __shfl_sync(0xffffffffu, hi, 0);
        const float row_scale = fmaxf((hi - lo) * (1.0f / 255.0f), 1.0e-6f);
        if (lane == 0) scale[(size_t)q * pq_m + m] = row_scale;
        for (int code = lane; code < kPqK; code += 32) {
            const float normalized = (lut[base + code] - lo) / row_scale;
            lut_u8[base + code] = static_cast<uint8_t>(
                min(255, max(0, __float2int_rn(normalized))));
        }
        if (lane == 0) {
            local_lo_sum += lo;
            local_inflation += row_scale;
        }
    }
    if (lane == 0) {
        lo_parts[warp] = local_lo_sum;
        inflation_parts[warp] = local_inflation;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total_lo = 0.0f;
        float total_inflation = 0.0f;
        for (int i = 0; i < min(pq_m, int(blockDim.x >> 5)); ++i) {
            total_lo += lo_parts[i];
            total_inflation += inflation_parts[i];
        }
        lo_sum[q] = total_lo;
        // The ADC expression multiplies the LUT sum by -2. Nearest rounding
        // contributes at most scale[m]/2 per row, hence sum(scale) is a safe
        // distance-domain threshold inflation. Plain L2 uses half that bound.
        inflation[q] = total_inflation * distance_multiplier;
    }
}

template <int DSub, bool PlainL2, typename Book>
void launch_query_lut(
    const float* d_queries,
    const Book* d_codebook,
    int n_query,
    int pq_m,
    float* d_lut,
    cudaStream_t stream) {
    build_query_codeword_dot_lut_kernel<DSub, PlainL2><<<n_query, 256, 0, stream>>>(
        d_queries, d_codebook, pq_m, d_lut);
}

}  // namespace

template<typename Book>
void build_query_lut(
    const float* d_queries,
    const Book* d_codebook,
    int n_query,
    int pq_m,
    int pq_d_sub_padded,
    float* d_query_codeword_dot_lut,
    cudaStream_t stream,
    paraselect::search::PQQueryLutKind kind) {
    if (n_query <= 0) return;
    if (!d_queries || !d_codebook || !d_query_codeword_dot_lut || pq_m <= 0) {
        throw std::invalid_argument(
            "pq_build_query_lut_gpu: missing required input");
    }
    const auto dispatch = [&](auto plain_tag) {
        constexpr bool PlainL2 = decltype(plain_tag)::value;
        switch (pq_d_sub_padded) {
            case 2: launch_query_lut<2, PlainL2>(d_queries, d_codebook, n_query, pq_m, d_query_codeword_dot_lut, stream); break;
            case 4: launch_query_lut<4, PlainL2>(d_queries, d_codebook, n_query, pq_m, d_query_codeword_dot_lut, stream); break;
            case 5: launch_query_lut<5, PlainL2>(d_queries, d_codebook, n_query, pq_m, d_query_codeword_dot_lut, stream); break;
            case 6: launch_query_lut<6, PlainL2>(d_queries, d_codebook, n_query, pq_m, d_query_codeword_dot_lut, stream); break;
            case 8: launch_query_lut<8, PlainL2>(d_queries, d_codebook, n_query, pq_m, d_query_codeword_dot_lut, stream); break;
            case 16: launch_query_lut<16, PlainL2>(d_queries, d_codebook, n_query, pq_m, d_query_codeword_dot_lut, stream); break;
            default: throw std::invalid_argument("pq_build_query_lut_gpu: unsupported dsub");
        }
    };
    if (kind == paraselect::search::PQQueryLutKind::PlainL2) dispatch(std::true_type{});
    else if (kind == paraselect::search::PQQueryLutKind::ResidualDot) dispatch(std::false_type{});
    else throw std::invalid_argument("pq_build_query_lut_gpu: invalid score kind");
    CUDA_LAUNCH_CHECK("build_query_codeword_dot_lut_kernel");
}

void pq_build_query_lut_gpu(const float* q,const float* book,int nq,int m,int d,
    float* lut,cudaStream_t stream,paraselect::search::PQQueryLutKind kind) {
    build_query_lut(q,book,nq,m,d,lut,stream,kind);
}
void pq_build_query_lut_gpu(const float* q,const __half* book,int nq,int m,int d,
    float* lut,cudaStream_t stream,paraselect::search::PQQueryLutKind kind) {
    build_query_lut(q,book,nq,m,d,lut,stream,kind);
}

void pq_quantize_query_lut_u8_gpu(
    const float* d_query_codeword_dot_lut,
    int n_query,
    int pq_m,
    uint8_t* d_query_codeword_dot_lut_u8,
    float* d_query_lut_scale,
    float* d_query_lut_lo_sum,
    float* d_query_lut_inflation,
    cudaStream_t stream,
    paraselect::search::PQQueryLutKind kind) {
    if (n_query <= 0) return;
    if (!d_query_codeword_dot_lut || !d_query_codeword_dot_lut_u8 ||
        !d_query_lut_scale || !d_query_lut_lo_sum || !d_query_lut_inflation ||
        pq_m <= 0 || pq_m > 64) {
        throw std::invalid_argument(
            "pq_quantize_query_lut_u8_gpu: invalid input");
    }
    if (kind != paraselect::search::PQQueryLutKind::PlainL2 &&
        kind != paraselect::search::PQQueryLutKind::ResidualDot)
        throw std::invalid_argument("pq_quantize_query_lut_u8_gpu: invalid score kind");
    quantize_query_codeword_dot_lut_u8_kernel<<<n_query, 256, 0, stream>>>(
        d_query_codeword_dot_lut, pq_m, d_query_codeword_dot_lut_u8,
        d_query_lut_scale, d_query_lut_lo_sum, d_query_lut_inflation,
        kind == paraselect::search::PQQueryLutKind::PlainL2 ? 0.5f : 1.0f);
    CUDA_LAUNCH_CHECK("quantize_query_codeword_dot_lut_u8_kernel");
}
