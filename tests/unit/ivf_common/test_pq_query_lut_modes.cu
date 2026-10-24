#include "search/kernels/fine/gpu_fine/pq_adc/pq_query_lut.cuh"
#include "utils/check_cuda_error.cuh"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>

void check(int m, int dsub, paraselect::search::PQQueryLutKind kind) {
    const int nq = 3, size = nq * m * 256;
    std::vector<float> queries(nq * m * dsub), codebook(m * 256 * dsub);
    for (int i = 0; i < int(queries.size()); ++i) queries[i] = float(i % 23 - 11) / 4;
    for (int i = 0; i < int(codebook.size()); ++i) codebook[i] = float(i % 31 - 15) / 8;
    float *dq, *dc, *dl, *ds, *dlo, *di; uint8_t* du;
    CUDA_CHECK(cudaMalloc(&dq, queries.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, codebook.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dl, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&du, size));
    CUDA_CHECK(cudaMalloc(&ds, nq * m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dlo, nq * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&di, nq * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dq, queries.data(), queries.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc, codebook.data(), codebook.size() * sizeof(float), cudaMemcpyHostToDevice));
    pq_build_query_lut_gpu(dq, dc, nq, m, dsub, dl, 0, kind);
    pq_quantize_query_lut_u8_gpu(dl, nq, m, du, ds, dlo, di, 0, kind);
    std::vector<float> lut(size), scales(nq * m), offsets(nq), inflation(nq);
    std::vector<uint8_t> u8(size);
    CUDA_CHECK(cudaMemcpy(lut.data(), dl, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(u8.data(), du, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scales.data(), ds, scales.size()*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(offsets.data(), dlo, nq*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(inflation.data(), di, nq*sizeof(float), cudaMemcpyDeviceToHost));
    const bool plain = kind == paraselect::search::PQQueryLutKind::PlainL2;
    for (int q = 0; q < nq; ++q) {
        float sum_lo = 0, sum_scale = 0;
        for (int sub = 0; sub < m; ++sub) {
            float lo = INFINITY;
            for (int c = 0; c < 256; ++c) {
                float ref = 0;
                for (int d = 0; d < dsub; ++d) {
                    const float a = queries[(q*m+sub)*dsub+d], b = codebook[(sub*256+c)*dsub+d];
                    ref += plain ? (a-b)*(a-b) : a*b;
                }
                const int index = (q*m+sub)*256+c;
                if (std::abs(lut[index]-ref) > 1e-5f) throw std::runtime_error("LUT disagrees with scalar reference");
                lo = std::min(lo, ref);
            }
            for (int c = 0; c < 256; ++c) {
                const int index = (q*m+sub)*256+c;
                const float restored = lo + scales[q*m+sub] * u8[index];
                if (std::abs(restored-lut[index]) > scales[q*m+sub]*0.5001f+1e-5f)
                    throw std::runtime_error("affine rounding bound exceeded");
            }
            sum_lo += lo; sum_scale += scales[q*m+sub];
        }
        if (std::abs(offsets[q]-sum_lo)>1e-4f ||
            std::abs(inflation[q]-sum_scale*(plain?0.5f:1.f))>1e-5f)
            throw std::runtime_error("wrong score-domain offset/inflation");
    }
    for (float* p : {dq,dc,dl,ds,dlo,di}) CUDA_CHECK(cudaFree(p));
    CUDA_CHECK(cudaFree(du));
}
int main() {
    for (auto kind : {paraselect::search::PQQueryLutKind::ResidualDot, paraselect::search::PQQueryLutKind::PlainL2}) {
        check(16, 8, kind); check(20, 5, kind); check(20, 8, kind);
    }
    std::cout << "PASS: M16/M20, plain L2/residual dot, affine U8 and inflation bounds\n";
}
