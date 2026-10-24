#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "common/test_context.cuh"
#include "search/ivfpq/ivfpq_search_context.h"
#include "search/ivfpq/pipeline/ivfpq_pipeline.cuh"
#include "search/ivfpq/stage/ivfpq_stages.cuh"

namespace {

struct SearchOutput {
    std::vector<idx_t> ids;
    std::vector<float> distances;
};

SearchOutput run_search(
    const char* fine_storage,
    const std::vector<float>& centroids,
    const std::vector<idx_t>& offsets,
    const std::vector<idx_t>& reordered_ids,
    const std::vector<uint8_t>& codes,
    const std::vector<float>& codebook,
    const std::vector<float>& queries,
    int nlist,
    int n_vectors,
    int n_query,
    int dim,
    int pq_m,
    int pq_k,
    int topk,
    int batch_size,
    bool fp16 = false)
{
    SearchOutput out;
    out.ids.assign((size_t)n_query * topk, (idx_t)-1);
    out.distances.assign(
        (size_t)n_query * topk, std::numeric_limits<float>::max());

    IVFPQSearchContext search;
    search.index.coarse = {centroids.data(), nlist, dim};
    search.index.lists = {
        offsets.data(), reordered_ids.data(), nlist, (idx_t)n_vectors};
    search.index.storage.pq.codes = codes.data();
    search.index.storage.pq.codebook = codebook.data();
    search.index.storage.pq.subvectors = pq_m;
    search.index.storage.pq.codebook_size = pq_k;
    search.options.n_rerank = topk;
    search.options.rerank_dtype = IVFExactRerankDType::Disabled;
    search.options.resident_codes = std::string(fine_storage) == "resident";
    search.options.resident_combo = "auto";
    search.options.adc_strategy = paraselect::search::PQAdcSearchStrategy::Dense;
    search.queries = {queries.data(), nullptr, n_query, dim};
    search.params = {nlist, topk, batch_size, 0, fp16};
    search.output = {out.ids.data(), out.distances.data(), nullptr};
    try {
        ivfpq_prepare_index(&search);
        ivfpq_prepare(&search);
        ivfpq_run_pipeline(&search);
        std::fill(out.ids.begin(), out.ids.end(), (idx_t)-1);
        std::fill(
            out.distances.begin(),
            out.distances.end(),
            std::numeric_limits<float>::max());
        ivfpq_run_pipeline(&search);
    } catch (...) {
        ivfpq_release(&search);
        throw;
    }
    ivfpq_release(&search);
    return out;
}

}  // namespace

int main() {
    paraselect::test::TestContext ctx(
        "ivfpq_resident_no_rerank", paraselect::test::TestKind::Integration);

    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        ctx.skip(std::string("CUDA device unavailable: ") + cudaGetErrorString(status));
        return ctx.exit_code();
    }

    constexpr int nlist = 4;
    constexpr int n_vectors = 128;
    constexpr int n_query = 5;
    constexpr int dim = 8;
    constexpr int pq_m = 4;
    constexpr int pq_k = 256;
    constexpr int topk = 10;
    constexpr int batch_size = 3;
    constexpr int dsub = dim / pq_m;

    std::vector<float> centroids((size_t)nlist * dim, 0.0f);
    std::vector<idx_t> offsets = {0, 32, 64, 96, 128};
    std::vector<idx_t> reordered_ids(n_vectors);
    std::vector<uint8_t> codes((size_t)n_vectors * pq_m, 0);
    std::vector<float> codebook((size_t)pq_m * pq_k * dsub, 0.0f);
    std::vector<float> queries((size_t)n_query * dim, 0.0f);

    for (int i = 0; i < n_vectors; ++i) {
        reordered_ids[(size_t)i] = (idx_t)(1000 + i * 3);
        codes[(size_t)i * pq_m] = (uint8_t)i;
    }
    for (int code = 0; code < pq_k; ++code) {
        for (int d = 0; d < dsub; ++d) {
            codebook[(size_t)code * dsub + d] =
                (float)code * 0.125f + (float)d * 0.01f;
        }
    }
    for (int q = 0; q < n_query; ++q) {
        queries[(size_t)q * dim] = (float)q * 0.0625f;
    }

    ctx.run_case("cpu_lut_matches_gpu_dense_adc", [&] {
        const SearchOutput cpu = run_search(
            "host", centroids, offsets, reordered_ids, codes, codebook, queries,
            nlist, n_vectors, n_query, dim, pq_m, pq_k, topk, batch_size);
        const SearchOutput gpu = run_search(
            "resident", centroids, offsets, reordered_ids, codes, codebook, queries,
            nlist, n_vectors, n_query, dim, pq_m, pq_k, topk, batch_size);

        for (size_t i = 0; i < cpu.ids.size(); ++i) {
            ctx.check(cpu.ids[i] == gpu.ids[i],
                      "GPU dense ADC ID differs from CPU LUT");
            const float tolerance = 1e-4f * std::max(1.0f, std::fabs(cpu.distances[i]));
            ctx.check(std::fabs(cpu.distances[i] - gpu.distances[i]) <= tolerance,
                      "GPU dense ADC distance differs from CPU LUT");
        }
    });

    ctx.run_case("fp16_coarse_matches_fp32", [&] {
        const auto fp32 = run_search(
            "resident", centroids, offsets, reordered_ids, codes, codebook, queries,
            nlist, n_vectors, n_query, dim, pq_m, pq_k, topk, batch_size);
        const auto fp16 = run_search(
            "resident", centroids, offsets, reordered_ids, codes, codebook, queries,
            nlist, n_vectors, n_query, dim, pq_m, pq_k, topk, batch_size, true);
        for (size_t i = 0; i < fp32.ids.size(); ++i) {
            ctx.check(fp32.ids[i] == fp16.ids[i], "FP16 coarse changed full-probe output");
            ctx.check(std::fabs(fp32.distances[i] - fp16.distances[i]) < 1e-4f,
                      "FP16 coarse changed full-probe distance");
        }
    });
    return ctx.exit_code();
}
