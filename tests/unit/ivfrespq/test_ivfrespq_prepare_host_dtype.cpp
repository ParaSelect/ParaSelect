#include "search/ivfrespq/stage/ivfrespq_stages.cuh"

#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check_mode(int mode) {
    constexpr int count = 16, dim = 16, m = 2, codewords = 256;
    const idx_t offsets[] = {0, 8, count};
    std::vector<idx_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<float> base(count * dim), query(dim), centers(2 * dim);
    std::vector<uint8_t> base_u8(base.size()), query_u8(dim), codes(count * m);
    std::vector<float> codebook(m * codewords * (dim / m));
    for (size_t i = 0; i < base.size(); ++i) base[i] = base_u8[i] = i % 251;
    for (int i = 0; i < dim; ++i) query[i] = query_u8[i] = i;
    idx_t result_id = -1;
    float result_distance = 0;
    IVFResSearchState arg;
    arg.lists = {offsets, ids.data(), 2, count};
    arg.exact_storage = IVFExactStorageView::from_host({
        mode == 1 ? nullptr : base.data(), mode == 1 ? base_u8.data() : nullptr,
        count, dim});
    arg.h_queries_fp32 = query.data();
    arg.h_queries_u8 = mode == 1 ? query_u8.data() : nullptr;
    arg.h_cluster_centers = centers.data();
    arg.pq = {codes.data(), nullptr, codebook.data(), m, codewords};
    arg.pq_M = m;
    arg.pq_K = codewords;
    arg.n_query = 1;
    arg.n_dim = dim;
    arg.n_probes = 1;
    arg.k = 1;
    arg.query_batch_size = 1;
    arg.pq_rerank_n = 8;
    arg.pq_use_float_rerank = mode == 0;
    arg.fine_backend = IVFResPQFineBackend::CpuLut;
    arg.cpu.rerank_threads = 1;
    arg.h_topk_index_out = &result_id;
    arg.h_topk_dist_out = &result_distance;
    IVFRESPQSearchCtx ctx;
    try {
        ivfrespq_prepare(&ctx, &arg);
        if (mode == 0) {
            require(ctx.owned_base_u8.empty(), "FP32 rerank created an unused U8 base");
            require(ctx.owned_query_u8.empty(), "FP32 rerank created unused U8 queries");
            require(ctx.fine.assets.h_base_fp32 == base.data(), "FP32 base pointer changed");
            require(ctx.fine.assets.h_queries_fp32 == query.data(), "FP32 query pointer changed");
        } else if (mode == 1) {
            require(ctx.owned_base_u8.empty() && ctx.owned_query_u8.empty(),
                    "Native U8 rerank unnecessarily copied input arrays");
            require(ctx.fine.assets.h_base_u8 == base_u8.data(), "Native U8 base pointer changed");
            require(ctx.fine.assets.h_queries_u8 == query_u8.data(), "Native U8 query pointer changed");
        } else {
            require(ctx.owned_base_u8 == base_u8, "Legacy FP32-to-U8 base conversion changed");
            require(ctx.owned_query_u8 == query_u8, "Legacy FP32-to-U8 query conversion changed");
            require(ctx.fine.assets.h_base_u8 == ctx.owned_base_u8.data(), "Converted base is not used");
            require(ctx.fine.assets.h_queries_u8 == ctx.owned_query_u8.data(), "Converted queries are not used");
        }
    } catch (...) {
        ivfrespq_search_ctx_release(&ctx);
        throw;
    }
    ivfrespq_search_ctx_release(&ctx);
}
}  // namespace

int main() {
    try {
        for (int mode : {0, 1, 2}) check_mode(mode);
        std::cout << "PASS prepare host FP32/native-U8/converted-U8 storage\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
