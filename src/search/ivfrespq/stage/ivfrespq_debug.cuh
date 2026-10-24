#ifndef PARASELECT_CUDA_SEARCH_STAGE_IVFRESPQ_STAGE2_DEBUG_CUH
#define PARASELECT_CUDA_SEARCH_STAGE_IVFRESPQ_STAGE2_DEBUG_CUH
struct RerankCandidateRef {
    int idx = -1;
    int q = -1;
    int slot = -1;
};

struct RerankCandidateLocalityStats {
    long long invalid = 0;
    long long duplicates = 0;
    long long in_probe_range = 0;
    long long adjacent_gap_sum = 0;
    long long adjacent_gap_count = 0;
    long long max_adjacent_gap = 0;
    long long max_probe_span = 0;
    double avg_probe_span = 0.0;
};

static RerankCandidateLocalityStats compute_rerank_candidate_locality(
    const int* candidate_idx,
    const idx_t* top_nprobe,
    const long long* cluster_offsets,
    const int* cluster_sizes,
    int batch_n,
    int nprobe,
    int global_k)
{
    RerankCandidateLocalityStats stats;
    if (!candidate_idx || !top_nprobe || !cluster_offsets || !cluster_sizes ||
        batch_n <= 0 || nprobe <= 0 || global_k <= 0) {
        return stats;
    }
    std::vector<int> sorted;
    sorted.reserve((size_t)global_k);
    std::unordered_set<int> seen;
    seen.reserve((size_t)global_k * 2);
    for (int q = 0; q < batch_n; ++q) {
        sorted.clear();
        seen.clear();
        const int* q_cands = candidate_idx + (size_t)q * global_k;
        const idx_t* q_probes = top_nprobe + (size_t)q * nprobe;
        for (int i = 0; i < global_k; ++i) {
            const int id = q_cands[i];
            if (id < 0) {
                stats.invalid += 1;
                continue;
            }
            if (!seen.insert(id).second) {
                stats.duplicates += 1;
            }
            sorted.push_back(id);
            for (int p = 0; p < nprobe; ++p) {
                const idx_t cid_idx = q_probes[p];
                if (cid_idx < 0) continue;
                const int cid = (int)cid_idx;
                const long long begin = cluster_offsets[cid];
                const long long end = begin + (long long)cluster_sizes[cid];
                if ((long long)id >= begin && (long long)id < end) {
                    stats.in_probe_range += 1;
                    const long long left = (long long)id - begin;
                    const long long right = end - 1 - (long long)id;
                    const long long span = std::max(left, right);
                    stats.max_probe_span = std::max(stats.max_probe_span, span);
                    stats.avg_probe_span += (double)span;
                    break;
                }
            }
        }
        std::sort(sorted.begin(), sorted.end());
        for (size_t i = 1; i < sorted.size(); ++i) {
            const long long gap = (long long)sorted[i] - (long long)sorted[i - 1];
            stats.adjacent_gap_sum += gap;
            stats.adjacent_gap_count += 1;
            stats.max_adjacent_gap = std::max(stats.max_adjacent_gap, gap);
        }
    }
    if (stats.in_probe_range > 0) {
        stats.avg_probe_span /= (double)stats.in_probe_range;
    }
    return stats;
}

static bool ivfrespq_debug_spacev_enabled() {
    const char* env = std::getenv("PARASELECT_IVFRESPQ_DEBUG_SPACEV");
    return env && *env && std::strcmp(env, "0") != 0;
}

static int ivfrespq_debug_batch_limit() {
    int limit = 4;
    if (const char* env = std::getenv("PARASELECT_IVFRESPQ_DEBUG_BATCH_LIMIT")) {
        if (*env) limit = std::atoi(env);
    }
    return std::max(0, limit);
}

static bool ivfrespq_debug_this_batch(int batch_start, int batch_n) {
    const int limit = ivfrespq_debug_batch_limit();
    return ivfrespq_debug_spacev_enabled() && limit > 0 && batch_start < limit * batch_n;
}

static void debug_dump_probe_ids(
    int batch_start,
    int batch_n,
    int nprobe,
    const idx_t* top_nprobe,
    const float* top_nprobe_dist = nullptr,
    const long long* top_nprobe_offsets = nullptr)
{
    const char* base = std::getenv("PARASELECT_RPQ_PROBE_DUMP_PATH");
    if (!base || !*base || !top_nprobe || batch_start < 0 || batch_n <= 0 || nprobe <= 0) {
        return;
    }
    long long query_offset = 0;
    if (const char* value = std::getenv("PARASELECT_QUERY_OFFSET")) {
        if (*value) query_offset = std::max(0LL, std::strtoll(value, nullptr, 10));
    }
    const long long absolute_query_start = query_offset + batch_start;
    std::ostringstream name;
    name << base << ".q" << absolute_query_start << ".i64";
    std::ofstream out(name.str(), std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "[IVFRESPQ] failed to open probe dump: " << name.str() << "\n";
        return;
    }
    const std::uint32_t version = top_nprobe_offsets ? 3u : (top_nprobe_dist ? 2u : 1u);
    const std::uint32_t query_start = static_cast<std::uint32_t>(absolute_query_start);
    const std::uint32_t query_count = static_cast<std::uint32_t>(batch_n);
    const std::uint32_t probe_count = static_cast<std::uint32_t>(nprobe);
    const std::uint32_t flags = (top_nprobe_dist ? 3u : 1u) | (top_nprobe_offsets ? 4u : 0u);
    out.write(top_nprobe_offsets ? "IAPRB003" : (top_nprobe_dist ? "IAPRB002" : "IAPRB001"), 8);
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&query_start), sizeof(query_start));
    out.write(reinterpret_cast<const char*>(&query_count), sizeof(query_count));
    out.write(reinterpret_cast<const char*>(&probe_count), sizeof(probe_count));
    out.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
    out.write(reinterpret_cast<const char*>(top_nprobe),
              static_cast<std::streamsize>(static_cast<size_t>(batch_n) *
                                            static_cast<size_t>(nprobe) * sizeof(idx_t)));
    if (top_nprobe_dist) {
        out.write(reinterpret_cast<const char*>(top_nprobe_dist),
                  static_cast<std::streamsize>(static_cast<size_t>(batch_n) *
                                                static_cast<size_t>(nprobe) * sizeof(float)));
    }
    if (top_nprobe_offsets) {
        out.write(reinterpret_cast<const char*>(top_nprobe_offsets),
                  static_cast<std::streamsize>(static_cast<size_t>(batch_n) *
                                                static_cast<size_t>(nprobe) * sizeof(long long)));
    }
}

static void debug_print_top_nprobe(
    const char* tag,
    int batch_start,
    int batch_n,
    int nprobe,
    const idx_t* top_nprobe,
    const float* top_nprobe_dist = nullptr,
    const long long* top_nprobe_offsets = nullptr)
{
    if (!top_nprobe) return;
    debug_dump_probe_ids(batch_start, batch_n, nprobe, top_nprobe, top_nprobe_dist,
                         top_nprobe_offsets);
    std::cerr << "[IVFRESPQ_DEBUG_SPACEV] " << tag
              << " batch_start=" << batch_start
              << " batch_n=" << batch_n
              << " nprobe=" << nprobe << "\n";
    const int rows = std::min(batch_n, 2);
    const int cols = std::min(nprobe, 16);
    for (int q = 0; q < rows; ++q) {
        std::cerr << "[IVFRESPQ_DEBUG_SPACEV] " << tag << " q=" << q << " cids=";
        for (int p = 0; p < cols; ++p) {
            if (p) std::cerr << ",";
            std::cerr << top_nprobe[(size_t)q * nprobe + p];
        }
        std::cerr << "\n";
    }
}

static void debug_print_candidates(
    const char* tag,
    int batch_start,
    int batch_n,
    int global_k,
    const int* candidate_idx)
{
    if (!candidate_idx) return;
    long long valid = 0;
    long long invalid = 0;
    int min_id = std::numeric_limits<int>::max();
    int max_id = -1;
    for (size_t i = 0, n = (size_t)batch_n * global_k; i < n; ++i) {
        const int id = candidate_idx[i];
        if (id < 0) {
            ++invalid;
        } else {
            ++valid;
            min_id = std::min(min_id, id);
            max_id = std::max(max_id, id);
        }
    }
    std::cerr << "[IVFRESPQ_DEBUG_SPACEV] " << tag
              << " batch_start=" << batch_start
              << " batch_n=" << batch_n
              << " cols=" << global_k
              << " valid=" << valid
              << " invalid=" << invalid
              << " min_id=" << (valid ? min_id : -1)
              << " max_id=" << (valid ? max_id : -1)
              << "\n";
    const int rows = std::min(batch_n, 2);
    const int cols = std::min(global_k, 16);
    for (int q = 0; q < rows; ++q) {
        std::cerr << "[IVFRESPQ_DEBUG_SPACEV] " << tag << " q=" << q << " ids=";
        for (int k = 0; k < cols; ++k) {
            if (k) std::cerr << ",";
            std::cerr << candidate_idx[(size_t)q * global_k + k];
        }
        std::cerr << "\n";
    }
}

#endif  // PARASELECT_CUDA_SEARCH_STAGE_IVFRESPQ_STAGE2_DEBUG_CUH
