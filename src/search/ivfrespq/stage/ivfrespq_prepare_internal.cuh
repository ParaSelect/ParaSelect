#pragma once
struct IVFRESPQSearchCtx;
// Fixed-capacity resources; released only by ivfrespq_search_ctx_release.
void ivfrespq_prepare_candidate_output(IVFRESPQSearchCtx* ctx);

struct IVFResSearchState;
void ivfrespq_validate_arg(const IVFResSearchState* arg);

void ivfrespq_upload_coarse_centers(IVFRESPQSearchCtx* ctx);

void prepare_fine_scan_workspace(IVFRESPQSearchCtx* ctx, const paraselect::search::ResidualPQGpuSearchSpec& rpq, int nprobe, size_t threshold_workspace_bytes);

int residual_pq_candidate_source_count(int nprobe, const paraselect::search::ResidualPQGpuSearchSpec& spec);
int residual_pq_max_vector_chunks_per_probe(const IVFResSearchState& arg, const paraselect::search::ResidualPQGpuSearchSpec& spec, const int* cluster_counts);

// Query-LUT seed selection uses threshold-owned scratch; dense seed kernels do not.
inline bool ivfrespq_uses_query_lut_seed(
    const IVFResSearchState& arg,
    const paraselect::search::ResidualPQGpuSearchSpec& spec) {
    return arg.residual_codec == paraselect::search::ResidualCodecKind::PQ &&
        spec.search_strategy == paraselect::search::PQAdcSearchStrategy::Threshold &&
        spec.adc_mode == paraselect::search::ResidualPQAdcMode::QueryLut;
}
