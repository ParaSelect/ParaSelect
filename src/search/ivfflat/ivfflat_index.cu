#include "search/ivfflat/ivfflat_index.h"

#include "dataset/common/dataset_spec.h"
#include "search/ivfflat/ivfflat_build.h"
#include "search/ivfflat/ivfflat_search.cuh"
#include "search/kernels/fine/cpu_fine/dispatch/cpu_ivf_kernel_combos.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace {

using Clock = std::chrono::steady_clock;

}  // namespace

namespace paraselect::search {

struct IVFFlatIndex::Impl {
    explicit Impl(IVFFlatBuildConfig requested) : config(std::move(requested)) {
        const auto build_start = Clock::now();
        if (config.fine_kind == IVFFlatFineKind::Cpu) {
            const std::string dataset_name =
                paraselect::dataset::benchmark_dataset_name(config.source.spec());
            const auto* combo =
                paraselect::cpu_fine::resolve_cpu_fine_combo_ptr_or_throw(
                    config.cpu_fine_combo.c_str(), dataset_name.c_str());
            const char* resolved =
                paraselect::cpu_fine::cpu_fine_combo_name_from_preset(
                    combo->layout, combo->kernel, combo->schedule_policy);
            if (!resolved) {
                throw std::invalid_argument("IVFFlat CPU fine combo has no public name");
            }
            config.cpu_fine_combo = resolved;
        }
        try {
            timing = paraselect::ivfflat::build_storage(config, &dataset);
        } catch (...) {
            dataset.release();
            throw;
        }
        index.coarse = {dataset.centroids, config.nlist, config.source.spec().dim};
        index.lists = {
            dataset.cluster_info.offsets,
            dataset.reordered_indices,
            config.nlist,
            static_cast<idx_t>(config.n_vectors)};
        index.storage.rowmajor_fp32 = dataset.get_row_major_fp32();
        index.storage.rowmajor_u8 = dataset.get_row_major_u8();
        index.storage.norms = dataset.vector_l2_norm;
        index.storage.aosoa_fp32 = dataset.get_aosoa_fp32();
        index.storage.aosoa_offsets = dataset.get_aosoa_offsets();
        index.storage.aosoa_tile =
            dataset.get_aosoa_tile() > 0 ? dataset.get_aosoa_tile() : 16;
        if (config.fine_kind == IVFFlatFineKind::Resident) {
            const auto& resident = dataset.get_gpu_resident_interleaved();
            index.storage.resident_fp32 = resident.d_vectors;
            index.storage.resident_u8 = resident.d_vectors_u8;
            index.storage.resident_norms = resident.d_norms;
            index.storage.resident_ids = resident.d_reordered_indices;
        }
        validate_ivf_index_shape(index.coarse, index.lists);
        options.fine_kind = config.fine_kind;
        options.threshold = config.threshold;
        options.cpu_fine_combo = config.cpu_fine_combo.c_str();
        options.resident_combo = config.resident_combo.c_str();
        options.resident_chunk_blocks = config.resident_chunk_blocks;
        options.overlap_cpu_pipeline = config.overlap_cpu_pipeline;
        build_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - build_start).count();
    }

    ~Impl() { dataset.release(); }

    IVFFlatBuildConfig config;
    ClusterDataset dataset;
    paraselect::ivfflat::BuildTiming timing;
    IVFFlatIndexView index;
    IVFFlatSearchOptions options;
    double build_ms = 0.0;
};

IVFFlatIndex IVFFlatIndex::build(IVFFlatBuildConfig config) {
    return IVFFlatIndex(std::make_unique<Impl>(std::move(config)));
}

IVFFlatIndex::IVFFlatIndex(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

IVFFlatIndex::~IVFFlatIndex() = default;
IVFFlatIndex::IVFFlatIndex(IVFFlatIndex&&) noexcept = default;
IVFFlatIndex& IVFFlatIndex::operator=(IVFFlatIndex&&) noexcept = default;

void IVFFlatIndex::search(
    const IVFQueryBatchView& queries,
    const IVFSearchParams& params,
    const IVFSearchOutputView& output,
    IVFSearchStats* stats) {
    ivfflat_search(
        {impl_->index,
         queries,
         params,
         output,
         impl_->options},
        stats);
}

size_t IVFFlatIndex::size() const { return impl_->config.n_vectors; }
int IVFFlatIndex::dim() const { return impl_->config.source.spec().dim; }
int IVFFlatIndex::nlist() const { return impl_->config.nlist; }
double IVFFlatIndex::build_time_ms() const { return impl_->build_ms; }
double IVFFlatIndex::kmeans_time_ms() const { return impl_->timing.kmeans_ms; }

}  // namespace paraselect::search
