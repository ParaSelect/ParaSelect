#include "quantization/pca/pca_storage.h"
#include "search/ivf_common/ivf_pca_refine.h"
#include "search/ivf_common/ivf_pca_resident.h"

#include <cuda_runtime.h>

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

template <typename Exception, typename Function>
bool throws(Function&& function) {
    try {
        function();
    } catch (const Exception&) {
        return true;
    } catch (...) {
    }
    return false;
}

bool equal(const std::vector<int>& actual, const std::vector<int>& expected) {
    if (actual == expected) return true;
    std::cerr << "expected:";
    for (int value : expected) std::cerr << ' ' << value;
    std::cerr << "\nactual:";
    for (int value : actual) std::cerr << ' ' << value;
    std::cerr << '\n';
    return false;
}

}  // namespace

int main() {
    auto fail = [](int line) { std::cerr << "PCA shortlist check failed at line " << line << '\n'; return 1; };
    if (paraselect::search::IVFPCAResidentConfig{}.dp4a_kernel !=
        paraselect::search::IVFPCADp4aKernelVersion::V2) {
        return fail(__LINE__);
    }
    using paraselect::search::IVFProbeScheduleStrategy;
    using paraselect::search::choose_ivf_probe_schedule_strategy;
    if (choose_ivf_probe_schedule_strategy(
            IVFProbeScheduleStrategy::Auto, 99, 3200) !=
            IVFProbeScheduleStrategy::Sparse ||
        choose_ivf_probe_schedule_strategy(
            IVFProbeScheduleStrategy::Auto, 100, 3200) !=
            IVFProbeScheduleStrategy::Dense ||
        choose_ivf_probe_schedule_strategy(
            IVFProbeScheduleStrategy::Sparse, 100, 100) !=
            IVFProbeScheduleStrategy::Sparse ||
        !throws<std::invalid_argument>([] {
            choose_ivf_probe_schedule_strategy(
                static_cast<IVFProbeScheduleStrategy>(99), 1, 32);
        })) {
        return fail(__LINE__);
    }

    // projection maps [x0, x1, x2] to [x0 + 2*x1 - 1, x2 + 1].
    const std::vector<float> projection = {
        1.0f, 2.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };
    const std::vector<float> bias = {-1.0f, 1.0f};
    const std::vector<float> scales = {0.5f, 2.0f};
    const std::vector<int8_t> vectors = {
        0, 0,    // dequantized [0, 0]
        4, 1,    // dequantized [2, 2]
        8, -1,   // dequantized [4, -2]
        -4, 2,   // dequantized [-2, 4]
    };
    const paraselect::quantization::PCAStorageView storage{
        vectors.data(), projection.data(), bias.data(), scales.data(),
        4, 3, 2};
    const std::vector<float> queries = {
        1.0f, 1.0f, 1.0f,  // projected [2, 2]
        0.0f, 0.0f, 3.0f,  // projected [-1, 4]
    };
    const std::vector<int> candidates = {
        2, 1, -1, -1, 0,
        0, 3, 2, -1, -1,
    };
    std::vector<int> output(4, -2);
    paraselect::search::IVFPCARefineStats stats;
    paraselect::search::ivf_pca_refine_cpu(
        storage, queries.data(), candidates.data(), 2, 5, 2, 1,
        output.data(), &stats);

    if (!equal(output, {1, 0, 3, 0})) return fail(__LINE__);
    if (std::abs(stats.avg_input_candidates - 3.0) > 1e-12 ||
        stats.threads != 1 || stats.query_transform_ms < 0.0 ||
        stats.select_ms < 0.0) {
        return fail(__LINE__);
    }

    int8_t* device_vectors = nullptr;
    float* device_queries = nullptr;
    int* device_candidates = nullptr;
    int* device_output = nullptr;
    if (cudaMalloc(&device_vectors, vectors.size() * sizeof(int8_t)) != cudaSuccess ||
        cudaMalloc(&device_queries, queries.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&device_candidates, candidates.size() * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&device_output, output.size() * sizeof(int)) != cudaSuccess ||
        cudaMemcpy(
            device_vectors, vectors.data(), vectors.size() * sizeof(int8_t),
            cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(
            device_queries, queries.data(), queries.size() * sizeof(float),
            cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(
            device_candidates, candidates.data(), candidates.size() * sizeof(int),
            cudaMemcpyHostToDevice) != cudaSuccess) {
        return fail(__LINE__);
    }
    paraselect::search::IVFPCAGpuRefiner gpu_refiner(storage, device_vectors, 2, 5, 2);
    paraselect::search::IVFPCARefineStats gpu_stats;
    gpu_refiner.run(
        device_queries, device_candidates, 2, 5, 5, 2, device_output,
        &gpu_stats, nullptr);
    std::vector<int> gpu_output(4, -2);
    if (cudaMemcpy(
            gpu_output.data(), device_output, gpu_output.size() * sizeof(int),
            cudaMemcpyDeviceToHost) != cudaSuccess ||
        !equal(gpu_output, output) || gpu_stats.avg_input_candidates != 3.0) {
        return fail(__LINE__);
    }
    gpu_refiner.run(
        device_queries, device_candidates, 2, 5, 3, 2, device_output,
        &gpu_stats, nullptr);
    if (cudaMemcpy(
            gpu_output.data(), device_output, gpu_output.size() * sizeof(int),
            cudaMemcpyDeviceToHost) != cudaSuccess ||
        !equal(gpu_output, {1, 2, 3, 0}) || gpu_stats.avg_input_candidates != 2.5) {
        return fail(__LINE__);
    }
    const std::vector<idx_t> list_offsets = {0, 2, 4};
    const std::vector<idx_t> reordered_ids = {10, 11, 12, 13};
    const paraselect::search::IVFListLayoutView lists{
        list_offsets.data(), reordered_ids.data(), 2, 4};
    const std::vector<idx_t> probes = {1, 0, 0, 1};
    std::vector<int> list_output(4, -2);
    paraselect::search::IVFPCAResidentConfig resident_config;
    resident_config.max_batch = 2;
    resident_config.max_nprobe = 2;
    resident_config.output_count = 2;
    resident_config.prepare_chunk_vectors = 2;
    resident_config.mode = paraselect::search::IVFPCAResidentMode::Full;
    resident_config.scan = paraselect::search::IVFPCAScanKind::FP32;
    resident_config.probe_schedule = IVFProbeScheduleStrategy::Dense;
    paraselect::search::IVFPCAResidentSearch resident_search(
        storage, lists, resident_config);
    paraselect::search::IVFPCAResidentStats resident_stats;
    resident_search.search(
        queries.data(), probes.data(), 2, 2, list_output.data(),
        &resident_stats, nullptr);
    if (!equal(list_output, {1, 0, 3, 1}) ||
        resident_stats.avg_scanned_vectors != 4.0 ||
        resident_stats.avg_reduce_input_candidates != 4.0) {
        return fail(__LINE__);
    }
    idx_t* device_probes = nullptr;
    if (cudaMalloc(&device_probes, probes.size() * sizeof(idx_t)) != cudaSuccess ||
        cudaMemcpy(device_probes, probes.data(), probes.size() * sizeof(idx_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return fail(__LINE__);
    }
    std::vector<int> device_probe_output(4, -2);
    resident_search.search_device_probes(
        queries.data(), {device_probes, 2, 2}, device_probe_output.data(),
        &resident_stats, nullptr);
    if (!equal(device_probe_output, list_output) ||
        resident_stats.avg_scanned_vectors != 4.0 ||
        resident_stats.avg_reduce_input_candidates != 4.0) {
        return fail(__LINE__);
    }
    resident_config.probe_schedule = IVFProbeScheduleStrategy::Sparse;
    paraselect::search::IVFPCAResidentSearch sparse_resident_search(
        storage, lists, resident_config);
    std::fill(device_probe_output.begin(), device_probe_output.end(), -2);
    sparse_resident_search.search_device_probes(
        queries.data(), {device_probes, 2, 2}, device_probe_output.data(),
        &resident_stats, nullptr);
    if (!equal(device_probe_output, list_output) ||
        resident_stats.probe_schedule != IVFProbeScheduleStrategy::Sparse ||
        resident_stats.avg_scanned_vectors != 4.0) {
        return fail(__LINE__);
    }
    const idx_t invalid_probe = 2;
    if (cudaMemcpy(device_probes, &invalid_probe, sizeof(idx_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        !throws<std::out_of_range>([&] {
            sparse_resident_search.search_device_probes(
                queries.data(), {device_probes, 2, 2},
                device_probe_output.data(), nullptr, nullptr);
        }) ||
        cudaMemcpy(device_probes, probes.data(), probes.size() * sizeof(idx_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return fail(__LINE__);
    }
    resident_config.output_count = 64;
    paraselect::search::IVFPCAResidentSearch resident_search_k64(
        storage, lists, resident_config);
    std::vector<int> list_output_k64(128, -2);
    resident_search_k64.search(
        queries.data(), probes.data(), 2, 2, list_output_k64.data(),
        &resident_stats, nullptr);
    if (list_output_k64[0] != 1 || list_output_k64[1] != 0 ||
        list_output_k64[64] != 3 || list_output_k64[65] != 1) {
        std::cerr << "resident PCA k=64 merge failed: "
                  << list_output_k64[0] << ' ' << list_output_k64[1] << ' '
                  << list_output_k64[64] << ' ' << list_output_k64[65] << '\n';
        return fail(__LINE__);
    }
    resident_config.output_count = 2;
    resident_config.mode = paraselect::search::IVFPCAResidentMode::Threshold;
    resident_config.probe_schedule = IVFProbeScheduleStrategy::Sparse;
    resident_config.threshold_seed_probes = 1;
    resident_config.threshold_candidate_capacity = 8;
    paraselect::search::IVFPCAResidentSearch threshold_search(
        storage, lists, resident_config);
    std::fill(list_output.begin(), list_output.end(), -2);
    threshold_search.search(
        queries.data(), probes.data(), 2, 2, list_output.data(),
        &resident_stats, nullptr);
    if (!equal(list_output, {1, 0, 3, 1}) ||
        resident_stats.avg_reduce_input_candidates < 2.0) {
        return fail(__LINE__);
    }
    std::fill(device_probe_output.begin(), device_probe_output.end(), -2);
    threshold_search.search_device_probes(
        queries.data(), {device_probes, 2, 2}, device_probe_output.data(),
        &resident_stats, nullptr);
    if (!equal(device_probe_output, list_output)) return fail(__LINE__);

    constexpr int dim64 = 64;
    constexpr int count64 = 320;
    std::vector<float> projection64(dim64 * dim64, 0.0f);
    std::vector<float> bias64(dim64);
    std::vector<float> scales64(dim64);
    std::vector<float> query64(dim64);
    std::vector<int8_t> vectors64(count64 * dim64);
    for (int d = 0; d < dim64; ++d) {
        projection64[d * dim64 + d] = 1.0f;
        bias64[d] = 0.01f * static_cast<float>(d % 7);
        scales64[d] = 0.25f + 0.01f * static_cast<float>(d % 13);
        query64[d] = static_cast<float>((d * 17 + 5) % 31) - 15.0f;
    }
    for (int row = 0; row < count64; ++row) {
        for (int d = 0; d < dim64; ++d) {
            uint32_t value = static_cast<uint32_t>(row + 1) * 0x9e3779b9u;
            value ^= static_cast<uint32_t>(d + 1) * 0x85ebca6bu;
            value ^= value >> 16;
            value *= 0x7feb352du;
            value ^= value >> 15;
            vectors64[row * dim64 + d] = static_cast<int8_t>(
                static_cast<int>(value % 101u) - 50);
        }
    }
    const paraselect::quantization::PCAStorageView storage64{
        vectors64.data(), projection64.data(), bias64.data(), scales64.data(),
        count64, dim64, dim64};
    const std::vector<idx_t> offsets64 = {0, count64 / 2, count64};
    std::vector<idx_t> ids64(count64);
    for (int i = 0; i < count64; ++i) ids64[i] = i;
    const paraselect::search::IVFListLayoutView lists64{offsets64.data(), ids64.data(), 2, count64};
    resident_config.max_batch = 1;
    resident_config.max_nprobe = 2;
    resident_config.output_count = 64;
    resident_config.mode = paraselect::search::IVFPCAResidentMode::Full;
    paraselect::search::IVFPCAResidentSearch search64(
        storage64, lists64, resident_config);
    const std::vector<idx_t> probes64 = {0, 1};
    std::vector<int> actual64(64, -1);
    search64.search(query64.data(), probes64.data(), 1, 2, actual64.data(),
                    nullptr, nullptr);
    std::vector<std::pair<float, int>> expected64;
    for (int row = 0; row < count64; ++row) {
        float distance = 0.0f;
        for (int d = 0; d < dim64; ++d) {
            const float projected = query64[d] + bias64[d];
            const float value = static_cast<float>(vectors64[row * dim64 + d]) * scales64[d];
            const float delta = projected - value;
            distance += delta * delta;
        }
        expected64.emplace_back(distance, row);
    }
    std::sort(expected64.begin(), expected64.end());
    for (int rank = 0; rank < 64; ++rank) {
        if (actual64[rank] != expected64[rank].second) {
            std::cerr << "resident PCA 64D mismatch at rank " << rank
                      << ": expected " << expected64[rank].second
                      << " actual " << actual64[rank] << '\n';
            return fail(__LINE__);
        }
    }

    std::vector<float> weighted64(dim64);
    float query_norm64 = 0.0f;
    float max_abs_weighted64 = 0.0f;
    for (int d = 0; d < dim64; ++d) {
        const float projected = query64[d] + bias64[d];
        weighted64[d] = projected * scales64[d];
        query_norm64 += projected * projected;
        max_abs_weighted64 = std::max(max_abs_weighted64, std::abs(weighted64[d]));
    }
    const float query_dot_scale64 = max_abs_weighted64 > 0.0f
        ? max_abs_weighted64 / 127.0f
        : 1.0f;
    std::vector<int8_t> quantized_query64(dim64);
    for (int d = 0; d < dim64; ++d) {
        const int value = std::max(-127, std::min(
            127, static_cast<int>(std::nearbyint(weighted64[d] / query_dot_scale64))));
        quantized_query64[d] = static_cast<int8_t>(value);
    }
    std::vector<std::pair<float, int>> expected_dp4a64;
    for (int row = 0; row < count64; ++row) {
        int dot = 0;
        float vector_norm = 0.0f;
        for (int d = 0; d < dim64; ++d) {
            const int8_t code = vectors64[row * dim64 + d];
            dot += static_cast<int>(quantized_query64[d]) * static_cast<int>(code);
            const float value = static_cast<float>(code) * scales64[d];
            vector_norm += value * value;
        }
        expected_dp4a64.emplace_back(
            query_norm64 + vector_norm - 2.0f * query_dot_scale64 * dot,
            row);
    }
    std::sort(expected_dp4a64.begin(), expected_dp4a64.end());
    resident_config.scan = paraselect::search::IVFPCAScanKind::I8Dp4a;
    resident_config.dp4a_kernel =
        paraselect::search::IVFPCADp4aKernelVersion::V1;
    paraselect::search::IVFPCAResidentSearch dp4a_v1_search64(
        storage64, lists64, resident_config);
    std::fill(actual64.begin(), actual64.end(), -1);
    dp4a_v1_search64.search(
        query64.data(), probes64.data(), 1, 2, actual64.data(), nullptr, nullptr);
    for (int rank = 0; rank < 64; ++rank) {
        if (actual64[rank] != expected_dp4a64[rank].second) {
            std::cerr << "resident PCA dp4a mismatch at rank " << rank
                      << ": expected " << expected_dp4a64[rank].second
                      << " actual " << actual64[rank] << '\n';
            return fail(__LINE__);
        }
    }
    resident_config.mode = paraselect::search::IVFPCAResidentMode::Threshold;
    resident_config.threshold_seed_probes = 1;
    resident_config.threshold_candidate_capacity = 512;
    paraselect::search::IVFPCAResidentSearch dp4a_v1_threshold_search64(
        storage64, lists64, resident_config);
    std::vector<int> threshold64(64, -1);
    dp4a_v1_threshold_search64.search(
        query64.data(), probes64.data(), 1, 2, threshold64.data(), nullptr, nullptr);
    if (threshold64 != actual64) {
        return fail(__LINE__);
    }
    resident_config.mode = paraselect::search::IVFPCAResidentMode::Full;
    resident_config.dp4a_kernel =
        paraselect::search::IVFPCADp4aKernelVersion::V2;
    paraselect::search::IVFPCAResidentSearch dp4a_v2_search64(
        storage64, lists64, resident_config);
    std::vector<int> actual_v2_64(64, -1);
    dp4a_v2_search64.search(
        query64.data(), probes64.data(), 1, 2, actual_v2_64.data(), nullptr, nullptr);
    if (actual_v2_64 != actual64) {
        return fail(__LINE__);
    }
    resident_config.mode = paraselect::search::IVFPCAResidentMode::Threshold;
    paraselect::search::IVFPCAResidentSearch dp4a_v2_threshold_search64(
        storage64, lists64, resident_config);
    std::fill(threshold64.begin(), threshold64.end(), -1);
    dp4a_v2_threshold_search64.search(
        query64.data(), probes64.data(), 1, 2, threshold64.data(), nullptr, nullptr);
    if (threshold64 != actual_v2_64) {
        return fail(__LINE__);
    }

    auto paired_config = resident_config;
    paired_config.max_batch = 8;
    paired_config.max_nprobe = 1;
    paired_config.dp4a_kernel =
        paraselect::search::IVFPCADp4aKernelVersion::V2;
    std::vector<float> paired_queries(8 * dim64);
    for (int query = 0; query < 8; ++query) {
        std::copy(query64.begin(), query64.end(),
                  paired_queries.begin() + (size_t)query * dim64);
    }
    const std::vector<idx_t> paired_probes(8, 0);
    std::vector<int> paired_v2(8 * 64, -1);
    paraselect::search::IVFPCAResidentSearch paired_v2_search(
        storage64, lists64, paired_config);
    paired_v2_search.search(
        paired_queries.data(), paired_probes.data(), 8, 1,
        paired_v2.data(), nullptr, nullptr);

    paired_config.dp4a_kernel =
        paraselect::search::IVFPCADp4aKernelVersion::V3;
    paraselect::search::IVFPCAResidentSearch paired_v3_search(
        storage64, lists64, paired_config);
    std::vector<int> paired_v3(8 * 64, -1);
    paired_v3_search.search(
        paired_queries.data(), paired_probes.data(), 8, 1,
        paired_v3.data(), nullptr, nullptr);
    if (paired_v3 != paired_v2) {
        std::cerr << "resident PCA DP4A V3 paired host probes mismatch\n";
        return fail(__LINE__);
    }
    idx_t* paired_device_probes = nullptr;
    if (cudaMalloc(&paired_device_probes,
                   paired_probes.size() * sizeof(idx_t)) != cudaSuccess ||
        cudaMemcpy(paired_device_probes, paired_probes.data(),
                   paired_probes.size() * sizeof(idx_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return fail(__LINE__);
    }
    std::fill(paired_v3.begin(), paired_v3.end(), -1);
    paired_v3_search.search_device_probes(
        paired_queries.data(), {paired_device_probes, 8, 1},
        paired_v3.data(), nullptr, nullptr);
    cudaFree(paired_device_probes);
    if (paired_v3 != paired_v2) {
        std::cerr << "resident PCA DP4A V3 paired device probes mismatch\n";
        return fail(__LINE__);
    }

    const auto check_large_shortlist = [&](int k,
                                            paraselect::search::IVFPCAScanKind scan,
                                            paraselect::search::IVFPCAResidentMode mode,
                                            paraselect::search::IVFPCADp4aKernelVersion version,
                                            const std::vector<std::pair<float, int>>& expected,
                                            const char* label) {
        auto config = resident_config;
        config.output_count = k;
        config.scan = scan;
        config.mode = mode;
        config.dp4a_kernel = version;
        config.threshold_seed_probes = 1;
        config.threshold_candidate_capacity = 512;
        paraselect::search::IVFPCAResidentSearch search(storage64, lists64, config);
        std::vector<int> actual((size_t)k, -1);
        search.search(
            query64.data(), probes64.data(), 1, 2, actual.data(), nullptr, nullptr);
        for (int rank = 0; rank < k; ++rank) {
            if (actual[rank] != expected[rank].second) {
                std::cerr << label << " k=" << k << " mismatch at rank " << rank
                          << ": expected " << expected[rank].second
                          << " actual " << actual[rank] << '\n';
                return false;
            }
        }
        std::fill(actual.begin(), actual.end(), -1);
        search.search_device_probes(
            query64.data(), {device_probes, 1, 2}, actual.data(), nullptr, nullptr);
        for (int rank = 0; rank < k; ++rank) {
            if (actual[rank] != expected[rank].second) {
                std::cerr << label << " device probes k=" << k
                          << " mismatch at rank " << rank << ": expected "
                          << expected[rank].second << " actual " << actual[rank]
                          << '\n';
                return false;
            }
        }
        return true;
    };
    for (int k : {128, 256}) {
        if (!check_large_shortlist(
                k, paraselect::search::IVFPCAScanKind::FP32,
                paraselect::search::IVFPCAResidentMode::Full,
                paraselect::search::IVFPCADp4aKernelVersion::V2,
                expected64, "resident PCA fp32 full") ||
            !check_large_shortlist(
                k, paraselect::search::IVFPCAScanKind::FP32,
                paraselect::search::IVFPCAResidentMode::Threshold,
                paraselect::search::IVFPCADp4aKernelVersion::V2,
                expected64, "resident PCA fp32 threshold") ||
            !check_large_shortlist(
                k, paraselect::search::IVFPCAScanKind::I8Dp4a,
                paraselect::search::IVFPCAResidentMode::Full,
                paraselect::search::IVFPCADp4aKernelVersion::V1,
                expected_dp4a64, "resident PCA dp4a v1 full") ||
            !check_large_shortlist(
                k, paraselect::search::IVFPCAScanKind::I8Dp4a,
                paraselect::search::IVFPCAResidentMode::Threshold,
                paraselect::search::IVFPCADp4aKernelVersion::V1,
                expected_dp4a64, "resident PCA dp4a v1 threshold") ||
            !check_large_shortlist(
                k, paraselect::search::IVFPCAScanKind::I8Dp4a,
                paraselect::search::IVFPCAResidentMode::Full,
                paraselect::search::IVFPCADp4aKernelVersion::V2,
                expected_dp4a64, "resident PCA dp4a v2 full") ||
            !check_large_shortlist(
                k, paraselect::search::IVFPCAScanKind::I8Dp4a,
                paraselect::search::IVFPCAResidentMode::Threshold,
                paraselect::search::IVFPCADp4aKernelVersion::V2,
                expected_dp4a64, "resident PCA dp4a v2 threshold")) {
            return fail(__LINE__);
        }
    }
    auto invalid_large_config = resident_config;
    invalid_large_config.output_count = 257;
    if (!throws<std::invalid_argument>([&] {
            paraselect::search::IVFPCAResidentSearch invalid_large_search(
                storage64, lists64, invalid_large_config);
        })) {
        return fail(__LINE__);
    }
    cudaFree(device_output);
    cudaFree(device_probes);
    cudaFree(device_candidates);
    cudaFree(device_queries);
    cudaFree(device_vectors);

    std::vector<int> padded(4, -2);
    const std::vector<int> sparse = {1, -1, -1, -1, -1};
    paraselect::search::ivf_pca_refine_cpu(
        storage, queries.data(), sparse.data(), 1, 5, 4, 1,
        padded.data(), nullptr);
    if (!equal(padded, {1, -1, -1, -1})) return fail(__LINE__);

    paraselect::quantization::PCAStorageView bad_scale = storage;
    const std::vector<float> invalid_scales = {0.5f, 0.0f};
    bad_scale.scales = invalid_scales.data();
    if (!throws<std::invalid_argument>([&] {
            paraselect::search::ivf_pca_refine_cpu(
                bad_scale, queries.data(), candidates.data(), 1, 5, 2, 1,
                output.data(), nullptr);
        })) {
        return fail(__LINE__);
    }

    const std::vector<int> out_of_range = {0, 4};
    if (!throws<std::out_of_range>([&] {
            paraselect::search::ivf_pca_refine_cpu(
                storage, queries.data(), out_of_range.data(), 1, 2, 1, 1,
                output.data(), nullptr);
        })) {
        return fail(__LINE__);
    }

    const std::vector<int> duplicate = {1, 1};
    if (!throws<std::invalid_argument>([&] {
            paraselect::search::ivf_pca_refine_cpu(
                storage, queries.data(), duplicate.data(), 1, 2, 1, 1,
                output.data(), nullptr);
        })) {
        return fail(__LINE__);
    }

    if (!throws<std::invalid_argument>([&] {
            paraselect::search::ivf_pca_refine_cpu(
                storage, queries.data(), candidates.data(), 1, 5, 6, 1,
                output.data(), nullptr);
        })) {
        return fail(__LINE__);
    }
    return 0;
}
