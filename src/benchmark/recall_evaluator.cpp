#include "benchmark/recall_evaluator.h"

#include "dataset/common/dataset_source.h"
#include "benchmark/recall_method.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace paraselect {
namespace benchmark {
namespace {

struct ExactRecallDiskData {
    std::vector<float> queries;
    std::vector<float> base_vectors;
    std::unordered_map<idx_t, size_t> base_offsets;
    int dim = 0;
};

dataset::DatasetSource open_recall_dataset(const std::string& dataset_name) {
    return dataset::open_dataset(
        dataset_name, dataset::parse_dataset_size_label(dataset_name));
}

void copy_as_float(dataset::VectorView view, std::vector<float>* output) {
    output->resize(view.n * static_cast<size_t>(view.dim));
    if (view.dtype == dataset::VectorDType::F32) {
        std::copy(view.fp32(), view.fp32() + output->size(), output->begin());
    } else {
        std::transform(view.u8(), view.u8() + output->size(), output->begin(),
                       [](uint8_t value) { return static_cast<float>(value); });
    }
}

bool load_queries_as_float(
    const dataset::DatasetSource& source,
    size_t count,
    std::vector<float>* output,
    int* dim_out) {
    if (!output || !dim_out || count == 0) return false;
    const dataset::VectorBuffer queries = source.load_queries(count);
    if (queries.view().n != count) return false;
    copy_as_float(queries.view(), output);
    *dim_out = queries.view().dim;
    return true;
}

bool load_selected_vectors_as_float(
    const dataset::DatasetSource& source,
    const std::vector<idx_t>& ids,
    int n_base_limit,
    std::vector<float>* vectors,
    std::unordered_map<idx_t, size_t>* offsets,
    int* dim_out) {
    if (!vectors || !offsets || !dim_out || ids.empty()) return false;
    std::vector<uint32_t> source_ids;
    source_ids.reserve(ids.size());
    for (idx_t id : ids) {
        if (id >= 0 && id < n_base_limit &&
            static_cast<size_t>(id) < source.spec().n_vectors) {
            source_ids.push_back(static_cast<uint32_t>(id));
        }
    }
    if (source_ids.empty()) return false;

    const dataset::VectorBuffer selected = source.load_base_rows(source_ids);
    copy_as_float(selected.view(), vectors);
    offsets->clear();
    offsets->reserve(source_ids.size() * 2);
    for (size_t row = 0; row < source_ids.size(); ++row) {
        (*offsets)[static_cast<idx_t>(source_ids[row])] = row;
    }
    *dim_out = selected.view().dim;
    return true;
}

ExactRecallResult compute_threshold_recall_from_disk_gt_data(
    const ExactRecallDiskData& data,
    const idx_t* result_indices,
    const idx_t* gt_indices,
    int n_query,
    int k_result,
    int gt_k,
    int distance_mode) {
    ExactRecallResult out;
    if (data.queries.empty() || data.base_vectors.empty() || data.dim <= 0 ||
        !result_indices || !gt_indices || n_query <= 0 || k_result <= 0 || gt_k <= 0) {
        return out;
    }

    const int recall_ks[3] = {1, 5, 10};
    long long hit_counts[3] = {0, 0, 0};
    const float eps = distance_mode == L2_DISTANCE ? 1e-3f : 1e-5f;
    const int threads = resolve_exact_recall_threads(n_query);
    const int chunk = (n_query + threads - 1) / threads;
    std::vector<std::thread> workers;
    std::vector<std::array<long long, 3>> local_hits((size_t)threads);

    auto base_ptr = [&](idx_t id) -> const float* {
        const auto it = data.base_offsets.find(id);
        if (it == data.base_offsets.end()) return nullptr;
        return data.base_vectors.data() + it->second * (size_t)data.dim;
    };

    for (int tid = 0; tid < threads; ++tid) {
        const int q_begin = tid * chunk;
        const int q_end = std::min(n_query, q_begin + chunk);
        workers.emplace_back([&, tid, q_begin, q_end]() {
            auto& hits = local_hits[(size_t)tid];
            hits = {0, 0, 0};
            for (int q = q_begin; q < q_end; ++q) {
                const float* qv = data.queries.data() + (size_t)q * data.dim;
                for (int rk_i = 0; rk_i < 3; ++rk_i) {
                    const int rk = recall_ks[rk_i];
                    if (rk > k_result || rk > gt_k) continue;
                    const idx_t gt_id = gt_indices[(size_t)q * gt_k + (rk - 1)];
                    const float* gt_bv = base_ptr(gt_id);
                    if (!gt_bv) continue;
                    const float threshold = exact_distance_for_recall(qv, gt_bv, data.dim, distance_mode) + eps;
                    for (int j = 0; j < rk; ++j) {
                        const idx_t id = result_indices[(size_t)q * k_result + j];
                        const float* bv = base_ptr(id);
                        if (!bv) continue;
                        const float result_dist = exact_distance_for_recall(qv, bv, data.dim, distance_mode);
                        if (result_dist <= threshold) {
                            hits[(size_t)rk_i]++;
                        }
                    }
                }
            }
        });
    }
    for (auto& worker : workers) worker.join();
    for (const auto& hits : local_hits) {
        for (int i = 0; i < 3; ++i) hit_counts[i] += hits[(size_t)i];
    }

    out.recall_at_1 = (gt_k >= 1 && k_result >= 1)
        ? static_cast<float>(hit_counts[0]) / static_cast<float>(n_query)
        : -1.0f;
    out.recall_at_5 = (gt_k >= 5 && k_result >= 5)
        ? static_cast<float>(hit_counts[1]) / static_cast<float>(n_query * 5)
        : -1.0f;
    out.recall_at_10 = (gt_k >= 10 && k_result >= 10)
        ? static_cast<float>(hit_counts[2]) / static_cast<float>(n_query * 10)
        : -1.0f;
    return out;
}

ExactRecallResult compute_exact_threshold_recall_from_disk_data(
    const ExactRecallDiskData& data,
    const idx_t* result_indices,
    const idx_t* gt_indices,
    int n_query,
    int k_result,
    int gt_k,
    int distance_mode) {
    ExactRecallResult out;
    if (data.queries.empty() || data.base_vectors.empty() || data.dim <= 0 ||
        !result_indices || !gt_indices || n_query <= 0 || k_result <= 0 || gt_k <= 0) {
        return out;
    }

    const int recall_ks[3] = {1, 5, 10};
    long long hit_counts[3] = {0, 0, 0};
    const float eps = distance_mode == L2_DISTANCE ? 1e-3f : 1e-5f;
    const int threads = resolve_exact_recall_threads(n_query);
    const int chunk = (n_query + threads - 1) / threads;
    std::vector<std::thread> workers;
    std::vector<std::array<long long, 3>> local_hits((size_t)threads);

    auto base_ptr = [&](idx_t id) -> const float* {
        const auto it = data.base_offsets.find(id);
        if (it == data.base_offsets.end()) return nullptr;
        return data.base_vectors.data() + it->second * (size_t)data.dim;
    };

    for (int tid = 0; tid < threads; ++tid) {
        const int q_begin = tid * chunk;
        const int q_end = std::min(n_query, q_begin + chunk);
        workers.emplace_back([&, tid, q_begin, q_end]() {
            auto& hits = local_hits[(size_t)tid];
            hits = {0, 0, 0};
            float result_dist[10];
            for (int q = q_begin; q < q_end; ++q) {
                const float* qv = data.queries.data() + (size_t)q * data.dim;
                const int max_k = std::min(k_result, 10);
                for (int j = 0; j < max_k; ++j) {
                    result_dist[j] = std::numeric_limits<float>::infinity();
                    const idx_t id = result_indices[(size_t)q * k_result + j];
                    const float* bv = base_ptr(id);
                    if (bv) {
                        result_dist[j] = exact_distance_for_recall(qv, bv, data.dim, distance_mode);
                    }
                }

                for (int rk_i = 0; rk_i < 3; ++rk_i) {
                    const int rk = recall_ks[rk_i];
                    if (rk > k_result || rk > gt_k) continue;
                    const idx_t gt_id = gt_indices[(size_t)q * gt_k + (rk - 1)];
                    const float* gt_bv = base_ptr(gt_id);
                    if (!gt_bv) continue;
                    const float threshold = exact_distance_for_recall(qv, gt_bv, data.dim, distance_mode) + eps;
                    for (int j = 0; j < rk; ++j) {
                        if (result_dist[j] <= threshold) {
                            hits[(size_t)rk_i]++;
                        }
                    }
                }
            }
        });
    }
    for (auto& worker : workers) worker.join();
    for (const auto& hits : local_hits) {
        for (int i = 0; i < 3; ++i) hit_counts[i] += hits[(size_t)i];
    }

    const char* debug_env = std::getenv("PARASELECT_RECALL_DEBUG");
    if (debug_env && debug_env[0] == '1' && n_query > 0) {
        const float* qv = data.queries.data();
        std::cout << "[SearchBenchmark:ExactRecallDebug] q=0";
        for (int rk : {1, 5, 10}) {
            if (rk <= k_result && rk <= gt_k) {
                const idx_t gt_id = gt_indices[(size_t)(rk - 1)];
                const float* gt_bv = base_ptr(gt_id);
                const float threshold = gt_bv
                    ? exact_distance_for_recall(qv, gt_bv, data.dim, distance_mode)
                    : std::numeric_limits<float>::infinity();
                std::cout << " gt" << rk << "_id=" << gt_id
                          << " gt" << rk << "_dist=" << threshold;
            }
        }
        const int print_k = std::min(k_result, 10);
        std::cout << " results=";
        for (int j = 0; j < print_k; ++j) {
            const idx_t id = result_indices[(size_t)j];
            const float* bv = base_ptr(id);
            const float d = bv
                ? exact_distance_for_recall(qv, bv, data.dim, distance_mode)
                : std::numeric_limits<float>::infinity();
            if (j) std::cout << ';';
            std::cout << id << ':' << d;
        }
        std::cout << std::endl;
    }

    out.recall_at_1 = (gt_k >= 1 && k_result >= 1) ?
        static_cast<float>(hit_counts[0]) / static_cast<float>(n_query) : -1.0f;
    out.recall_at_5 = (gt_k >= 5 && k_result >= 5) ?
        static_cast<float>(hit_counts[1]) / static_cast<float>(n_query * 5) : -1.0f;
    out.recall_at_10 = (gt_k >= 10 && k_result >= 10) ?
        static_cast<float>(hit_counts[2]) / static_cast<float>(n_query * 10) : -1.0f;
    return out;
}

}  // namespace

float compute_idx_intersection_recall(
    const idx_t* gt,
    int n_gt,
    const idx_t* result,
    int n_query,
    int k_result,
    int recall_k) {
    if (!gt || !result || n_query <= 0 || k_result < recall_k || recall_k <= 0 || n_gt < recall_k) {
        return -1.0f;
    }

    int total_hits = 0;
    for (int q = 0; q < n_query; ++q) {
        const idx_t* gt_row = gt + (size_t)q * n_gt;
        const idx_t* res_row = result + (size_t)q * k_result;

        int hits = 0;
        for (int j = 0; j < recall_k; ++j) {
            const idx_t res_idx = res_row[j];
            if (res_idx < 0) continue;
            for (int i = 0; i < recall_k; ++i) {
                if (gt_row[i] == res_idx) {
                    hits++;
                    break;
                }
            }
        }
        total_hits += hits;
    }

    return static_cast<float>(total_hits) / static_cast<float>(n_query * recall_k);
}

ExactRecallResult compute_idx_intersection_recalls(
    const idx_t* gt,
    int gt_k,
    const idx_t* result,
    int n_query,
    int k_result) {
    ExactRecallResult out;
    out.recall_at_1 = (gt_k >= 1 && k_result >= 1)
        ? compute_idx_intersection_recall(gt, gt_k, result, n_query, k_result, 1)
        : -1.0f;
    out.recall_at_5 = (gt_k >= 5 && k_result >= 5)
        ? compute_idx_intersection_recall(gt, gt_k, result, n_query, k_result, 5)
        : -1.0f;
    out.recall_at_10 = (gt_k >= 10 && k_result >= 10)
        ? compute_idx_intersection_recall(gt, gt_k, result, n_query, k_result, 10)
        : -1.0f;
    return out;
}

ExactRecallResult compute_adc_threshold_recall_from_disk_gt_data(
    const ExactRecallDiskData& data,
    const float* result_distances,
    const idx_t* gt_indices,
    int n_query,
    int k_result,
    int gt_k,
    int distance_mode) {
    ExactRecallResult out;
    if (data.queries.empty() || data.base_vectors.empty() || data.dim <= 0 ||
        !result_distances || !gt_indices || n_query <= 0 || k_result <= 0 || gt_k <= 0) {
        return out;
    }

    const int recall_ks[3] = {1, 5, 10};
    long long hit_counts[3] = {0, 0, 0};
    const float eps = distance_mode == L2_DISTANCE ? 1e-3f : 1e-5f;
    const int threads = resolve_exact_recall_threads(n_query);
    const int chunk = (n_query + threads - 1) / threads;
    std::vector<std::thread> workers;
    std::vector<std::array<long long, 3>> local_hits((size_t)threads);

    auto base_ptr = [&](idx_t id) -> const float* {
        const auto it = data.base_offsets.find(id);
        if (it == data.base_offsets.end()) return nullptr;
        return data.base_vectors.data() + it->second * (size_t)data.dim;
    };

    for (int tid = 0; tid < threads; ++tid) {
        const int q_begin = tid * chunk;
        const int q_end = std::min(n_query, q_begin + chunk);
        workers.emplace_back([&, tid, q_begin, q_end]() {
            auto& hits = local_hits[(size_t)tid];
            hits = {0, 0, 0};
            for (int q = q_begin; q < q_end; ++q) {
                const float* qv = data.queries.data() + (size_t)q * data.dim;
                for (int rk_i = 0; rk_i < 3; ++rk_i) {
                    const int rk = recall_ks[rk_i];
                    if (rk > k_result || rk > gt_k) continue;
                    const idx_t gt_id = gt_indices[(size_t)q * gt_k + (rk - 1)];
                    const float* gt_bv = base_ptr(gt_id);
                    if (!gt_bv) continue;
                    const float threshold = exact_distance_for_recall(qv, gt_bv, data.dim, distance_mode) + eps;
                    for (int j = 0; j < rk; ++j) {
                        const float adc_dist = result_distances[(size_t)q * k_result + j];
                        if (std::isfinite(adc_dist) && adc_dist <= threshold) {
                            hits[(size_t)rk_i]++;
                        }
                    }
                }
            }
        });
    }
    for (auto& worker : workers) worker.join();
    for (const auto& hits : local_hits) {
        for (int i = 0; i < 3; ++i) hit_counts[i] += hits[(size_t)i];
    }

    out.recall_at_1 = (gt_k >= 1 && k_result >= 1)
        ? static_cast<float>(hit_counts[0]) / static_cast<float>(n_query)
        : -1.0f;
    out.recall_at_5 = (gt_k >= 5 && k_result >= 5)
        ? static_cast<float>(hit_counts[1]) / static_cast<float>(n_query * 5)
        : -1.0f;
    out.recall_at_10 = (gt_k >= 10 && k_result >= 10)
        ? static_cast<float>(hit_counts[2]) / static_cast<float>(n_query * 10)
        : -1.0f;
    return out;
}

bool compute_disk_adc_threshold_recall(
    const RecallEvaluationConfig& config,
    const float* result_distances,
    const idx_t* gt_indices,
    int n_query,
    int k_result,
    int gt_k,
    ExactRecallResult* out) {
    if (!out || !result_distances || !gt_indices) return false;
    try {
        const std::string& dataset_name = config.dataset_name;
        const dataset::DatasetSource source = open_recall_dataset(dataset_name);
        const idx_t n_total_vectors = static_cast<idx_t>(source.base_info().n);

        ExactRecallDiskData data;
        int query_dim = 0;
        if (!load_queries_as_float(source, static_cast<size_t>(n_query),
            &data.queries, &query_dim)) {
            std::cerr << "[SearchBenchmark] ADC threshold recall failed to load queries from disk: "
                      << source.query_path() << std::endl;
            return false;
        }

        std::vector<idx_t> ids;
        ids.reserve((size_t)n_query * 3);
        for (int q = 0; q < n_query; ++q) {
            for (int rk : {1, 5, 10}) {
                if (rk <= gt_k && rk <= k_result) {
                    const idx_t id = gt_indices[(size_t)q * gt_k + (rk - 1)];
                    if (id >= 0 && id < n_total_vectors) ids.push_back(id);
                }
            }
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

        int base_dim = 0;
        if (!load_selected_vectors_as_float(source, ids, n_total_vectors,
                                            &data.base_vectors, &data.base_offsets, &base_dim)) {
            std::cerr << "[SearchBenchmark] ADC threshold recall failed to load GT base vectors from disk: "
                      << source.base_path() << std::endl;
            return false;
        }
        if (query_dim != base_dim || query_dim != source.spec().dim) {
            std::cerr << "[SearchBenchmark] ADC threshold recall dimension mismatch: query_dim=" << query_dim
                      << " base_dim=" << base_dim << " dataset_dim=" << source.spec().dim << std::endl;
            return false;
        }
        data.dim = query_dim;
        *out = compute_adc_threshold_recall_from_disk_gt_data(
            data, result_distances, gt_indices, n_query, k_result, gt_k, config.distance_mode);
        std::cout << "[SearchBenchmark] ADC threshold recall disk exact threshold vectors: unique_gt_ids="
                  << ids.size() << ", query_vectors=" << n_query
                  << ", threads=" << resolve_exact_recall_threads(n_query) << std::endl;
        return out->recall_at_1 >= 0.0f || out->recall_at_5 >= 0.0f || out->recall_at_10 >= 0.0f;
    } catch (const std::exception& e) {
        std::cerr << "[SearchBenchmark] ADC threshold recall from disk failed: " << e.what() << std::endl;
        return false;
    }
}

bool compute_disk_threshold_recall(
    const RecallEvaluationConfig& config,
    const idx_t* result_indices,
    const idx_t* gt_indices,
    int n_query,
    int k_result,
    int gt_k,
    ExactRecallResult* out) {
    if (!out || !result_indices || !gt_indices) return false;
    try {
        const std::string& dataset_name = config.dataset_name;
        const dataset::DatasetSource source = open_recall_dataset(dataset_name);
        const idx_t n_total_vectors = static_cast<idx_t>(source.base_info().n);

        ExactRecallDiskData data;
        int query_dim = 0;
        if (!load_queries_as_float(source, static_cast<size_t>(n_query),
            &data.queries, &query_dim)) {
            std::cerr << "[SearchBenchmark] Threshold recall failed to load queries from disk: "
                      << source.query_path() << std::endl;
            return false;
        }

        std::vector<idx_t> ids;
        ids.reserve((size_t)n_query * (size_t)(k_result + 3));
        for (int q = 0; q < n_query; ++q) {
            for (int j = 0; j < k_result; ++j) {
                const idx_t id = result_indices[(size_t)q * k_result + j];
                if (id >= 0 && id < n_total_vectors) ids.push_back(id);
            }
            for (int rk : {1, 5, 10}) {
                if (rk <= gt_k && rk <= k_result) {
                    const idx_t id = gt_indices[(size_t)q * gt_k + (rk - 1)];
                    if (id >= 0 && id < n_total_vectors) ids.push_back(id);
                }
            }
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

        int base_dim = 0;
        if (!load_selected_vectors_as_float(source, ids, n_total_vectors,
                                            &data.base_vectors, &data.base_offsets, &base_dim)) {
            std::cerr << "[SearchBenchmark] Threshold recall failed to load GT base vectors from disk: "
                      << source.base_path() << std::endl;
            return false;
        }
        if (query_dim != base_dim || query_dim != source.spec().dim) {
            std::cerr << "[SearchBenchmark] Threshold recall dimension mismatch: query_dim=" << query_dim
                      << " base_dim=" << base_dim << " dataset_dim=" << source.spec().dim << std::endl;
            return false;
        }
        data.dim = query_dim;
        *out = compute_threshold_recall_from_disk_gt_data(
            data, result_indices, gt_indices, n_query, k_result, gt_k, config.distance_mode);
        std::cout << "[SearchBenchmark] Threshold recall disk exact vectors: unique_ids=" << ids.size()
                  << ", query_vectors=" << n_query
                  << ", threads=" << resolve_exact_recall_threads(n_query) << std::endl;
        return out->recall_at_1 >= 0.0f || out->recall_at_5 >= 0.0f || out->recall_at_10 >= 0.0f;
    } catch (const std::exception& e) {
        std::cerr << "[SearchBenchmark] Threshold recall from disk failed: " << e.what() << std::endl;
        return false;
    }
}

bool compute_disk_exact_recall(
    const RecallEvaluationConfig& config,
    const idx_t* result_indices,
    const idx_t* gt_indices,
    int n_query,
    int k_result,
    int gt_k,
    ExactRecallResult* out) {
    if (!out || !result_indices || !gt_indices) return false;
    try {
        const std::string& dataset_name = config.dataset_name;
        const dataset::DatasetSource source = open_recall_dataset(dataset_name);
        const idx_t n_total_vectors = static_cast<idx_t>(source.base_info().n);

        ExactRecallDiskData data;
        int query_dim = 0;
        if (!load_queries_as_float(source, static_cast<size_t>(n_query),
            &data.queries, &query_dim)) {
            std::cerr << "[SearchBenchmark] Exact recall failed to load queries from disk: "
                      << source.query_path() << std::endl;
            return false;
        }

        const int max_result_k = std::min(k_result, 10);
        std::vector<idx_t> ids;
        ids.reserve((size_t)n_query * (size_t)(max_result_k + 3));
        for (int q = 0; q < n_query; ++q) {
            for (int j = 0; j < max_result_k; ++j) {
                const idx_t id = result_indices[(size_t)q * k_result + j];
                if (id >= 0 && id < n_total_vectors) ids.push_back(id);
            }
            for (int rk : {1, 5, 10}) {
                if (rk <= gt_k && rk <= k_result) {
                    const idx_t id = gt_indices[(size_t)q * gt_k + (rk - 1)];
                    if (id >= 0 && id < n_total_vectors) ids.push_back(id);
                }
            }
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

        int base_dim = 0;
        if (!load_selected_vectors_as_float(source, ids, n_total_vectors,
                                            &data.base_vectors, &data.base_offsets, &base_dim)) {
            std::cerr << "[SearchBenchmark] Exact recall failed to load selected base vectors from disk: "
                      << source.base_path() << std::endl;
            return false;
        }
        if (query_dim != base_dim || query_dim != source.spec().dim) {
            std::cerr << "[SearchBenchmark] Exact recall dimension mismatch: query_dim=" << query_dim
                      << " base_dim=" << base_dim << " dataset_dim=" << source.spec().dim << std::endl;
            return false;
        }
        data.dim = query_dim;
        *out = compute_exact_threshold_recall_from_disk_data(
            data, result_indices, gt_indices, n_query, k_result, gt_k, config.distance_mode);
        std::cout << "[SearchBenchmark] Exact recall disk vectors: unique_base_ids=" << ids.size()
                  << ", query_vectors=" << n_query
                  << ", threads=" << resolve_exact_recall_threads(n_query) << std::endl;
        return out->recall_at_1 >= 0.0f || out->recall_at_5 >= 0.0f || out->recall_at_10 >= 0.0f;
    } catch (const std::exception& e) {
        std::cerr << "[SearchBenchmark] Exact recall from disk failed: " << e.what() << std::endl;
        return false;
    }
}

}  // namespace benchmark
}  // namespace paraselect
