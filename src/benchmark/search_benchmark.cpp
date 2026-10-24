#include "search_benchmark.h"
#include "dataset/common/dataset_source.h"
#include "benchmark/recall_evaluator.h"
#include "benchmark/recall_evaluation.h"
#include "benchmark/recall_method.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <limits>

namespace paraselect {

namespace {

bool str_eq(const char* lhs, const char* rhs) {
    return std::strcmp(lhs, rhs) == 0;
}


}  // namespace

int SearchBenchmark::parseRecallMethod(const char* value) {
    if (!value || value[0] == '\0') {
        return IVF_RECALL_IDX_INTERSECTION;
    }
    if (str_eq(value, "idx_intersection") || str_eq(value, "idx") || str_eq(value, "0")) {
        return IVF_RECALL_IDX_INTERSECTION;
    }
    if (str_eq(value, "threshold") || str_eq(value, "1")) {
        return IVF_RECALL_THRESHOLD;
    }
    if (str_eq(value, "exact_recompute") || str_eq(value, "exact") || str_eq(value, "2")) {
        return IVF_RECALL_EXACT_RECOMPUTE;
    }
    if (str_eq(value, "adc_threshold") || str_eq(value, "adc") || str_eq(value, "3")) {
        return IVF_RECALL_ADC_THRESHOLD;
    }
    return IVF_RECALL_IDX_INTERSECTION;
}

int SearchBenchmark::recallMethodFromEnv(const char* env_name) {
    if (!env_name || env_name[0] == '\0') {
        env_name = "PARASELECT_RECALL_METHOD";
    }
    return parseRecallMethod(std::getenv(env_name));
}

float SearchBenchmark::computeRecall(
    const idx_t* gt, int n_gt,
    const idx_t* result, int n_query, int k_result) {
    return benchmark::compute_idx_intersection_recall(gt, n_gt, result, n_query, k_result, k_result);
}

bool SearchBenchmark::loadGroundTruth(
    const std::string& dataset_name,
    idx_t*& gt_data, float*& gt_distances, int& n_queries, int& k_gt) {
    gt_data = nullptr;
    gt_distances = nullptr;
    n_queries = 0;
    k_gt = 0;
    if (dataset_name.empty() || dataset_name == "DEFAULT") {
        std::cerr << "[SearchBenchmark] Invalid dataset name: " << dataset_name << std::endl;
        return false;
    }
    try {
        const dataset::DatasetSource source = dataset::open_dataset(
            dataset_name, dataset::parse_dataset_size_label(dataset_name));
        const dataset::GroundTruth groundtruth = source.load_groundtruth();
        n_queries = static_cast<int>(groundtruth.n);
        k_gt = groundtruth.k;
        gt_data = new idx_t[groundtruth.ids.size()];
        std::transform(groundtruth.ids.begin(), groundtruth.ids.end(), gt_data,
                       [](int id) { return static_cast<idx_t>(id); });
        if (!groundtruth.distances.empty()) {
            gt_distances = new float[groundtruth.distances.size()];
            std::copy(groundtruth.distances.begin(), groundtruth.distances.end(), gt_distances);
        }
        return true;
    } catch (const std::exception& error) {
        delete[] gt_data;
        delete[] gt_distances;
        gt_data = nullptr;
        gt_distances = nullptr;
        std::cerr << "[SearchBenchmark] Failed to load ground truth for "
                  << dataset_name << ": " << error.what() << std::endl;
        return false;
    }
}

bool SearchBenchmark::appendCsvRow(
    const std::string& path,
    const std::vector<std::string>& columns,
    const std::unordered_map<std::string, double>& numeric_fields,
    const std::unordered_map<std::string, std::string>& string_fields) {
    const bool needs_header = [&]() {
        std::ifstream in(path);
        return !in.good() || in.peek() == std::ifstream::traits_type::eof();
    }();

    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) {
        std::cerr << "[SearchBenchmark] Failed to open CSV for append: " << path << std::endl;
        return false;
    }

    // CSV is a data artifact: retain integer counts and round-trip doubles.
    out.precision(std::numeric_limits<double>::max_digits10);

    if (needs_header) {
        for (size_t i = 0; i < columns.size(); ++i) {
            if (i) out << ',';
            out << columns[i];
        }
        out << '\n';
    }

    for (size_t i = 0; i < columns.size(); ++i) {
        if (i) out << ',';
        const auto& key = columns[i];
        auto sit = string_fields.find(key);
        if (sit != string_fields.end()) {
            out << sit->second;
            continue;
        }
        auto nit = numeric_fields.find(key);
        if (nit != numeric_fields.end()) {
            out << nit->second;
        }
    }
    out << '\n';
    return true;
}

bool SearchBenchmark::resetCsv(const std::string& path) {
    std::ofstream out(path, std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "[SearchBenchmark] Failed to reset CSV: " << path << std::endl;
        return false;
    }
    return true;
}

std::string SearchBenchmark::defaultAnnDataDir() {
    const char* v = std::getenv("ANN_BENCHMARKS_DATA_DIR");
    if (v && *v) return std::string(v);
    return "/data/raw_dataset";
}

} // namespace paraselect

namespace paraselect::benchmark {

RecallEvaluationResult evaluate_recall(
    const RecallEvaluationConfig& config,
    const idx_t* indices,
    const float* distances,
    int n_query,
    int k_result) {
    if (!indices || n_query <= 0 || k_result <= 0) {
        throw std::invalid_argument("recall evaluation requires non-empty search results");
    }
    idx_t* gt_raw = nullptr;
    float* gt_distances_raw = nullptr;
    int gt_n_queries = 0;
    int gt_k = 0;
    if (!SearchBenchmark::loadGroundTruth(
            config.dataset_name,
            gt_raw,
            gt_distances_raw,
            gt_n_queries,
            gt_k)) {
        throw std::runtime_error(
            "ground truth is unavailable for " + config.dataset_name);
    }
    std::unique_ptr<idx_t[]> gt(gt_raw);
    std::unique_ptr<float[]> gt_distances(gt_distances_raw);
    size_t query_offset = 0;
    if (const char* value = std::getenv("PARASELECT_QUERY_OFFSET")) {
        if (value[0] != '\0') {
            try {
                query_offset = static_cast<size_t>(std::stoull(value));
            } catch (const std::exception&) {
                throw std::invalid_argument("PARASELECT_QUERY_OFFSET must be a non-negative integer");
            }
        }
    }
    if (query_offset > static_cast<size_t>(gt_n_queries) ||
        static_cast<size_t>(n_query) > static_cast<size_t>(gt_n_queries) - query_offset) {
        throw std::runtime_error("ground truth is smaller than query offset/request");
    }
    if (gt_k < k_result) {
        throw std::runtime_error(
            "ground truth shape is smaller than the recall request for " +
            config.dataset_name);
    }

    const int method = normalize_recall_method(config.recall_method);
    ExactRecallResult recall;
    if (method == IVF_RECALL_IDX_INTERSECTION) {
        recall = compute_idx_intersection_recalls(
            gt.get() + query_offset * static_cast<size_t>(gt_k),
            gt_k, indices, n_query, k_result);
    } else if (method == IVF_RECALL_EXACT_RECOMPUTE) {
        if (!compute_disk_exact_recall(
                config, indices,
                gt.get() + query_offset * static_cast<size_t>(gt_k),
                n_query, k_result, gt_k, &recall)) {
            throw std::runtime_error(
                "exact recall evaluation failed for " + config.dataset_name);
        }
    } else if (method == IVF_RECALL_ADC_THRESHOLD) {
        if (!compute_disk_adc_threshold_recall(
                config, distances,
                gt.get() + query_offset * static_cast<size_t>(gt_k),
                n_query, k_result, gt_k, &recall)) {
            throw std::runtime_error(
                "ADC threshold recall evaluation failed for " + config.dataset_name);
        }
    } else if (!compute_disk_threshold_recall(
                   config, indices,
                gt.get() + query_offset * static_cast<size_t>(gt_k),
                n_query, k_result, gt_k, &recall)) {
        throw std::runtime_error(
            "threshold recall evaluation failed for " + config.dataset_name);
    }

    RecallEvaluationResult out;
    out.available = true;
    out.recall_at_1 = recall.recall_at_1;
    out.recall_at_5 = recall.recall_at_5;
    out.recall_at_10 = recall.recall_at_10;
    out.method = recall_method_name(method);
    return out;
}

} // namespace paraselect::benchmark
