#ifndef PARASELECT_CUDA_DATASET_COMMON_DATASET_SPEC_H
#define PARASELECT_CUDA_DATASET_COMMON_DATASET_SPEC_H

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

#include "clustering/kmeans/kmeans_policy.cuh"
#include "vector_dtype.h"

#ifndef PARASELECT_DISTANCE_TYPE_DEFINED
#define PARASELECT_DISTANCE_TYPE_DEFINED
enum DistanceType {
    L2_DISTANCE = 0,
    COSINE_DISTANCE = 1
};
#endif

namespace paraselect {
namespace dataset {

struct DatasetSpec {
    std::string name;
    std::string size_label;
    size_t n_vectors;
    int dim;
    int default_n_query;
    DistanceType metric;
    VectorDType dtype;
};

inline std::string to_lower_ascii(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

inline std::string to_upper_ascii(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return s;
}

DatasetSpec describe_dataset(
    const std::string& name,
    const std::string& size_label);

inline std::string parse_dataset_size_label(const std::string& name) {
    const std::string lower = to_lower_ascii(name);
    const size_t first_digit = lower.find_first_of("0123456789");
    if (first_digit == std::string::npos) {
        throw std::runtime_error("Dataset size is missing: " + name);
    }

    size_t pos = first_digit;
    while (pos < lower.size() && std::isdigit(static_cast<unsigned char>(lower[pos]))) {
        ++pos;
    }
    if (pos >= lower.size() || (lower[pos] != 'm' && lower[pos] != 'b')) {
        throw std::runtime_error("Dataset size must use M or B suffix: " + name);
    }

    std::string label = lower.substr(first_digit, pos - first_digit + 1);
    if (label == "1000m") return "1B";
    return to_upper_ascii(label);
}

inline size_t parse_dataset_size(const std::string& name) {
    const std::string label = parse_dataset_size_label(name);
    const char suffix = label.back();
    const size_t value = static_cast<size_t>(std::stoull(label.substr(0, label.size() - 1)));
    if (suffix == 'M') return value * 1000000ULL;
    if (suffix == 'B') return value * 1000000000ULL;
    throw std::runtime_error("Unsupported dataset size suffix: " + label);
}

inline std::string distance_metric_name(DistanceType metric) {
    switch (metric) {
        case L2_DISTANCE: return "euclidean";
        case COSINE_DISTANCE: return "cosine";
    }
    throw std::runtime_error("Unknown distance metric");
}

inline std::string display_dataset_name(const DatasetSpec& spec) {
    return to_upper_ascii(spec.name) + spec.size_label;
}

inline std::string dataset_key_name(const DatasetSpec& spec) {
    return display_dataset_name(spec) + "-" + std::to_string(spec.dim) + "-" +
           distance_metric_name(spec.metric);
}

inline std::string groundtruth_dataset_name(const DatasetSpec& spec) {
    return spec.name + spec.size_label;
}

inline std::string benchmark_dataset_name(const DatasetSpec& spec) {
    return groundtruth_dataset_name(spec);
}

inline KMeansSpec make_kmeans_spec_for_dataset(
    bool use_minibatch,
    int kmeans_iters,
    float train_sample_ratio,
    int train_max_samples,
    KMeansAssignOperator assign_operator = KMeansAssignOperator::Kmeans,
    KMeansComputePrecision precision = KMeansComputePrecision::Tf32,
    KMeansTrainStrategy train_strategy = KMeansTrainStrategy::FullLloyd,
    int progressive_stage1_samples = 0,
    int progressive_stage2_samples = 0,
    int fast_ivf_train_samples = 0,
    KMeansFinalAssignStrategy final_assign_strategy = KMeansFinalAssignStrategy::Exact) {
    return make_kmeans_spec(
        use_minibatch,
        kmeans_iters,
        train_sample_ratio,
        train_max_samples,
        assign_operator,
        precision,
        train_strategy,
        progressive_stage1_samples,
        progressive_stage2_samples,
        fast_ivf_train_samples,
        final_assign_strategy);
}

} // namespace dataset
} // namespace paraselect

#endif // PARASELECT_CUDA_DATASET_COMMON_DATASET_SPEC_H
