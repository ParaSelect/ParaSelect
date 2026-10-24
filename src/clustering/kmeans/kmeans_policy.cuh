#ifndef PARASELECT_CUDA_CLUSTERING_KMEANS_POLICY_CUH
#define PARASELECT_CUDA_CLUSTERING_KMEANS_POLICY_CUH

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

enum class KMeansAssignOperator {
    Kmeans,
    FlashKmeans
};

enum class KMeansFinalAssignStrategy {
    Exact,
    Cagra
};

enum class KMeansTrainStrategy {
    FullLloyd,
    MiniBatch,
    Subset,
    ProgressivePrefix,
    FastIVF,
    CagraLloyd,
    Hierarchical
};

enum class KMeansCagraProfile {
    FastV1
};

enum class KMeansCagraBuildAlgo {
    Auto,
    IvfPq,
    NnDescent
};

enum class KMeansComputePrecision {
    Tf32,
    Fp16
};

enum class KMeansReorderPolicy {
    Standard,
    NumaFirstTouch
};

struct KMeansSpec {
    KMeansAssignOperator assign_operator = KMeansAssignOperator::Kmeans;
    KMeansTrainStrategy train_strategy = KMeansTrainStrategy::FullLloyd;
    KMeansComputePrecision precision = KMeansComputePrecision::Tf32;
    int kmeans_iters = 20;
    float train_sample_ratio = 1.0f;
    int train_max_samples = 0;
    int progressive_stage1_samples = 0;
    int progressive_stage2_samples = 0;
    int fast_ivf_train_samples = 0;
    int cagra_train_samples = 0;
    KMeansCagraProfile cagra_profile = KMeansCagraProfile::FastV1;
    KMeansFinalAssignStrategy final_assign_strategy = KMeansFinalAssignStrategy::Exact;
    KMeansReorderPolicy reorder_policy = KMeansReorderPolicy::Standard;
    bool cluster_only = false;
    float hierarchical_alpha = 0.0f;
};

inline const char* kmeans_assign_operator_tag(KMeansAssignOperator op) {
    switch (op) {
        case KMeansAssignOperator::Kmeans: return "kmeans";
        case KMeansAssignOperator::FlashKmeans: return "flash";
    }
    throw std::runtime_error("unknown KMeansAssignOperator");
}

inline const char* kmeans_final_assign_strategy_tag(KMeansFinalAssignStrategy strategy) {
    switch (strategy) {
        case KMeansFinalAssignStrategy::Exact: return "exact";
        case KMeansFinalAssignStrategy::Cagra: return "cagra";
    }
    throw std::runtime_error("unknown KMeansFinalAssignStrategy");
}

inline const char* kmeans_train_strategy_tag(KMeansTrainStrategy strategy) {
    switch (strategy) {
        case KMeansTrainStrategy::FullLloyd: return "full";
        case KMeansTrainStrategy::MiniBatch: return "minibatch";
        case KMeansTrainStrategy::Subset: return "subset";
        case KMeansTrainStrategy::ProgressivePrefix: return "progressive-prefix";
        case KMeansTrainStrategy::FastIVF: return "fast-ivf";
        case KMeansTrainStrategy::CagraLloyd: return "cagra";
        case KMeansTrainStrategy::Hierarchical: return "hierarchical";
    }
    throw std::runtime_error("unknown KMeansTrainStrategy");
}

inline const char* kmeans_cagra_profile_tag(KMeansCagraProfile profile) {
    switch (profile) {
        case KMeansCagraProfile::FastV1: return "fast-v1";
    }
    throw std::runtime_error("unknown KMeansCagraProfile");
}


inline const char* kmeans_reorder_policy_tag(KMeansReorderPolicy policy) {
    switch (policy) {
        case KMeansReorderPolicy::Standard: return "standard";
        case KMeansReorderPolicy::NumaFirstTouch: return "numa";
    }
    throw std::runtime_error("unknown KMeansReorderPolicy");
}

inline const char* kmeans_compute_precision_tag(KMeansComputePrecision precision) {
    switch (precision) {
        case KMeansComputePrecision::Tf32: return "tf32";
        case KMeansComputePrecision::Fp16: return "fp16";
    }
    throw std::runtime_error("unknown KMeansComputePrecision");
}

inline std::string kmeans_format_ratio_for_tag(float ratio) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << ratio;
    std::string out = oss.str();
    while (!out.empty() && out.back() == '0') out.pop_back();
    if (!out.empty() && out.back() == '.') out.pop_back();
    std::replace(out.begin(), out.end(), '.', 'p');
    return out;
}

inline KMeansTrainStrategy infer_kmeans_train_strategy(
    bool use_minibatch,
    float train_sample_ratio,
    int train_max_samples) {
    if (use_minibatch) return KMeansTrainStrategy::MiniBatch;
    if (train_max_samples > 0 || train_sample_ratio < 1.0f) return KMeansTrainStrategy::Subset;
    return KMeansTrainStrategy::FullLloyd;
}

inline bool kmeans_uses_flash_assign(KMeansAssignOperator op) {
    return op == KMeansAssignOperator::FlashKmeans;
}

inline KMeansSpec make_kmeans_spec(
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
    KMeansFinalAssignStrategy final_assign_strategy = KMeansFinalAssignStrategy::Exact,
    bool cluster_only = false,
    KMeansReorderPolicy reorder_policy = KMeansReorderPolicy::Standard) {
    KMeansSpec spec;
    spec.assign_operator = assign_operator;
    spec.train_strategy = (train_strategy == KMeansTrainStrategy::FullLloyd)
        ? infer_kmeans_train_strategy(use_minibatch, train_sample_ratio, train_max_samples)
        : train_strategy;
    spec.precision = precision;
    spec.kmeans_iters = kmeans_iters;
    spec.train_sample_ratio = train_sample_ratio;
    spec.train_max_samples = train_max_samples;
    spec.progressive_stage1_samples = progressive_stage1_samples;
    spec.progressive_stage2_samples = progressive_stage2_samples;
    spec.fast_ivf_train_samples = fast_ivf_train_samples;
    spec.final_assign_strategy = final_assign_strategy;
    spec.reorder_policy = reorder_policy;
    spec.cluster_only = cluster_only;
    return spec;
}

inline std::string make_kmeans_algo_tag(const KMeansSpec& spec) {
    if (spec.train_strategy == KMeansTrainStrategy::CagraLloyd) {
        return "kmeans-cagra-" + std::string(kmeans_cagra_profile_tag(spec.cagra_profile)) +
               "-train" + std::to_string(spec.cagra_train_samples) +
               "-iters" + std::to_string(spec.kmeans_iters);
    }
    std::string mode = kmeans_train_strategy_tag(spec.train_strategy);
    if (std::fabs(spec.train_sample_ratio - 1.0f) > 1e-6f) {
        mode += "-ratio" + kmeans_format_ratio_for_tag(spec.train_sample_ratio);
    }
    if (spec.train_max_samples > 0) {
        mode += "-sample" + std::to_string(spec.train_max_samples);
    }
    if (spec.train_strategy == KMeansTrainStrategy::ProgressivePrefix) {
        mode += "-stages" + std::to_string(spec.progressive_stage1_samples) +
                "_" + std::to_string(spec.progressive_stage2_samples);
    }
    if (spec.train_strategy == KMeansTrainStrategy::FastIVF) {
        mode += "-train" + std::to_string(spec.fast_ivf_train_samples);
        mode += "-final-" + std::string(kmeans_final_assign_strategy_tag(spec.final_assign_strategy));
    }
    if (spec.cluster_only) {
        mode += "-cluster-only";
    }
    if (spec.train_strategy == KMeansTrainStrategy::Hierarchical &&
        spec.hierarchical_alpha > 0.0f) {
        mode += "-alpha" + kmeans_format_ratio_for_tag(spec.hierarchical_alpha);
    }
    return "kmeans-" + mode +
           "-op-" + kmeans_assign_operator_tag(spec.assign_operator) +
           "-prec-" + kmeans_compute_precision_tag(spec.precision) +
           "-iters" + std::to_string(spec.kmeans_iters);
}

#endif // PARASELECT_CUDA_CLUSTERING_KMEANS_POLICY_CUH
