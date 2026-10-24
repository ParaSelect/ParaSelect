#pragma once
#include "clustering/kmeans/kmeans_policy.cuh"
#include <nlohmann/json.hpp>

// Structured selection parameters; the published layout ID owns row identity.
inline std::string make_kmeans_cache_recipe(const KMeansSpec& spec) {
    nlohmann::json recipe{{"family","kmeans"},
        {"strategy",kmeans_train_strategy_tag(spec.train_strategy)},
        {"iters",spec.kmeans_iters}};
    if (spec.train_strategy == KMeansTrainStrategy::CagraLloyd) {
        recipe["profile"] = kmeans_cagra_profile_tag(spec.cagra_profile);
        recipe["training_limit"] = spec.cagra_train_samples;
        return recipe.dump();
    }
    recipe["operator"] = kmeans_assign_operator_tag(spec.assign_operator);
    recipe["precision"] = kmeans_compute_precision_tag(spec.precision);
    recipe["cluster_only"] = spec.cluster_only;
    // Preserve the six-decimal training-ratio contract of existing assets.
    recipe["sample_ratio"] = std::fabs(spec.train_sample_ratio - 1.0f) > 1e-6f
        ? std::round(double(spec.train_sample_ratio) * 1e6) / 1e6 : 1.0;
    recipe["sample_limit"] = std::max(0,spec.train_max_samples);
    if (spec.train_strategy == KMeansTrainStrategy::ProgressivePrefix) {
        recipe["stage1_rows"] = spec.progressive_stage1_samples;
        recipe["stage2_rows"] = spec.progressive_stage2_samples;
    }
    if (spec.train_strategy == KMeansTrainStrategy::FastIVF) {
        recipe["training_limit"] = spec.fast_ivf_train_samples;
        recipe["final_assign"] = kmeans_final_assign_strategy_tag(spec.final_assign_strategy);
    }
    if (spec.train_strategy == KMeansTrainStrategy::Hierarchical)
        recipe["alpha"] = spec.hierarchical_alpha > 0
            ? std::round(double(spec.hierarchical_alpha)*1e6)/1e6 : 0.0;
    return recipe.dump();
}
