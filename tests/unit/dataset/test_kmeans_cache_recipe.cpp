#include "clustering/kmeans/kmeans_cache_recipe.h"
#include <iostream>
#include <vector>
int main() {
    std::vector<KMeansSpec> cases;
    KMeansSpec s;cases.push_back(s);
    s.train_strategy=KMeansTrainStrategy::MiniBatch;cases.push_back(s);
    s.train_strategy=KMeansTrainStrategy::Subset;s.train_sample_ratio=.1f;cases.push_back(s);
    s.train_max_samples=1000;cases.push_back(s);
    s=KMeansSpec{};s.train_strategy=KMeansTrainStrategy::ProgressivePrefix;
    s.progressive_stage1_samples=1000;s.progressive_stage2_samples=2000;cases.push_back(s);
    s=KMeansSpec{};s.train_strategy=KMeansTrainStrategy::FastIVF;
    s.fast_ivf_train_samples=50000000;cases.push_back(s);
    s.cluster_only=true;cases.push_back(s);
    s=KMeansSpec{};s.train_strategy=KMeansTrainStrategy::Hierarchical;
    s.hierarchical_alpha=.1f;cases.push_back(s);
    s=KMeansSpec{};s.train_strategy=KMeansTrainStrategy::CagraLloyd;
    s.cagra_train_samples=50000000;cases.push_back(s);
    for (const auto& spec:cases)
        std::cout << nlohmann::json{{"legacy",make_kmeans_algo_tag(spec)},
            {"build",nlohmann::json::parse(make_kmeans_cache_recipe(spec))}}.dump() << '\n';
}
