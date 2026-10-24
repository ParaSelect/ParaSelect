#include <string>

#include "common/test_context.cuh"
#include "src/clustering/kmeans/kmeans_policy.cuh"

namespace {

void test_kmeans_algo_tags(paraselect::test::TestContext& ctx) {
    KMeansSpec spec{};
    ctx.check_eq(make_kmeans_algo_tag(spec), std::string("kmeans-full-op-kmeans-prec-tf32-iters20"),
                 "default full-data KMeans tag");

    spec.train_strategy = KMeansTrainStrategy::MiniBatch;
    ctx.check_eq(make_kmeans_algo_tag(spec), std::string("kmeans-minibatch-op-kmeans-prec-tf32-iters20"),
                 "minibatch KMeans tag");

    spec = KMeansSpec{};
    spec.train_strategy = KMeansTrainStrategy::Subset;
    spec.train_max_samples = 1000000;
    ctx.check_eq(make_kmeans_algo_tag(spec), std::string("kmeans-subset-sample1000000-op-kmeans-prec-tf32-iters20"),
                 "sample-limited KMeans tag");

    spec = KMeansSpec{};
    spec.train_strategy = KMeansTrainStrategy::Subset;
    spec.train_sample_ratio = 0.1f;
    ctx.check_eq(make_kmeans_algo_tag(spec), std::string("kmeans-subset-ratio0p1-op-kmeans-prec-tf32-iters20"),
                 "ratio KMeans tag");

    spec = KMeansSpec{};
    spec.train_strategy = KMeansTrainStrategy::MiniBatch;
    spec.train_max_samples = 1000000;
    spec.kmeans_iters = 30;
    ctx.check_eq(make_kmeans_algo_tag(spec), std::string("kmeans-minibatch-sample1000000-op-kmeans-prec-tf32-iters30"),
                 "minibatch sample-limited KMeans tag");

    spec = KMeansSpec{};
    spec.assign_operator = KMeansAssignOperator::FlashKmeans;
    ctx.check_eq(make_kmeans_algo_tag(spec), std::string("kmeans-full-op-flash-prec-tf32-iters20"),
                 "flash KMeans operator tag");

    spec = KMeansSpec{};
    spec.precision = KMeansComputePrecision::Fp16;
    ctx.check_eq(make_kmeans_algo_tag(spec), std::string("kmeans-full-op-kmeans-prec-fp16-iters20"),
                 "fp16 KMeans precision tag");

    spec = KMeansSpec{};
    spec.train_strategy = KMeansTrainStrategy::Hierarchical;
    spec.hierarchical_alpha = 0.1f;
    ctx.check_eq(make_kmeans_algo_tag(spec), std::string("kmeans-hierarchical-alpha0p1-op-kmeans-prec-tf32-iters20"),
                 "hierarchical KMeans tag");

    spec = KMeansSpec{};
    spec.train_strategy = KMeansTrainStrategy::ProgressivePrefix;
    spec.progressive_stage1_samples = 50;
    spec.progressive_stage2_samples = 80;
    ctx.check_eq(make_kmeans_algo_tag(spec),
                 std::string("kmeans-progressive-prefix-stages50_80-op-kmeans-prec-tf32-iters20"),
                 "progressive-prefix KMeans tag");

    spec = KMeansSpec{};
    spec.train_strategy = KMeansTrainStrategy::FastIVF;
    spec.fast_ivf_train_samples = 50000000;
    spec.final_assign_strategy = KMeansFinalAssignStrategy::Cagra;
    ctx.check_eq(make_kmeans_algo_tag(spec),
                 std::string("kmeans-fast-ivf-train50000000-final-cagra-op-kmeans-prec-tf32-iters20"),
                 "fast-ivf KMeans tag");

    spec = KMeansSpec{};
    spec.train_strategy = KMeansTrainStrategy::CagraLloyd;
    spec.cagra_train_samples = 100000000;
    spec.kmeans_iters = 5;
    ctx.check_eq(make_kmeans_algo_tag(spec),
                 std::string("kmeans-cagra-fast-v1-train100000000-iters5"),
                 "CAGRA Lloyd profile tag");
}

void test_legacy_constructor(paraselect::test::TestContext& ctx) {
    KMeansSpec full = make_kmeans_spec(false, 20, 1.0f, 0);
    ctx.check(full.train_strategy == KMeansTrainStrategy::FullLloyd,
              "full legacy args should infer FullLloyd");

    KMeansSpec minibatch = make_kmeans_spec(true, 20, 1.0f, 0);
    ctx.check(minibatch.train_strategy == KMeansTrainStrategy::MiniBatch,
              "minibatch legacy args should infer MiniBatch");

    KMeansSpec subset = make_kmeans_spec(false, 20, 0.1f, 0);
    ctx.check(subset.train_strategy == KMeansTrainStrategy::Subset,
              "sample ratio should infer Subset");

    KMeansSpec sample_limited = make_kmeans_spec(false, 20, 1.0f, 1000000);
    ctx.check(sample_limited.train_strategy == KMeansTrainStrategy::Subset,
              "sample limit should infer Subset");

    KMeansSpec progressive = make_kmeans_spec(
        false, 20, 1.0f, 0,
        KMeansAssignOperator::Kmeans, KMeansComputePrecision::Tf32,
        KMeansTrainStrategy::ProgressivePrefix, 50, 80);
    ctx.check(progressive.train_strategy == KMeansTrainStrategy::ProgressivePrefix,
              "explicit strategy should select ProgressivePrefix");
    ctx.check_eq(progressive.progressive_stage1_samples, 50,
                 "progressive stage1 should be stored");
    ctx.check_eq(progressive.progressive_stage2_samples, 80,
                 "progressive stage2 should be stored");

    KMeansSpec fast_ivf = make_kmeans_spec(
        false, 5, 1.0f, 0,
        KMeansAssignOperator::Kmeans, KMeansComputePrecision::Tf32,
        KMeansTrainStrategy::FastIVF, 0, 0, 50000000,
        KMeansFinalAssignStrategy::Cagra);
    ctx.check(fast_ivf.train_strategy == KMeansTrainStrategy::FastIVF,
              "explicit strategy should select FastIVF");
    ctx.check_eq(fast_ivf.fast_ivf_train_samples, 50000000,
                 "FastIVF train sample count should be stored");
    ctx.check(fast_ivf.final_assign_strategy == KMeansFinalAssignStrategy::Cagra,
              "FastIVF final assign strategy should be stored");
}

}  // namespace

int main() {
    paraselect::test::TestContext ctx("kmeans_spec", paraselect::test::TestKind::Unit);
    ctx.run_case("kmeans_algo_tags", [&] { test_kmeans_algo_tags(ctx); });
    ctx.run_case("legacy_constructor", [&] { test_legacy_constructor(ctx); });
    return ctx.exit_code();
}
