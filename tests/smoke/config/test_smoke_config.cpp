#include <cstdlib>
#include <cstring>
#include <string>

#include "common/test_context.cuh"
#include "cpu/utils/cpu_affinity.h"
#include "src/dataset/common/dataset_spec.h"
#include "src/search/kernels/fine/cpu_fine/dispatch/cpu_ivf_kernel_combos.h"

namespace {

class EnvGuard {
public:
    explicit EnvGuard(const char* name) : name_(name) {
        const char* value = std::getenv(name);
        if (value) {
            had_value_ = true;
            old_value_ = value;
        }
    }

    ~EnvGuard() {
        if (had_value_) {
            setenv(name_, old_value_.c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }

private:
    const char* name_;
    bool had_value_ = false;
    std::string old_value_;
};

void test_thread_resolver(paraselect::test::TestContext& ctx) {
    EnvGuard fine_guard("PARASELECT_CPU_FINE_THREADS");
    EnvGuard omp_guard("OMP_NUM_THREADS");

    unsetenv("PARASELECT_CPU_FINE_THREADS");
    unsetenv("OMP_NUM_THREADS");
    const int default_threads = paraselect::resolve_cpu_fine_thread_count();
    ctx.check(default_threads >= 1,
              "default thread count should be positive");
    ctx.check_eq(default_threads, paraselect::default_worker_thread_count(),
                 "default thread count should use physical cores with scheduler reserve");

    setenv("OMP_NUM_THREADS", "8", 1);
    ctx.check_eq(paraselect::resolve_cpu_fine_thread_count(), 8,
                 "OMP_NUM_THREADS should be used when explicit env is unset");

    setenv("PARASELECT_CPU_FINE_THREADS", "16", 1);
    ctx.check_eq(paraselect::resolve_cpu_fine_thread_count(), 16,
                 "PARASELECT_CPU_FINE_THREADS should override OMP_NUM_THREADS");

    setenv("PARASELECT_CPU_FINE_THREADS", "invalid", 1);
    ctx.check_eq(paraselect::resolve_cpu_fine_thread_count(), 8,
                 "invalid explicit thread env should fall back to OMP_NUM_THREADS");

    setenv("PARASELECT_CPU_FINE_THREADS", "0", 1);
    unsetenv("OMP_NUM_THREADS");
    ctx.check(paraselect::resolve_cpu_fine_thread_count() >= 1,
              "non-positive env values should fall back to physical cores");
}

void test_cpu_fine_combo_resolver(paraselect::test::TestContext& ctx) {
    using namespace paraselect::cpu_fine;

    ctx.check_eq(std::string(resolve_default_cpu_fine_combo_name("SIFT100M")),
                 std::string("u8_cluster_dynamic"),
                 "SIFT default combo should use u8 non-NUMA baseline preset");
    ctx.check_eq(std::string(resolve_default_cpu_fine_combo_name("DEEP100M")),
                 std::string("fp32_cluster_dynamic"),
                 "DEEP default combo should use fp32 AVX non-NUMA baseline preset");
    ctx.check_eq(std::string(resolve_default_cpu_fine_combo_name("TEXT100M")),
                 std::string("fp32_cluster_dynamic"),
                 "TEXT default combo should use fp32 AVX non-NUMA baseline preset");

    CpuFineComboPreset preset{};
    ctx.check(lookup_cpu_fine_combo_preset("u8_cluster_dynamic_numa", &preset),
              "u8_cluster_dynamic_numa preset should exist");
    ctx.check_eq(preset.combo_id, 8, "u8_cluster_dynamic_numa combo id should be stable");
    CpuFineComboPreset fp32_numa{};
    ctx.check(lookup_cpu_fine_combo_preset("fp32_cluster_dynamic_numa", &fp32_numa),
              "fp32_cluster_dynamic_numa preset should exist");
    ctx.check_eq(fp32_numa.combo_id, 7,
                 "fp32_cluster_dynamic_numa combo id should be stable");
    ctx.check(!lookup_cpu_fine_combo_preset("not_a_combo", &preset),
              "invalid combo preset lookup should fail");
}

void test_dataset_spec(paraselect::test::TestContext& ctx) {
    using namespace paraselect::dataset;

    const DatasetSpec text = describe_dataset("text", "100M");
    ctx.check_eq(text.n_vectors, 100000000ULL, "TEXT100M vector count should parse");
    ctx.check_eq(display_dataset_name(text), std::string("TEXT100M"), "display dataset name should be stable");
    ctx.check_eq(dataset_key_name(text), std::string("TEXT100M-200-euclidean"),
                 "cache dataset name should be stable");
    ctx.check_eq(groundtruth_dataset_name(text), std::string("text100M"),
                 "groundtruth dataset name should be stable");
    ctx.check_eq(benchmark_dataset_name(text), std::string("text100M"),
                 "benchmark dataset name should be stable");

    const DatasetSpec deep = describe_dataset("deep", "1000M");
    ctx.check_eq(display_dataset_name(deep), std::string("DEEP1B"),
                 "1000M should normalize to 1B");
    ctx.check_eq(dataset_key_name(deep), std::string("DEEP1B-96-euclidean"),
                 "configured cache dataset name should be stable");

    KMeansSpec cache{};
    ctx.check_eq(make_kmeans_algo_tag(cache), std::string("kmeans-full-op-kmeans-prec-tf32-iters20"),
                 "default KMeans algo tag should be stable");
    cache.train_strategy = KMeansTrainStrategy::Subset;
    cache.train_sample_ratio = 0.1f;
    ctx.check_eq(make_kmeans_algo_tag(cache), std::string("kmeans-subset-ratio0p1-op-kmeans-prec-tf32-iters20"),
                 "sample ratio KMeans algo tag should be stable");
}

}  // namespace

int main() {
    paraselect::test::TestContext ctx("smoke_config", paraselect::test::TestKind::Smoke);
    ctx.run_case("thread_resolver", [&] { test_thread_resolver(ctx); });
    ctx.run_case("cpu_fine_combo_resolver", [&] { test_cpu_fine_combo_resolver(ctx); });
    ctx.run_case("dataset_spec", [&] { test_dataset_spec(ctx); });
    return ctx.exit_code();
}
