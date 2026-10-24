#ifndef PARASELECT_UNIT_TEST_CONTEXT_CUH
#define PARASELECT_UNIT_TEST_CONTEXT_CUH

#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace paraselect {
namespace test {

enum class TestKind {
    Unit,
    Smoke,
    Integration,
    Benchmark,
    Experiment,
};

inline const char* test_kind_name(TestKind kind) {
    switch (kind) {
        case TestKind::Unit: return "unit";
        case TestKind::Smoke: return "smoke";
        case TestKind::Integration: return "integration";
        case TestKind::Benchmark: return "benchmark";
        case TestKind::Experiment: return "experiment";
    }
    return "unknown";
}

struct TestFailure {
    std::string case_name;
    std::string message;
};

struct TestMetric {
    std::string name;
    double value;
};

struct TestArtifact {
    std::string name;
    std::string path;
};

enum class TestOutcome {
    Pass,
    Fail,
    Skip,
};

class TestContext {
public:
    TestContext(std::string suite_name, TestKind kind)
        : suite_name_(std::move(suite_name)), kind_(kind) {}

    const std::string& suite_name() const { return suite_name_; }
    TestKind kind() const { return kind_; }
    const std::string& current_case() const { return current_case_; }
    bool passed() const { return failures_.empty(); }
    int failures() const { return static_cast<int>(failures_.size()); }
    int checks() const { return checks_; }
    int cases() const { return cases_; }
    int skipped() const { return skipped_; }

    void begin_case(const std::string& name) {
        current_case_ = name;
        ++cases_;
        std::cout << "[ RUN      ] " << name << "\n";
    }

    void end_case() {
        if (current_case_.empty()) return;
        std::cout << "[     DONE ] " << current_case_ << "\n";
        current_case_.clear();
    }

    template <typename Func>
    bool run_case(const std::string& name, Func&& func) {
        begin_case(name);
        const int failures_before = failures();
        const int skipped_before = skipped_;
        try {
            func();
        } catch (const std::exception& e) {
            fail(std::string("unexpected exception: ") + e.what());
        } catch (...) {
            fail("unexpected non-std exception");
        }
        const TestOutcome outcome = case_outcome(failures_before, skipped_before);
        print_case_outcome(outcome, name);
        current_case_.clear();
        return outcome == TestOutcome::Pass;
    }

    bool check(bool condition, const std::string& message) {
        ++checks_;
        if (!condition) {
            fail(message);
            return false;
        }
        return true;
    }

    bool require(bool condition, const std::string& message) {
        return check(condition, message);
    }

    template <typename A, typename B>
    bool check_eq(const A& actual, const B& expected, const std::string& message) {
        if (actual == expected) {
            ++checks_;
            return true;
        }
        std::ostringstream oss;
        oss << message << " (actual=" << actual << ", expected=" << expected << ")";
        ++checks_;
        fail(oss.str());
        return false;
    }

    bool check_near(double actual,
                    double expected,
                    double abs_tol,
                    const std::string& message,
                    double rel_tol = 0.0) {
        ++checks_;
        const double abs_err = std::fabs(actual - expected);
        const double scale = std::max(std::fabs(actual), std::fabs(expected));
        const bool abs_ok = abs_err <= abs_tol;
        const bool rel_ok = rel_tol > 0.0 && abs_err <= rel_tol * scale;
        if (abs_ok || rel_ok) {
            return true;
        }
        std::ostringstream oss;
        oss << message << " (actual=" << actual << ", expected=" << expected
            << ", abs_err=" << abs_err << ", abs_tol=" << abs_tol;
        if (rel_tol > 0.0) oss << ", rel_tol=" << rel_tol;
        oss << ")";
        fail(oss.str());
        return false;
    }

    bool check_pass_rate(double pass_rate, double min_pass_rate, const std::string& message) {
        add_metric(message + ".pass_rate", pass_rate);
        return check(pass_rate >= min_pass_rate, message);
    }

    void skip(const std::string& reason) {
        ++skipped_;
        notes_.push_back("skip " + (current_case_.empty() ? suite_name_ : current_case_) + ": " + reason);
    }

    void fail(const std::string& message) {
        failures_.push_back({current_case_.empty() ? suite_name_ : current_case_, message});
        std::cerr << "[  FAILED  ] "
                  << (current_case_.empty() ? suite_name_ : current_case_)
                  << ": " << message << "\n";
    }

    void add_tag(const std::string& tag) {
        tags_.push_back(tag);
    }

    void add_metric(const std::string& name, double value) {
        metrics_.push_back({name, value});
    }

    void add_param(const std::string& name, const std::string& value) {
        params_[name] = value;
    }

    void add_param(const std::string& name, int value) {
        params_[name] = std::to_string(value);
    }

    void add_param(const std::string& name, long long value) {
        params_[name] = std::to_string(value);
    }

    void add_param(const std::string& name, double value) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << value;
        params_[name] = oss.str();
    }

    void add_artifact(const std::string& name, const std::string& path) {
        artifacts_.push_back({name, path});
    }

    void add_note(const std::string& note) {
        notes_.push_back(note);
    }

    void print_summary() const {
        std::cout << "========================================\n";
        std::cout << "[ParaSelect test] suite=" << suite_name_
                  << " kind=" << test_kind_name(kind_)
                  << " cases=" << cases_
                  << " skipped=" << skipped_
                  << " checks=" << checks_
                  << " failures=" << failures_.size() << "\n";

        if (!tags_.empty()) {
            std::cout << "[ParaSelect test] tags";
            for (const auto& tag : tags_) {
                std::cout << " " << tag;
            }
            std::cout << "\n";
        }

        if (!params_.empty()) {
            std::cout << "[ParaSelect test] params\n";
            for (const auto& param : params_) {
                std::cout << "  " << param.first << "=" << param.second << "\n";
            }
        }

        if (!metrics_.empty()) {
            std::cout << "[ParaSelect test] metrics\n";
            for (const auto& metric : metrics_) {
                std::cout << "  " << metric.name << "="
                          << std::fixed << std::setprecision(6)
                          << metric.value << "\n";
            }
        }

        if (!artifacts_.empty()) {
            std::cout << "[ParaSelect test] artifacts\n";
            for (const auto& artifact : artifacts_) {
                std::cout << "  " << artifact.name << "=" << artifact.path << "\n";
            }
        }

        if (!notes_.empty()) {
            std::cout << "[ParaSelect test] notes\n";
            for (const auto& note : notes_) {
                std::cout << "  " << note << "\n";
            }
        }

        if (!failures_.empty()) {
            std::cout << "[ParaSelect test] failures\n";
            for (const auto& failure : failures_) {
                std::cout << "  " << failure.case_name << ": " << failure.message << "\n";
            }
        }

        std::cout << (passed() ? "[ParaSelect test] PASS\n" : "[ParaSelect test] FAIL\n");
        std::cout << "========================================\n";
    }

    int exit_code() const {
        print_summary();
        return passed() ? 0 : 1;
    }

private:
    TestOutcome case_outcome(int failures_before, int skipped_before) const {
        if (failures() != failures_before) return TestOutcome::Fail;
        if (skipped_ != skipped_before) return TestOutcome::Skip;
        return TestOutcome::Pass;
    }

    void print_case_outcome(TestOutcome outcome, const std::string& name) const {
        switch (outcome) {
            case TestOutcome::Pass:
                std::cout << "[       OK ] " << name << "\n";
                break;
            case TestOutcome::Fail:
                std::cout << "[  FAILED  ] " << name << "\n";
                break;
            case TestOutcome::Skip:
                std::cout << "[  SKIPPED ] " << name << "\n";
                break;
        }
    }

    std::string suite_name_;
    TestKind kind_;
    std::string current_case_;
    int cases_ = 0;
    int checks_ = 0;
    int skipped_ = 0;
    std::vector<TestFailure> failures_;
    std::vector<std::string> tags_;
    std::map<std::string, std::string> params_;
    std::vector<TestMetric> metrics_;
    std::vector<TestArtifact> artifacts_;
    std::vector<std::string> notes_;
};

}  // namespace test
}  // namespace paraselect

#define PARASELECT_CHECK(CTX, EXPR) \
    (CTX).check(static_cast<bool>(EXPR), #EXPR)

#define PARASELECT_CHECK_MSG(CTX, EXPR, MSG) \
    (CTX).check(static_cast<bool>(EXPR), (MSG))

#define PARASELECT_REQUIRE(CTX, EXPR) \
    do { \
        if (!(CTX).require(static_cast<bool>(EXPR), #EXPR)) { \
            return; \
        } \
    } while (0)

#endif  // PARASELECT_UNIT_TEST_CONTEXT_CUH
