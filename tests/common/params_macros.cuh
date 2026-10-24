// ============================================================================
//  (Parameter Sweep Macros) - Hydra-like style
// ============================================================================

#ifndef PARAMS_MACROS_CUH
#define PARAMS_MACROS_CUH

#include <any>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace paraselect::test_params {

inline std::unordered_map<std::string, std::any>& cli_overrides() {
    static std::unordered_map<std::string, std::any> overrides;
    return overrides;
}

template <typename T>
void set_cli_override(const char* name, const T& value) {
    cli_overrides()[name] = value;
}

template <typename T>
void clear_cli_override(const char* name) {
    (void)sizeof(T);
    cli_overrides().erase(name);
}

template <typename T>
std::vector<T> values(const char* name, std::initializer_list<T> defaults) {
    auto& overrides = cli_overrides();
    auto it = overrides.find(name);
    if (it == overrides.end()) {
        return std::vector<T>(defaults);
    }
    const T* value = std::any_cast<T>(&it->second);
    if (!value) {
        throw std::runtime_error(std::string("PARAM CLI override type mismatch: ") + name);
    }
    return {*value};
}

}  // namespace paraselect::test_params

/**
 * ()
 *
 * :
 * PARAM_1D(k, 8, 16, 32, 64, 128, 256) {
 *     bool result = test_warpsort(10000, 1024, k);
 *     check_pass("Test with k=" + std::to_string(k), result);
 * }
 *
 * @param VAR
 * @param ... (,)
 */
#define PARAM_1D(VAR, ...) \
    for (auto VAR : ::paraselect::test_params::values("" #VAR, {__VA_ARGS__}))

/**
 * ()
 *
 * :
 * int idx = 0;
 * PARAM_1D(k, 8, 16, 32, 64, 128, 256) {
 *     COUT_ENDL("[", idx++, "] Testing with k=", k);
 *     bool result = test_warpsort(10000, 1024, k);
 * }
 *
 * :
 */

/**
 *
 *
 * :
 * PARAM_2D(k, (16, 32, 64), batch, (10, 100, 1000)) {
 *     COUT_ENDL("Testing k=", k, "batch=", batch);
 *     test_warpsort(batch, 1024, k);
 * }
 *
 * : () ,
 *
 * @param VAR1
 * @param VALUES1 ()
 * @param VAR2
 * @param VALUES2 ()
 */
#define PARAM_2D(VAR1, VALUES1, VAR2, VALUES2) \
    for (auto VAR1 : ::paraselect::test_params::values("" #VAR1, {UNWRAP VALUES1})) \
        for (auto VAR2 : ::paraselect::test_params::values("" #VAR2, {UNWRAP VALUES2}))

/* : */
#define UNWRAP(...) __VA_ARGS__

/**
 *
 *
 * :
 * PARAM_3D(k, (16, 32), batch, (10, 100), len, (512, 1024)) {
 *     test_warpsort(batch, len, k);
 * }
 *
 * @param VAR1
 * @param VALUES1 ()
 * @param VAR2
 * @param VALUES2 ()
 * @param VAR3
 * @param VALUES3 ()
 */
#define PARAM_3D(VAR1, VALUES1, VAR2, VALUES2, VAR3, VALUES3) \
    for (auto VAR1 : ::paraselect::test_params::values("" #VAR1, {UNWRAP VALUES1})) \
        for (auto VAR2 : ::paraselect::test_params::values("" #VAR2, {UNWRAP VALUES2})) \
            for (auto VAR3 : ::paraselect::test_params::values("" #VAR3, {UNWRAP VALUES3}))

#define PARAM_4D(VAR1, VALUES1, VAR2, VALUES2, VAR3, VALUES3, VAR4, VALUES4) \
    for (auto VAR1 : ::paraselect::test_params::values("" #VAR1, {UNWRAP VALUES1})) \
        for (auto VAR2 : ::paraselect::test_params::values("" #VAR2, {UNWRAP VALUES2})) \
            for (auto VAR3 : ::paraselect::test_params::values("" #VAR3, {UNWRAP VALUES3})) \
                for (auto VAR4 : ::paraselect::test_params::values("" #VAR4, {UNWRAP VALUES4}))

/**
 *
 *
 * :
 * PARAM_5D(k, (16, 32), batch, (10, 100), len, (512, 1024), dim, (128, 256), nprobe, (10, 100)) {
 *     test_function(batch, len, dim, k, nprobe);
 * }
 *
 * @param VAR1
 * @param VALUES1 ()
 * @param VAR2
 * @param VALUES2 ()
 * @param VAR3
 * @param VALUES3 ()
 * @param VAR4
 * @param VALUES4 ()
 * @param VAR5
 * @param VALUES5 ()
 */
#define PARAM_5D(VAR1, VALUES1, VAR2, VALUES2, VAR3, VALUES3, VAR4, VALUES4, VAR5, VALUES5) \
    for (auto VAR1 : ::paraselect::test_params::values("" #VAR1, {UNWRAP VALUES1})) \
        for (auto VAR2 : ::paraselect::test_params::values("" #VAR2, {UNWRAP VALUES2})) \
            for (auto VAR3 : ::paraselect::test_params::values("" #VAR3, {UNWRAP VALUES3})) \
                for (auto VAR4 : ::paraselect::test_params::values("" #VAR4, {UNWRAP VALUES4})) \
                    for (auto VAR5 : ::paraselect::test_params::values("" #VAR5, {UNWRAP VALUES5}))

/**
 *
 *
 * :
 * PARAM_6D(k, (16, 32), batch, (10, 100), len, (512, 1024), dim, (128, 256), nprobe, (10, 100), nlists, (100, 1000)) {
 *     test_function(batch, len, dim, k, nprobe, nlists);
 * }
 *
 * @param VAR1
 * @param VALUES1 ()
 * @param VAR2
 * @param VALUES2 ()
 * @param VAR3
 * @param VALUES3 ()
 * @param VAR4
 * @param VALUES4 ()
 * @param VAR5
 * @param VALUES5 ()
 * @param VAR6
 * @param VALUES6 ()
 */
#define PARAM_6D(VAR1, VALUES1, VAR2, VALUES2, VAR3, VALUES3, VAR4, VALUES4, VAR5, VALUES5, VAR6, VALUES6) \
    for (auto VAR1 : ::paraselect::test_params::values("" #VAR1, {UNWRAP VALUES1})) \
        for (auto VAR2 : ::paraselect::test_params::values("" #VAR2, {UNWRAP VALUES2})) \
            for (auto VAR3 : ::paraselect::test_params::values("" #VAR3, {UNWRAP VALUES3})) \
                for (auto VAR4 : ::paraselect::test_params::values("" #VAR4, {UNWRAP VALUES4})) \
                    for (auto VAR5 : ::paraselect::test_params::values("" #VAR5, {UNWRAP VALUES5})) \
                        for (auto VAR6 : ::paraselect::test_params::values("" #VAR6, {UNWRAP VALUES6}))

// ============================================================================
//  N (,)
// ============================================================================

/**
 * :
 *  auto , int, float, bool, enum
 */
#define PARAM_IMPL_1(VAR, VALUES) \
    for (auto VAR : ::paraselect::test_params::values("" #VAR, {UNWRAP VALUES}))

#define PARAM_IMPL_2(VAR, VALUES, ...) \
    for (auto VAR : ::paraselect::test_params::values("" #VAR, {UNWRAP VALUES})) \
        PARAM_IMPL_1(__VA_ARGS__)

#define PARAM_IMPL_3(VAR, VALUES, ...) \
    for (auto VAR : ::paraselect::test_params::values("" #VAR, {UNWRAP VALUES})) \
        PARAM_IMPL_2(__VA_ARGS__)

#define PARAM_IMPL_4(VAR, VALUES, ...) \
    for (auto VAR : ::paraselect::test_params::values("" #VAR, {UNWRAP VALUES})) \
        PARAM_IMPL_3(__VA_ARGS__)

#define PARAM_IMPL_5(VAR, VALUES, ...) \
    for (auto VAR : ::paraselect::test_params::values("" #VAR, {UNWRAP VALUES})) \
        PARAM_IMPL_4(__VA_ARGS__)

#define PARAM_IMPL_6(VAR, VALUES, ...) \
    for (auto VAR : ::paraselect::test_params::values("" #VAR, {UNWRAP VALUES})) \
        PARAM_IMPL_5(__VA_ARGS__)

#define PARAM_IMPL_7(VAR, VALUES, ...) \
    for (auto VAR : ::paraselect::test_params::values("" #VAR, {UNWRAP VALUES})) \
        PARAM_IMPL_6(__VA_ARGS__)

#define PARAM_IMPL_8(VAR, VALUES, ...) \
    for (auto VAR : ::paraselect::test_params::values("" #VAR, {UNWRAP VALUES})) \
        PARAM_IMPL_7(__VA_ARGS__)

#define PARAM_IMPL_9(VAR, VALUES, ...) \
    for (auto VAR : ::paraselect::test_params::values("" #VAR, {UNWRAP VALUES})) \
        PARAM_IMPL_8(__VA_ARGS__)

#define PARAM_IMPL_10(VAR, VALUES, ...) \
    for (auto VAR : ::paraselect::test_params::values("" #VAR, {UNWRAP VALUES})) \
        PARAM_IMPL_9(__VA_ARGS__)

/**
 * :(2:VAR, VALUES)
 */
#define PARAM_COUNT_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, N, ...) N
#define PARAM_COUNT(...) PARAM_COUNT_IMPL(__VA_ARGS__, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1)

/**
 * :
 */
#define PARAM_SELECT(N) PARAM_IMPL_##N
#define PARAM_SELECT_CALL(N, ...) PARAM_SELECT(N)(__VA_ARGS__)

/**
 *  N (:int, float, bool, enum )
 *
 * :
 * // int
 * PARAM(dim, (128, 256), n, (10000, 100000), k, (10, 100)) {
 *     test_function(n, dim, k);
 * }
 *
 * // bool  -  true/false
 * PARAM(use_fp16, (true, false), batch, (1, 10, 100)) {
 *     test_function(batch, use_fp16);
 * }
 *
 * // enum  -
 * PARAM(model_type, (ModelType::BALANCED, ModelType::NAIVE), ...) {
 *     // model_type  ModelType
 * }
 *
 * :
 * - :(, )
 * - :(value1, value2, ...)
 * -  1-10
 * - (int, float, bool, enum )
 *
 * @param ...  (, )
 */
#define PARAM(...) PARAM_SELECT_CALL(PARAM_COUNT(__VA_ARGS__), __VA_ARGS__)

// ============================================================================
//
// ============================================================================

/**
 * PARAM_FOR - PARAM ,
 */
#define PARAM_FOR(...) PARAM(__VA_ARGS__)

/**
 * PARAM_GRID - PARAM ,
 */
#define PARAM_GRID(...) PARAM(__VA_ARGS__)

// ============================================================================
// C++ ()
// ============================================================================

#include <vector>
#include <initializer_list>

namespace param_sweep {

/**
 *
 */
template<typename T>
struct Param {
    const char* name;
    std::vector<T> values;

    Param(const char* n, std::initializer_list<T> vals)
        : name(n), values(vals) {}
};

/**
 * :
 */
template<typename Func, typename... Args>
void param_for_impl(Func&& func, Args&&... args) {
    func(std::forward<Args>(args)...);
}

/**
 *
 */
template<typename Func, typename T, typename... Rest>
void param_for_impl(Func&& func, const Param<T>& param, const Rest&... rest) {
    for (const auto& value : param.values) {
        param_for_impl(func, rest..., value);
    }
}

/**
 * :
 * param_for(
 *     [&](int dim, int n, int k) {
 *         test_function(n, dim, k);
 *     },
 *     Param<int>("dim", {128, 256}),
 *     Param<int>("n", {100000, 1000000}),
 *     Param<int>("k", {10, 100})
 * );
 */
template<typename Func, typename... Params>
void param_for(Func&& func, const Params&... params) {
    param_for_impl(std::forward<Func>(func), params...);
}

} // namespace param_sweep

#endif
