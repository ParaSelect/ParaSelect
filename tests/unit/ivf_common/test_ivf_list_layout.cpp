#include "common/test_context.cuh"
#include "search/ivf_common/ivf_list_layout.h"

#include <climits>
#include <stdexcept>
#include <vector>

namespace {

using paraselect::search::IVFListLayoutView;
using paraselect::search::derive_ivf_list_counts;
using paraselect::search::validate_ivf_list_layout;

template <typename Fn>
void check_invalid(paraselect::test::TestContext& ctx, Fn&& fn, const char* message) {
    bool rejected = false;
    try {
        fn();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    ctx.check(rejected, message);
}

void test_valid_layout(paraselect::test::TestContext& ctx) {
    const std::vector<idx_t> offsets{0, 3, 3, 8};
    const std::vector<idx_t> ids(8);
    const IVFListLayoutView lists{offsets.data(), ids.data(), 3, 8};
    const std::vector<int> counts = derive_ivf_list_counts(lists);
    ctx.check_eq(counts.size(), size_t{3}, "one count per list");
    ctx.check_eq(counts[0], 3, "first list count");
    ctx.check_eq(counts[1], 0, "empty list count");
    ctx.check_eq(counts[2], 5, "last list count");
}

void test_invalid_layouts(paraselect::test::TestContext& ctx) {
    const std::vector<idx_t> ids(8);
    check_invalid(ctx, [&] {
        validate_ivf_list_layout({nullptr, ids.data(), 3, 8});
    }, "null offsets rejected");

    const idx_t nonzero[] = {1, 3, 8};
    check_invalid(ctx, [&] {
        validate_ivf_list_layout({nonzero, ids.data(), 2, 8});
    }, "nonzero initial offset rejected");

    const idx_t descending[] = {0, 5, 4};
    check_invalid(ctx, [&] {
        validate_ivf_list_layout({descending, ids.data(), 2, 4});
    }, "descending offsets rejected");

    const idx_t wrong_terminal[] = {0, 3, 7};
    check_invalid(ctx, [&] {
        validate_ivf_list_layout({wrong_terminal, ids.data(), 2, 8});
    }, "terminal mismatch rejected");

    const idx_t valid[] = {0, 8};
    check_invalid(ctx, [&] {
        validate_ivf_list_layout({valid, nullptr, 1, 8});
    }, "missing reordered IDs rejected");

    const idx_t oversized[] = {0, static_cast<idx_t>(INT_MAX) + 1};
    check_invalid(ctx, [&] {
        validate_ivf_list_layout({oversized, ids.data(), 1, oversized[1]});
    }, "int32 backend overflow rejected");
}

}  // namespace

int main() {
    paraselect::test::TestContext ctx("ivf_list_layout", paraselect::test::TestKind::Unit);
    ctx.run_case("valid_layout", [&] { test_valid_layout(ctx); });
    ctx.run_case("invalid_layouts", [&] { test_invalid_layouts(ctx); });
    return ctx.exit_code();
}
