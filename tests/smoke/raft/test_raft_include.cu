#include "common/test_context.cuh"
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <cuvs/selection/select_k.hpp>

int main() {
    paraselect::test::TestContext ctx("raft_include", paraselect::test::TestKind::Smoke);
    ctx.add_tag("rapids");
    ctx.add_tag("compile");
    ctx.run_case("headers", [&] {
        ctx.check(true, "RAFT/cuVS headers are available at compile time");
    });
    return ctx.exit_code();
}
