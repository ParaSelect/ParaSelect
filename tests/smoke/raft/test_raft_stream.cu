#include "common/test_context.cuh"
#include "utils/check_cuda_error.cuh"
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <cuvs/selection/select_k.hpp>

//  stream  API
int main() {
    paraselect::test::TestContext ctx("raft_stream", paraselect::test::TestKind::Smoke);
    ctx.add_tag("rapids");
    ctx.add_tag("cuda");

    ctx.run_case("set_and_sync_cuda_stream", [&] {
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        raft::resources res;
        raft::resource::set_cuda_stream(res, rmm::cuda_stream_view(stream));
        raft::resource::sync_stream(res);

        CUDA_CHECK(cudaStreamDestroy(stream));
        ctx.check(true, "RAFT stream API accepts the configured CUDA stream");
    });

    return ctx.exit_code();
}
