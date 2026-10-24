#include "clustering/flash_kmeans/flash_kmeans_assign_v8.cuh"
#include "utils/check_cuda_error.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

namespace paraselect {
namespace clustering {
namespace kmeans {

namespace {

bool allow_stage1_debug_override() {
    const char* value = std::getenv("PARASELECT_FLASH_KMEANS_ALLOW_UNVALIDATED_STAGE1");
    return value &&
           std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0;
}

template <int TileM, int TileN, int TileK, int Stages>
using CutlassAssignKernel = typename cutlass::gemm::kernel::DefaultGemm<
    float,
    cutlass::layout::RowMajor,
    128 / cutlass::sizeof_bits<float>::value,
    float,
    cutlass::layout::ColumnMajor,
    128 / cutlass::sizeof_bits<float>::value,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TileM, TileN, TileK>,
    cutlass::gemm::GemmShape<64, 64, TileK>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
        float,
        128 / cutlass::sizeof_bits<float>::value,
        float,
        float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
    Stages,
    false,
    cutlass::arch::OpMultiplyAdd,
    cutlass::gemm::SharedMemoryClearOption::kNone>::GemmKernel;

template <
    typename ThreadblockShape,
    int ThreadCount,
    typename OutputTileIterator,
    typename ElementAccumulator,
    bool UseSharedReduce>
class FlashAssignMinVisitor {
public:
    static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;
    static int const kIterations = OutputTileIterator::kIterations;
    using ElementOutput = float;
    using AccumulatorFragment =
        cutlass::Array<ElementAccumulator, kElementsPerAccess>;

    struct Arguments {
        const float* xnorm2 = nullptr;
        const float* cnorm2 = nullptr;
        unsigned long long* partial = nullptr;
        int n_tiles = 0;

        Arguments() = default;
        Arguments(
            const float* xnorm2_,
            const float* cnorm2_,
            unsigned long long* partial_,
            int n_tiles_)
            : xnorm2(xnorm2_),
              cnorm2(cnorm2_),
              partial(partial_),
              n_tiles(n_tiles_) {}
    };

    struct Params {
        const float* xnorm2 = nullptr;
        const float* cnorm2 = nullptr;
        unsigned long long* partial = nullptr;
        int n_tiles = 0;

        CUTLASS_HOST_DEVICE
        Params() = default;

        CUTLASS_HOST_DEVICE
        explicit Params(Arguments const& args)
            : xnorm2(args.xnorm2),
              cnorm2(args.cnorm2),
              partial(args.partial),
              n_tiles(args.n_tiles) {}
    };

    struct SharedStorage {
        unsigned long long row_min[ThreadblockShape::kM];
    };

private:
    Params const& params_;
    SharedStorage* shared_ = nullptr;
    cutlass::MatrixCoord extent_;
    cutlass::MatrixCoord thread_start_;
    int thread_idx_ = 0;
    int row_block_start_ = 0;
    int tile_n_ = 0;
    float row_best_ = 3.4028234663852886e38f;
    int row_idx_ = -1;
    int row_global_ = -1;

public:
    CUTLASS_DEVICE
    FlashAssignMinVisitor(
        Params const& params,
        SharedStorage& shared_storage,
        cutlass::MatrixCoord const& problem_size,
        int thread_idx,
        int,
        int,
        cutlass::MatrixCoord const& threadblock_offset = cutlass::MatrixCoord(0, 0))
        : params_(params),
          shared_(&shared_storage),
          extent_(problem_size),
          thread_start_(
              OutputTileIterator::ThreadMap::initial_offset(thread_idx) +
              threadblock_offset),
          thread_idx_(thread_idx),
          row_block_start_(threadblock_offset.row()),
          tile_n_(threadblock_offset.column() / ThreadblockShape::kN) {}

    CUTLASS_DEVICE
    void set_k_partition(int, int) {}

    CUTLASS_DEVICE
    void set_batch_index(int) {}

    CUTLASS_DEVICE
    void begin_epilogue() {
        if constexpr (UseSharedReduce) {
            constexpr unsigned long long kInfPacked = 0x7f7fffffull << 32;
            for (int local_row = thread_idx_; local_row < ThreadblockShape::kM;
                 local_row += ThreadCount) {
                shared_->row_min[local_row] = kInfPacked;
            }
            __syncthreads();
        }
    }

    CUTLASS_DEVICE
    void begin_step(int) {}

    CUTLASS_DEVICE
    void begin_row(int) {
        row_best_ = 3.4028234663852886e38f;
        row_idx_ = -1;
        row_global_ = -1;
    }

    CUTLASS_DEVICE
    void visit(
        int iter_idx,
        int,
        int,
        int frag_idx,
        AccumulatorFragment const& accum)
    {
        const cutlass::MatrixCoord thread_offset =
            thread_start_ +
            cutlass::MatrixCoord(iter_idx * OutputTileIterator::Shape::kRow, 0) +
            OutputTileIterator::ThreadMap::iteration_offset(frag_idx);
        const int row = thread_offset.row();
        const int col_base = thread_offset.column();
        if (row >= extent_.row()) return;

        const float xn = params_.xnorm2[row];
        row_global_ = row;

        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < kElementsPerAccess; ++e) {
            const int col = col_base + e;
            if (col >= extent_.column()) break;
            float dist = xn + params_.cnorm2[col] - 2.0f * (float)accum[e];
            dist = fmaxf(dist, 0.0f);
            if (dist < row_best_) {
                row_best_ = dist;
                row_idx_ = col;
            }
        }
    }

    CUTLASS_DEVICE
    void end_row(int) {
        if (row_idx_ >= 0 && tile_n_ < params_.n_tiles) {
            const unsigned long long packed =
                ((unsigned long long)__float_as_uint(row_best_) << 32) |
                (unsigned int)row_idx_;
            if constexpr (UseSharedReduce) {
                const int local_row = row_global_ - row_block_start_;
                if (local_row >= 0 && local_row < ThreadblockShape::kM) {
                    atomicMin(shared_->row_min + local_row, packed);
                }
            } else {
                atomicMin(
                    params_.partial + (size_t)row_global_ * params_.n_tiles + tile_n_,
                    packed);
            }
        }
    }

    CUTLASS_DEVICE
    void end_step(int) {}

    CUTLASS_DEVICE
    void end_epilogue() {
        if constexpr (UseSharedReduce) {
            __syncthreads();
            if (tile_n_ >= params_.n_tiles) return;
            for (int local_row = thread_idx_; local_row < ThreadblockShape::kM;
                 local_row += ThreadCount) {
                const int row = row_block_start_ + local_row;
                if (row < extent_.row()) {
                    params_.partial[(size_t)row * params_.n_tiles + tile_n_] =
                        shared_->row_min[local_row];
                }
            }
        }
    }
};

template <
    typename Mma_,
    typename Epilogue_,
    typename ThreadblockSwizzle_>
struct FlashAssignGemmWithMinVisitor {
    using Mma = Mma_;
    using Epilogue = Epilogue_;
    using EpilogueVisitor = typename Epilogue::Visitor;
    using ThreadblockSwizzle = ThreadblockSwizzle_;

    using ElementA = typename Mma::IteratorA::Element;
    using LayoutA = typename Mma::IteratorA::Layout;
    using TensorRefA = cutlass::TensorRef<ElementA, LayoutA>;
    using ElementB = typename Mma::IteratorB::Element;
    using LayoutB = typename Mma::IteratorB::Layout;
    using TensorRefB = cutlass::TensorRef<ElementB, LayoutB>;

    using ThreadblockShape = typename Mma::Shape;
    using WarpCount = typename Mma::WarpCount;
    static int const kThreadCount = 32 * WarpCount::kCount;

    struct Arguments {
        cutlass::gemm::GemmCoord problem_size;
        TensorRefA ref_A;
        TensorRefB ref_B;
        typename EpilogueVisitor::Arguments epilogue_visitor;

        Arguments() = default;
        Arguments(
            cutlass::gemm::GemmCoord problem_size_,
            TensorRefA ref_A_,
            TensorRefB ref_B_,
            typename EpilogueVisitor::Arguments epilogue_visitor_)
            : problem_size(problem_size_),
              ref_A(ref_A_),
              ref_B(ref_B_),
              epilogue_visitor(epilogue_visitor_) {}
    };

    struct Params {
        cutlass::gemm::GemmCoord problem_size;
        cutlass::gemm::GemmCoord grid_tiled_shape;
        int swizzle_log_tile = 0;
        typename Mma::IteratorA::Params params_A;
        typename Mma::IteratorB::Params params_B;
        void* ptr_A = nullptr;
        void* ptr_B = nullptr;
        typename EpilogueVisitor::Params epilogue_visitor;

        CUTLASS_HOST_DEVICE
        Params() = default;

        Params(Arguments const& args)
            : problem_size(args.problem_size),
              params_A(args.ref_A.layout()),
              params_B(args.ref_B.layout()),
              ptr_A(args.ref_A.data()),
              ptr_B(args.ref_B.data()),
              epilogue_visitor(args.epilogue_visitor)
        {
            ThreadblockSwizzle swizzle;
            grid_tiled_shape = swizzle.get_tiled_shape(
                args.problem_size,
                {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
                1);
            swizzle_log_tile = swizzle.get_log_tile(grid_tiled_shape);
        }
    };

    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        struct {
            typename Epilogue::SharedStorage epilogue;
            typename EpilogueVisitor::SharedStorage visitor;
        } epilogue;
    };

    CUTLASS_DEVICE
    void operator()(Params const& params, SharedStorage& shared_storage) {
        ThreadblockSwizzle swizzle;
        cutlass::gemm::GemmCoord tile_offset =
            swizzle.get_tile_offset(params.swizzle_log_tile);

        if (params.grid_tiled_shape.m() <= tile_offset.m() ||
            params.grid_tiled_shape.n() <= tile_offset.n()) {
            return;
        }

        const int offset_k = 0;
        const int problem_size_k = params.problem_size.k();
        ElementA* ptr_A = static_cast<ElementA*>(params.ptr_A);
        ElementB* ptr_B = static_cast<ElementB*>(params.ptr_B);

        cutlass::MatrixCoord tb_offset_A{
            tile_offset.m() * Mma::Shape::kM,
            offset_k,
        };
        cutlass::MatrixCoord tb_offset_B{
            offset_k,
            tile_offset.n() * Mma::Shape::kN,
        };

        const int thread_idx = threadIdx.x;
        typename Mma::IteratorA iterator_A(
            params.params_A,
            ptr_A,
            {params.problem_size.m(), problem_size_k},
            thread_idx,
            tb_offset_A);
        typename Mma::IteratorB iterator_B(
            params.params_B,
            ptr_B,
            {problem_size_k, params.problem_size.n()},
            thread_idx,
            tb_offset_B);

        const int warp_idx = cutlass::canonical_warp_idx_sync();
        const int lane_idx = threadIdx.x % 32;

        Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
        typename Mma::FragmentC accumulators;
        accumulators.clear();

        const int gemm_k_iterations =
            (problem_size_k - offset_k + Mma::Shape::kK - 1) / Mma::Shape::kK;
        mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

        tile_offset = swizzle.get_tile_offset(params.swizzle_log_tile);
        cutlass::MatrixCoord threadblock_offset(
            tile_offset.m() * Mma::Shape::kM,
            tile_offset.n() * Mma::Shape::kN);

        EpilogueVisitor visitor(
            params.epilogue_visitor,
            shared_storage.epilogue.visitor,
            params.problem_size.mn(),
            thread_idx,
            warp_idx,
            lane_idx,
            threadblock_offset);

        Epilogue epilogue(
            shared_storage.epilogue.epilogue,
            thread_idx,
            warp_idx,
            lane_idx);
        epilogue(visitor, accumulators);
    }
};

using CutlassAssignKernel64x128x32s1 = CutlassAssignKernel<64, 128, 32, 1>;
using CutlassAssignKernel64x128x32s2 = CutlassAssignKernel<64, 128, 32, 2>;
using CutlassAssignKernel64x256x16s1 = CutlassAssignKernel<64, 256, 16, 1>;
using CutlassAssignKernel64x256x16s2 = CutlassAssignKernel<64, 256, 16, 2>;
using CutlassAssignKernel64x256x32s1 = CutlassAssignKernel<64, 256, 32, 1>;
using CutlassAssignKernel64x256x32s2 = CutlassAssignKernel<64, 256, 32, 2>;
using CutlassAssignKernel128x128x16s4 = CutlassAssignKernel<128, 128, 16, 4>;
using CutlassAssignKernel128x256x16s4 = CutlassAssignKernel<128, 256, 16, 4>;
using CutlassAssignKernel128x128x32s1 = CutlassAssignKernel<128, 128, 32, 1>;
using CutlassAssignKernel128x256x32s1 = CutlassAssignKernel<128, 256, 32, 1>;
using CutlassAssignKernel128x128x32s2 = CutlassAssignKernel<128, 128, 32, 2>;
using CutlassAssignKernel128x256x32s2 = CutlassAssignKernel<128, 256, 32, 2>;

using FlashAssignMinVisitorV8Global128 = FlashAssignMinVisitor<
    typename CutlassAssignKernel128x128x16s4::Mma::Shape,
    CutlassAssignKernel128x128x16s4::kThreadCount,
    typename CutlassAssignKernel128x128x16s4::Epilogue::OutputTileIterator,
    float,
    false>;

using FlashAssignMinEpilogueV8Global128 =
    typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
        FlashAssignMinVisitorV8Global128,
        typename CutlassAssignKernel128x128x16s4::Epilogue>::Epilogue;

using FlashAssignMinGemmV8Global128 = FlashAssignGemmWithMinVisitor<
    typename CutlassAssignKernel128x128x16s4::Mma,
    FlashAssignMinEpilogueV8Global128,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>;

using FlashAssignMinVisitorV8Shared128 = FlashAssignMinVisitor<
    typename CutlassAssignKernel128x128x16s4::Mma::Shape,
    CutlassAssignKernel128x128x16s4::kThreadCount,
    typename CutlassAssignKernel128x128x16s4::Epilogue::OutputTileIterator,
    float,
    true>;

using FlashAssignMinEpilogueV8Shared128 =
    typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
        FlashAssignMinVisitorV8Shared128,
        typename CutlassAssignKernel128x128x16s4::Epilogue>::Epilogue;

using FlashAssignMinGemmV8Shared128 = FlashAssignGemmWithMinVisitor<
    typename CutlassAssignKernel128x128x16s4::Mma,
    FlashAssignMinEpilogueV8Shared128,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>;

using FlashAssignMinVisitorV8Global256 = FlashAssignMinVisitor<
    typename CutlassAssignKernel128x256x16s4::Mma::Shape,
    CutlassAssignKernel128x256x16s4::kThreadCount,
    typename CutlassAssignKernel128x256x16s4::Epilogue::OutputTileIterator,
    float,
    false>;

using FlashAssignMinEpilogueV8Global256 =
    typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
        FlashAssignMinVisitorV8Global256,
        typename CutlassAssignKernel128x256x16s4::Epilogue>::Epilogue;

using FlashAssignMinGemmV8Global256 = FlashAssignGemmWithMinVisitor<
    typename CutlassAssignKernel128x256x16s4::Mma,
    FlashAssignMinEpilogueV8Global256,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>;

using FlashAssignMinVisitorV8Shared256 = FlashAssignMinVisitor<
    typename CutlassAssignKernel128x256x16s4::Mma::Shape,
    CutlassAssignKernel128x256x16s4::kThreadCount,
    typename CutlassAssignKernel128x256x16s4::Epilogue::OutputTileIterator,
    float,
    true>;

using FlashAssignMinEpilogueV8Shared256 =
    typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
        FlashAssignMinVisitorV8Shared256,
        typename CutlassAssignKernel128x256x16s4::Epilogue>::Epilogue;

using FlashAssignMinGemmV8Shared256 = FlashAssignGemmWithMinVisitor<
    typename CutlassAssignKernel128x256x16s4::Mma,
    FlashAssignMinEpilogueV8Shared256,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>;

#define DEFINE_V8_ALIAS(SUFFIX, KERNEL) \
using FlashAssignMinVisitorV8Global##SUFFIX = FlashAssignMinVisitor< \
    typename KERNEL::Mma::Shape, \
    KERNEL::kThreadCount, \
    typename KERNEL::Epilogue::OutputTileIterator, \
    float, \
    false>; \
using FlashAssignMinEpilogueV8Global##SUFFIX = \
    typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue< \
        FlashAssignMinVisitorV8Global##SUFFIX, \
        typename KERNEL::Epilogue>::Epilogue; \
using FlashAssignMinGemmV8Global##SUFFIX = FlashAssignGemmWithMinVisitor< \
    typename KERNEL::Mma, \
    FlashAssignMinEpilogueV8Global##SUFFIX, \
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>; \
using FlashAssignMinVisitorV8Shared##SUFFIX = FlashAssignMinVisitor< \
    typename KERNEL::Mma::Shape, \
    KERNEL::kThreadCount, \
    typename KERNEL::Epilogue::OutputTileIterator, \
    float, \
    true>; \
using FlashAssignMinEpilogueV8Shared##SUFFIX = \
    typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue< \
        FlashAssignMinVisitorV8Shared##SUFFIX, \
        typename KERNEL::Epilogue>::Epilogue; \
using FlashAssignMinGemmV8Shared##SUFFIX = FlashAssignGemmWithMinVisitor< \
    typename KERNEL::Mma, \
    FlashAssignMinEpilogueV8Shared##SUFFIX, \
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>;

DEFINE_V8_ALIAS(64x128x32s1, CutlassAssignKernel64x128x32s1)
DEFINE_V8_ALIAS(64x128x32s2, CutlassAssignKernel64x128x32s2)
DEFINE_V8_ALIAS(64x256x16s1, CutlassAssignKernel64x256x16s1)
DEFINE_V8_ALIAS(64x256x16s2, CutlassAssignKernel64x256x16s2)
DEFINE_V8_ALIAS(64x256x32s1, CutlassAssignKernel64x256x32s1)
DEFINE_V8_ALIAS(64x256x32s2, CutlassAssignKernel64x256x32s2)
DEFINE_V8_ALIAS(128x128x32s1, CutlassAssignKernel128x128x32s1)
DEFINE_V8_ALIAS(128x128x32s2, CutlassAssignKernel128x128x32s2)
DEFINE_V8_ALIAS(128x256x32s1, CutlassAssignKernel128x256x32s1)
DEFINE_V8_ALIAS(128x256x32s2, CutlassAssignKernel128x256x32s2)

#undef DEFINE_V8_ALIAS

template <typename Gemm, typename Visitor>
void launch_flash_assign_min_gemm_impl(
    const float* d_data,
    const float* d_centroids,
    const float* d_xnorm2,
    const float* d_cnorm2,
    int n,
    int k,
    int dim,
    int n_tiles,
    unsigned long long* d_partial_packed,
    cudaStream_t stream,
    cudaDeviceProp const& prop,
    int max_optin,
    const char* name)
{
    typename Gemm::Arguments args(
        cutlass::gemm::GemmCoord(n, k, dim),
        {const_cast<float*>(d_data), dim},
        {const_cast<float*>(d_centroids), dim},
        typename Visitor::Arguments(
            d_xnorm2,
            d_cnorm2,
            d_partial_packed,
            n_tiles));

    typename Gemm::Params params(args);
    dim3 grid =
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>()
            .get_grid_shape(params.grid_tiled_shape);
    dim3 block(Gemm::kThreadCount, 1, 1);
    const int smem_size = (int)sizeof(typename Gemm::SharedStorage);
    if (smem_size > max_optin) {
        std::cerr << "[flash_assign_v8] shared memory exceeds device limit"
                  << " kernel=" << name
                  << " smem=" << smem_size
                  << " max_optin=" << max_optin << std::endl;
        CUDA_CHECK(cudaErrorInvalidConfiguration);
    }
    if (smem_size > (int)prop.sharedMemPerBlock) {
        CUDA_CHECK(cudaFuncSetAttribute(
            cutlass::Kernel<Gemm>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
    }
    cutlass::Kernel<Gemm><<<grid, block, smem_size, stream>>>(params);
}

}  // namespace

void launch_flash_assign_min_gemm_v8(
    const float* d_data,
    const float* d_centroids,
    const float* d_xnorm2,
    const float* d_cnorm2,
    int n,
    int k,
    int dim,
    int n_tiles,
    int point_tile,
    int centroid_tile,
    int cutlass_k_tile,
    int cutlass_stages,
    int shared_reduce_min_tiles,
    unsigned long long* d_partial_packed,
    cudaStream_t stream)
{
    const bool use_shared_reduce = n_tiles >= shared_reduce_min_tiles;
    const int tile_m = (point_tile == 64) ? 64 : 128;
    const int tile_n = (centroid_tile == 128) ? 128 : 256;
    const bool use_k16_s1 = (cutlass_k_tile == 16 && cutlass_stages == 1);
    const bool use_k16_s2 = (cutlass_k_tile == 16 && cutlass_stages == 2);
    const bool use_k32_s1 = (cutlass_k_tile == 32 && cutlass_stages == 1);
    const bool use_k32_s2 = (cutlass_k_tile == 32 && cutlass_stages == 2);
    if (cutlass_stages == 1 && !allow_stage1_debug_override()) {
        std::cerr << "[flash_assign_v8] unsupported parameter combination"
                  << " point_tile=" << point_tile
                  << " centroid_tile=" << centroid_tile
                  << " cutlass_k_tile=" << cutlass_k_tile
                  << " cutlass_stages=" << cutlass_stages
                  << " reason=stage1_failed_top1_validation"
                  << std::endl;
        CUDA_CHECK(cudaErrorInvalidConfiguration);
    }
    if (tile_n == 128 && use_k32_s1) {
        std::cerr << "[flash_assign_v8] unsupported parameter combination"
                  << " point_tile=" << point_tile
                  << " centroid_tile=" << centroid_tile
                  << " cutlass_k_tile=" << cutlass_k_tile
                  << " cutlass_stages=" << cutlass_stages
                  << " reason=BK128_K32_S1_failed_validation"
                  << std::endl;
        CUDA_CHECK(cudaErrorInvalidConfiguration);
    }
    cudaDeviceProp prop{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    const int max_optin = prop.sharedMemPerBlockOptin > 0
        ? prop.sharedMemPerBlockOptin
        : prop.sharedMemPerBlock;

    if (tile_m == 64 && tile_n == 128) {
        if (use_shared_reduce && use_k32_s1) {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Shared64x128x32s1,
                FlashAssignMinVisitorV8Shared64x128x32s1>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_shared_64x128x32s1");
        } else if (!use_shared_reduce && use_k32_s1) {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Global64x128x32s1,
                FlashAssignMinVisitorV8Global64x128x32s1>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_global_64x128x32s1");
        } else if (use_shared_reduce) {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Shared64x128x32s2,
                FlashAssignMinVisitorV8Shared64x128x32s2>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_shared_64x128x32s2");
        } else {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Global64x128x32s2,
                FlashAssignMinVisitorV8Global64x128x32s2>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_global_64x128x32s2");
        }
    } else if (tile_m == 64 && tile_n == 256) {
        if (use_shared_reduce && use_k16_s1) {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Shared64x256x16s1,
                FlashAssignMinVisitorV8Shared64x256x16s1>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_shared_64x256x16s1");
        } else if (!use_shared_reduce && use_k16_s1) {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Global64x256x16s1,
                FlashAssignMinVisitorV8Global64x256x16s1>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_global_64x256x16s1");
        } else if (use_shared_reduce && use_k16_s2) {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Shared64x256x16s2,
                FlashAssignMinVisitorV8Shared64x256x16s2>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_shared_64x256x16s2");
        } else if (!use_shared_reduce && use_k16_s2) {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Global64x256x16s2,
                FlashAssignMinVisitorV8Global64x256x16s2>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_global_64x256x16s2");
        } else if (use_shared_reduce && use_k32_s1) {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Shared64x256x32s1,
                FlashAssignMinVisitorV8Shared64x256x32s1>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_shared_64x256x32s1");
        } else if (!use_shared_reduce && use_k32_s1) {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Global64x256x32s1,
                FlashAssignMinVisitorV8Global64x256x32s1>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_global_64x256x32s1");
        } else if (use_shared_reduce) {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Shared64x256x32s2,
                FlashAssignMinVisitorV8Shared64x256x32s2>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_shared_64x256x32s2");
        } else {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Global64x256x32s2,
                FlashAssignMinVisitorV8Global64x256x32s2>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_global_64x256x32s2");
        }
    } else if (tile_n == 128) {
        if (use_shared_reduce && use_k32_s1) {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Shared128x128x32s1,
                FlashAssignMinVisitorV8Shared128x128x32s1>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_shared_128x128x32s1");
        } else if (!use_shared_reduce && use_k32_s1) {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Global128x128x32s1,
                FlashAssignMinVisitorV8Global128x128x32s1>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_global_128x128x32s1");
        } else if (use_shared_reduce && use_k32_s2) {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Shared128x128x32s2,
                FlashAssignMinVisitorV8Shared128x128x32s2>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_shared_128x128x32s2");
        } else if (!use_shared_reduce && use_k32_s2) {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Global128x128x32s2,
                FlashAssignMinVisitorV8Global128x128x32s2>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_global_128x128x32s2");
        } else if (use_shared_reduce) {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Shared128,
                FlashAssignMinVisitorV8Shared128>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_shared_128");
        } else {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Global128,
                FlashAssignMinVisitorV8Global128>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_global_128");
        }
    } else {
        if (use_shared_reduce && use_k32_s1) {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Shared128x256x32s1,
                FlashAssignMinVisitorV8Shared128x256x32s1>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_shared_128x256x32s1");
        } else if (!use_shared_reduce && use_k32_s1) {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Global128x256x32s1,
                FlashAssignMinVisitorV8Global128x256x32s1>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_global_128x256x32s1");
        } else if (use_shared_reduce && use_k32_s2) {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Shared128x256x32s2,
                FlashAssignMinVisitorV8Shared128x256x32s2>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_shared_128x256x32s2");
        } else if (!use_shared_reduce && use_k32_s2) {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Global128x256x32s2,
                FlashAssignMinVisitorV8Global128x256x32s2>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_global_128x256x32s2");
        } else if (use_shared_reduce) {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Shared256,
                FlashAssignMinVisitorV8Shared256>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_shared_256");
        } else {
            launch_flash_assign_min_gemm_impl<
                FlashAssignMinGemmV8Global256,
                FlashAssignMinVisitorV8Global256>(
                d_data, d_centroids, d_xnorm2, d_cnorm2, n, k, dim, n_tiles,
                d_partial_packed, stream, prop, max_optin, "v8_global_256");
        }
    }
    CUDA_LAUNCH_CHECK("flash_assign_min_gemm_v8");
}

}  // namespace kmeans
}  // namespace clustering
}  // namespace paraselect
