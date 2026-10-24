#include "residual_encoding.cuh"
#include "turboquant_1bit_encode.cuh"
#include "utils/check_cuda_error.cuh"
#include <algorithm>
#include <stdexcept>

namespace paraselect::search::turboquant {
namespace {
// Build scratch is released before the first query, including on exceptions.
template<class T> struct BuildBuffer {
    T* ptr = nullptr;
    explicit BuildBuffer(size_t count) { CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T))); }
    ~BuildBuffer() { if (ptr) cudaFree(ptr); }
    BuildBuffer(const BuildBuffer&) = delete;
    BuildBuffer& operator=(const BuildBuffer&) = delete;
};
constexpr unsigned kRotationSeed = 17u;
// The database is encoded in bounded row chunks so the host U8 corpus never
// has to be resident on the device as a whole. Codes and scales are written
// straight into the full device buffers at their global row offsets.
constexpr int kEncodeChunkRows = 1 << 20;
}
ResidualEncoding::ResidualEncoding(const ResidualCodecBuildInput& in)
    : dim_(in.dim), max_batch_(in.max_batch) {
    if (!in.vectors || !in.centroids || !in.offsets || in.rows <= 0 ||
        in.lists <= 0 || in.dim != 128 || in.max_batch <= 0) {
        throw std::invalid_argument("sign residual codec requires valid U8/SIFT128 input");
    }
    if ((in.cached_codes == nullptr) != (in.cached_scales == nullptr)) {
        throw std::invalid_argument(
            "sign residual codec cache requires cached_codes and cached_scales together");
    }
    try {
        const int words_per_code = in.dim / 32;
        const int chunk_rows = std::min(kEncodeChunkRows, in.rows);
        BuildBuffer<float> centroids((size_t)in.lists * in.dim);
        BuildBuffer<long long> offsets((size_t)in.lists + 1);
        BuildBuffer<unsigned char> chunk_vectors((size_t)chunk_rows * in.dim);
        CUDA_CHECK(cudaMemcpy(
            centroids.ptr, in.centroids, (size_t)in.lists * in.dim * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            offsets.ptr, in.offsets, ((size_t)in.lists + 1) * sizeof(long long),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&codes_, (size_t)in.rows * words_per_code * sizeof(std::uint32_t)));
        CUDA_CHECK(cudaMalloc(&scales_, (size_t)in.rows * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&rotated_centroids_, (size_t)in.lists * in.dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&rotated_queries_, (size_t)in.max_batch * in.dim * sizeof(float)));
        if (in.cached_codes && in.cached_scales) {
            // Cache hit: upload the pre-encoded rows instead of re-encoding.
            CUDA_CHECK(cudaMemcpy(
                codes_, in.cached_codes,
                (size_t)in.rows * words_per_code * sizeof(std::uint32_t),
                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(
                scales_, in.cached_scales, (size_t)in.rows * sizeof(float),
                cudaMemcpyHostToDevice));
        } else {
            for (int start = 0; start < in.rows; start += chunk_rows) {
                const int count = std::min(chunk_rows, in.rows - start);
                CUDA_CHECK(cudaMemcpy(
                    chunk_vectors.ptr, in.vectors + (size_t)start * in.dim,
                    (size_t)count * in.dim, cudaMemcpyHostToDevice));
                ivf_res_turboquant_encode_database_u8_gpu(
                    {chunk_vectors.ptr, centroids.ptr, offsets.ptr, count, in.lists,
                     in.dim, in.dim, kRotationSeed, start},
                    {nullptr, codes_, scales_, nullptr}, nullptr);
            }
        }
        turboquant_1bit_encode_gpu({centroids.ptr, in.lists, in.dim, in.dim, kRotationSeed},
            {rotated_centroids_, nullptr, nullptr, nullptr}, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
    } catch (...) {
        release();
        throw;
    }
}
void ResidualEncoding::release() noexcept {
    if (codes_) cudaFree(codes_);
    if (scales_) cudaFree(scales_);
    if (rotated_centroids_) cudaFree(rotated_centroids_);
    if (rotated_queries_) cudaFree(rotated_queries_);
    codes_ = nullptr;
    scales_ = rotated_centroids_ = rotated_queries_ = nullptr;
}
ResidualEncoding::~ResidualEncoding() { release(); }
SignResidualDeviceView ResidualEncoding::prepare_queries(
    const float* queries, int count, cudaStream_t stream) {
    if (!queries || count <= 0 || count > max_batch_)
        throw std::invalid_argument("sign residual query capacity exceeded");
    turboquant_1bit_encode_gpu({queries, count, dim_, dim_, kRotationSeed},
        {rotated_queries_, nullptr, nullptr, nullptr}, stream);
    return {codes_, scales_, rotated_queries_, rotated_centroids_};
}
} // namespace paraselect::search::turboquant
