#include "dataset/temporary_exact_store.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace paraselect::dataset {
using namespace paraselect::search;
namespace {

constexpr size_t kBlockSize = 4096;
constexpr char kMagic[8] = {'I', 'V', 'F', 'E', 'X', 'R', '1', '\0'};

struct TemporaryExactHeader {
    char magic[8];
    uint32_t version;
    uint32_t dtype;
    uint64_t count;
    uint32_t dim;
    uint32_t reserved;
    uint64_t row_stride;
    uint64_t payload_offset;
    uint64_t block_size;
};

IVFExactRerankDType rerank_dtype(VectorDType dtype) {
    switch (dtype) {
        case VectorDType::U8: return IVFExactRerankDType::U8;
        case VectorDType::F32: return IVFExactRerankDType::FP32;
    }
    throw std::invalid_argument("unsupported temporary exact store dtype");
}

size_t next_power_of_two(size_t value) {
    if (value == 0) throw std::invalid_argument("temporary exact row size is zero");
    size_t out = 1;
    while (out < value) {
        if (out > std::numeric_limits<size_t>::max() / 2) {
            throw std::overflow_error("temporary exact row stride overflow");
        }
        out *= 2;
    }
    return out;
}

size_t align_up(size_t value, size_t alignment) {
    if (value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        throw std::overflow_error("temporary exact file size overflow");
    }
    return (value + alignment - 1) / alignment * alignment;
}

void throw_errno(const char* operation) {
    throw std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
}

void write_header(
    int fd,
    IVFExactRerankDType dtype,
    idx_t count,
    int dim,
    size_t row_stride,
    size_t file_size) {
    void* block = nullptr;
    if (::posix_memalign(&block, kBlockSize, kBlockSize) != 0 || !block) {
        throw std::bad_alloc();
    }
    std::memset(block, 0, kBlockSize);
    TemporaryExactHeader header{};
    std::memcpy(header.magic, kMagic, sizeof(kMagic));
    header.version = 1;
    header.dtype = static_cast<uint32_t>(dtype);
    header.count = static_cast<uint64_t>(count);
    header.dim = static_cast<uint32_t>(dim);
    header.row_stride = static_cast<uint64_t>(row_stride);
    header.payload_offset = kBlockSize;
    header.block_size = kBlockSize;
    std::memcpy(block, &header, sizeof(header));
    const ssize_t written = ::pwrite(fd, block, kBlockSize, 0);
    std::free(block);
    if (written != static_cast<ssize_t>(kBlockSize)) {
        if (written < 0) throw_errno("write temporary exact header");
        throw std::runtime_error("short write of temporary exact header");
    }
    if (::ftruncate(fd, static_cast<off_t>(file_size)) != 0) {
        throw_errno("resize temporary exact store");
    }
    const int allocate_result = ::posix_fallocate(fd, 0, static_cast<off_t>(file_size));
    if (allocate_result != 0) {
        throw std::runtime_error(
            std::string("allocate temporary exact store: ") +
            std::strerror(allocate_result));
    }
}

}  // namespace

TemporaryExactStore::~TemporaryExactStore() { reset(); }

TemporaryExactStore::TemporaryExactStore(TemporaryExactStore&& other) noexcept {
    *this = std::move(other);
}

TemporaryExactStore& TemporaryExactStore::operator=(TemporaryExactStore&& other) noexcept {
    if (this == &other) return *this;
    reset();
    fd_ = other.fd_;
    dtype_ = other.dtype_;
    count_ = other.count_;
    dim_ = other.dim_;
    row_stride_ = other.row_stride_;
    payload_offset_ = other.payload_offset_;
    block_size_ = other.block_size_;
    build_ms_ = other.build_ms_;
    other.fd_ = -1;
    other.dtype_ = IVFExactRerankDType::Disabled;
    return *this;
}

void TemporaryExactStore::reset() noexcept {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
    dtype_ = IVFExactRerankDType::Disabled;
}

TemporaryExactStore TemporaryExactStore::build(
    const DatasetSource& source,
    const idx_t* reordered_ids,
    idx_t count,
    int dim,
    const TemporaryExactStoreOptions& options) {
    if (!reordered_ids || count <= 0 || dim <= 0) {
        throw std::invalid_argument("invalid temporary exact store shape");
    }
    if (static_cast<uint64_t>(count) > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("temporary exact store supports at most UINT32_MAX rows");
    }
    const auto info = source.base_info();
    if (info.dim != dim || info.n < static_cast<size_t>(count)) {
        throw std::invalid_argument("temporary exact store source shape mismatch");
    }

    const IVFExactRerankDType dtype = rerank_dtype(source.spec().dtype);
    const auto begin = std::chrono::steady_clock::now();
    const size_t row_bytes =
        static_cast<size_t>(dim) * vector_dtype_size(source.spec().dtype);
    const size_t row_stride = next_power_of_two(row_bytes);
    if (row_stride > kBlockSize || kBlockSize % row_stride != 0) {
        throw std::invalid_argument(
            "temporary exact row stride must divide 4096 bytes");
    }
    if (static_cast<size_t>(count) >
        (std::numeric_limits<size_t>::max() - kBlockSize) / row_stride) {
        throw std::overflow_error("temporary exact file size overflow");
    }
    const size_t payload_bytes = static_cast<size_t>(count) * row_stride;
    const size_t file_size = kBlockSize + align_up(payload_bytes, kBlockSize);

    std::vector<uint32_t> source_to_row(static_cast<size_t>(count), UINT32_MAX);
    for (uint32_t row = 0; row < static_cast<uint32_t>(count); ++row) {
        const idx_t source_id = reordered_ids[row];
        if (source_id < 0 || source_id >= count) {
            throw std::invalid_argument("temporary exact reordered id is out of range");
        }
        uint32_t& destination = source_to_row[static_cast<size_t>(source_id)];
        if (destination != UINT32_MAX) {
            throw std::invalid_argument("temporary exact reordered ids are not a permutation");
        }
        destination = row;
    }

    const std::string directory = options.directory.empty() ? ".tmp" : options.directory;
    std::filesystem::create_directories(directory);
    std::string pattern = directory + "/ivf-exact-XXXXXX";
    std::vector<char> path(pattern.begin(), pattern.end());
    path.push_back('\0');
    int build_fd = ::mkstemp(path.data());
    if (build_fd < 0) throw_errno("create temporary exact store");

    void* mapping = MAP_FAILED;
    try {
        write_header(build_fd, dtype, count, dim, row_stride, file_size);
        mapping = ::mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, build_fd, 0);
        if (mapping == MAP_FAILED) throw_errno("map temporary exact store");
        uint8_t* payload = static_cast<uint8_t*>(mapping) + kBlockSize;
        const size_t chunk_rows = std::max<size_t>(1, options.source_chunk_rows);
        std::vector<uint8_t> source_rows(
            std::min(chunk_rows, static_cast<size_t>(count)) * row_bytes);
        for (size_t source_begin = 0;
             source_begin < static_cast<size_t>(count);
             source_begin += chunk_rows) {
            const size_t rows =
                std::min(chunk_rows, static_cast<size_t>(count) - source_begin);
            if (source.read_base_range(source_begin, rows, source_rows.data()) != rows) {
                throw std::runtime_error("short read while building temporary exact store");
            }
            for (size_t local = 0; local < rows; ++local) {
                const uint32_t row = source_to_row[source_begin + local];
                uint8_t* destination = payload + static_cast<size_t>(row) * row_stride;
                std::memcpy(destination, source_rows.data() + local * row_bytes, row_bytes);
                if (row_stride > row_bytes) {
                    std::memset(destination + row_bytes, 0, row_stride - row_bytes);
                }
            }
        }
        if (::msync(mapping, file_size, MS_SYNC) != 0) {
            throw_errno("flush temporary exact store mapping");
        }
        if (::munmap(mapping, file_size) != 0) {
            throw_errno("unmap temporary exact store");
        }
        mapping = MAP_FAILED;
        if (::fdatasync(build_fd) != 0) throw_errno("sync temporary exact store");
        const int advise_result = ::posix_fadvise(
            build_fd, 0, static_cast<off_t>(file_size), POSIX_FADV_DONTNEED);
        if (advise_result != 0) {
            throw std::runtime_error(
                std::string("drop temporary exact store page cache: ") +
                std::strerror(advise_result));
        }
        if (::close(build_fd) != 0) throw_errno("close temporary exact build file");
        build_fd = -1;

        int direct_fd = ::open(path.data(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (direct_fd < 0) throw_errno("open temporary exact store with O_DIRECT");
        if (::unlink(path.data()) != 0) {
            const int saved_errno = errno;
            ::close(direct_fd);
            errno = saved_errno;
            throw_errno("unlink active temporary exact store");
        }

        TemporaryExactStore out;
        out.fd_ = direct_fd;
        out.dtype_ = dtype;
        out.count_ = count;
        out.dim_ = dim;
        out.row_stride_ = row_stride;
        out.payload_offset_ = kBlockSize;
        out.block_size_ = kBlockSize;
        out.build_ms_ = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        return out;
    } catch (...) {
        if (mapping != MAP_FAILED) ::munmap(mapping, file_size);
        if (build_fd >= 0) ::close(build_fd);
        ::unlink(path.data());
        throw;
    }
}

IVFFileStorageView TemporaryExactStore::view() const {
    if (fd_ < 0 || dtype_ == IVFExactRerankDType::Disabled) {
        throw std::logic_error("temporary exact store is not initialized");
    }
    return {dtype_, fd_, count_, dim_, row_stride_, payload_offset_, block_size_};
}

}  // namespace paraselect::dataset
