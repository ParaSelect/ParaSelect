#include "pca_cache.h"

#include "cache_fs.h"
#include "cache_manifest.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <filesystem>

namespace paraselect::dataset {
namespace {

cache_manifest::Json manifest_identity(const std::string& dataset, const PCACacheSpec& spec) {
    auto result = cache_manifest::identity("pca", dataset);
    result.update({{"n_vectors", spec.n_vectors}, {"source_dim", spec.source_dim},
                   {"dim", spec.dim}, {"train_n", spec.train_n},
                   {"cluster_layout_id", spec.cluster_layout_id},
                   {"quantization", spec.quantization}});
    return result;
}

template <typename T>
void write_binary_atomic(
    const std::string& path,
    uint32_t n,
    uint32_t dim,
    const T* values) {
    const std::string temporary = path + ".tmp";
    FILE* file = std::fopen(temporary.c_str(), "wb");
    if (!file) throw std::runtime_error("failed to create PCA cache file: " + temporary);
    const size_t count = static_cast<size_t>(n) * dim;
    bool ok = std::fwrite(&n, sizeof(n), 1, file) == 1 &&
              std::fwrite(&dim, sizeof(dim), 1, file) == 1 &&
              std::fwrite(values, sizeof(T), count, file) == count;
    if (std::fclose(file) != 0) ok = false;
    if (!ok || std::rename(temporary.c_str(), path.c_str()) != 0) {
        std::remove(temporary.c_str());
        throw std::runtime_error("failed to commit PCA cache file: " + path);
    }
}

void write_metadata_atomic(
    const std::string& path,
    const std::string& dataset_name,
    const PCACacheSpec& spec) {
    const std::string temporary = path + ".tmp";
    std::ofstream output(temporary);
    if (!output) throw std::runtime_error("failed to create PCA cache metadata");
    output << "{\n"
           << "  \"format\": \"paraselect-pca-cache\",\n"
           << "  \"version\": 1,\n"
           << "  \"dataset_name\": \"" << dataset_name << "\",\n"
           << "  \"n_vectors\": " << spec.n_vectors << ",\n"
           << "  \"source_dim\": " << spec.source_dim << ",\n"
           << "  \"dim\": " << spec.dim << ",\n"
           << "  \"train_n\": " << spec.train_n << ",\n"
           << "  \"cluster_layout_id\": \"" << spec.cluster_layout_id << "\",\n"
           << "  \"quantization\": \"" << spec.quantization << "\",\n"
           << "  \"vectors_layout\": \"ivf-local-id-major\"\n"
           << "}\n";
    output.close();
    if (!output || std::rename(temporary.c_str(), path.c_str()) != 0) {
        std::remove(temporary.c_str());
        throw std::runtime_error("failed to commit PCA cache metadata");
    }
}

}  // namespace

struct PCACacheWriter::Impl {
    Impl(
        std::string output_directory,
        std::string name,
        PCACacheSpec value)
        : directory(std::move(output_directory)),
          dataset_name(std::move(name)),
          spec(std::move(value)),
          vectors_path(directory + "/vectors.i8bin"),
          temporary_path(vectors_path + ".tmp") {
        const size_t count = static_cast<size_t>(spec.n_vectors) * spec.dim;
        bytes = sizeof(uint32_t) * 2 + count * sizeof(int8_t);
        fd = ::open(temporary_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd < 0 || ::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
            cleanup();
            throw std::runtime_error("failed to allocate PCA vector cache");
        }
        mapping = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mapping == MAP_FAILED) {
            mapping = nullptr;
            cleanup();
            throw std::runtime_error("failed to map PCA vector cache");
        }
        auto* header = static_cast<uint32_t*>(mapping);
        header[0] = static_cast<uint32_t>(spec.n_vectors);
        header[1] = static_cast<uint32_t>(spec.dim);
        data = reinterpret_cast<int8_t*>(header + 2);
    }

    ~Impl() { cleanup(); }

    void cleanup() {
        if (mapping) ::munmap(mapping, bytes);
        if (fd >= 0) ::close(fd);
        if (!committed) std::remove(temporary_path.c_str());
        mapping = nullptr;
        data = nullptr;
        fd = -1;
    }

    std::string directory;
    std::string dataset_name;
    PCACacheSpec spec;
    std::string vectors_path;
    std::string temporary_path;
    int fd = -1;
    void* mapping = nullptr;
    int8_t* data = nullptr;
    size_t bytes = 0;
    bool committed = false;
};

PCACacheWriter::PCACacheWriter(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

PCACacheWriter::~PCACacheWriter() = default;
PCACacheWriter::PCACacheWriter(PCACacheWriter&&) noexcept = default;
PCACacheWriter& PCACacheWriter::operator=(PCACacheWriter&&) noexcept = default;

int8_t* PCACacheWriter::vectors() {
    if (!impl_ || !impl_->data) throw std::logic_error("PCA cache writer is closed");
    return impl_->data;
}

void PCACacheWriter::commit(
    const std::vector<float>& projection,
    const std::vector<float>& bias,
    const std::vector<float>& scales) {
    if (!impl_ || !impl_->mapping || impl_->committed) {
        throw std::logic_error("PCA cache writer cannot be committed");
    }
    if (projection.size() !=
            static_cast<size_t>(impl_->spec.dim) * impl_->spec.source_dim ||
        bias.size() != static_cast<size_t>(impl_->spec.dim) ||
        scales.size() != static_cast<size_t>(impl_->spec.dim)) {
        throw std::invalid_argument("PCA cache model shape mismatch");
    }
    if (::msync(impl_->mapping, impl_->bytes, MS_SYNC) != 0 ||
        ::munmap(impl_->mapping, impl_->bytes) != 0) {
        throw std::runtime_error("failed to flush PCA vector cache");
    }
    impl_->mapping = nullptr;
    impl_->data = nullptr;
    if (::fsync(impl_->fd) != 0 || ::close(impl_->fd) != 0) {
        impl_->fd = -1;
        throw std::runtime_error("failed to sync PCA vector cache");
    }
    impl_->fd = -1;
    if (std::rename(impl_->temporary_path.c_str(), impl_->vectors_path.c_str()) != 0) {
        throw std::runtime_error("failed to commit PCA vector cache");
    }
    write_binary_atomic(
        impl_->directory + "/projection.fbin",
        static_cast<uint32_t>(impl_->spec.dim),
        static_cast<uint32_t>(impl_->spec.source_dim),
        projection.data());
    write_binary_atomic(
        impl_->directory + "/bias.fbin", 1,
        static_cast<uint32_t>(impl_->spec.dim), bias.data());
    write_binary_atomic(
        impl_->directory + "/scales.fbin", 1,
        static_cast<uint32_t>(impl_->spec.dim), scales.data());
    write_metadata_atomic(
        impl_->directory + "/meta.json", impl_->dataset_name, impl_->spec);
    cache_manifest::publish(impl_->directory,
        manifest_identity(impl_->dataset_name, impl_->spec));
    impl_->committed = true;
}

PCACache::PCACache(std::string directory, std::string dataset_name)
    : directory_(std::move(directory)), dataset_name_(std::move(dataset_name)) {
    if (directory_.empty() || dataset_name_.empty())
        throw std::invalid_argument("PCA cache requires root and dataset");
}

PCACacheMappedResult PCACache::try_map(
    const PCACacheSpec& spec) const {
    PCACacheMappedResult output;
    output.request = spec;
    output.entry_dir = cache_manifest::find(directory_, manifest_identity(dataset_name_, spec));
    if (output.entry_dir.empty()) return output;
    try {
        output.vectors = MappedArray<int8_t>::open_binary(output.entry_dir + "/vectors.i8bin", spec.n_vectors, spec.dim);
        output.projection = MappedArray<float>::open_binary(output.entry_dir + "/projection.fbin", spec.dim, spec.source_dim);
        output.bias = MappedArray<float>::open_binary(output.entry_dir + "/bias.fbin", 1, spec.dim);
        output.scales = MappedArray<float>::open_binary(output.entry_dir + "/scales.fbin", 1, spec.dim);
    } catch (const std::exception&) { return output; }
    output.hit = true;
    return output;
}

PCACacheWriter PCACache::create(
    const PCACacheSpec& spec) const {
    if (spec.n_vectors <= 0 || spec.source_dim <= 0 ||
        spec.dim <= 0 || spec.dim > spec.source_dim || spec.train_n <= 0 ||
        spec.train_n > spec.n_vectors || spec.cluster_layout_id.empty() ||
        spec.quantization != "symmetric-per-dimension-int8-v1") {
        throw std::invalid_argument("invalid PCA cache specification");
    }
    const std::string path = cache_manifest::canonical_entry(
        directory_, manifest_identity(dataset_name_, spec));
    if (!cache_fs::ensure_dir_exists(path)) {
        throw std::runtime_error("failed to create PCA cache directory");
    }
    return PCACacheWriter(std::make_unique<PCACacheWriter::Impl>(
        path, dataset_name_, spec));
}

}  // namespace paraselect::dataset
