#include "dataset_source.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
namespace paraselect::dataset {
namespace {
enum class VectorFileFormat { Fbin, Bvecs, U8bin, I8bin };
enum class GroundTruthFileFormat { IbinFbin, IvecsFvecs };

struct DatasetDefinition {
    const char* name;
    int dim;
    int default_n_query;
    size_t billion_scale_vectors;
    VectorDType dtype;
    DistanceType metric;
    VectorFileFormat vector_format;
    GroundTruthFileFormat groundtruth_format;
    const char* base_env;
    const char* query_env;
    const char* groundtruth_index_env;
    const char* groundtruth_distance_env;
};

const std::array<DatasetDefinition, 0> kDatasets{};

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

const DatasetDefinition& find_dataset(const std::string& name) {
    const std::string normalized = lower_ascii(name);
    for (const DatasetDefinition& dataset : kDatasets) {
        const std::string_view id(dataset.name);
        if (normalized == id || normalized.rfind(id, 0) == 0) return dataset;
    }
    throw std::runtime_error("unknown dataset: " + name);
}

std::string required_path(const char* env_name) {
    const char* value = std::getenv(env_name);
    if (!value || value[0] == '\0') {
        throw std::runtime_error(
            std::string("missing required dataset path environment variable: ") + env_name);
    }
    return value;
}

std::string optional_path(const char* env_name) {
    const char* value = std::getenv(env_name);
    return (value && value[0] != '\0') ? std::string(value) : std::string();
}

std::string path_key(const char* pattern, const std::string& size_label) {
    std::string result(pattern);
    constexpr std::string_view token = "{size}";
    const size_t position = result.find(token);
    if (position == std::string::npos) {
        throw std::runtime_error("ground truth path pattern is invalid");
    }
    result.replace(position, token.size(), parse_dataset_size_label(size_label));
    return result;
}

void free_pinned(void* data) {
    BinaryFormat::free_pinned(data);
}
bool has_suffix(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}
bool is_header_byte_format(VectorFileFormat format) {
    return format == VectorFileFormat::U8bin || format == VectorFileFormat::I8bin;
}
VectorFileFormat resolve_vector_file_format(
    const std::string& path, VectorFileFormat declared_format) {
    if (has_suffix(path, ".bvecs")) return VectorFileFormat::Bvecs;
    if (has_suffix(path, ".u8bin")) return VectorFileFormat::U8bin;
    if (has_suffix(path, ".i8bin")) return VectorFileFormat::I8bin;
    if (has_suffix(path, ".fbin")) return VectorFileFormat::Fbin;
    if (has_suffix(path, ".bin") && declared_format != VectorFileFormat::Bvecs) {
        return declared_format;
    }
    throw std::runtime_error("unsupported vector file format: " + path);
}
VectorDType normalized_vector_dtype(VectorFileFormat format) {
    return format == VectorFileFormat::Fbin ? VectorDType::F32 : VectorDType::U8;
}
void validate_vector_dtype(
    const std::string& path, VectorFileFormat format, VectorDType expected) {
    if (normalized_vector_dtype(format) != expected) {
        throw std::runtime_error("vector file dtype does not match dataset spec: " + path);
    }
}
void normalize_i8_as_u8(uint8_t* data, size_t count) {
    for (size_t i = 0; i < count; ++i) data[i] ^= 0x80;
}
size_t load_header_bytes_as_u8(
    const std::string& path,
    VectorFileFormat format,
    uint8_t* output,
    size_t begin,
    size_t count,
    int dim) {
    const size_t loaded = BinaryFormat::load_header_u8bin_chunk_u8(
        path, output, begin, count);
    if (format == VectorFileFormat::I8bin) {
        normalize_i8_as_u8(output, loaded * static_cast<size_t>(dim));
    }
    return loaded;
}
} // namespace

struct DatasetSourceImpl {
    explicit DatasetSourceImpl(const DatasetDefinition& value) : definition(value) {}
    DatasetDefinition definition;
};

DatasetSpec describe_dataset(const std::string& name, const std::string& size_label) {
    const DatasetDefinition& dataset = find_dataset(name);
    const std::string normalized_size = parse_dataset_size_label(size_label);
    const size_t n_vectors = normalized_size == "1B"
        ? dataset.billion_scale_vectors
        : parse_dataset_size(normalized_size);
    return {dataset.name, normalized_size, n_vectors,
            dataset.dim, dataset.default_n_query, dataset.metric, dataset.dtype};
}

DatasetSource::DatasetSource(DatasetSpec spec)
    : spec_(std::move(spec)),
      impl_(std::make_shared<DatasetSourceImpl>(find_dataset(spec_.name))) {
    const DatasetDefinition& dataset = impl_->definition;
    if (spec_.dtype != dataset.dtype) {
        throw std::invalid_argument("dataset dtype does not match its definition");
    }
    if (spec_.dim != dataset.dim || spec_.metric != dataset.metric) {
        throw std::invalid_argument("dataset shape or metric does not match its definition");
    }
    base_path_ = required_path(dataset.base_env);
    query_path_ = required_path(dataset.query_env);
    const std::string gt_index_key = path_key(dataset.groundtruth_index_env, spec_.size_label);
    const std::string gt_distance_key = path_key(dataset.groundtruth_distance_env, spec_.size_label);
    groundtruth_index_path_ = required_path(gt_index_key.c_str());
    groundtruth_distance_path_ = optional_path(gt_distance_key.c_str());
}

DatasetInfo DatasetSource::vector_info(const std::string& path) const {
    const VectorFileFormat format = resolve_vector_file_format(path, impl_->definition.vector_format);
    validate_vector_dtype(path, format, spec_.dtype);
    if (is_header_byte_format(format)) {
        if (!BinaryFormat::is_header_u8bin(path)) {
            throw std::runtime_error("invalid header byte vector file: " + path);
        }
        return BinaryFormat::read_header_ibin(path);
    }
    return format == VectorFileFormat::Fbin
               ? BinaryFormat::read_header_fbin(path)
               : BinaryFormat::get_file_info(path);
}

DatasetInfo DatasetSource::base_info() const {
    DatasetInfo info = vector_info(base_path_);
    if (info.n < spec_.n_vectors) {
        throw std::runtime_error(
            "base file has " + std::to_string(info.n) + " vectors, but " +
            display_dataset_name(spec_) + " requires " + std::to_string(spec_.n_vectors) +
            ": " + base_path_);
    }
    info.n = spec_.n_vectors;
    info.data_size = info.n * static_cast<size_t>(info.dim) * vector_dtype_size(spec_.dtype);
    return info;
}

bool DatasetSource::base_has_signed_byte_encoding() const {
    return resolve_vector_file_format(base_path_, impl_->definition.vector_format) ==
           VectorFileFormat::I8bin;
}

DatasetInfo DatasetSource::query_info() const {
    DatasetInfo info = vector_info(query_path_);
    info.data_size = info.n * static_cast<size_t>(info.dim) * vector_dtype_size(spec_.dtype);
    return info;
}
VectorBuffer DatasetSource::load_vectors(const std::string& path, size_t limit) const {
    const VectorFileFormat format = resolve_vector_file_format(path, impl_->definition.vector_format);
    const DatasetInfo file_info = vector_info(path);
    const size_t n = limit ? std::min(limit, file_info.n) : file_info.n;
    void* data = nullptr;
    size_t loaded = 0;
    if (is_header_byte_format(format)) {
        auto* typed = static_cast<uint8_t*>(
            BinaryFormat::allocate_pinned(n * static_cast<size_t>(file_info.dim)));
        try {
            loaded = load_header_bytes_as_u8(
                path, format, typed, 0, n, file_info.dim);
        } catch (...) {
            free_pinned(typed);
            throw;
        }
        data = typed;
    } else if (format == VectorFileFormat::Bvecs) {
        uint8_t* typed = nullptr;
        const DatasetInfo info = BinaryFormat::load_bvecs_pinned_u8(path, &typed, n);
        data = typed;
        loaded = info.n;
    } else {
        float* typed = nullptr;
        const DatasetInfo info = BinaryFormat::load_fbin_pinned(path, &typed, n);
        data = typed;
        loaded = info.n;
    }
    if (file_info.dim != spec_.dim || loaded != n) {
        free_pinned(data);
        throw std::runtime_error("dataset vector shape mismatch: " + path);
    }
    return {spec_.dtype, data, loaded, file_info.dim, free_pinned};
}

VectorBuffer DatasetSource::load_base(size_t limit) const {
    (void)base_info();
    const size_t n = limit ? std::min(limit, spec_.n_vectors) : spec_.n_vectors;
    return load_vectors(base_path_, n);
}

VectorBuffer DatasetSource::load_queries(size_t limit) const {
    return load_vectors(query_path_, limit ? limit : impl_->definition.default_n_query);
}

size_t DatasetSource::read_base_range(size_t begin, size_t count, void* output) const {
    const VectorFileFormat format = resolve_vector_file_format(base_path_, impl_->definition.vector_format);
    validate_vector_dtype(base_path_, format, spec_.dtype);
    if (!output && count != 0) throw std::invalid_argument("null base range output");
    if (begin > spec_.n_vectors || count > spec_.n_vectors - begin) {
        throw std::out_of_range("base range outside dataset");
    }
    if (is_header_byte_format(format)) {
        return load_header_bytes_as_u8(
            base_path_, format, static_cast<uint8_t*>(output),
            begin, count, spec_.dim);
    }
    if (format == VectorFileFormat::Bvecs) {
        return BinaryFormat::load_bvecs_chunk_u8(
            base_path_, static_cast<uint8_t*>(output), begin, count);
    }
    return BinaryFormat::load_fbin_chunk(
        base_path_, static_cast<float*>(output), begin, count);
}

VectorBuffer DatasetSource::load_base_rows(
    const std::vector<uint32_t>& ids, size_t begin, size_t count) const {
    if (begin > ids.size()) throw std::out_of_range("base row range outside id list");
    const size_t n = count ? std::min(count, ids.size() - begin) : ids.size() - begin;
    const size_t row_bytes = static_cast<size_t>(spec_.dim) * vector_dtype_size(spec_.dtype);
    void* data = BinaryFormat::allocate_pinned(n * row_bytes);
    try {
        std::vector<std::pair<uint32_t, size_t>> ordered;
        ordered.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            const uint32_t id = ids[begin + i];
            if (id >= spec_.n_vectors) throw std::out_of_range("base id outside dataset");
            ordered.emplace_back(id, i);
        }
        std::sort(ordered.begin(), ordered.end());
        constexpr size_t kMaxReadRows = 65536;
        const size_t id_span = ordered.empty() ? 0 : (size_t)ordered.back().first - ordered.front().first + 1;
        const bool scan_windows = id_span > 0 && ordered.size() * 200 >= id_span;
        std::vector<uint8_t> scratch;
        for (size_t first = 0; first < ordered.size();) {
            size_t last = first;
            while (last + 1 < ordered.size() &&
                   (scan_windows || ordered[last + 1].first <= ordered[last].first + 1) &&
                   static_cast<size_t>(ordered[last + 1].first) - ordered[first].first < kMaxReadRows) {
                ++last;
            }
            const size_t rows = static_cast<size_t>(ordered[last].first) - ordered[first].first + 1;
            scratch.resize(rows * row_bytes);
            if (read_base_range(ordered[first].first, rows, scratch.data()) != rows) {
                throw std::runtime_error("short base row read");
            }
            for (size_t i = first; i <= last; ++i) {
                const size_t source_offset = static_cast<size_t>(ordered[i].first) - ordered[first].first;
                std::memcpy(static_cast<uint8_t*>(data) + ordered[i].second * row_bytes,
                            scratch.data() + source_offset * row_bytes, row_bytes);
            }
            first = last + 1;
        }
    } catch (...) {
        free_pinned(data);
        throw;
    }
    return {spec_.dtype, data, n, spec_.dim, free_pinned};
}

GroundTruth DatasetSource::load_groundtruth(size_t query_limit) const {
    if (has_suffix(groundtruth_index_path_, ".bin") &&
        !has_suffix(groundtruth_index_path_, ".ibin") &&
        !has_suffix(groundtruth_index_path_, ".fbin")) {
        int* raw_ids = nullptr;
        float* raw_distances = nullptr;
        int n = 0;
        int k = 0;
        if (!BinaryFormat::load_groundtruth_bin_to_host(
                groundtruth_index_path_, &raw_ids, &raw_distances, n, k, query_limit)) {
            throw std::runtime_error(
                "failed to load combined groundtruth: " + groundtruth_index_path_);
        }
        std::unique_ptr<int[]> ids(raw_ids);
        std::unique_ptr<float[]> distances(raw_distances);
        GroundTruth result;
        result.n = static_cast<size_t>(n);
        result.k = k;
        result.ids.assign(ids.get(), ids.get() + result.n * static_cast<size_t>(k));
        result.distances.assign(
            distances.get(), distances.get() + result.n * static_cast<size_t>(k));
        return result;
    }

    int* raw_ids = nullptr;
    int n = 0;
    int k = 0;
    const bool ids_loaded = has_suffix(groundtruth_index_path_, ".ivecs")
                                ? BinaryFormat::load_ivecs_to_host(groundtruth_index_path_, &raw_ids, n, k)
                                : BinaryFormat::load_ibin_to_host(groundtruth_index_path_, &raw_ids, n, k);
    if (!ids_loaded) {
        throw std::runtime_error("failed to load groundtruth indices: " + groundtruth_index_path_);
    }
    std::unique_ptr<int[]> ids(raw_ids);
    const size_t keep = query_limit ? std::min(query_limit, static_cast<size_t>(n))
                                    : static_cast<size_t>(n);
    GroundTruth result;
    result.n = keep;
    result.k = k;
    result.ids.assign(ids.get(), ids.get() + keep * static_cast<size_t>(k));

    float* raw_distances = nullptr;
    int distance_n = 0;
    int distance_k = 0;
    const bool distances_loaded = has_suffix(groundtruth_distance_path_, ".fvecs")
                                      ? BinaryFormat::load_fvecs_to_host(
                                            groundtruth_distance_path_, &raw_distances, distance_n, distance_k)
                                      : BinaryFormat::load_fbin_to_host(
                                            groundtruth_distance_path_, &raw_distances, distance_n, distance_k);
    std::unique_ptr<float[]> distances(raw_distances);
    if (distances_loaded) {
        if (distance_n != n || distance_k != k) {
            throw std::runtime_error("groundtruth index/distance shape mismatch");
        }
        result.distances.assign(
            distances.get(), distances.get() + keep * static_cast<size_t>(k));
    }
    return result;
}

DatasetSource open_dataset(const DatasetSpec& spec) {
    return DatasetSource(spec);
}

DatasetSource open_dataset(const std::string& name, const std::string& size_label) {
    return DatasetSource(describe_dataset(name, size_label));
}

} // namespace paraselect::dataset
