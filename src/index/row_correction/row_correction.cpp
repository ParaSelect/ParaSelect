#include "index/row_correction/row_correction.h"

#include <algorithm>
#include "dataset/cache/cache_manifest.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>

namespace paraselect::row_correction {
namespace {

constexpr const char* kPayloadName = "row_correction.u16a";
constexpr const char* kFp32Name = "row_correction.f32";
constexpr const char* kManifestName = "row_correction.meta.json";

std::string path(const std::string& entry_dir, const char* name) {
    return entry_dir + "/" + name;
}

bool file_size_is(const std::string& file, std::size_t expected) {
    struct stat st {};
    return ::stat(file.c_str(), &st) == 0 &&
           static_cast<std::size_t>(st.st_size) == expected;
}

using Json = dataset::cache_manifest::Json;

Json manifest_identity(const Manifest& m) {
    return {{"format", "paraselect-row-correction-cache"}, {"version", kManifestVersion},
        {"dataset_name", m.dataset_name},
        {"parent_pq_manifest", Json::parse(m.parent_manifest_json)},
        {"cluster_layout_id", m.cluster_layout_id}, {"codec", m.codec},
        {"row_layout", m.row_layout}, {"rows", m.rows}, {"dim", m.dim},
        {"nlist", m.nlist}, {"code_width", m.code_width},
        {"codebook_size", m.codebook_size}, {"subvector_dim", m.subvector_dim},
        {"encoding", "u16_affine"}, {"payload", "row_correction.u16a"}};
}

void validate_manifest_shape(const Manifest& m) {
    if (m.dataset_name.empty() || m.parent_manifest_json.empty() ||
        m.cluster_layout_id.empty() || m.codec.empty() || m.row_layout.empty() ||
        m.rows == 0 || m.rows > static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max()) ||
        m.dim <= 0 || m.nlist <= 0 || m.code_width <= 0 ||
        m.codebook_size <= 0 || m.codebook_size > 256 ||
        m.subvector_dim <= 0 || m.code_width * m.subvector_dim != m.dim) {
        throw std::invalid_argument("invalid row-correction manifest");
    }
    const auto parent = Json::parse(m.parent_manifest_json);
    auto required = dataset::cache_manifest::identity("residual_pq", m.dataset_name);
    required.update({{"n_vectors", m.rows}, {"dim", m.dim}, {"nlist", m.nlist},
        {"m", m.code_width}, {"k", m.codebook_size}, {"dsub", m.subvector_dim},
        {"cluster_layout_id", m.cluster_layout_id}, {"codes_layout", m.row_layout}});
    if (!dataset::cache_manifest::matches(parent, required))
        throw std::invalid_argument("row-correction parent manifest mismatch");
}

void write_manifest(const std::string& file, const Manifest& m) {
    std::ofstream out(file, std::ios::trunc);
    if (!out) throw std::runtime_error("failed to create row-correction manifest");
    out << manifest_identity(m).dump(2) << std::endl;
    out.close();
    if (!out) throw std::runtime_error("failed to write row-correction manifest");
}

void commit(const std::string& temporary, const std::string& destination) {
    if (std::rename(temporary.c_str(), destination.c_str()) != 0) {
        std::remove(temporary.c_str());
        throw std::runtime_error("failed to commit row-correction cache file");
    }
}

}  // namespace

bool manifest_matches(const std::string& entry_dir, const Manifest& expected) {
    try {
        validate_manifest_shape(expected);
    } catch (const std::exception&) {
        return false;
    }
    std::ifstream in(path(entry_dir, kManifestName));
    if (!in) return false;
    const std::string content(
        (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    try {
        const auto actual = Json::parse(content);
        if (actual.contains("parent_pq_tag") || actual.contains("parent_index_tag") ||
            !dataset::cache_manifest::matches(actual, manifest_identity(expected))) return false;
        if (actual.at("parent_pq_manifest") != Json::parse(expected.parent_manifest_json))
            return false;
        std::ifstream parent_file(entry_dir + "/manifest.json");
        if (!parent_file || Json::parse(parent_file) != Json::parse(expected.parent_manifest_json))
            return false;
    } catch (const Json::exception&) { return false; }
    return file_size_is(
        path(entry_dir, kPayloadName),
        sizeof(U16Header) + static_cast<std::size_t>(expected.rows) * sizeof(std::uint16_t));
}

Asset Asset::open(const std::string& entry_dir, const Manifest& expected) {
    validate_manifest_shape(expected);
    if (!manifest_matches(entry_dir, expected)) {
        throw std::invalid_argument("row-correction cache manifest mismatch");
    }
    Asset result;
    result.mapping_ = dataset::ReadOnlyMapping::open(path(entry_dir, kPayloadName));
    const auto* header = static_cast<const U16Header*>(result.mapping_->data());
    if (std::memcmp(header->magic, kU16Magic, sizeof(header->magic)) != 0 ||
        header->version != kU16Version || header->header_bytes != sizeof(*header) ||
        header->count != expected.rows || !std::isfinite(header->scale) ||
        !(header->scale > 0.0f) || !std::isfinite(header->offset)) {
        throw std::invalid_argument("row-correction payload header mismatch");
    }
    result.view_.u16 = reinterpret_cast<const std::uint16_t*>(
        static_cast<const std::uint8_t*>(result.mapping_->data()) + sizeof(*header));
    result.view_.scale = header->scale;
    result.view_.offset = header->offset;
    result.view_.count = static_cast<std::int64_t>(expected.rows);
    return result;
}

BuildResult build_pq(
    const std::string& entry_dir,
    const Manifest& manifest,
    const PQBuildInput& input) {
    validate_manifest_shape(manifest);
    if (manifest.codec != "residual_pq") {
        throw std::invalid_argument("PQ row-correction builder requires residual_pq codec");
    }
    if (!input.centroids || !input.cluster_offsets || !input.codes || !input.codebook) {
        throw std::invalid_argument("row-correction builder input is incomplete");
    }

    using Clock = std::chrono::steady_clock;
    BuildResult result;
    auto start = Clock::now();
    std::vector<float> values(static_cast<std::size_t>(manifest.rows));
    for (int cluster = 0; cluster < manifest.nlist; ++cluster) {
        const std::int64_t begin = input.cluster_offsets[cluster];
        const std::int64_t end = input.cluster_offsets[cluster + 1];
        if (begin < 0 || end < begin ||
            static_cast<std::uint64_t>(end) > manifest.rows) {
            throw std::invalid_argument("row-correction cluster offsets are invalid");
        }
        const float* centroid = input.centroids +
            static_cast<std::size_t>(cluster) * manifest.dim;
        for (std::int64_t row = begin; row < end; ++row) {
            const std::uint8_t* code = input.codes +
                static_cast<std::size_t>(row) * manifest.code_width;
            double correction = 0.0;
            for (int sub = 0; sub < manifest.code_width; ++sub) {
                const float* codeword = input.codebook +
                    (static_cast<std::size_t>(sub) * manifest.codebook_size + code[sub]) *
                    manifest.subvector_dim;
                const float* centroid_sub = centroid + sub * manifest.subvector_dim;
                for (int d = 0; d < manifest.subvector_dim; ++d) {
                    correction += static_cast<double>(codeword[d]) * codeword[d] +
                        2.0 * static_cast<double>(centroid_sub[d]) * codeword[d];
                }
            }
            values[static_cast<std::size_t>(row)] = static_cast<float>(correction);
        }
    }
    result.compute_s = std::chrono::duration<double>(Clock::now() - start).count();

    start = Clock::now();
    const auto bounds = std::minmax_element(values.begin(), values.end());
    if (!std::isfinite(*bounds.first) || !std::isfinite(*bounds.second)) {
        throw std::runtime_error("row-correction values contain non-finite data");
    }
    const float offset = std::floor(*bounds.first);
    const float range = *bounds.second - offset;
    const float scale = range > 0.0f
        ? std::nextafter(range / 65534.0f, std::numeric_limits<float>::infinity())
        : 1.0f;
    if (!(scale > 0.0f) || !std::isfinite(scale)) {
        throw std::runtime_error("row-correction range cannot be quantized");
    }
    std::vector<std::uint16_t> encoded(values.size());
    for (std::size_t row = 0; row < values.size(); ++row) {
        const double quantized = std::ceil(
            (static_cast<double>(values[row]) - offset) / scale);
        if (quantized < 0.0 || quantized > 65534.0) {
            throw std::runtime_error("row-correction quantization overflow");
        }
        encoded[row] = static_cast<std::uint16_t>(quantized);
    }
    result.quantize_s = std::chrono::duration<double>(Clock::now() - start).count();

    start = Clock::now();
    const std::string payload_tmp = path(entry_dir, kPayloadName) + ".tmp";
    const std::string fp32_tmp = path(entry_dir, kFp32Name) + ".tmp";
    const std::string manifest_tmp = path(entry_dir, kManifestName) + ".tmp";
    U16Header header{};
    std::memcpy(header.magic, kU16Magic, sizeof(header.magic));
    header.version = kU16Version;
    header.header_bytes = sizeof(header);
    header.count = manifest.rows;
    header.scale = scale;
    header.offset = offset;
    {
        std::ofstream out(payload_tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to create row-correction payload");
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));
        out.write(reinterpret_cast<const char*>(encoded.data()),
                  static_cast<std::streamsize>(encoded.size() * sizeof(std::uint16_t)));
        out.close();
        if (!out) throw std::runtime_error("failed to write row-correction payload");
    }
    {
        std::ofstream out(fp32_tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to create row-correction FP32 audit file");
        const std::uint32_t rows = static_cast<std::uint32_t>(manifest.rows);
        const std::uint32_t dim = 1;
        out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        out.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(float)));
        out.close();
        if (!out) throw std::runtime_error("failed to write row-correction FP32 audit file");
    }
    write_manifest(manifest_tmp, manifest);
    commit(payload_tmp, path(entry_dir, kPayloadName));
    commit(fp32_tmp, path(entry_dir, kFp32Name));
    commit(manifest_tmp, path(entry_dir, kManifestName));
    result.commit_s = std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}

void invalidate(const std::string& entry_dir) {
    std::remove(path(entry_dir, kManifestName).c_str());
    std::remove(path(entry_dir, kPayloadName).c_str());
    std::remove(path(entry_dir, kFp32Name).c_str());
}

}  // namespace paraselect::row_correction
