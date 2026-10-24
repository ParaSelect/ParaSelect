#ifndef PARASELECT_INDEX_ROW_CORRECTION_H
#define PARASELECT_INDEX_ROW_CORRECTION_H

#include "common/row_correction.h"
#include "dataset/common/mapped_array.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace paraselect::row_correction {

inline constexpr std::uint32_t kManifestVersion = 3;

struct Manifest {
    std::string dataset_name;
    std::string parent_manifest_json;  // Serialized authoritative PQ manifest, not a tag.
    std::string cluster_layout_id;
    std::string codec;
    std::string row_layout;
    std::uint64_t rows = 0;
    int dim = 0;
    int nlist = 0;
    int code_width = 0;
    int codebook_size = 0;
    int subvector_dim = 0;
};

struct PQBuildInput {
    const float* centroids = nullptr;
    const std::int64_t* cluster_offsets = nullptr;
    const std::uint8_t* codes = nullptr;
    const float* codebook = nullptr;
};

struct BuildResult {
    bool cache_hit = false;
    double compute_s = 0.0;
    double quantize_s = 0.0;
    double commit_s = 0.0;
};

class Asset {
public:
    static Asset open(const std::string& entry_dir, const Manifest& expected);

    const StorageView& view() const noexcept { return view_; }
    bool empty() const noexcept { return view_.empty(); }

private:
    std::shared_ptr<dataset::ReadOnlyMapping> mapping_;
    StorageView view_;
};

bool manifest_matches(const std::string& entry_dir, const Manifest& expected);
BuildResult build_pq(
    const std::string& entry_dir,
    const Manifest& manifest,
    const PQBuildInput& input);
void invalidate(const std::string& entry_dir);

}  // namespace paraselect::row_correction

#endif  // PARASELECT_INDEX_ROW_CORRECTION_H
