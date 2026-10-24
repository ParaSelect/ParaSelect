#ifndef PARASELECT_DATASET_COMMON_DATASET_SOURCE_H
#define PARASELECT_DATASET_COMMON_DATASET_SOURCE_H
#include "dataset_spec.h"
#include "binary_format.h"
#include "vector_buffer.h"
#include <string>
#include <memory>
#include <vector>
namespace paraselect::dataset {
struct DatasetSourceImpl;
struct GroundTruth { size_t n=0; int k=0; std::vector<int> ids; std::vector<float> distances; };
class DatasetSource {
public:
    explicit DatasetSource(DatasetSpec spec);
    const DatasetSpec& spec() const { return spec_; }
    const std::string& base_path() const { return base_path_; }
    const std::string& query_path() const { return query_path_; }
    const std::string& groundtruth_index_path() const { return groundtruth_index_path_; }
    const std::string& groundtruth_distance_path() const { return groundtruth_distance_path_; }
    bool base_has_signed_byte_encoding() const;
    DatasetInfo base_info() const;
    DatasetInfo query_info() const;
    VectorBuffer load_base(size_t limit=0) const;
    VectorBuffer load_queries(size_t limit=0) const;
    GroundTruth load_groundtruth(size_t query_limit=0) const;
    size_t read_base_range(size_t begin, size_t count, void* output) const;
    VectorBuffer load_base_rows(const std::vector<uint32_t>& ids, size_t begin = 0, size_t count = 0) const;
private:
    VectorBuffer load_vectors(const std::string&, size_t) const;
    DatasetInfo vector_info(const std::string& path) const;
    DatasetSpec spec_;
    std::shared_ptr<const DatasetSourceImpl> impl_;
    std::string base_path_;
    std::string query_path_;
    std::string groundtruth_index_path_;
    std::string groundtruth_distance_path_;
};
DatasetSource open_dataset(const DatasetSpec& spec);
DatasetSource open_dataset(const std::string& name, const std::string& size_label);
}
#endif
