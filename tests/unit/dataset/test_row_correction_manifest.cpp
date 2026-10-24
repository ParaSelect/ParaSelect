#include "index/row_correction/row_correction.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unistd.h>
using namespace paraselect::row_correction;
using Json = nlohmann::json;
static void check(bool ok) { if (!ok) throw std::runtime_error("row correction manifest contract"); }
int main() {
    char pattern[] = "/tmp/row-correction-manifest-XXXXXX";
    const std::filesystem::path root(mkdtemp(pattern));
    const Json parent = {{"schema","paraselect.cache"},{"schema_version",2},
        {"asset_type","residual_pq"},{"dataset","synthetic"},{"n_vectors",2},
        {"dim",2},{"nlist",1},{"m",1},{"k",2},{"dsub",2},
        {"cluster_layout_id","parent-a"},{"codes_layout","cluster-reordered-[N,M]"}};
    std::ofstream(root/"manifest.json") << parent.dump();
    Manifest m{"synthetic",parent.dump(),"parent-a","residual_pq",
        "cluster-reordered-[N,M]",2,2,1,1,2,2};
    const float centers[] = {0,0}, book[] = {1,2,3,4};
    const std::int64_t offsets[] = {0,2};
    const std::uint8_t codes[] = {0,1};
    build_pq(root.string(), m, {centers,offsets,codes,book});
    auto manifest = Json::parse(std::ifstream(root/"row_correction.meta.json"));
    check(!manifest.contains("parent_pq_tag"));
    check(manifest.at("parent_pq_manifest") == parent);
    check(manifest_matches(root.string(),m));
    auto legacy=manifest; legacy["parent_index_tag"]="obsolete";
    std::ofstream(root/"row_correction.meta.json") << legacy.dump();
    check(!manifest_matches(root.string(),m));
    std::ofstream(root/"row_correction.meta.json") << manifest.dump();
    check(manifest_matches(root.string(),m));
    auto incorrect=manifest; incorrect["dim"]=20;
    std::ofstream(root/"row_correction.meta.json") << incorrect.dump(2);
    check(!manifest_matches(root.string(),m));
    incorrect=manifest; incorrect["parent_pq_manifest"]["cluster_layout_id"]="other";
    std::ofstream(root/"row_correction.meta.json") << incorrect.dump(2);
    check(!manifest_matches(root.string(),m));
    incorrect=manifest; incorrect["parent_pq_manifest"]["unexpected_identity"]=true;
    std::ofstream(root/"row_correction.meta.json") << incorrect.dump();
    check(!manifest_matches(root.string(),m));
    std::ofstream(root/"row_correction.meta.json") << manifest.dump(2);
    auto changed=parent; changed["m"]=2;
    std::ofstream(root/"manifest.json") << changed.dump();
    check(!manifest_matches(root.string(),m));
    std::ofstream(root/"manifest.json") << parent.dump();
    check(Asset::open(root.string(),m).view().count==2);
    std::filesystem::remove_all(root);
}
