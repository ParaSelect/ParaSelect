#include "dataset/cache/cluster_cache.h"
#include "dataset/cache/cache_manifest.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unistd.h>
using namespace paraselect::dataset;
static void check(bool value) { if (!value) throw std::runtime_error("cluster manifest publication"); }
int main() {
    char pattern[]="/tmp/cluster-manifest-XXXXXX";
    const std::filesystem::path root(mkdtemp(pattern));
    ClusterCache cache(root.string());
    const ClusterCacheSpec spec{2,1,2,R"({"family":"synthetic","revision":1})"};
    const float centroids[]={0,0};
    const idx_t ids[]={0,1}, offsets[]={0,2};
    const int counts[]={2};
    cache.save("synthetic",spec,centroids,ids,counts,offsets,1);
    const auto entry=cache_manifest::find(root.string(),cache_manifest::identity("cluster","synthetic"));
    auto manifest=cache_manifest::Json::parse(std::ifstream(entry+"/manifest.json"));
    check(manifest.contains("cluster_layout_id"));
    const auto mapped=cache.try_map("synthetic",spec);
    check(mapped.hit && mapped.layout_id==manifest.at("cluster_layout_id"));
    const auto staged=root/".building-regression";
    std::filesystem::copy(entry,staged,std::filesystem::copy_options::recursive);
    std::filesystem::remove(staged/"manifest.json");
    std::filesystem::remove_all(entry);
    const auto published=cache.publish_staged(staged.string(),"synthetic",spec);
    check(!std::filesystem::exists(staged));
    check(cache.try_map("synthetic",spec).hit);
    check(cache_manifest::Json::parse(std::ifstream(published+"/manifest.json"))==manifest);
    auto missing=manifest; missing.erase("cluster_layout_id");
    std::ofstream(published+"/manifest.json") << missing.dump();
    check(!cache.try_map("synthetic",spec).hit);
    auto malformed=manifest; malformed["cluster_layout_id"]=123;
    std::ofstream(published+"/manifest.json") << malformed.dump();
    check(!cache.try_map("synthetic",spec).hit);
    std::ofstream(published+"/manifest.json") << manifest.dump();
    const auto broken=root/".building-broken";
    std::filesystem::copy(published,broken,std::filesystem::copy_options::recursive);
    std::filesystem::remove(broken/"manifest.json");
    std::filesystem::resize_file(broken/"reordered_indices.ibin",8);
    bool rejected=false;
    try { cache.publish_staged(broken.string(),"synthetic",spec); }
    catch (const std::runtime_error&) { rejected=true; }
    check(rejected && !std::filesystem::exists(broken/"manifest.json"));
    std::filesystem::remove_all(root);
}
