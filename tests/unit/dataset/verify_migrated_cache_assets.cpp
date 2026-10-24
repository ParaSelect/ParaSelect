#include "dataset/cache/cluster_cache.h"
#include "dataset/cache/turboquant_cache.h"
#include "dataset/cache/pca_cache.h"
#include "dataset/cache/residual_pq_cache.h"
#include "dataset/cache/cache_manifest.h"
#include "index/row_correction/row_correction.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace paraselect::dataset;
namespace fs=std::filesystem;
using Json=cache_manifest::Json;
int main(int argc,char** argv) try {
    if(argc!=2)throw std::runtime_error("usage: verify_migrated_cache_assets CACHE_ROOT");
    const fs::path root(argv[1]);int clusters=0,pqs=0,corrections=0,tqs=0,pcas=0;
    Json skipped=Json::array();
    for(const auto& file:fs::recursive_directory_iterator(root)) {
        if(file.path().filename()!="manifest.json")continue;
        const auto entry=file.path().parent_path();
        const auto manifest=Json::parse(std::ifstream(file.path()));
        const auto type=manifest.at("asset_type").get<std::string>();
        const auto dataset=manifest.at("dataset").get<std::string>();
        if(type=="cluster") {
            ClusterCache cache(root.string());
            const ClusterCacheSpec spec{manifest.at("n_vectors").get<int>(),
                manifest.at("nlist").get<int>(),manifest.at("dim").get<int>(),
                manifest.at("build").dump(),manifest.at("cluster_layout_id").get<std::string>()};
            const auto asset=cache.try_map(dataset,spec);
            if(!asset.hit||fs::path(asset.entry_dir)!=entry)
                throw std::runtime_error("cluster did not load from its authoritative manifest: "+entry.string());
            ++clusters;
        } else if(type=="pq"||type=="residual_pq") {
            PQCacheRequest spec;
            spec.n_vectors=manifest.at("n_vectors");spec.n_lists=manifest.at("nlist");
            spec.dim=manifest.at("dim");spec.cluster_layout_id=manifest.at("cluster_layout_id");
            spec.pq_m=manifest.at("m");spec.pq_k=manifest.at("k");spec.pq_iters=manifest.at("pq_iters");
            spec.pq_type=manifest.at("pq_type");spec.codes_layout=manifest.at("codes_layout");
            spec.trainer=manifest.at("trainer");spec.training_rows=manifest.at("training").at("rows");
            spec.training_sampling=manifest.at("training").at("sampling");
            if(manifest.contains("transform")) {
                spec.opq_iters=manifest.at("transform").at("iters");
                spec.opq_train_rows=manifest.at("transform").at("training_rows");
                spec.opq_seed=manifest.at("transform").at("seed");
            }
            ResidualPQCache cache(entry.parent_path().string(),dataset);
            const auto asset=cache.try_map(spec);
            if(!asset.hit||fs::path(asset.entry_dir)!=entry)
                throw std::runtime_error("PQ did not load from its authoritative manifest: "+entry.string());
            ++pqs;
            if(fs::exists(entry/"row_correction.meta.json")) {
                const paraselect::row_correction::Manifest row{dataset,manifest.dump(),spec.cluster_layout_id,
                    "residual_pq",spec.codes_layout,static_cast<uint64_t>(spec.n_vectors),spec.dim,
                    spec.n_lists,spec.pq_m,spec.pq_k,spec.dsub()};
                const auto correction=paraselect::row_correction::Asset::open(entry.string(),row);
                if(correction.view().count!=spec.n_vectors)throw std::runtime_error("row correction row count mismatch");
                ++corrections;
            }
        } else if(type=="turboquant") {
            TurboQuantCacheRequest spec;
            spec.n_vectors=manifest.at("n_vectors");spec.n_lists=manifest.at("nlist");
            spec.dim=manifest.at("dim");spec.cluster_layout_id=manifest.at("cluster_layout_id");
            spec.rotation_seed=manifest.at("rotation_seed");spec.builder=manifest.at("builder");
            spec.codes_layout=manifest.at("codes_layout");
            TurboQuantCache cache(entry.parent_path().string(),dataset);
            const auto asset=cache.try_map(spec);
            if(!asset.hit||fs::path(asset.entry_dir)!=entry)
                throw std::runtime_error("TQ did not load from its authoritative manifest: "+entry.string());
            ++tqs;
        } else if(type=="pca") {
            PCACacheSpec spec;
            spec.n_vectors=manifest.at("n_vectors");spec.source_dim=manifest.at("source_dim");
            spec.dim=manifest.at("dim");spec.train_n=manifest.at("train_n");
            spec.cluster_layout_id=manifest.at("cluster_layout_id");spec.quantization=manifest.at("quantization");
            PCACache cache(entry.parent_path().string(),dataset);
            const auto asset=cache.try_map(spec);
            if(!asset.hit||fs::path(asset.entry_dir)!=entry)
                throw std::runtime_error("PCA did not load from its authoritative manifest: "+entry.string());
            ++pcas;
        } else {
            skipped.push_back({{"type",type},{"entry",entry.string()}});
        }
    }
    if(!clusters||!pqs)throw std::runtime_error("empty validation scope");
    std::cout<<Json{{"clusters",clusters},{"pq_assets",pqs},{"row_corrections",corrections},{"turboquant_assets",tqs},{"pca_assets",pcas},{"skipped",skipped}}.dump()<<'\n';
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
