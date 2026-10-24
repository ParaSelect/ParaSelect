#pragma once

#include "utils/pch.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace paraselect {

class SearchBenchmark {
public:
    //  recall: ground truth
    // @param k recall@k  k
    // @param gt ground truth  [n_query, k_gt], top-k
    // @param n_gt ground truth  k ( gt )
    // @param result  [n_query, k_result]
    // @param n_query
    // @param k_result  k ( result )
    static float computeRecall(
        const idx_t* gt, int n_gt,
        const idx_t* result, int n_query, int k_result);

    //  ground truth  neighbors  distances
    // @param dataset_name ( "sift100M", "deep100M", "text100M")
    // @param gt_data :ground truth ( delete[] )
    // @param gt_distances :ground truth ( delete[] )
    // @param n_queries :
    // @param k_gt :ground truth  k
    // @return
    static bool loadGroundTruth(
        const std::string& dataset_name,
        idx_t*& gt_data, float*& gt_distances, int& n_queries, int& k_gt);

    static bool appendCsvRow(
        const std::string& path,
        const std::vector<std::string>& columns,
        const std::unordered_map<std::string, double>& numeric_fields,
        const std::unordered_map<std::string, std::string>& string_fields = {});
    static bool resetCsv(const std::string& path);

    // ():ANN_BENCHMARKS_DATA_DIR
    static std::string defaultAnnDataDir();

    static int parseRecallMethod(const char* value);
    static int recallMethodFromEnv(const char* env_name = "PARASELECT_RECALL_METHOD");

};

} // namespace paraselect
