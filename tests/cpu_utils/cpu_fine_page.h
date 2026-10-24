/* ============================================================================
 * CPU  ()
 * ============================================================================ */
static void cpu_ground_truth(
    int n_query, int n_dim, int k, int n_probes,
    const float* h_query,
    const float* h_query_norm, //  qnorm,
    const float* h_vectors_rm,
    const std::vector<int>& h_pvoc,
    const std::vector<int>& h_pvc,
    const std::vector<int>& phys_pages,
    const std::vector<std::vector<int>>& query_to_clusters,
    std::vector<float>& gt_dist,
    std::vector<int>& gt_index
) {
    // 1.  Cluster  phys_pages
    int n_cluster = h_pvc.size();
    std::vector<int> cluster_page_offset(n_cluster, 0);
    int current_offset = 0;
    for (int c = 0; c < n_cluster; ++c) {
        cluster_page_offset[c] = current_offset;
        current_offset += (h_pvc[c] + 255) / 256;
    }

    // 2.  Query
    for (int q = 0; q < n_query; ++q) {
        const float* q_vec = h_query + q * n_dim;
        float qn = h_query_norm[q];
        float qn_sq = qn * qn;

        std::vector<std::pair<float, int>> candidates; // <distance, compact_idx>

        //  query  probe(cluster)
        for (int p = 0; p < n_probes; ++p) {
            int c = query_to_clusters[q][p];
            int vec_count = h_pvc[c];
            int compact_off = h_pvoc[c];
            int num_pages = (vec_count + 255) / 256;

            //  Cluster
            for (int epid = 0; epid < num_pages; ++epid) {
                int phys_page = phys_pages[cluster_page_offset[c] + epid];
                int vec_in_page = std::min(256, vec_count - epid * 256);

                //
                for (int v = 0; v < vec_in_page; ++v) {
                    const float* db_vec = h_vectors_rm + (phys_page * 256 + v) * n_dim;

                    float dot = 0.0f;
                    float vnorm_sq = 0.0f;
                    for (int d = 0; d < n_dim; ++d) {
                        float v_val = db_vec[d];
                        dot += q_vec[d] * v_val;
                        vnorm_sq += v_val * v_val;
                    }

                    //  Kernel
                    float l2 = qn_sq + vnorm_sq - 2.0f * dot;
                    int compact_idx = compact_off + epid * 256 + v;

                    candidates.push_back({l2, compact_idx});
                }
            }
        }

        // 3.  Top-K
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) {
                // ,
                if (std::abs(a.first - b.first) < 1e-5f) return a.second < b.second;
                return a.first < b.first;
            });

        // 4.
        for (int i = 0; i < k; ++i) {
            if (i < candidates.size()) {
                gt_dist[q * k + i] = candidates[i].first;
                gt_index[q * k + i] = candidates[i].second;
            } else {
                gt_dist[q * k + i] = paraselect::warpsort_utils::upper_bound<float>();
                gt_index[q * k + i] = -1;
            }
        }
    }
}
