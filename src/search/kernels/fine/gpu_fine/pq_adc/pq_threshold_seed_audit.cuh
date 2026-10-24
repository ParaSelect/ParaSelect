#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>

namespace paraselect::search::detail {
// Host-only diagnostic, called after the seed stream has synchronized.
// Sums of per-query hashes are independent of how queries are batched.
inline void audit_threshold_seed(
    const idx_t* probes, const int* seed_idx,
    const float* seed_dist, int nq, int np, int k) {
    std::vector<float> d((size_t)nq * k);
    std::vector<idx_t> p((size_t)nq * np);
    std::vector<int> ids((size_t)nq * k);
    CUDA_CHECK(cudaMemcpy(p.data(), probes, p.size()*sizeof(idx_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ids.data(), seed_idx, ids.size()*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(d.data(), seed_dist, d.size()*sizeof(float), cudaMemcpyDeviceToHost));
    auto mix = [](uint64_t h, uint64_t v) { return (h ^ v) * UINT64_C(1099511628211); };
    auto bits = [](float x) { uint32_t v; std::memcpy(&v, &x, sizeof(v)); return v; };
    uint64_t probe_hash = 0, result_hash = 0;
    unsigned unsorted = 0, invalid = 0;
    double cutoff_sum = 0;
    for (int i=0; i<nq; ++i) {
        uint64_t h = UINT64_C(14695981039346656037);
        for (int j=0; j<np; ++j) h=mix(h,static_cast<uint64_t>(p[(size_t)i*np+j]));
        probe_hash += h;
        bool bad_order=false;
        for (int j=0; j<k; ++j) {
            const size_t at=(size_t)i*k+j;
            invalid += ids[at]<0;
            bad_order |= j>0 && d[at]<d[at-1];
            h=mix(mix(h,static_cast<uint64_t>(ids[at])),bits(d[at]));
        }
        result_hash += h;
        unsorted += bad_order;
        cutoff_sum += d[(size_t)i*k+k-1];
    }
    std::fprintf(stderr,"[PQ_SEED_AUDIT] queries=%d cutoff_sum=%.17g probe_hash=%llu result_hash=%llu unsorted_queries=%u invalid_ids=%u\n",
        nq,cutoff_sum,(unsigned long long)probe_hash,(unsigned long long)result_hash,unsorted,invalid);
}
}
