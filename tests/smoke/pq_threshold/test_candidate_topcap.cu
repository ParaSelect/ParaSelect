#include "pq_candidate_topcap.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <vector>

static void check(cudaError_t e) { if (e != cudaSuccess) std::abort(); }

int main() {
    constexpr int capacity = 512;
    constexpr int queries = 3;
    int *d_ids = nullptr, *d_counts = nullptr;
    float *d_dist = nullptr;
    check(cudaMalloc(&d_ids, sizeof(int) * capacity * queries));
    check(cudaMalloc(&d_dist, sizeof(float) * capacity * queries));
    check(cudaMalloc(&d_counts, sizeof(int) * queries));
    std::vector<int> ids(capacity * queries), counts{512, 317, 0};
    std::vector<float> dist(capacity * queries);
    for (int q = 0; q < queries; ++q) for (int i = 0; i < capacity; ++i) {
        ids[q * capacity + i] = i;
        dist[q * capacity + i] = float(capacity - i) + float(q) * 0.001f;
    }
    check(cudaMemcpy(d_ids, ids.data(), sizeof(int) * ids.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_dist, dist.data(), sizeof(float) * dist.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_counts, counts.data(), sizeof(int) * counts.size(), cudaMemcpyHostToDevice));
    paraselect::search::pq_candidate_topcap(d_ids, d_dist, d_counts, queries, capacity, 64, 0);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(ids.data(), d_ids, sizeof(int) * ids.size(), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(dist.data(), d_dist, sizeof(float) * dist.size(), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(counts.data(), d_counts, sizeof(int) * counts.size(), cudaMemcpyDeviceToHost));
    if (counts != std::vector<int>({64, 64, 0})) return 1;
    for (int q = 0; q < 2; ++q) for (int i = 0; i < 64; ++i) {
        const int expected = (q == 0 ? capacity : 317) - 1 - i;
        if (ids[q * capacity + i] != expected) {
            std::fprintf(stderr, "q=%d i=%d got=%d expected=%d dist=%f\\n", q, i,
                         ids[q * capacity + i], expected,
                         dist[q * capacity + i]);
            return 2;
        }
    }
    std::puts("PASS top-cap CUB ordering and counts");
    cudaFree(d_ids); cudaFree(d_dist); cudaFree(d_counts);
    return 0;
}
