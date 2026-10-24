#include "search/ivf_common/ivf_probe_schedule.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct Shape {
    int nlist;
    int batch;
    int nprobe;
};

std::vector<idx_t> make_probes(const Shape& shape) {
    std::vector<idx_t> probes((size_t)shape.batch * shape.nprobe);
    for (int query = 0; query < shape.batch; ++query) {
        for (int rank = 0; rank < shape.nprobe; ++rank) {
            probes[(size_t)query * shape.nprobe + rank] =
                (static_cast<int64_t>(query) * 104729 +
                 static_cast<int64_t>(rank) * 8191) % shape.nlist;
        }
    }
    return probes;
}

double run_strategy(
    const Shape& shape,
    paraselect::search::IVFProbeScheduleStrategy strategy,
    int repeats, int entry_width = 8) {
    const std::vector<idx_t> probes = make_probes(shape);
    std::vector<int> list_counts((size_t)shape.nlist);
    for (int cluster = 0; cluster < shape.nlist; ++cluster) {
        list_counts[cluster] = cluster % 17 + 1;
    }
    // Force aggregate counts beyond uint32 without allocating vector data.
    list_counts[0] = 1 << 30;
    idx_t* device_probes = nullptr;
    int* device_list_counts = nullptr;
    check_cuda(cudaMalloc(&device_probes, probes.size() * sizeof(idx_t)),
               "allocate schedule test probes");
    check_cuda(cudaMalloc(
                   &device_list_counts, list_counts.size() * sizeof(int)),
               "allocate schedule test list counts");
    check_cuda(cudaMemcpy(
                   device_probes, probes.data(), probes.size() * sizeof(idx_t),
                   cudaMemcpyHostToDevice),
               "copy schedule test probes");
    check_cuda(cudaMemcpy(
                   device_list_counts, list_counts.data(),
                   list_counts.size() * sizeof(int), cudaMemcpyHostToDevice),
               "copy schedule test list counts");

    double elapsed_ms = 0.0;
    {
        paraselect::search::IVFProbeScheduleBuilder builder(
            shape.nlist, static_cast<int>(probes.size()), device_list_counts,
            strategy);
        const paraselect::search::IVFDeviceProbeView view{
            device_probes, shape.batch, shape.nprobe};
        (void)builder.build(view, shape.nprobe, nullptr);
        const auto begin = std::chrono::steady_clock::now();
        paraselect::search::IVFProbeScheduleResult result;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            result = builder.build(view, shape.nprobe, nullptr);
        }
        elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count() / repeats;

        unsigned long long expected_scanned = 0;
        std::vector<int> expected_references(shape.nlist, 0);
        for (idx_t cluster : probes) {
            expected_scanned += list_counts[cluster];
            ++expected_references[cluster];
        }
        if (result.scanned_vectors != expected_scanned ||
            result.strategy != strategy || result.view.max_list_queries !=
                *std::max_element(expected_references.begin(), expected_references.end())) {
            throw std::runtime_error("IVF probe schedule aggregate mismatch");
        }

        const int refs = static_cast<int>(probes.size());
        const int entries = result.view.entry_count;
        std::vector<int> clusters((size_t)entries);
        std::vector<int> starts((size_t)entries);
        std::vector<int> counts((size_t)entries);
        std::vector<int> queries((size_t)refs);
        std::vector<int> ranks((size_t)refs);
        check_cuda(cudaMemcpy(
                       clusters.data(), result.view.clusters,
                       clusters.size() * sizeof(int), cudaMemcpyDeviceToHost),
                   "copy schedule test clusters");
        check_cuda(cudaMemcpy(
                       starts.data(), result.view.starts,
                       starts.size() * sizeof(int), cudaMemcpyDeviceToHost),
                   "copy schedule test starts");
        check_cuda(cudaMemcpy(
                       counts.data(), result.view.counts,
                       counts.size() * sizeof(int), cudaMemcpyDeviceToHost),
                   "copy schedule test counts");
        check_cuda(cudaMemcpy(
                       queries.data(), result.view.queries,
                       queries.size() * sizeof(int), cudaMemcpyDeviceToHost),
                   "copy schedule test queries");
        check_cuda(cudaMemcpy(
                       ranks.data(), result.view.probe_ranks,
                       ranks.size() * sizeof(int), cudaMemcpyDeviceToHost),
                   "copy schedule test probe ranks");
        std::vector<uint8_t> seen((size_t)refs, 0);
        int visited = 0;
        for (int entry = 0; entry < entries; ++entry) {
            if (counts[entry] <= 0 || counts[entry] > entry_width || starts[entry] < 0 ||
                starts[entry] + counts[entry] > refs) {
                throw std::runtime_error("invalid IVF probe schedule entry");
            }
            for (int offset = 0; offset < counts[entry]; ++offset) {
                const int ref = starts[entry] + offset;
                const int query = queries[ref];
                const int rank = ranks[ref];
                if (query < 0 || query >= shape.batch || rank < 0 ||
                    rank >= shape.nprobe ||
                    probes[(size_t)query * shape.nprobe + rank] !=
                        clusters[entry]) {
                    throw std::runtime_error("invalid IVF probe schedule payload");
                }
                const int source = query * shape.nprobe + rank;
                if (seen[source] != 0) {
                    throw std::runtime_error("duplicate IVF probe schedule reference");
                }
                seen[source] = 1;
                ++visited;
            }
        }
        if (visited != refs ||
            std::find(seen.begin(), seen.end(), uint8_t{0}) != seen.end()) {
            throw std::runtime_error("missing IVF probe schedule reference");
        }
        // A reused workspace must not retain the previous batch's maximum.
        const auto single = builder.build({device_probes, 1, shape.nprobe}, 1, nullptr);
        if (single.view.max_list_queries != 1)
            throw std::runtime_error("stale IVF list query maximum");
    }

    check_cuda(cudaFree(device_probes), "release schedule test probes");
    check_cuda(cudaFree(device_list_counts),
               "release schedule test list counts");
    return elapsed_ms;
}

void test_padded_probe_schedule() {
    constexpr int nlist = 16;
    constexpr int batch = 2;
    constexpr int stride = 6;
    const std::vector<idx_t> probes = {
        1, 2, 3, -1, -1, -1,
        4, 5, -1, -1, -1, -1,
    };
    const std::vector<int> list_counts((size_t)nlist, 10);
    idx_t* device_probes = nullptr;
    int* device_list_counts = nullptr;
    check_cuda(cudaMalloc(&device_probes, probes.size() * sizeof(idx_t)),
               "allocate padded schedule probes");
    check_cuda(cudaMalloc(
                   &device_list_counts, list_counts.size() * sizeof(int)),
               "allocate padded schedule list counts");
    check_cuda(cudaMemcpy(
                   device_probes, probes.data(), probes.size() * sizeof(idx_t),
                   cudaMemcpyHostToDevice),
               "copy padded schedule probes");
    check_cuda(cudaMemcpy(
                   device_list_counts, list_counts.data(),
                   list_counts.size() * sizeof(int), cudaMemcpyHostToDevice),
               "copy padded schedule list counts");

    for (const auto strategy : {
             paraselect::search::IVFProbeScheduleStrategy::Dense,
             paraselect::search::IVFProbeScheduleStrategy::Sparse}) {
        paraselect::search::IVFProbeScheduleBuilder builder(
            nlist, static_cast<int>(probes.size()), device_list_counts, strategy);
        bool rejected_without_padding = false;
        try {
            (void)builder.build(
                {device_probes, batch, stride}, stride, nullptr, false);
        } catch (const std::out_of_range&) {
            rejected_without_padding = true;
        }
        if (!rejected_without_padding) {
            throw std::runtime_error("padded IVF probes bypassed the strict schedule gate");
        }
        const auto result = builder.build(
            {device_probes, batch, stride}, stride, nullptr, true);
        if (result.reference_count != 5 || result.scanned_vectors != 50 ||
            result.view.max_list_queries != 1) {
            throw std::runtime_error("padded IVF probe schedule aggregate mismatch");
        }
        const auto empty = builder.build({device_probes + 3, 1, stride}, 3, nullptr, true);
        if (empty.reference_count != 0 || empty.view.entry_count != 0 ||
            empty.view.max_list_queries != 0)
            throw std::runtime_error("empty IVF schedule retained a query maximum");
    }

    check_cuda(cudaFree(device_probes), "release padded schedule probes");
    check_cuda(cudaFree(device_list_counts),
               "release padded schedule list counts");
}

void test_entry_storage_bound() {
    int* counts = nullptr;
    check_cuda(cudaMalloc(&counts, 128 * sizeof(int)), "allocate list counts");
    size_t before = 0, after = 0, total = 0;
    check_cuda(cudaMemGetInfo(&before, &total), "memory before schedule");
    {
        paraselect::search::IVFProbeScheduleBuilder builder(
            128, 1 << 20, counts, paraselect::search::IVFProbeScheduleStrategy::Dense);
        check_cuda(cudaMemGetInfo(&after, &total), "memory after schedule");
        // 8 MiB for refs plus roughly 2.5 MiB for entries, with rounding room.
        // Reserving five entry arrays per reference instead exceeds 28 MiB.
        if (before > after && before - after > (16ULL << 20))
            throw std::runtime_error("schedule entry storage scales as references");
    }
    check_cuda(cudaFree(counts), "free list counts");
}

}  // namespace

int main() {
    try {
        for(int width : {16,32,64,10000}) {
            setenv("PARASELECT_RPQ_PROBE_ENTRY_WIDTH",std::to_string(width).c_str(),1);
            for(auto strategy : {paraselect::search::IVFProbeScheduleStrategy::Dense,
                                 paraselect::search::IVFProbeScheduleStrategy::Sparse})
                run_strategy({1,129,1},strategy,1,width);
        }
        setenv("PARASELECT_RPQ_PROBE_ENTRY_WIDTH","8",1);
        test_entry_storage_bound();
        test_padded_probe_schedule();
        const std::vector<Shape> shapes = {
            {1, 2048, 1},
            {32, 32, 32},
            {1031, 3, 67},
            {524288, 2048, 512},
            {4096, 8, 128},
            {131072, 1, 4},
            {131072, 1, 64},
            {131072, 8, 64},
            {131072, 8, 224},
            {131072, 32, 256},
            {131072, 128, 256},
            {131072, 256, 384},
            {131072, 2048, 512},
            {524288, 1, 4},
            {524288, 1, 64},
            {524288, 8, 64},
            {524288, 8, 224},
            {524288, 32, 256},
            {524288, 128, 256},
            {524288, 256, 384},
        };
        std::cout << "nlist,batch,nprobe,refs,dense_ms,sparse_ms\n";
        for (const Shape& shape : shapes) {
            const int refs = shape.batch * shape.nprobe;
            const int repeats = refs <= 2048 ? 100 : (refs <= 100000 ? 30 : 5);
            const double dense = run_strategy(
                shape, paraselect::search::IVFProbeScheduleStrategy::Dense,
                repeats);
            const double sparse = run_strategy(
                shape, paraselect::search::IVFProbeScheduleStrategy::Sparse,
                repeats);
            std::cout << shape.nlist << ',' << shape.batch << ',' << shape.nprobe
                      << ',' << refs << ',' << dense << ',' << sparse << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[test_ivf_probe_schedule] " << error.what() << '\n';
        return 1;
    }
}
