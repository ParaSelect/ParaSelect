/**
 * CPU
 */

#include "bandwidth_profiler.h"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <omp.h>

namespace paraselect {
namespace cpu {

//
static double g_theoretical_bw_gb_s = 0.0;
static bool g_theoretical_bw_cached = false;

void set_global_theoretical_bandwidth(double bw_gb_s) {
    g_theoretical_bw_gb_s = bw_gb_s;
    g_theoretical_bw_cached = true;
}

static std::string get_cpu_model() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") == 0) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                return line.substr(pos + 2);
            }
        }
    }
    return "Unknown";
}

static double get_memory_frequency_mts() {
    FILE* fp = popen("dmidecode -t memory | grep -i 'speed' | head -1", "r");
    if (!fp) return 0.0;

    char buf[128];
    if (fgets(buf, sizeof(buf), fp)) {
        std::string line(buf);
        pclose(fp);

        size_t pos = line.find("MT/s");
        if (pos != std::string::npos) {
            size_t start = line.rfind(' ', pos);
            if (start != std::string::npos) {
                try {
                    return std::stod(line.substr(start + 1, pos - start - 1));
                } catch (...) {}
            }
        }
    }
    pclose(fp);
    return 0.0;
}

static int get_memory_channels() {
    FILE* fp = popen("dmidecode -t memory | grep -c 'Bank Locator'", "r");
    if (!fp) return 2;

    int channels = 2;
    if (fscanf(fp, "%d", &channels) != 1) {
        channels = 2;
    }
    pclose(fp);
    return channels > 0 ? channels : 2;
}

static double estimate_theoretical_bandwidth() {
    std::string cpu_model = get_cpu_model();
    double mem_freq = get_memory_frequency_mts();
    int channels = get_memory_channels();

    double bw_gb_s = 0.0;

    if (mem_freq > 0) {
        const int bus_width = 64;
        bw_gb_s = mem_freq * channels * bus_width / 8.0 / 1000.0;
    } else {
        if (cpu_model.find("Xeon") != std::string::npos) {
            if (cpu_model.find("Gold") != std::string::npos || cpu_model.find("Platinum") != std::string::npos) {
                bw_gb_s = 102.0;
            } else if (cpu_model.find("Silver") != std::string::npos) {
                bw_gb_s = 76.8;
            } else {
                bw_gb_s = 51.2;
            }
        } else if (cpu_model.find("AMD") != std::string::npos) {
            if (cpu_model.find("EPYC") != std::string::npos) {
                bw_gb_s = 102.0;
            } else {
                bw_gb_s = 51.2;
            }
        } else {
            bw_gb_s = 51.2;
        }
    }

    std::cout << "[BandwidthEst] CPU: " << cpu_model << std::endl;
    if (mem_freq > 0) {
        std::cout << "[BandwidthEst] Memory: " << mem_freq << " MT/s, " << channels << " channels" << std::endl;
    }
    std::cout << "[BandwidthEst] Estimated theoretical bandwidth: " << bw_gb_s << " GB/s" << std::endl;

    return bw_gb_s;
}

double get_global_theoretical_bandwidth() {
    return g_theoretical_bw_gb_s;
}

void BandwidthStats::print(const std::string& label) const {
    std::cout << "\n" << std::string(50, '-') << std::endl;
    if (!label.empty()) {
        std::cout << "[" << label << "]" << std::endl;
    }
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  :         " << elapsed_seconds << " s" << std::endl;
    std::cout << "  :         " << (bytes_read / 1e9) << " GB" << std::endl;
    std::cout << "  :         " << (bytes_written / 1e9) << " GB" << std::endl;
    std::cout << "  :         " << (total_bytes / 1e9) << " GB" << std::endl;
    std::cout << "  :         " << bandwidth_gb_s << " GB/s" << std::endl;
    if (theoretical_bw_gb_s > 0) {
        std::cout << "  :         " << theoretical_bw_gb_s << " GB/s" << std::endl;
        std::cout << "  :       " << utilization_percent << " %" << std::endl;
    }
    if (num_operations > 0) {
        std::cout << "  :         " << num_operations << std::endl;
        std::cout << "  :   " << bytes_per_operation << " bytes" << std::endl;
        std::cout << "  :       " << std::scientific << (num_operations / elapsed_seconds) << std::endl;
    }
    std::cout << std::string(50, '-') << std::endl;
}

BandwidthProfiler::BandwidthProfiler()
    : bytes_read_(0)
    , bytes_written_(0)
    , num_operations_(0)
    , theoretical_bw_gb_s_(0.0)
    , running_(false) {}

void BandwidthProfiler::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
    bytes_read_ = 0;
    bytes_written_ = 0;
    num_operations_ = 0;
    running_ = true;
}

BandwidthStats BandwidthProfiler::stop() {
    auto end_time = std::chrono::high_resolution_clock::now();
    running_ = false;

    BandwidthStats stats;
    stats.elapsed_seconds = std::chrono::duration<double>(end_time - start_time_).count();
    stats.bytes_read = bytes_read_;
    stats.bytes_written = bytes_written_;
    stats.total_bytes = bytes_read_ + bytes_written_;
    stats.bandwidth_gb_s = stats.total_bytes / (stats.elapsed_seconds * 1e9);
    stats.theoretical_bw_gb_s = theoretical_bw_gb_s_ > 0 ? theoretical_bw_gb_s_ : g_theoretical_bw_gb_s;
    stats.utilization_percent = (stats.bandwidth_gb_s / stats.theoretical_bw_gb_s) * 100.0;
    stats.num_operations = num_operations_;
    stats.bytes_per_operation = num_operations_ > 0 ? (double)stats.total_bytes / num_operations_ : 0;

    return stats;
}

void BandwidthProfiler::add_read(uint64_t bytes) {
    bytes_read_ += bytes;
}

void BandwidthProfiler::add_write(uint64_t bytes) {
    bytes_written_ += bytes;
}

void BandwidthProfiler::add_operation(uint64_t count) {
    num_operations_ += count;
}

void BandwidthProfiler::set_theoretical_bandwidth(double bw_gb_s) {
    theoretical_bw_gb_s_ = bw_gb_s;
}

BandwidthStats BandwidthProfiler::get_current_stats() const {
    BandwidthStats stats;
    if (running_) {
        auto current_time = std::chrono::high_resolution_clock::now();
        stats.elapsed_seconds = std::chrono::duration<double>(current_time - start_time_).count();
    } else {
        stats.elapsed_seconds = 0;
    }
    stats.bytes_read = bytes_read_;
    stats.bytes_written = bytes_written_;
    stats.total_bytes = bytes_read_ + bytes_written_;
    stats.bandwidth_gb_s = stats.elapsed_seconds > 0 ? stats.total_bytes / (stats.elapsed_seconds * 1e9) : 0;
    stats.theoretical_bw_gb_s = theoretical_bw_gb_s_ > 0 ? theoretical_bw_gb_s_ : g_theoretical_bw_gb_s;
    stats.utilization_percent = stats.theoretical_bw_gb_s > 0 ? (stats.bandwidth_gb_s / stats.theoretical_bw_gb_s) * 100.0 : 0;
    stats.num_operations = num_operations_;
    stats.bytes_per_operation = num_operations_ > 0 ? (double)stats.total_bytes / num_operations_ : 0;
    return stats;
}

ScopedBandwidthProfiler::ScopedBandwidthProfiler(const std::string& name)
    : name_(name) {
    profiler_.start();
}

ScopedBandwidthProfiler::~ScopedBandwidthProfiler() {
    auto stats = profiler_.stop();
    stats.print(name_);
}

/**
 * STREAM-like
 */
double measure_memory_bandwidth_stream() {
    if (!g_theoretical_bw_cached) {
        double estimated_bw = estimate_theoretical_bandwidth();
        if (estimated_bw > 0) {
            g_theoretical_bw_cached = true;
            return estimated_bw;
        }
    }

    const size_t array_size = 100000000; // 100M elements
    const int ntimes = 10;

    size_t bytes = array_size * sizeof(double);
    double* a = (double*)aligned_alloc(64, bytes);
    double* b = (double*)aligned_alloc(64, bytes);
    double* c = (double*)aligned_alloc(64, bytes);

    //
    #pragma omp parallel for
    for (size_t i = 0; i < array_size; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }

    double scalar = 3.0;

    //
    #pragma omp parallel for
    for (size_t i = 0; i < array_size; i++) {
        a[i] = b[i];
    }

    // Triad: a[i] = b[i] + scalar * c[i] ()
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < ntimes; t++) {
        #pragma omp parallel for
        for (size_t i = 0; i < array_size; i++) {
            a[i] = b[i] + scalar * c[i];
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    double time = std::chrono::duration<double>(t2 - t1).count();
    double bandwidth = (3.0 * bytes * ntimes) / (time * 1e9); // 2+1

    free(a);
    free(b);
    free(c);

    //
    set_global_theoretical_bandwidth(bandwidth);

    return bandwidth;
}

} // namespace cpu
} // namespace paraselect
