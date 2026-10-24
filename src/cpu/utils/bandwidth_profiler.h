/**
 * CPU
 *
 * CPU
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <string>

namespace paraselect {
namespace cpu {

/**
 *
 */
struct BandwidthStats {
    double elapsed_seconds;          //
    uint64_t bytes_read;             //
    uint64_t bytes_written;          //
    uint64_t total_bytes;            //
    double bandwidth_gb_s;           //  GB/s
    double theoretical_bw_gb_s;      //  GB/s()
    double utilization_percent;      //  %
    uint64_t num_operations;         // ()
    double bytes_per_operation;      //

    void print(const std::string& label = "") const;
};

/**
 *  -
 */
class BandwidthProfiler {
public:
    BandwidthProfiler();

    //
    void start();

    //
    BandwidthStats stop();

    //
    void add_read(uint64_t bytes);

    //
    void add_write(uint64_t bytes);

    //
    void add_operation(uint64_t count = 1);

    // ()
    void set_theoretical_bandwidth(double bw_gb_s);

    // ()
    BandwidthStats get_current_stats() const;

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    uint64_t bytes_read_;
    uint64_t bytes_written_;
    uint64_t num_operations_;
    double theoretical_bw_gb_s_;
    bool running_;
};

/**
 *
 */
class ScopedBandwidthProfiler {
public:
    explicit ScopedBandwidthProfiler(const std::string& name);
    ~ScopedBandwidthProfiler();

    void add_read(uint64_t bytes) { profiler_.add_read(bytes); }
    void add_write(uint64_t bytes) { profiler_.add_write(bytes); }
    void add_operation(uint64_t count = 1) { profiler_.add_operation(count); }

private:
    std::string name_;
    BandwidthProfiler profiler_;
};

/**
 * ()
 */
void set_global_theoretical_bandwidth(double bw_gb_s);
double get_global_theoretical_bandwidth();

/**
 * (STREAM-like)
 */
double measure_memory_bandwidth_stream();

} // namespace cpu
} // namespace paraselect

//
#define IVF_BANDWIDTH_PROFILE(name) paraselect::cpu::ScopedBandwidthProfiler _bw_prof(name)
#define IVF_BANDWIDTH_ADD_READ(bytes) _bw_prof.add_read(bytes)
#define IVF_BANDWIDTH_ADD_WRITE(bytes) _bw_prof.add_write(bytes)
#define IVF_BANDWIDTH_ADD_OP(count) _bw_prof.add_operation(count)
