#ifndef PARASELECT_CPU_FINE_EXECUTOR_H
#define PARASELECT_CPU_FINE_EXECUTOR_H

#include "search/kernels/fine/cpu_fine/executor/cpu_fine_batch_view.h"
#include "search/kernels/fine/cpu_fine/schedulers/thread_stats.h"

#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>

namespace paraselect {
namespace cpu_fine {

using CpuFineBatchRunFn = void (*)(void*, const CpuFineBatchView&);

struct CpuFineBatchJob {
    void* owner = nullptr;
    CpuFineBatchRunFn run = nullptr;
    CpuFineBatchView view{};
};

class CpuFineExecutor {
public:
    explicit CpuFineExecutor(bool async_worker = false);
    CpuFineExecutor(const CpuFineExecutor&) = delete;
    CpuFineExecutor& operator=(const CpuFineExecutor&) = delete;
    ~CpuFineExecutor() noexcept;

    void submit(const CpuFineBatchJob& job);
    void wait();
    paraselect::cpu_fine::thread_stats_detail::FineTimingSnapshot latest_timing() const;
    double latest_stage_ms() const;

private:
    void worker_loop();
    void run_job(const CpuFineBatchJob& job);
    void stop_worker_noexcept() noexcept;

    bool async_worker_ = false;
    std::thread worker_;
    std::mutex mu_;
    std::condition_variable cv_;
    CpuFineBatchJob pending_job_{};
    bool has_job_ = false;
    bool job_done_ = true;
    bool stop_ = false;
    std::exception_ptr error_;
    paraselect::cpu_fine::thread_stats_detail::FineTimingSnapshot latest_timing_{};
    double latest_stage_ms_ = 0.0;
};

}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_CPU_FINE_EXECUTOR_H
