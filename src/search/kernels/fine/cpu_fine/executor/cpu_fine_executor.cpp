#include "search/kernels/fine/cpu_fine/executor/cpu_fine_executor.h"


#include <stdexcept>
#include <chrono>

namespace paraselect {
namespace cpu_fine {

CpuFineExecutor::CpuFineExecutor(bool async_worker)
    : async_worker_(async_worker)
{
    if (async_worker_) {
        worker_ = std::thread([this]() { worker_loop(); });
    }
}

CpuFineExecutor::~CpuFineExecutor() noexcept {
    stop_worker_noexcept();
}

void CpuFineExecutor::submit(const CpuFineBatchJob& job) {
    if (!job.owner || !job.run) {
        throw std::invalid_argument("CpuFineExecutor::submit requires a valid owner and callback");
    }
    if (!async_worker_) {
        run_job(job);
        return;
    }

    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [this]() { return job_done_ && !has_job_; });
    pending_job_ = job;
    error_ = nullptr;
    job_done_ = false;
    has_job_ = true;
    lock.unlock();
    cv_.notify_one();
}

void CpuFineExecutor::wait() {
    if (!async_worker_) return;

    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [this]() { return job_done_; });
    auto error = error_;
    error_ = nullptr;
    lock.unlock();
    if (error) std::rethrow_exception(error);
}

void CpuFineExecutor::run_job(const CpuFineBatchJob& job) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    job.run(job.owner, job.view);
    const auto t1 = std::chrono::high_resolution_clock::now();
    latest_stage_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
    latest_timing_ = paraselect::cpu_fine::thread_stats_detail::get_latest_fine_timing();
}

paraselect::cpu_fine::thread_stats_detail::FineTimingSnapshot CpuFineExecutor::latest_timing() const {
    return latest_timing_;
}

double CpuFineExecutor::latest_stage_ms() const {
    return latest_stage_ms_;
}

void CpuFineExecutor::worker_loop() {
    for (;;) {
        CpuFineBatchJob job{};
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [this]() { return has_job_ || stop_; });
            if (stop_ && !has_job_) return;
            job = pending_job_;
            has_job_ = false;
        }

        std::exception_ptr error;
        try {
            run_job(job);
        } catch (...) {
            error = std::current_exception();
        }

        {
            std::lock_guard<std::mutex> lock(mu_);
            error_ = error;
            job_done_ = true;
        }
        cv_.notify_all();
    }
}

void CpuFineExecutor::stop_worker_noexcept() noexcept {
    if (!async_worker_ || !worker_.joinable()) return;
    try {
        wait();
    } catch (...) {
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_.notify_one();
    worker_.join();
}

}  // namespace cpu_fine
}  // namespace paraselect
