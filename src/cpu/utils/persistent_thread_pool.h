#ifndef PARASELECT_UTILS_PERSISTENT_THREAD_POOL_H
#define PARASELECT_UTILS_PERSISTENT_THREAD_POOL_H

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace paraselect {

class PersistentThreadPool {
public:
    explicit PersistentThreadPool(int n_threads)
        : n_threads_(std::max(1, n_threads))
    {
        workers_.reserve((size_t)n_threads_);
        for (int tid = 0; tid < n_threads_; ++tid) {
            workers_.emplace_back([this, tid]() { worker_loop(tid); });
        }
    }

    PersistentThreadPool(const PersistentThreadPool&) = delete;
    PersistentThreadPool& operator=(const PersistentThreadPool&) = delete;

    ~PersistentThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
            generation_++;
        }
        cv_start_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    int thread_count() const { return n_threads_; }

    void run_raw_job(void* job_ptr, void (*thunk)(void*, int)) {
        run_raw(job_ptr, thunk);
    }

    template <typename Fn>
    void run(Fn&& fn) {
        using FnNoRef = typename std::remove_reference<Fn>::type;
        FnNoRef* job_ptr = &fn;
        run_raw(
            job_ptr,
            [](void* opaque, int tid) {
                (*static_cast<FnNoRef*>(opaque))(tid);
            });
    }

private:
    using JobThunk = void (*)(void*, int);

    void run_raw(void* job_ptr, JobThunk thunk) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            job_ptr_ = job_ptr;
            job_thunk_ = thunk;
            error_ = nullptr;
            remaining_ = n_threads_;
            generation_++;
        }
        cv_start_.notify_all();

        std::unique_lock<std::mutex> lock(mu_);
        cv_done_.wait(lock, [this]() { return remaining_ == 0; });
        auto error = error_;
        job_ptr_ = nullptr;
        job_thunk_ = nullptr;
        error_ = nullptr;
        lock.unlock();
        if (error) std::rethrow_exception(error);
    }

    void worker_loop(int tid) {
        int seen_generation = 0;
        for (;;) {
            void* job_ptr = nullptr;
            JobThunk job_thunk = nullptr;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_start_.wait(lock, [&]() { return stop_ || generation_ != seen_generation; });
                if (stop_) return;
                seen_generation = generation_;
                job_ptr = job_ptr_;
                job_thunk = job_thunk_;
            }

            std::exception_ptr error;
            try {
                if (job_thunk) job_thunk(job_ptr, tid);
            } catch (...) {
                error = std::current_exception();
            }

            {
                std::lock_guard<std::mutex> lock(mu_);
                if (error && !error_) error_ = error;
                remaining_--;
            }
            cv_done_.notify_one();
        }
    }

    int n_threads_ = 1;
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    void* job_ptr_ = nullptr;
    JobThunk job_thunk_ = nullptr;
    int generation_ = 0;
    int remaining_ = 0;
    bool stop_ = false;
    std::exception_ptr error_;
};

inline PersistentThreadPool& persistent_thread_pool_for_threads(int n_threads) {
    thread_local std::vector<std::unique_ptr<PersistentThreadPool>> pools;
    n_threads = std::max(1, n_threads);
    if ((int)pools.size() <= n_threads) pools.resize((size_t)n_threads + 1);
    if (!pools[(size_t)n_threads]) {
        pools[(size_t)n_threads] = std::make_unique<PersistentThreadPool>(n_threads);
    }
    return *pools[(size_t)n_threads];
}

}  // namespace paraselect

#endif  // PARASELECT_UTILS_PERSISTENT_THREAD_POOL_H
