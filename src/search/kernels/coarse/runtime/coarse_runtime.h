#pragma once
#include "dispatch.h"
#include "search/kernels/coarse/coarse.h"
#include <cstddef>
#include <memory>

namespace paraselect::coarse {
// Owns all scratch. Inputs/outputs belong to the caller. One stream at a time;
// destroy only after the stream finishes. No environment parsing in execution.
class Workspace {
public:
    explicit Workspace(current::Plan plan);
    Workspace(const Request&, int request, bool asynchronous_only=false);
    // Re-select for the actual shape; scratch grows only when capacity is missing.
    void prepare(const Request&);
    ~Workspace();
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    const current::Plan& plan() const;
    std::size_t workspace_bytes() const;
    std::size_t fallback_queries() const;
    // Test-only admission override: -1 underfill, 1 overflow, 2 mixed, 0 normal.
    void set_admission_test_mode(int mode);
private:
    friend void search(const Request&, Workspace&, Timing*);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace paraselect::coarse
