#pragma once

#include <cstdlib>
#include <cstring>
#include <cerrno>

#include <fcntl.h>
#include <unistd.h>

namespace paraselect {
namespace profiling {

inline void perf_control(const char* command) {
    const char* path = std::getenv("PARASELECT_PERF_CTL");
    if (!path || path[0] == '\0') return;

    const int fd = ::open(path, O_WRONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) return;

    const ssize_t command_bytes = ::write(fd, command, std::strlen(command));
    const ssize_t newline_bytes = ::write(fd, "\n", 1);
    (void)command_bytes;
    (void)newline_bytes;
    ::close(fd);
}

class PerfControlScope {
public:
    PerfControlScope() { perf_control("enable"); }
    ~PerfControlScope() { perf_control("disable"); }

    PerfControlScope(const PerfControlScope&) = delete;
    PerfControlScope& operator=(const PerfControlScope&) = delete;
};

}  // namespace profiling
}  // namespace paraselect
