#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace paraselect::ivf_cache {

enum class CachePolicy { Require, Build };

inline CachePolicy cache_policy() {
    const char* value = std::getenv("PARASELECT_CACHE_POLICY");
    if (!value || std::string(value) == "require") return CachePolicy::Require;
    if (std::string(value) == "build") return CachePolicy::Build;
    throw std::invalid_argument("PARASELECT_CACHE_POLICY must be require or build");
}

inline bool cache_required() { return cache_policy() == CachePolicy::Require; }

[[noreturn]] inline void cache_miss(const std::string& entry) {
    throw std::runtime_error("required cache is missing or incompatible: " + entry);
}

inline void require_build_policy(const std::string& entry) {
    if (cache_required()) cache_miss(entry);
}

}  // namespace paraselect::ivf_cache
