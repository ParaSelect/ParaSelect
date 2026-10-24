#pragma once

#include <nlohmann/json.hpp>
#include <filesystem>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <iomanip>
#include <sstream>
#include <unistd.h>

namespace paraselect::dataset::cache_manifest {
using Json = nlohmann::json;

// Match structured fields, never formatting, directory names or numeric prefixes.
inline bool matches(const Json& actual, const Json& expected) {
    if (!actual.is_object() || !expected.is_object()) return false;
    for (auto it = expected.begin(); it != expected.end(); ++it) {
        const auto found = actual.find(it.key());
        if (found == actual.end()) return false;
        if (it->is_object()) {
            if (!matches(*found, *it)) return false;
        } else if (found->type() != it->type() &&
                   !(found->is_number_integer() && it->is_number_integer())) {
            return false;
        } else if (*found != *it) return false;
    }
    return true;
}

inline Json identity(const std::string& kind, const std::string& dataset) {
    return {{"schema", "paraselect.cache"}, {"schema_version", 2},
            {"asset_type", kind}, {"dataset", dataset}};
}

// Directory names are deliberately excluded from identity.  Assets may be
// nested below a cluster entry and may use any short storage name.
inline std::string find(const std::string& root, const Json& request) {
    if (root.empty() || !std::filesystem::exists(root)) return {};
    std::string match;
    for (const auto& item : std::filesystem::recursive_directory_iterator(root)) {
        if (!item.is_directory()) continue;
        const auto path = item.path() / "manifest.json";
        if (!std::filesystem::exists(path)) continue;
        std::ifstream input(path);
        if (!input) throw std::runtime_error("cannot read cache manifest: " + path.string());
        Json manifest;
        try { input >> manifest; }
        catch (const Json::exception& e) {
            throw std::runtime_error("invalid cache manifest " + path.string() + ": " + e.what());
        }
        if (!matches(manifest, request)) continue;
        if (!match.empty()) throw std::runtime_error(
            "ambiguous cache assets: " + match + " and " + item.path().string());
        match = item.path().string();
    }
    return match;
}

inline std::string storage_id(const Json& manifest) {
    // FNV-1a is stable across processes and toolchains; the manifest remains
    // the authoritative identity and this value is only a compact path name.
    const std::string bytes = manifest.dump();
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char c : bytes) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    std::ostringstream out;
    out << "asset-" << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

inline std::string canonical_entry(const std::string& root, const Json& manifest) {
    return (std::filesystem::path(root) / storage_id(manifest)).string();
}

// Publish only after all payloads have been committed. Caller owns the build lock.
inline void publish(const std::string& entry, const Json& manifest) {
    const auto target = std::filesystem::path(entry) / "manifest.json";
    const auto temporary = target.string() + ".tmp." + std::to_string(::getpid());
    try {
        std::ofstream output(temporary);
        output << manifest.dump(2) << '\n';
        output.close();
        if (!output) throw std::runtime_error("cannot write cache manifest: " + temporary);
        std::filesystem::rename(temporary, target);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}
}  // namespace paraselect::dataset::cache_manifest
