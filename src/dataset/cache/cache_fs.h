#ifndef PARASELECT_CUDA_DATASET_CACHE_CACHE_FS_H
#define PARASELECT_CUDA_DATASET_CACHE_CACHE_FS_H

#include <cerrno>
#include <dirent.h>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace paraselect {
namespace dataset {
namespace cache_fs {

inline bool ensure_dir_exists(const std::string& dir) {
    if (dir.empty()) return false;
    struct stat st;
    if (stat(dir.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }

    std::string current;
    if (dir.front() == '/') current = "/";
    std::stringstream ss(dir);
    std::string part;
    while (std::getline(ss, part, '/')) {
        if (part.empty()) continue;
        if (!current.empty() && current.back() != '/') current += '/';
        current += part;
        if (stat(current.c_str(), &st) == 0) continue;
        if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) return false;
    }
    return true;
}

inline bool remove_dir_recursive(const std::string& path) {
    struct stat root_st;
    if (lstat(path.c_str(), &root_st) != 0) return errno == ENOENT;
    if (!S_ISDIR(root_st.st_mode)) return unlink(path.c_str()) == 0;

    DIR* dir = opendir(path.c_str());
    if (!dir) return false;

    bool ok = true;
    struct dirent* entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        const std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        const std::string child = path + "/" + name;
        struct stat st;
        if (lstat(child.c_str(), &st) != 0) {
            ok = false;
            continue;
        }
        if (S_ISDIR(st.st_mode)) {
            ok = remove_dir_recursive(child) && ok;
        } else if (unlink(child.c_str()) != 0) {
            ok = false;
        }
    }
    closedir(dir);
    if (rmdir(path.c_str()) != 0) ok = false;
    return ok;
}

}  // namespace cache_fs
}  // namespace dataset
}  // namespace paraselect

#endif  // PARASELECT_CUDA_DATASET_CACHE_CACHE_FS_H
