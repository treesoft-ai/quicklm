#pragma once

#include "tensor.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

#ifdef _WIN32
#include <windows.h>
#endif

// A simple cross-platform memory-mapped file wrapper (using Windows API on Windows)
struct MappedFile {
    std::string path;
    size_t file_size = 0;
    void* mapped_data = nullptr;

#ifdef _WIN32
    HANDLE file_handle = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle = NULL;
#else
    int file_fd = -1;
#endif

    MappedFile() = default;
    ~MappedFile();

    bool open(const std::string& filepath);
    void close();
};

struct SafetensorsTensorInfo {
    std::string dtype;
    std::vector<int> shape;
    size_t start_offset = 0;
    size_t end_offset = 0;
};

class SafetensorsLoader {
public:
    SafetensorsLoader() = default;
    ~SafetensorsLoader() = default;

    // Load the safetensors metadata and memory-map the file
    bool open(const std::string& filepath);
    void close();

    // Retrieve a Tensor view from the mapped file
    Tensor get_tensor(const std::string& name);

    // Retrieve a tensor keeping BF16 storage as raw bf16 (no FP32 upcast).
    // For F32 tensors this behaves like get_tensor. Used for large weight
    // matrices to halve resident size and decode-time memory bandwidth.
    Tensor get_tensor_keep_bf16(const std::string& name);

    // Check if a tensor exists
    bool has_tensor(const std::string& name) const;

    // Get list of all tensor names
    std::vector<std::string> get_tensor_names() const;

private:
    std::shared_ptr<MappedFile> mapped_file = nullptr;
    std::unordered_map<std::string, SafetensorsTensorInfo> tensor_infos;
    size_t binary_start_offset = 0;
};
