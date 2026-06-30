#include "safetensors.hpp"
#include "json_parser.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <iostream>

#ifndef _WIN32
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

// --- MappedFile Implementation ---

MappedFile::~MappedFile() {
    close();
}

#ifdef _WIN32
bool MappedFile::open(const std::string& filepath) {
    path = filepath;
    file_handle = CreateFileA(
        filepath.c_str(), 
        GENERIC_READ, 
        FILE_SHARE_READ, 
        NULL, 
        OPEN_EXISTING, 
        FILE_ATTRIBUTE_NORMAL, 
        NULL
    );
    if (file_handle == INVALID_HANDLE_VALUE) {
        return false;
    }

    LARGE_INTEGER size_li;
    if (!GetFileSizeEx(file_handle, &size_li)) {
        CloseHandle(file_handle);
        file_handle = INVALID_HANDLE_VALUE;
        return false;
    }
    file_size = static_cast<size_t>(size_li.QuadPart);

    mapping_handle = CreateFileMappingA(file_handle, NULL, PAGE_READONLY, 0, 0, NULL);
    if (mapping_handle == NULL) {
        CloseHandle(file_handle);
        file_handle = INVALID_HANDLE_VALUE;
        return false;
    }

    mapped_data = MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0);
    if (mapped_data == nullptr) {
        CloseHandle(mapping_handle);
        mapping_handle = NULL;
        CloseHandle(file_handle);
        file_handle = INVALID_HANDLE_VALUE;
        return false;
    }

    return true;
}

void MappedFile::close() {
    if (mapped_data != nullptr) {
        UnmapViewOfFile(mapped_data);
        mapped_data = nullptr;
    }
    if (mapping_handle != NULL) {
        CloseHandle(mapping_handle);
        mapping_handle = NULL;
    }
    if (file_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle);
        file_handle = INVALID_HANDLE_VALUE;
    }
}
#else
bool MappedFile::open(const std::string& filepath) {
    path = filepath;
    file_fd = ::open(filepath.c_str(), O_RDONLY);
    if (file_fd < 0) {
        return false;
    }

    file_size = lseek(file_fd, 0, SEEK_END);
    if (file_size == (size_t)-1) {
        ::close(file_fd);
        file_fd = -1;
        return false;
    }

    mapped_data = mmap(NULL, file_size, PROT_READ, MAP_SHARED, file_fd, 0);
    if (mapped_data == MAP_FAILED) {
        ::close(file_fd);
        file_fd = -1;
        mapped_data = nullptr;
        return false;
    }

    return true;
}

void MappedFile::close() {
    if (mapped_data != nullptr) {
        munmap(mapped_data, file_size);
        mapped_data = nullptr;
    }
    if (file_fd >= 0) {
        ::close(file_fd);
        file_fd = -1;
    }
}
#endif

// --- SafetensorsLoader Implementation ---

bool SafetensorsLoader::open(const std::string& filepath) {
    mapped_file = std::make_shared<MappedFile>();
    if (!mapped_file->open(filepath)) {
        mapped_file = nullptr;
        return false;
    }

    if (mapped_file->file_size < 8) {
        close();
        return false;
    }

    // Read header size (first 8 bytes, uint64_t little-endian)
    uint64_t header_size = 0;
    std::memcpy(&header_size, mapped_file->mapped_data, 8);

    if (8 + header_size > mapped_file->file_size) {
        close();
        return false;
    }

    binary_start_offset = 8 + header_size;

    // Parse JSON header string view
    std::string_view json_str(static_cast<char*>(mapped_file->mapped_data) + 8, header_size);
    try {
        JsonValue root = JsonParser::parse(json_str);
        if (!root.is_object()) {
            close();
            return false;
        }

        for (const auto& pair : root.obj_val) {
            const std::string& name = pair.first;
            if (name == "__metadata__") {
                continue;
            }

            const JsonValue& info_val = pair.second;
            if (!info_val.is_object()) {
                continue;
            }

            SafetensorsTensorInfo info;
            info.dtype = info_val["dtype"].str_val;

            const JsonValue& shape_val = info_val["shape"];
            if (shape_val.is_array()) {
                for (size_t i = 0; i < shape_val.arr_val.size(); ++i) {
                    info.shape.push_back(static_cast<int>(shape_val[i].num_val));
                }
            }

            const JsonValue& offsets_val = info_val["data_offsets"];
            if (offsets_val.is_array() && offsets_val.arr_val.size() == 2) {
                info.start_offset = static_cast<size_t>(offsets_val[0].num_val);
                info.end_offset = static_cast<size_t>(offsets_val[1].num_val);
            }

            tensor_infos[name] = std::move(info);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing safetensors JSON header: " << e.what() << std::endl;
        close();
        return false;
    }

    return true;
}

void SafetensorsLoader::close() {
    if (mapped_file) {
        mapped_file->close();
        mapped_file = nullptr;
    }
    tensor_infos.clear();
    binary_start_offset = 0;
}

Tensor SafetensorsLoader::get_tensor(const std::string& name) {
    auto it = tensor_infos.find(name);
    if (it == tensor_infos.end()) {
        throw std::runtime_error("Tensor not found in safetensors file: " + name);
    }

    const auto& info = it->second;
    size_t abs_start = binary_start_offset + info.start_offset;
    size_t abs_end = binary_start_offset + info.end_offset;

    if (abs_end > mapped_file->file_size) {
        throw std::runtime_error("Tensor '" + name + "' offset goes out of file boundaries");
    }

    if (info.dtype == "F32") {
        float* data_ptr = reinterpret_cast<float*>(static_cast<char*>(mapped_file->mapped_data) + abs_start);
        return Tensor(info.shape, data_ptr);
    } else if (info.dtype == "BF16") {
        long long num_elements = 1;
        for (int dim : info.shape) {
            num_elements *= dim;
        }

        auto data_owner = std::make_shared<std::vector<float>>(num_elements);
        const uint16_t* raw_bf16 = reinterpret_cast<const uint16_t*>(
            static_cast<const char*>(mapped_file->mapped_data) + abs_start
        );

        for (long long i = 0; i < num_elements; ++i) {
            uint16_t val = raw_bf16[i];
            uint32_t temp = (static_cast<uint32_t>(val) << 16);
            float f;
            std::memcpy(&f, &temp, sizeof(float));
            (*data_owner)[i] = f;
        }

        Tensor t;
        t.shape = info.shape;
        t.compute_strides();
        t.data_owner = data_owner;
        t.data = data_owner->data();
        return t;
    } else {
        throw std::runtime_error("Unsupported tensor datatype for '" + name + "': " + info.dtype +
                                 " (only F32 and BF16 supported)");
    }
}

Tensor SafetensorsLoader::get_tensor_keep_bf16(const std::string& name) {
    auto it = tensor_infos.find(name);
    if (it == tensor_infos.end()) {
        throw std::runtime_error("Tensor not found in safetensors file: " + name);
    }

    const auto& info = it->second;
    if (info.dtype != "BF16") {
        // F32 (or anything else) goes through the normal path / upcast.
        return get_tensor(name);
    }

    size_t abs_start = binary_start_offset + info.start_offset;
    size_t abs_end = binary_start_offset + info.end_offset;
    if (abs_end > mapped_file->file_size) {
        throw std::runtime_error("Tensor '" + name + "' offset goes out of file boundaries");
    }

    // Zero-copy bf16 view directly into the memory mapping (no upcast, no alloc).
    Tensor t;
    t.shape = info.shape;
    t.compute_strides();
    t.bf16_data = reinterpret_cast<const uint16_t*>(
        static_cast<const char*>(mapped_file->mapped_data) + abs_start);
    return t;
}

bool SafetensorsLoader::has_tensor(const std::string& name) const {
    return tensor_infos.find(name) != tensor_infos.end();
}

std::vector<std::string> SafetensorsLoader::get_tensor_names() const {
    std::vector<std::string> names;
    for (const auto& pair : tensor_infos) {
        names.push_back(pair.first);
    }
    return names;
}
