#include "safetensors.hpp"
#include "json_parser.hpp"
#include "math_ops.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <immintrin.h>

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

// --- AVX2 helpers for fused BF16/F32 -> INT8 quantization ---
// Upcast 8 packed bf16 values to a __m256 of fp32 (same bit-shift trick as the
// scalar loop in get_tensor(): bf16 is just the top 16 bits of an fp32).
static inline __m256 st_load_bf16_8(const uint16_t* p) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(v), 16));
}

static inline float st_hmax(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    return _mm_cvtss_f32(m);
}

// Quantize one row of `K` bf16 values to symmetric per-block int8 (Q8_0-style:
// one scale per Tensor::INT8_BLOCK_SIZE-element block, scale = max(abs(block))/127).
// Reads directly from the mapped bf16 bytes; never materializes an intermediate
// FP32 array. `scales` must have room for ceil(K/INT8_BLOCK_SIZE) entries.
static void st_quantize_row_bf16(const uint16_t* row, int K, int8_t* out, float* scales) {
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const int BLOCK = Tensor::INT8_BLOCK_SIZE;
    int num_blocks = (K + BLOCK - 1) / BLOCK;

    for (int blk = 0; blk < num_blocks; ++blk) {
        int k0 = blk * BLOCK;
        int k1 = std::min(k0 + BLOCK, K);

        __m256 vmax = _mm256_setzero_ps();
        int k = k0;
        for (; k + 8 <= k1; k += 8) {
            vmax = _mm256_max_ps(vmax, _mm256_and_ps(st_load_bf16_8(row + k), abs_mask));
        }
        float max_abs = st_hmax(vmax);
        for (; k < k1; ++k) {
            uint32_t bits = static_cast<uint32_t>(row[k]) << 16;
            float f;
            std::memcpy(&f, &bits, sizeof(float));
            max_abs = std::max(max_abs, std::fabs(f));
        }

        float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        scales[blk] = scale;
        float inv_scale = 1.0f / scale;
        __m256 vinv = _mm256_set1_ps(inv_scale);

        k = k0;
        for (; k + 8 <= k1; k += 8) {
            __m256 v = _mm256_mul_ps(st_load_bf16_8(row + k), vinv);
            v = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256i vi = _mm256_cvtps_epi32(v);
            alignas(32) int32_t tmp[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), vi);
            for (int i = 0; i < 8; ++i) {
                int q = tmp[i];
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
                out[k + i] = static_cast<int8_t>(q);
            }
        }
        for (; k < k1; ++k) {
            uint32_t bits = static_cast<uint32_t>(row[k]) << 16;
            float f;
            std::memcpy(&f, &bits, sizeof(float));
            int q = static_cast<int>(std::lround(f * inv_scale));
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
            out[k] = static_cast<int8_t>(q);
        }
    }
}

// Same as st_quantize_row_bf16 but for a raw FP32 row (no upcast needed).
static void st_quantize_row_f32(const float* row, int K, int8_t* out, float* scales) {
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const int BLOCK = Tensor::INT8_BLOCK_SIZE;
    int num_blocks = (K + BLOCK - 1) / BLOCK;

    for (int blk = 0; blk < num_blocks; ++blk) {
        int k0 = blk * BLOCK;
        int k1 = std::min(k0 + BLOCK, K);

        __m256 vmax = _mm256_setzero_ps();
        int k = k0;
        for (; k + 8 <= k1; k += 8) {
            vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(row + k), abs_mask));
        }
        float max_abs = st_hmax(vmax);
        for (; k < k1; ++k) {
            max_abs = std::max(max_abs, std::fabs(row[k]));
        }

        float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        scales[blk] = scale;
        float inv_scale = 1.0f / scale;
        __m256 vinv = _mm256_set1_ps(inv_scale);

        k = k0;
        for (; k + 8 <= k1; k += 8) {
            __m256 v = _mm256_mul_ps(_mm256_loadu_ps(row + k), vinv);
            v = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256i vi = _mm256_cvtps_epi32(v);
            alignas(32) int32_t tmp[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), vi);
            for (int i = 0; i < 8; ++i) {
                int q = tmp[i];
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
                out[k + i] = static_cast<int8_t>(q);
            }
        }
        for (; k < k1; ++k) {
            int q = static_cast<int>(std::lround(row[k] * inv_scale));
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
            out[k] = static_cast<int8_t>(q);
        }
    }
}

// Quantize one row of `K` bf16 values to symmetric per-block int4 (Q4_0-style:
// one scale per Tensor::INT4_BLOCK_SIZE-element block, scale = max(abs(block))/8,
// values clamped to [-8,7] and biased to an unsigned nibble [0,15]). Packs two
// values per output byte (low nibble = even column, high nibble = odd column).
static void st_quantize_row_bf16_int4(const uint16_t* row, int K, uint8_t* out, float* scales) {
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const int BLOCK = Tensor::INT4_BLOCK_SIZE;
    int num_blocks = (K + BLOCK - 1) / BLOCK;

    for (int blk = 0; blk < num_blocks; ++blk) {
        int k0 = blk * BLOCK;
        int k1 = std::min(k0 + BLOCK, K);

        __m256 vmax = _mm256_setzero_ps();
        int k = k0;
        for (; k + 8 <= k1; k += 8) {
            vmax = _mm256_max_ps(vmax, _mm256_and_ps(st_load_bf16_8(row + k), abs_mask));
        }
        float max_abs = st_hmax(vmax);
        for (; k < k1; ++k) {
            uint32_t bits = static_cast<uint32_t>(row[k]) << 16;
            float f;
            std::memcpy(&f, &bits, sizeof(float));
            max_abs = std::max(max_abs, std::fabs(f));
        }

        float scale = (max_abs > 0.0f) ? (max_abs / 8.0f) : 1.0f;
        scales[blk] = scale;
        float inv_scale = 1.0f / scale;
        __m256 vinv = _mm256_set1_ps(inv_scale);

        // Quantize+clamp 8 elements at a time via AVX2 (mirrors st_quantize_row_bf16),
        // then pack pairs into nibbles scalar (cheap bit ops, no per-element lround).
        alignas(32) int32_t tmp[8];
        k = k0;
        for (; k + 8 <= k1; k += 8) {
            __m256 v = _mm256_mul_ps(st_load_bf16_8(row + k), vinv);
            v = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256i vi = _mm256_cvtps_epi32(v);
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), vi);
            for (int i = 0; i < 8; i += 2) {
                int q0 = tmp[i] < -8 ? -8 : (tmp[i] > 7 ? 7 : tmp[i]);
                int q1 = tmp[i + 1] < -8 ? -8 : (tmp[i + 1] > 7 ? 7 : tmp[i + 1]);
                out[(k + i) / 2] = static_cast<uint8_t>(((q1 + 8) << 4) | ((q0 + 8) & 0x0F));
            }
        }
        // Scalar tail; block size is even so kk+1 is in range except for a
        // defensive odd-K tail (not expected in practice).
        for (int kk = k; kk < k1; kk += 2) {
            uint32_t bits0 = static_cast<uint32_t>(row[kk]) << 16;
            float f0;
            std::memcpy(&f0, &bits0, sizeof(float));
            int q0 = static_cast<int>(std::lround(f0 * inv_scale));
            q0 = q0 < -8 ? -8 : (q0 > 7 ? 7 : q0);
            uint8_t n0 = static_cast<uint8_t>(q0 + 8);

            uint8_t n1 = 8;
            if (kk + 1 < k1) {
                uint32_t bits1 = static_cast<uint32_t>(row[kk + 1]) << 16;
                float f1;
                std::memcpy(&f1, &bits1, sizeof(float));
                int q1 = static_cast<int>(std::lround(f1 * inv_scale));
                q1 = q1 < -8 ? -8 : (q1 > 7 ? 7 : q1);
                n1 = static_cast<uint8_t>(q1 + 8);
            }
            out[kk / 2] = static_cast<uint8_t>((n1 << 4) | (n0 & 0x0F));
        }
    }
}

// Same as st_quantize_row_bf16_int4 but for a raw FP32 row (no upcast needed).
static void st_quantize_row_f32_int4(const float* row, int K, uint8_t* out, float* scales) {
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const int BLOCK = Tensor::INT4_BLOCK_SIZE;
    int num_blocks = (K + BLOCK - 1) / BLOCK;

    for (int blk = 0; blk < num_blocks; ++blk) {
        int k0 = blk * BLOCK;
        int k1 = std::min(k0 + BLOCK, K);

        __m256 vmax = _mm256_setzero_ps();
        int k = k0;
        for (; k + 8 <= k1; k += 8) {
            vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(row + k), abs_mask));
        }
        float max_abs = st_hmax(vmax);
        for (; k < k1; ++k) {
            max_abs = std::max(max_abs, std::fabs(row[k]));
        }

        float scale = (max_abs > 0.0f) ? (max_abs / 8.0f) : 1.0f;
        scales[blk] = scale;
        float inv_scale = 1.0f / scale;
        __m256 vinv = _mm256_set1_ps(inv_scale);

        alignas(32) int32_t tmp[8];
        k = k0;
        for (; k + 8 <= k1; k += 8) {
            __m256 v = _mm256_mul_ps(_mm256_loadu_ps(row + k), vinv);
            v = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256i vi = _mm256_cvtps_epi32(v);
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), vi);
            for (int i = 0; i < 8; i += 2) {
                int q0 = tmp[i] < -8 ? -8 : (tmp[i] > 7 ? 7 : tmp[i]);
                int q1 = tmp[i + 1] < -8 ? -8 : (tmp[i + 1] > 7 ? 7 : tmp[i + 1]);
                out[(k + i) / 2] = static_cast<uint8_t>(((q1 + 8) << 4) | ((q0 + 8) & 0x0F));
            }
        }
        for (int kk = k; kk < k1; kk += 2) {
            int q0 = static_cast<int>(std::lround(row[kk] * inv_scale));
            q0 = q0 < -8 ? -8 : (q0 > 7 ? 7 : q0);
            uint8_t n0 = static_cast<uint8_t>(q0 + 8);

            uint8_t n1 = 8;
            if (kk + 1 < k1) {
                int q1 = static_cast<int>(std::lround(row[kk + 1] * inv_scale));
                q1 = q1 < -8 ? -8 : (q1 > 7 ? 7 : q1);
                n1 = static_cast<uint8_t>(q1 + 8);
            }
            out[kk / 2] = static_cast<uint8_t>((n1 << 4) | (n0 & 0x0F));
        }
    }
}

Tensor SafetensorsLoader::get_tensor_keep_int8(const std::string& name) {
    auto it = tensor_infos.find(name);
    if (it == tensor_infos.end()) {
        throw std::runtime_error("Tensor not found in safetensors file: " + name);
    }

    const auto& info = it->second;
    if (info.dtype != "F32" && info.dtype != "BF16") {
        throw std::runtime_error("Unsupported tensor datatype for '" + name + "': " + info.dtype +
                                 " (only F32 and BF16 supported for int8 quantization)");
    }

    size_t abs_start = binary_start_offset + info.start_offset;
    size_t abs_end = binary_start_offset + info.end_offset;
    if (abs_end > mapped_file->file_size) {
        throw std::runtime_error("Tensor '" + name + "' offset goes out of file boundaries");
    }

    int rows = info.shape.empty() ? 1 : info.shape[0];
    int K = info.shape.empty() ? 0 : info.shape.back();
    int num_blocks_per_row = (K + Tensor::INT8_BLOCK_SIZE - 1) / Tensor::INT8_BLOCK_SIZE;

    Tensor t;
    t.shape = info.shape;
    t.compute_strides();
    t.int8_owner = std::make_shared<std::vector<int8_t>>((size_t)rows * K);
    t.int8_scales = std::make_shared<std::vector<float>>((size_t)rows * num_blocks_per_row);
    int8_t* out = t.int8_owner->data();
    float* scales = t.int8_scales->data();

    const char* base = static_cast<const char*>(mapped_file->mapped_data) + abs_start;
    bool is_bf16 = (info.dtype == "BF16");

    auto quantize_rows = [&](int r0, int r1) {
        if (is_bf16) {
            const uint16_t* raw = reinterpret_cast<const uint16_t*>(base);
            for (int r = r0; r < r1; ++r) {
                st_quantize_row_bf16(raw + (size_t)r * K, K, out + (size_t)r * K,
                                     scales + (size_t)r * num_blocks_per_row);
            }
        } else {
            const float* raw = reinterpret_cast<const float*>(base);
            for (int r = r0; r < r1; ++r) {
                st_quantize_row_f32(raw + (size_t)r * K, K, out + (size_t)r * K,
                                    scales + (size_t)r * num_blocks_per_row);
            }
        }
    };

    ThreadPool* pool = math::get_thread_pool();
    int nthreads = math::get_thread_pool_size();
    if (pool && nthreads > 1 && rows >= nthreads) {
        int chunk = (rows + nthreads - 1) / nthreads;
        std::vector<std::future<void>> futures;
        futures.reserve(nthreads);
        for (int tid = 0; tid < nthreads; ++tid) {
            int r0 = tid * chunk;
            int r1 = std::min(r0 + chunk, rows);
            if (r0 >= r1) break;
            futures.push_back(pool->enqueue([&quantize_rows, r0, r1]() { quantize_rows(r0, r1); }));
        }
        for (auto& f : futures) f.get();
    } else {
        quantize_rows(0, rows);
    }

    t.int8_data = out;
    return t;
}

Tensor SafetensorsLoader::get_tensor_keep_int4(const std::string& name) {
    auto it = tensor_infos.find(name);
    if (it == tensor_infos.end()) {
        throw std::runtime_error("Tensor not found in safetensors file: " + name);
    }

    const auto& info = it->second;
    if (info.dtype != "F32" && info.dtype != "BF16") {
        throw std::runtime_error("Unsupported tensor datatype for '" + name + "': " + info.dtype +
                                 " (only F32 and BF16 supported for int4 quantization)");
    }

    size_t abs_start = binary_start_offset + info.start_offset;
    size_t abs_end = binary_start_offset + info.end_offset;
    if (abs_end > mapped_file->file_size) {
        throw std::runtime_error("Tensor '" + name + "' offset goes out of file boundaries");
    }

    int rows = info.shape.empty() ? 1 : info.shape[0];
    int K = info.shape.empty() ? 0 : info.shape.back();
    int num_blocks_per_row = (K + Tensor::INT4_BLOCK_SIZE - 1) / Tensor::INT4_BLOCK_SIZE;
    int packed_bytes_per_row = (K + 1) / 2;

    Tensor t;
    t.shape = info.shape;
    t.compute_strides();
    t.int4_owner = std::make_shared<std::vector<uint8_t>>((size_t)rows * packed_bytes_per_row);
    t.int4_scales = std::make_shared<std::vector<float>>((size_t)rows * num_blocks_per_row);
    uint8_t* out = t.int4_owner->data();
    float* scales = t.int4_scales->data();

    const char* base = static_cast<const char*>(mapped_file->mapped_data) + abs_start;
    bool is_bf16 = (info.dtype == "BF16");

    auto quantize_rows = [&](int r0, int r1) {
        if (is_bf16) {
            const uint16_t* raw = reinterpret_cast<const uint16_t*>(base);
            for (int r = r0; r < r1; ++r) {
                st_quantize_row_bf16_int4(raw + (size_t)r * K, K, out + (size_t)r * packed_bytes_per_row,
                                          scales + (size_t)r * num_blocks_per_row);
            }
        } else {
            const float* raw = reinterpret_cast<const float*>(base);
            for (int r = r0; r < r1; ++r) {
                st_quantize_row_f32_int4(raw + (size_t)r * K, K, out + (size_t)r * packed_bytes_per_row,
                                         scales + (size_t)r * num_blocks_per_row);
            }
        }
    };

    ThreadPool* pool = math::get_thread_pool();
    int nthreads = math::get_thread_pool_size();
    if (pool && nthreads > 1 && rows >= nthreads) {
        int chunk = (rows + nthreads - 1) / nthreads;
        std::vector<std::future<void>> futures;
        futures.reserve(nthreads);
        for (int tid = 0; tid < nthreads; ++tid) {
            int r0 = tid * chunk;
            int r1 = std::min(r0 + chunk, rows);
            if (r0 >= r1) break;
            futures.push_back(pool->enqueue([&quantize_rows, r0, r1]() { quantize_rows(r0, r1); }));
        }
        for (auto& f : futures) f.get();
    } else {
        quantize_rows(0, rows);
    }

    t.int4_data = out;
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
