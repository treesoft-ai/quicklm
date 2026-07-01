#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <cstdint>

// A lightweight multi-dimensional tensor representation in C++17.
// Can either own its data (via std::vector) or view external memory.
//
// Weight matrices may optionally be stored in bf16 instead of FP32 to halve the
// memory bandwidth at decode time (batch-1 matmul is bandwidth-bound). When
// `bf16_data` is set, `data` is null and the matmul kernel upcasts bf16->FP32
// in-register. The upcast is exact (bf16 is the high 16 bits of FP32), so results
// are identical to the FP32 path.
struct Tensor {
    std::vector<int> shape;
    std::vector<int> strides;
    float* data = nullptr;
    std::shared_ptr<std::vector<float>> data_owner = nullptr;

    // Optional bf16 backing store (raw FP32-high-half bits). Mutually exclusive
    // with `data` for matmul B operands.
    const uint16_t* bf16_data = nullptr;
    std::shared_ptr<std::vector<uint16_t>> bf16_owner = nullptr;

    bool is_bf16() const { return bf16_data != nullptr; }

    // Block size for int8 quantization scales (Q8_0-style): one scale per this
    // many contiguous elements within a row, rather than one scale per row.
    static constexpr int INT8_BLOCK_SIZE = 32;

    // Optional int8 backing store: symmetric per-block (Q8_0-style) quantization.
    // For a row of length K, num_blocks = ceil(K / INT8_BLOCK_SIZE); row r's
    // dequantized value at column k is
    // `int8_data[r*K+k] * int8_scales[r*num_blocks + k/INT8_BLOCK_SIZE]`.
    // Mutually exclusive with `data` and `bf16_data` for matmul B operands.
    const int8_t* int8_data = nullptr;
    std::shared_ptr<std::vector<int8_t>> int8_owner = nullptr;
    std::shared_ptr<std::vector<float>> int8_scales = nullptr;

    bool is_int8() const { return int8_data != nullptr; }

    // Block size for int4 quantization scales (Q4_0-style): same block-per-scale
    // granularity as INT8_BLOCK_SIZE.
    static constexpr int INT4_BLOCK_SIZE = 32;

    // Optional int4 backing store: symmetric per-block (Q4_0-style) quantization,
    // two values packed per byte (low nibble = even column, high nibble = odd
    // column). For a row of length K, num_blocks = ceil(K / INT4_BLOCK_SIZE); row
    // r's dequantized value at column k is
    // `(nibble - 8) * int4_scales[r*num_blocks + k/INT4_BLOCK_SIZE]`, where
    // `nibble = (int4_data[r*K/2 + k/2] >> ((k%2)*4)) & 0xF` (unsigned, range
    // [0,15], biased by +8 from the signed quantized value in [-8,7]). Mutually
    // exclusive with `data`, `bf16_data`, and `int8_data` for matmul B operands.
    const uint8_t* int4_data = nullptr;
    std::shared_ptr<std::vector<uint8_t>> int4_owner = nullptr;
    std::shared_ptr<std::vector<float>> int4_scales = nullptr;

    bool is_int4() const { return int4_data != nullptr; }

    Tensor() = default;

    // Construct a tensor that owns its memory (zero-initialized)
    explicit Tensor(const std::vector<int>& shp) : shape(shp) {
        long long total_size = 1;
        for (int dim : shape) {
            total_size *= dim;
        }
        data_owner = std::make_shared<std::vector<float>>(total_size, 0.0f);
        data = data_owner->data();
        compute_strides();
    }

    // Construct a tensor that views external memory
    Tensor(const std::vector<int>& shp, float* external_data) : shape(shp), data(external_data) {
        compute_strides();
    }

    void compute_strides() {
        strides.resize(shape.size());
        int stride = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    int size() const {
        if (shape.empty()) return 0;
        int s = 1;
        for (int dim : shape) s *= dim;
        return s;
    }

    // Helper to get element index based on indices
    int get_index(const std::vector<int>& indices) const {
        if (indices.size() != shape.size()) {
            throw std::runtime_error("Tensor index mismatch: expected rank " + 
                                     std::to_string(shape.size()) + ", got " + 
                                     std::to_string(indices.size()));
        }
        int idx = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw std::runtime_error("Tensor index out of bounds");
            }
            idx += indices[i] * strides[i];
        }
        return idx;
    }

    float& operator()(const std::vector<int>& indices) {
        return data[get_index(indices)];
    }

    const float& operator()(const std::vector<int>& indices) const {
        return data[get_index(indices)];
    }
};
