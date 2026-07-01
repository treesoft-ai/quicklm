#pragma once

#include "tensor.hpp"
#include "safetensors.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>

// Manages weights loaded across one or more safetensors files in a model dir.
//
// Architecture-neutral: tensor lookups go through an optional name-remap hook so
// each architecture can translate the engine's canonical names (e.g. "model.*")
// to that checkpoint's actual storage names (e.g. Qwen's "model.language_model.*")
// without ModelWeights knowing anything model-specific. When no remap is set,
// names are looked up verbatim.
class ModelWeights {
public:
    ModelWeights() = default;
    ~ModelWeights() = default;

    // Set the per-architecture name remap. Given a canonical name, it returns an
    // alternate storage name to try when the verbatim lookup misses (or "" / the
    // same name for no alternative).
    using RemapFn = std::function<std::string(const std::string&)>;
    void set_remap(RemapFn fn) { remap = std::move(fn); }

    // Precision used for matmul weight operands returned by get_weight(). Chosen
    // at runtime, independent of what's stored in the safetensors files — the
    // on-disk checkpoint is never modified.
    enum class Precision { Default, Int8, Int4 };
    void set_precision(Precision p) { precision = p; }

    bool load_from_dir(const std::string& model_dir);
    Tensor get_tensor(const std::string& name) const;
    // Like get_tensor, but returns the matmul weight operand in the configured
    // Precision: Default keeps BF16 in bf16 (no FP32 upcast, zero-copy), Int8
    // quantizes to symmetric per-block (Q8_0-style) INT8 in RAM, Int4 quantizes
    // to symmetric per-block (Q4_0-style) packed INT4 in RAM.
    Tensor get_weight(const std::string& name) const;
    // Zero-copy bf16 view (no FP32 upcast/allocation), independent of the
    // configured Precision. For tensors that are read via gather (e.g. the
    // embedding table) rather than as a matmul operand, where quantizing to
    // int8 isn't useful but avoiding the full-tensor FP32 upcast still is.
    Tensor get_tensor_bf16(const std::string& name) const;
    bool has_tensor(const std::string& name) const;

private:
    // Locate the loader holding `name` (verbatim or via remap). Returns nullptr if
    // absent; writes the resolved storage name to `resolved`.
    SafetensorsLoader* find_loader(const std::string& name, std::string& resolved) const;

    std::vector<std::unique_ptr<SafetensorsLoader>> loaders;
    RemapFn remap;
    Precision precision = Precision::Default;
};
