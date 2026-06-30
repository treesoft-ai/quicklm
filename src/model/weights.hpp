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

    bool load_from_dir(const std::string& model_dir);
    Tensor get_tensor(const std::string& name) const;
    // Like get_tensor, but keeps BF16 weights in bf16 (no FP32 upcast). Use for
    // large 2D weight matrices that are consumed by matmul.
    Tensor get_weight(const std::string& name) const;
    bool has_tensor(const std::string& name) const;

private:
    // Locate the loader holding `name` (verbatim or via remap). Returns nullptr if
    // absent; writes the resolved storage name to `resolved`.
    SafetensorsLoader* find_loader(const std::string& name, std::string& resolved) const;

    std::vector<std::unique_ptr<SafetensorsLoader>> loaders;
    RemapFn remap;
};
