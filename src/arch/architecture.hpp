#pragma once

#include "config.hpp"
#include "modules.hpp"
#include "weights.hpp"
#include "json_parser.hpp"
#include <memory>
#include <string>

// An architecture descriptor teaches the generic decoder runtime how to assemble
// one model family. The engine (src/model/model.cpp) knows nothing model-specific;
// it selects a descriptor by model_type and delegates every architecture-dependent
// decision to it. Adding a new architecture = implementing this interface + one
// registration line in registry.cpp — the engine core never changes.
class IArchitecture {
public:
    virtual ~IArchitecture() = default;

    // Identifier matching, by config.json "model_type" (e.g. "qwen3_5").
    virtual bool matches(const std::string& model_type) const = 0;

    // Populate `cfg` from this model's own config.json field names and defaults.
    // `config_node` is the already-unwrapped node (text_config handled by caller).
    virtual void parse_config(const JsonValue& config_node, ModelConfig& cfg) const = 0;

    // Build decoder layer `i` by selecting + initializing this layer's modules
    // (norms, attention variant, MLP) from the loaded weights.
    virtual std::shared_ptr<DecoderLayer> build_layer(
        int i, const ModelConfig& cfg, const ModelWeights& weights, int max_seq_len) const = 0;

    // Canonical tensor names the runtime loads directly.
    virtual std::string embed_name() const { return "model.embed_tokens.weight"; }
    virtual std::string final_norm_name() const { return "model.norm.weight"; }
    virtual std::string lm_head_name() const { return "lm_head.weight"; }
    // Fallback embed name to use as a tied lm_head when lm_head_name() is absent.
    virtual std::string tied_lm_head_name() const { return "model.embed_tokens.weight"; }

    // Translate a canonical tensor name to this checkpoint's storage name when the
    // verbatim lookup misses (e.g. Qwen: "model." -> "model.language_model."). Return
    // the same name (or "") for no alternative. Installed on ModelWeights at load.
    virtual std::string remap_tensor_name(const std::string& name) const { return name; }

    // Built-in chat template used when the model dir has no chat_template.jinja and
    // tokenizer_config.json carries none. Empty string = engine uses raw prompt.
    virtual std::string default_chat_template() const { return ""; }
};
