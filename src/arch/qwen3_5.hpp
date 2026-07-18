#pragma once

#include "architecture.hpp"

// Architecture descriptor for Qwen3.5 — the hybrid full-attention / Gated-DeltaNet
// linear-attention model. This is the ONLY place "qwen3_5", the 3:1 layer pattern,
// the partial-rotary / QK-norm / output-gate wiring, and the
// model. -> model.language_model. tensor remap appear.
class Qwen3_5Architecture : public IArchitecture {
public:
    bool matches(const std::string& model_type) const override;
    void parse_config(const JsonValue& config_node, ModelConfig& cfg) const override;
    std::shared_ptr<DecoderLayer> build_layer(
        int i, const ModelConfig& cfg, const ModelWeights& weights, int max_seq_len) const override;

    std::string remap_tensor_name(const std::string& name) const override;
    std::string default_chat_template() const override;
};
