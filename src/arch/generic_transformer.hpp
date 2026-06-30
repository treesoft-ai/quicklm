#pragma once

#include "architecture.hpp"

// Architecture descriptor for standard decoder-only transformers that share
// the Llama weight-naming layout and standard GQA attention.
//
// Weight naming (canonical, no remap needed):
//   model.embed_tokens.weight
//   model.layers.{i}.input_layernorm.weight
//   model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
//   model.layers.{i}.post_attention_layernorm.weight
//   model.layers.{i}.mlp.{gate,up,down}_proj.weight
//   model.norm.weight / lm_head.weight
//
// Supported model_type values:
//   llama, mistral, qwen2, gemma, gemma2, gemma3, stablelm, granite,
//   internlm2, phi3, phi, cohere, starcoder2, smollm
//
// Optional features detected from weights at load time (no config flag needed):
//   - QK-norm  (self_attn.q_norm.weight present)
//   - QKV bias (self_attn.q_proj.bias present)
//   - Partial RoPE (partial_rotary_factor in config.json, default = 1.0)
//   - Tied LM head (lm_head absent → falls back to embed_tokens)
class GenericTransformerArchitecture : public IArchitecture {
public:
    bool matches(const std::string& model_type) const override;
    void parse_config(const JsonValue& config_node, ModelConfig& cfg) const override;
    std::shared_ptr<DecoderLayer> build_layer(
        int i, const ModelConfig& cfg, const ModelWeights& weights, int max_seq_len) const override;
};
