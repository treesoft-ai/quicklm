#pragma once

#include <string>
#include <vector>

// Architecture-neutral model configuration.
//
// Field set is the union of what the engine's modules need; each architecture
// descriptor (see src/arch/) fills it in from that model's own config.json field
// names and defaults. Defaults below are Qwen3.5-0.8B values, kept so a partially
// populated config still runs the reference model.
struct ModelConfig {
    std::string model_type;
    int vocab_size = 248320;
    int hidden_size = 1024;
    int intermediate_size = 3584;
    int num_hidden_layers = 24;
    int num_attention_heads = 8;   // Full-attention Q heads
    int num_key_value_heads = 2;   // Full-attention KV heads
    int head_dim = 256;            // Full-attention head dimension (explicit from config)
    int linear_num_heads = 16;     // Linear-attention heads
    int linear_head_dim = 128;     // Linear-attention head dimension
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    float partial_rotary_factor = 1.0f;   // 1.0 = full RoPE (standard); Qwen3.5 overrides to 0.25
    std::vector<std::string> layer_types;
};
