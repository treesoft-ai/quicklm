#include "generic_transformer.hpp"
#include "attention.hpp"

static bool string_matches_any(const std::string& s, std::initializer_list<const char*> list) {
    for (const char* item : list) {
        if (s == item) return true;
    }
    return false;
}

bool GenericTransformerArchitecture::matches(const std::string& model_type) const {
    return string_matches_any(model_type, {
        "llama", "mistral", "qwen2", "gemma", "gemma2", "gemma3",
        "stablelm", "granite", "internlm2", "phi3", "phi",
        "cohere", "starcoder2", "smollm",
    });
}

void GenericTransformerArchitecture::parse_config(const JsonValue& cfg_node, ModelConfig& cfg) const {
    cfg.model_type = cfg_node["model_type"].str_val;
    cfg.vocab_size = static_cast<int>(cfg_node["vocab_size"].num_val);
    cfg.hidden_size = static_cast<int>(cfg_node["hidden_size"].num_val);
    cfg.intermediate_size = static_cast<int>(cfg_node["intermediate_size"].num_val);
    cfg.num_hidden_layers = static_cast<int>(cfg_node["num_hidden_layers"].num_val);
    cfg.num_attention_heads = static_cast<int>(cfg_node["num_attention_heads"].num_val);

    cfg.num_key_value_heads = cfg_node.contains("num_key_value_heads")
        ? static_cast<int>(cfg_node["num_key_value_heads"].num_val)
        : cfg.num_attention_heads;  // MHA: KV heads == Q heads

    // Normalization epsilon — try all field names used across model families.
    if (cfg_node.contains("rms_norm_eps")) {
        cfg.rms_norm_eps = static_cast<float>(cfg_node["rms_norm_eps"].num_val);
    } else if (cfg_node.contains("layer_norm_eps")) {
        cfg.rms_norm_eps = static_cast<float>(cfg_node["layer_norm_eps"].num_val);
    } else if (cfg_node.contains("layer_norm_epsilon")) {
        cfg.rms_norm_eps = static_cast<float>(cfg_node["layer_norm_epsilon"].num_val);
    }
    // else keep ModelConfig default (1e-6)

    // Head dimension — explicit in config.json for Gemma/Phi3; derived for Llama/Mistral.
    cfg.head_dim = cfg_node.contains("head_dim")
        ? static_cast<int>(cfg_node["head_dim"].num_val)
        : cfg.hidden_size / cfg.num_attention_heads;

    // RoPE base frequency
    if (cfg_node.contains("rope_theta")) {
        cfg.rope_theta = static_cast<float>(cfg_node["rope_theta"].num_val);
    }
    // Partial rotary factor (Phi-3 uses 0.5; Llama/Mistral/Qwen2 omit it → defaults to 1.0)
    if (cfg_node.contains("partial_rotary_factor")) {
        cfg.partial_rotary_factor = static_cast<float>(cfg_node["partial_rotary_factor"].num_val);
    }

    // All layers are standard full attention.
    cfg.layer_types.assign(cfg.num_hidden_layers, "full_attention");
}

std::shared_ptr<DecoderLayer> GenericTransformerArchitecture::build_layer(
    int i, const ModelConfig& cfg, const ModelWeights& weights, int max_seq_len) const {

    const std::string p = "model.layers." + std::to_string(i) + ".";

    auto norm1 = std::make_shared<RMSNorm>();
    norm1->init(weights.get_tensor(p + "input_layernorm.weight"), cfg.rms_norm_eps);

    auto norm2 = std::make_shared<RMSNorm>();
    norm2->init(weights.get_tensor(p + "post_attention_layernorm.weight"), cfg.rms_norm_eps);

    auto mlp = std::make_shared<SwiGLU_MLP>();
    mlp->init(
        weights.get_weight(p + "mlp.gate_proj.weight"),
        weights.get_weight(p + "mlp.up_proj.weight"),
        weights.get_weight(p + "mlp.down_proj.weight")
    );

    // Optional QKV biases (e.g. Phi-3).
    Tensor q_bias, k_bias, v_bias;
    if (weights.has_tensor(p + "self_attn.q_proj.bias")) {
        q_bias = weights.get_tensor(p + "self_attn.q_proj.bias");
        k_bias = weights.get_tensor(p + "self_attn.k_proj.bias");
        v_bias = weights.get_tensor(p + "self_attn.v_proj.bias");
    }

    // Optional per-head QK-norm (e.g. Gemma 2, some Qwen2 variants).
    Tensor q_norm_w, k_norm_w;
    if (weights.has_tensor(p + "self_attn.q_norm.weight")) {
        q_norm_w = weights.get_tensor(p + "self_attn.q_norm.weight");
        k_norm_w = weights.get_tensor(p + "self_attn.k_norm.weight");
    }

    int rotary_dim = static_cast<int>(cfg.head_dim * cfg.partial_rotary_factor);

    // Reuse Qwen3_5Attention: it handles GQA, optional biases, optional QK-norm,
    // partial RoPE, and output gating — the gate is only applied when q_proj
    // emits 2*head_dim per head (q_head_dim > head_dim), which doesn't happen
    // for standard models.
    auto attn = std::make_shared<Qwen3_5Attention>();
    attn->init(
        weights.get_weight(p + "self_attn.q_proj.weight"),
        weights.get_weight(p + "self_attn.k_proj.weight"),
        weights.get_weight(p + "self_attn.v_proj.weight"),
        weights.get_weight(p + "self_attn.o_proj.weight"),
        q_bias, k_bias, v_bias,
        q_norm_w, k_norm_w,
        cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
        rotary_dim, cfg.rope_theta, max_seq_len
    );

    auto layer = std::make_shared<DecoderLayer>();
    layer->init(norm1, attn, norm2, mlp);
    return layer;
}
