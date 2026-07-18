#include "qwen3_5.hpp"
#include "attention.hpp"
#include <cmath>

bool Qwen3_5Architecture::matches(const std::string& model_type) const {
    // Accept the Qwen3.5 text backbone identifiers.
    return model_type == "qwen3_5" || model_type == "qwen3_5_moe" ||
           model_type == "qwen3_5_text" || model_type.rfind("qwen3_5", 0) == 0;
}

void Qwen3_5Architecture::parse_config(const JsonValue& config_node, ModelConfig& cfg) const {
    cfg.model_type = config_node["model_type"].str_val;
    cfg.vocab_size = static_cast<int>(config_node["vocab_size"].num_val);
    cfg.hidden_size = static_cast<int>(config_node["hidden_size"].num_val);
    cfg.intermediate_size = static_cast<int>(config_node["intermediate_size"].num_val);
    cfg.num_hidden_layers = static_cast<int>(config_node["num_hidden_layers"].num_val);
    cfg.num_attention_heads = static_cast<int>(config_node["num_attention_heads"].num_val);

    if (config_node.contains("num_key_value_heads")) {
        cfg.num_key_value_heads = static_cast<int>(config_node["num_key_value_heads"].num_val);
    } else {
        cfg.num_key_value_heads = cfg.num_attention_heads;
    }

    if (config_node.contains("rms_norm_eps")) {
        cfg.rms_norm_eps = static_cast<float>(config_node["rms_norm_eps"].num_val);
    }

    if (config_node.contains("rope_theta")) {
        cfg.rope_theta = static_cast<float>(config_node["rope_theta"].num_val);
    } else if (config_node.contains("rope_parameters")) {
        const JsonValue& rope_params = config_node["rope_parameters"];
        if (rope_params.contains("rope_theta")) {
            cfg.rope_theta = static_cast<float>(rope_params["rope_theta"].num_val);
        }
        if (rope_params.contains("partial_rotary_factor")) {
            cfg.partial_rotary_factor = static_cast<float>(rope_params["partial_rotary_factor"].num_val);
        }
    }
    if (config_node.contains("partial_rotary_factor")) {
        cfg.partial_rotary_factor = static_cast<float>(config_node["partial_rotary_factor"].num_val);
    }

    // Explicit head dims (Qwen3.5 has head_dim != hidden_size / num_attention_heads)
    if (config_node.contains("head_dim")) {
        cfg.head_dim = static_cast<int>(config_node["head_dim"].num_val);
    } else {
        cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
    }
    if (config_node.contains("linear_num_key_heads")) {
        cfg.linear_num_heads = static_cast<int>(config_node["linear_num_key_heads"].num_val);
    }
    if (config_node.contains("linear_key_head_dim")) {
        cfg.linear_head_dim = static_cast<int>(config_node["linear_key_head_dim"].num_val);
    }

    const JsonValue& layer_types_val = config_node["layer_types"];
    if (layer_types_val.is_array()) {
        for (size_t i = 0; i < layer_types_val.arr_val.size(); ++i) {
            cfg.layer_types.push_back(layer_types_val[i].str_val);
        }
    } else {
        // Default Qwen3.5 3:1 pattern if missing
        for (int i = 0; i < cfg.num_hidden_layers; ++i) {
            if (i % 4 == 3) {
                cfg.layer_types.push_back("full_attention");
            } else {
                cfg.layer_types.push_back("linear_attention");
            }
        }
    }
}

std::shared_ptr<DecoderLayer> Qwen3_5Architecture::build_layer(
    int i, const ModelConfig& cfg, const ModelWeights& weights, int max_seq_len) const {
    std::string layer_prefix = "model.layers." + std::to_string(i) + ".";

    auto norm1 = std::make_shared<RMSNorm>();
    norm1->init(weights.get_tensor(layer_prefix + "input_layernorm.weight"), cfg.rms_norm_eps);

    auto norm2 = std::make_shared<RMSNorm>();
    norm2->init(weights.get_tensor(layer_prefix + "post_attention_layernorm.weight"), cfg.rms_norm_eps);

    auto mlp = std::make_shared<SwiGLU_MLP>();
    mlp->init(
        weights.get_weight(layer_prefix + "mlp.gate_proj.weight"),
        weights.get_weight(layer_prefix + "mlp.up_proj.weight"),
        weights.get_weight(layer_prefix + "mlp.down_proj.weight")
    );

    std::shared_ptr<IAttention> attn = nullptr;
    const std::string& attn_type = cfg.layer_types[i];

    if (attn_type == "full_attention") {
        auto full_attn = std::make_shared<Qwen3_5Attention>();
        Tensor q_bias, k_bias, v_bias;
        if (weights.has_tensor(layer_prefix + "self_attn.q_proj.bias")) {
            q_bias = weights.get_tensor(layer_prefix + "self_attn.q_proj.bias");
            k_bias = weights.get_tensor(layer_prefix + "self_attn.k_proj.bias");
            v_bias = weights.get_tensor(layer_prefix + "self_attn.v_proj.bias");
        }
        Tensor q_norm_w, k_norm_w;
        if (weights.has_tensor(layer_prefix + "self_attn.q_norm.weight")) {
            q_norm_w = weights.get_tensor(layer_prefix + "self_attn.q_norm.weight");
            k_norm_w = weights.get_tensor(layer_prefix + "self_attn.k_norm.weight");
        }
        int rotary_dim = static_cast<int>(cfg.head_dim * cfg.partial_rotary_factor);
        // Pass num_attention_heads from config; init() derives q_head_dim from q_proj shape
        full_attn->init(
            weights.get_weight(layer_prefix + "self_attn.q_proj.weight"),
            weights.get_weight(layer_prefix + "self_attn.k_proj.weight"),
            weights.get_weight(layer_prefix + "self_attn.v_proj.weight"),
            weights.get_weight(layer_prefix + "self_attn.o_proj.weight"),
            q_bias, k_bias, v_bias,
            q_norm_w, k_norm_w,
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
            rotary_dim, cfg.rope_theta, max_seq_len
        );
        attn = full_attn;
    } else {
        auto delta_attn = std::make_shared<Qwen3_5GatedDeltaNet>();
        delta_attn->init(
            weights.get_weight(layer_prefix + "linear_attn.in_proj_qkv.weight"),
            weights.get_weight(layer_prefix + "linear_attn.in_proj_z.weight"),
            weights.get_weight(layer_prefix + "linear_attn.in_proj_b.weight"),
            weights.get_weight(layer_prefix + "linear_attn.in_proj_a.weight"),
            weights.get_tensor(layer_prefix + "linear_attn.conv1d.weight"),
            weights.get_tensor(layer_prefix + "linear_attn.norm.weight"),
            weights.get_tensor(layer_prefix + "linear_attn.A_log"),
            weights.get_tensor(layer_prefix + "linear_attn.dt_bias"),
            weights.get_weight(layer_prefix + "linear_attn.out_proj.weight"),
            cfg.linear_num_heads, cfg.linear_head_dim, max_seq_len
        );
        attn = delta_attn;
    }

    auto layer = std::make_shared<DecoderLayer>();
    layer->init(norm1, attn, norm2, mlp);
    return layer;
}

std::string Qwen3_5Architecture::remap_tensor_name(const std::string& name) const {
    // Multimodal Qwen3.5 checkpoints store the text backbone under
    // model.language_model.* — map our canonical model.* names onto it.
    if (name.rfind("model.", 0) == 0) {
        return "model.language_model." + name.substr(6);
    }
    return name;
}

std::string Qwen3_5Architecture::default_chat_template() const {
    // Minimal Jinja (within the chat_template renderer's supported subset) that
    // reproduces the engine's historical hardcoded prompt: a non-thinking assistant
    // turn primed with an empty <think></think> block. Used only when the model dir
    // exposes no chat_template.jinja the renderer can handle. Tags are adjacent so
    // only the string literals are emitted (no incidental whitespace).
    //
    // The assistant branch re-emits the same "<think>\n\n</think>\n\n" priming that
    // add_generation_prompt inserted when that turn was generated, instead of just
    // '<|im_start|>assistant\n' + content. Those priming tokens are physically part
    // of what got forwarded into the KV-cache/DeltaNet state for that turn, so a
    // history render that omits them is guaranteed to diverge from the live cache
    // right after the prior turn ends (--optimize kv-reuse would reset every turn).
    // Reproducing them here makes a later turn's render a true superset of the
    // earlier turn's cached tokens, so kv-reuse can actually extend the cache
    // instead of only ever falling back.
    return
        "{%- for message in messages %}"
        "{%- if message.role == 'system' %}"
        "{{- '<|im_start|>system\n' + message.content + '<|im_end|>\n' }}"
        "{%- elif message.role == 'user' %}"
        "{{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}"
        "{%- elif message.role == 'assistant' %}"
        "{{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' + message.content + '<|im_end|>\n' }}"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}"
        "{%- endif %}";
}
