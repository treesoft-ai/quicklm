#pragma once

#include "modules.hpp"
#include <vector>

// Grouped-Query Attention (GQA) layer for Qwen3.5
class Qwen3_5Attention : public IAttention {
public:
    Qwen3_5Attention() = default;
    virtual ~Qwen3_5Attention() = default;

    void init(
        const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& o,
        const Tensor& q_b, const Tensor& k_b, const Tensor& v_b,
        const Tensor& q_norm_w, const Tensor& k_norm_w,
        int n_heads, int n_kv_heads, int h_dim, int rot_dim, float r_theta, int max_seq_len
    );

    void forward(const Tensor& input, Tensor& output, Context& ctx) override;
    void reset_states() override;

private:
    Tensor q_proj;
    Tensor k_proj;
    Tensor v_proj;
    Tensor o_proj;

    Tensor q_bias;
    Tensor k_bias;
    Tensor v_bias;

    Tensor q_norm;   // per-head RMSNorm weight over head_dim (Qwen3 QK-Norm)
    Tensor k_norm;

    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;    // K/V head dim (e.g. 256)
    int q_head_dim = 0;  // Q head dim in the proj, = 2*head_dim (query | gate)
    int rotary_dim = 0;  // # of dims per head that get RoPE (partial rotary, e.g. 64)
    float rope_theta = 0.0f;

    // KV Cache
    Tensor k_cache;
    Tensor v_cache;
};

// Gated DeltaNet (Linear Attention) layer for Qwen3.5
class Qwen3_5GatedDeltaNet : public IAttention {
public:
    Qwen3_5GatedDeltaNet() = default;
    virtual ~Qwen3_5GatedDeltaNet() = default;

    void init(
        const Tensor& qkv, const Tensor& z, const Tensor& b, const Tensor& a,
        const Tensor& conv_w, const Tensor& norm_w, const Tensor& A_log_t, const Tensor& dt_bias_t,
        const Tensor& out, int n_heads, int h_dim, int max_seq_len
    );

    void forward(const Tensor& input, Tensor& output, Context& ctx) override;
    void reset_states() override;

private:
    Tensor in_proj_qkv;
    Tensor in_proj_z;
    Tensor in_proj_b;
    Tensor in_proj_a;
    Tensor conv1d_weight;
    Tensor norm_weight;
    Tensor A_log;
    Tensor dt_bias;
    Tensor out_proj;

    int num_heads = 0;
    int head_dim = 0;
    int hidden_size = 0;

    // Convolution State: [conv_dim, kernel_size - 1] (where kernel_size = 4, shape [6144, 3])
    Tensor conv_state;

    // Recurrent State: [num_heads, head_dim, head_dim]
    Tensor recurrent_state;
};
