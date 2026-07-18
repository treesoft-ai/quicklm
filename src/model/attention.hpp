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
    void prefetch_weights() const override;

private:
    // Batched/chunked path used when ctx.seq_len > 1 (see definition in
    // attention.cpp for why this is bit-identical to the single-position
    // path called seq_len times).
    void forward_chunk(const Tensor& input, Tensor& output, Context& ctx);

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

    // Persistent scratch backing for q/k/v projections and the attention
    // output (see scratch_view in modules.hpp). Shared by both the
    // single-position and chunked paths (views are re-shaped per call).
    std::vector<float> s_q, s_k, s_v, s_attn;
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
    void prefetch_weights() const override;

    // Speculative decoding reject-path support (design doc §9): conv_state and
    // recurrent_state are dense accumulators mutated in place every step, so a
    // rejected draft token's contribution can't be undone — save both before a
    // draft round, copy back on partial reject. Qwen3_5Attention needs no
    // equivalent (position-indexed K/V cache; stale speculative rows are inert).
    void snapshot_states() override;
    void restore_states() override;

private:
    // Batched/chunked path used when ctx.seq_len > 1. The projections batch
    // across rows (weight read once); the causal conv1d and recurrent
    // delta-rule updates stay the existing single-step primitives, called
    // once per row IN POSITION ORDER, since both are genuine sequential
    // recurrences (each row's update depends on the previous row's state).
    void forward_chunk(const Tensor& input, Tensor& output, Context& ctx);

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

    // Convolution State, tap-major ring buffer: [kernel_size - 1, conv_dim]
    // (one row per time tap, contiguous across channels — vectorizable).
    // conv_ring is the row index of the OLDEST tap; see
    // math::causal_conv1d_update_tapmajor.
    Tensor conv_state;
    int conv_ring = 0;

    // conv1d_weight transposed to [kernel_size, conv_dim] at init, matching the
    // tap-major state layout.
    Tensor conv1d_weight_t;

    // Recurrent State: [num_heads, head_dim, head_dim]
    Tensor recurrent_state;

    // Snapshot buffers for the speculative-decoding reject path (§9). Same
    // shapes as conv_state/recurrent_state; allocated lazily on the first
    // snapshot_states() call so non-speculative runs pay nothing. conv_ring is
    // part of the conv state and is snapshotted alongside.
    Tensor conv_state_snapshot;
    Tensor recurrent_state_snapshot;
    int conv_ring_snapshot = 0;

    // Persistent scratch backing for the per-forward intermediates (see
    // scratch_view in modules.hpp). Shared by both forward paths.
    std::vector<float> s_qkv, s_z, s_b, s_a, s_conv, s_attn;
};
