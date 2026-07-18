#include "attention.hpp"
#include "math_ops.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <iostream>

// --- Qwen3_5Attention (Standard GQA) Implementation ---

#include <immintrin.h>
#include <vector>
#include <atomic>

namespace {

// AVX2 dot product over a contiguous run (2 accumulators + scalar tail).
inline float dot_f32(const float* a, const float* b, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), a1);
    }
    for (; i + 8 <= n; i += 8) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
    }
    __m256 sum = _mm256_add_ps(a0, a1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float r = _mm_cvtss_f32(lo);
    for (; i < n; ++i) r += a[i] * b[i];
    return r;
}

// out[i] += w * v[i] over a contiguous run. Vectorized over i, so the
// accumulation order across positions p is unchanged — bit-identical to the
// scalar loop it replaces.
inline void axpy_f32(float w, const float* v, float* out, int n) {
    __m256 vw = _mm256_set1_ps(w);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i,
            _mm256_fmadd_ps(vw, _mm256_loadu_ps(v + i), _mm256_loadu_ps(out + i)));
    }
    for (; i < n; ++i) out[i] += w * v[i];
}

// One GQA head: scores over positions 0..pos, softmax, weighted V sum.
// Identical math/loop structure to the previous inline code, with the
// per-position inner loops vectorized over head_dim.
inline void gqa_attend_head(const float* q_head, const float* k_cache,
                            const float* v_cache, float* out_head,
                            int pos, int kv_size, int kv_off, int head_dim,
                            float scale) {
    std::vector<float> scores(pos + 1);
    for (int p = 0; p <= pos; ++p) {
        const float* k_cached = k_cache + (size_t)p * kv_size + kv_off;
        scores[p] = dot_f32(q_head, k_cached, head_dim) * scale;
    }

    float max_score = scores[0];
    for (int p = 1; p <= pos; ++p) {
        if (scores[p] > max_score) max_score = scores[p];
    }
    float exp_sum = 0.0f;
    for (int p = 0; p <= pos; ++p) {
        scores[p] = std::exp(scores[p] - max_score);
        exp_sum += scores[p];
    }
    for (int p = 0; p <= pos; ++p) {
        scores[p] /= exp_sum;
    }

    std::memset(out_head, 0, head_dim * sizeof(float));
    for (int p = 0; p <= pos; ++p) {
        const float* v_cached = v_cache + (size_t)p * kv_size + kv_off;
        axpy_f32(scores[p], v_cached, out_head, head_dim);
    }
}

// Run fn(h) for h in [0, num_heads), splitting heads across the fork-join pool
// when there is enough work to amortize dispatch. Heads are fully independent
// (disjoint outputs), so parallelization is bit-identical. Heads are handed
// out dynamically off an atomic counter so uneven per-head timing (or an OS
// preemption) doesn't strand a fixed chunk on one thread.
template <typename F>
inline void parallel_heads(int num_heads, long long work_per_head, F&& fn) {
    if (math::parallel_width() <= 1 || work_per_head * num_heads < (1LL << 15)) {
        for (int h = 0; h < num_heads; ++h) fn(h);
        return;
    }
    std::atomic<int> next_head{0};
    math::parallel_invoke([&](int, int) {
        for (;;) {
            int h = next_head.fetch_add(1, std::memory_order_relaxed);
            if (h >= num_heads) break;
            fn(h);
        }
    });
}

} // namespace

void Qwen3_5Attention::init(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& o,
    const Tensor& q_b, const Tensor& k_b, const Tensor& v_b,
    const Tensor& q_norm_w, const Tensor& k_norm_w,
    int n_heads, int n_kv_heads, int h_dim, int rot_dim, float r_theta, int max_seq_len
) {
    q_proj = q;
    k_proj = k;
    v_proj = v;
    o_proj = o;

    q_bias = q_b;
    k_bias = k_b;
    v_bias = v_b;

    q_norm = q_norm_w;
    k_norm = k_norm_w;

    num_heads = n_heads;
    num_kv_heads = n_kv_heads;
    head_dim = h_dim;
    rotary_dim = rot_dim;
    // q_proj emits [query | gate] per head (attn_output_gate=true) -> 2*head_dim wide
    q_head_dim = (q_proj.shape.size() >= 1) ? q_proj.shape[0] / num_heads : 2 * h_dim;
    rope_theta = r_theta;

    // Allocate KV Cache (uses K/V head_dim)
    k_cache = Tensor({max_seq_len, num_kv_heads, head_dim});
    v_cache = Tensor({max_seq_len, num_kv_heads, head_dim});
}

void Qwen3_5Attention::forward(const Tensor& input, Tensor& output, Context& ctx) {
    if (ctx.seq_len > 1) {
        forward_chunk(input, output, ctx);
        return;
    }

    // input is shape [hidden_size]
    // q_proj output per head = [query(head_dim) | gate(head_dim)] (query first, gate second)

    // 1. QKV Projections — q, k, v share the same input; batch under one barrier.
    Tensor q_t = scratch_view(s_q, {num_heads, q_head_dim});  // query+gate per head
    Tensor k_t = scratch_view(s_k, {num_kv_heads, head_dim});
    Tensor v_t = scratch_view(s_v, {num_kv_heads, head_dim});
    if (q_proj.is_bf16() && k_proj.is_bf16() && v_proj.is_bf16()) {
        Tensor in_flat({1, q_proj.shape.back()}, input.data);
        math::matmul_batched({&in_flat, &in_flat, &in_flat},
                             {&q_proj, &k_proj, &v_proj}, {&q_t, &k_t, &v_t});
    } else {
        math::matmul(input, q_proj, q_t, true);
        math::matmul(input, k_proj, k_t, true);
        math::matmul(input, v_proj, v_t, true);
    }
    if (q_bias.data) math::elementwise_add(q_t, q_bias);
    if (k_bias.data) math::elementwise_add(k_t, k_bias);
    if (v_bias.data) math::elementwise_add(v_t, v_bias);

    // 2. QK-Norm: per-head RMSNorm over head_dim (applied to the query half of each
    //    Q head and to each K head), BEFORE RoPE.
    if (q_norm.data) {
        for (int h = 0; h < num_heads; ++h) {
            float* q_head = q_t.data + h * q_head_dim;  // query is first head_dim dims
            float scale = 1.0f / std::sqrt((math::sumsq(q_head, head_dim) / head_dim) + 1e-6f);
            math::norm_apply_zc(q_head, q_norm.data, q_head, scale, head_dim);
        }
    }
    if (k_norm.data) {
        for (int h = 0; h < num_kv_heads; ++h) {
            float* k_head = k_t.data + h * head_dim;
            float scale = 1.0f / std::sqrt((math::sumsq(k_head, head_dim) / head_dim) + 1e-6f);
            math::norm_apply_zc(k_head, k_norm.data, k_head, scale, head_dim);
        }
    }

    // 3. Apply partial NeoX-style RoPE (only first rotary_dim dims per head).
    //    K: stride == head_dim (contiguous). Q: stride == q_head_dim, rotate query half only.
    math::apply_rope_neox(k_t, ctx.pos, rope_theta, head_dim, rotary_dim);
    for (int h = 0; h < num_heads; ++h) {
        // View the query half (first head_dim dims) of this Q head as a single head.
        Tensor q_head_view({1, head_dim}, q_t.data + h * q_head_dim);
        math::apply_rope_neox(q_head_view, ctx.pos, rope_theta, head_dim, rotary_dim);
    }

    // 4. Write k_t and v_t to the KV cache at position pos
    int pos = ctx.pos;
    if (pos >= k_cache.shape[0]) {
        throw std::runtime_error("KV cache index out of bounds: pos = " + std::to_string(pos) +
                                 ", max_seq_len = " + std::to_string(k_cache.shape[0]));
    }

    int kv_size = num_kv_heads * head_dim;
    std::memcpy(k_cache.data + pos * kv_size, k_t.data, kv_size * sizeof(float));
    std::memcpy(v_cache.data + pos * kv_size, v_t.data, kv_size * sizeof(float));

    // 5. Grouped-Query Attention
    // Output shape [num_heads, head_dim] (uses head_dim from V)
    Tensor attn_out = scratch_view(s_attn, {num_heads, head_dim});
    int group_size = num_heads / num_kv_heads;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // AVX2 scores/weighted-V per head, heads split across the thread pool
    // (each head writes a disjoint output slice).
    parallel_heads(num_heads, 2LL * (pos + 1) * head_dim, [&](int h) {
        int kv_h = h / group_size;
        gqa_attend_head(q_t.data + h * q_head_dim, k_cache.data, v_cache.data,
                        attn_out.data + h * head_dim, pos, kv_size,
                        kv_h * head_dim, head_dim, scale);
    });

    // 6. Output gate: present when q_proj emits [query|gate] per head (q_head_dim > head_dim).
    //    Standard GQA models have q_head_dim == head_dim and skip this entirely.
    if (q_head_dim > head_dim) {
        for (int h = 0; h < num_heads; ++h) {
            float* out_head = attn_out.data + h * head_dim;
            float* gate_head = q_t.data + h * q_head_dim + head_dim;
            for (int d = 0; d < head_dim; ++d) {
                float g = gate_head[d];
                out_head[d] *= 1.0f / (1.0f + std::exp(-g));
            }
        }
    }

    // 7. Output projection: attn_out is [num_heads, head_dim] = [8, 256] = 2048 total
    Tensor attn_out_flat({1, num_heads * head_dim}, attn_out.data);
    math::matmul(attn_out_flat, o_proj, output, true);
}

// Batched/chunked path: input/output are [seq_len, hidden_size], covering
// absolute positions ctx.pos .. ctx.pos+ctx.seq_len-1. Used to verify a whole
// speculative-decoding draft chunk against the target model in one pass
// instead of one forward() call per drafted token (see QI_roadmap.md).
//
// This computes EXACTLY the same thing as calling the single-position path
// above once per row, in row order: every reduction (QK-norm, softmax,
// output projection) still happens within one row's data via the identical
// per-row code, so results are bit-identical to the sequential path, not
// just numerically close. The only thing batching changes is that q/k/v/o
// projections now read each weight matrix once for all seq_len rows instead
// of once per row (matmul() already supports M>1 rows for this).
void Qwen3_5Attention::forward_chunk(const Tensor& input, Tensor& output, Context& ctx) {
    int K = ctx.seq_len;
    int base_pos = ctx.pos;

    // 1. QKV projections for all K rows in one batched matmul call each
    //    (weight matrix read once, applied to K activation rows).
    Tensor q_t = scratch_view(s_q, {K, num_heads, q_head_dim});
    Tensor k_t = scratch_view(s_k, {K, num_kv_heads, head_dim});
    Tensor v_t = scratch_view(s_v, {K, num_kv_heads, head_dim});
    math::matmul(input, q_proj, q_t, true);
    math::matmul(input, k_proj, k_t, true);
    math::matmul(input, v_proj, v_t, true);

    int q_row_size = num_heads * q_head_dim;
    int kv_row_size = num_kv_heads * head_dim;

    // 2-3. Bias, QK-norm, and RoPE are all per-row/per-head operations with
    // no cross-row reduction, so each row is processed with the exact same
    // per-row code the single-position path uses, just looped over rows.
    for (int r = 0; r < K; ++r) {
        int pos = base_pos + r;
        float* q_row = q_t.data + (size_t)r * q_row_size;
        float* k_row = k_t.data + (size_t)r * kv_row_size;
        float* v_row = v_t.data + (size_t)r * kv_row_size;

        if (q_bias.data) {
            Tensor q_row_view({q_row_size}, q_row);
            math::elementwise_add(q_row_view, q_bias);
        }
        if (k_bias.data) {
            Tensor k_row_view({kv_row_size}, k_row);
            math::elementwise_add(k_row_view, k_bias);
        }
        if (v_bias.data) {
            Tensor v_row_view({kv_row_size}, v_row);
            math::elementwise_add(v_row_view, v_bias);
        }

        if (q_norm.data) {
            for (int h = 0; h < num_heads; ++h) {
                float* q_head = q_row + h * q_head_dim;
                float scale = 1.0f / std::sqrt((math::sumsq(q_head, head_dim) / head_dim) + 1e-6f);
                math::norm_apply_zc(q_head, q_norm.data, q_head, scale, head_dim);
            }
        }
        if (k_norm.data) {
            for (int h = 0; h < num_kv_heads; ++h) {
                float* k_head = k_row + h * head_dim;
                float scale = 1.0f / std::sqrt((math::sumsq(k_head, head_dim) / head_dim) + 1e-6f);
                math::norm_apply_zc(k_head, k_norm.data, k_head, scale, head_dim);
            }
        }

        Tensor k_row_view({num_kv_heads, head_dim}, k_row);
        math::apply_rope_neox(k_row_view, pos, rope_theta, head_dim, rotary_dim);
        for (int h = 0; h < num_heads; ++h) {
            Tensor q_head_view({1, head_dim}, q_row + h * q_head_dim);
            math::apply_rope_neox(q_head_view, pos, rope_theta, head_dim, rotary_dim);
        }

        // 4. Write this row's K/V into the cache. Every row is written BEFORE
        // any row's attention output is computed below, so row r's attention
        // sees positions 0..base_pos+r including earlier rows of this same
        // chunk — identical to what the sequential single-position path
        // would have in cache by the time it reached position base_pos+r.
        if (pos >= k_cache.shape[0]) {
            throw std::runtime_error("KV cache index out of bounds: pos = " + std::to_string(pos) +
                                     ", max_seq_len = " + std::to_string(k_cache.shape[0]));
        }
        std::memcpy(k_cache.data + (size_t)pos * kv_row_size, k_row, kv_row_size * sizeof(float));
        std::memcpy(v_cache.data + (size_t)pos * kv_row_size, v_row, kv_row_size * sizeof(float));
    }

    // 5. Grouped-Query Attention, per row, per head — identical math/loop
    // structure to the single-position path, just with pos = base_pos + r.
    Tensor attn_out = scratch_view(s_attn, {K, num_heads, head_dim});
    int group_size = num_heads / num_kv_heads;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int r = 0; r < K; ++r) {
        int pos = base_pos + r;
        float* q_row = q_t.data + (size_t)r * q_row_size;
        float* out_row = attn_out.data + (size_t)r * num_heads * head_dim;

        // Same AVX2 + parallel-heads kernel as the single-position path.
        parallel_heads(num_heads, 2LL * (pos + 1) * head_dim, [&](int h) {
            int kv_h = h / group_size;
            gqa_attend_head(q_row + h * q_head_dim, k_cache.data, v_cache.data,
                            out_row + h * head_dim, pos, kv_row_size,
                            kv_h * head_dim, head_dim, scale);
        });

        // 6. Output gate (same condition/formula as the single-position path).
        if (q_head_dim > head_dim) {
            for (int h = 0; h < num_heads; ++h) {
                float* out_head = out_row + h * head_dim;
                float* gate_head = q_row + h * q_head_dim + head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    float g = gate_head[d];
                    out_head[d] *= 1.0f / (1.0f + std::exp(-g));
                }
            }
        }
    }

    // 7. Output projection for all K rows in one batched matmul call.
    // matmul() only reads A.shape.back() as the contraction dim, so a 3D
    // [K, num_heads, head_dim] tensor must be flattened to 2D first (same
    // reason the single-position path above builds attn_out_flat).
    Tensor attn_out_flat({K, num_heads * head_dim}, attn_out.data);
    math::matmul(attn_out_flat, o_proj, output, true);
}

void Qwen3_5Attention::reset_states() {
    std::memset(k_cache.data, 0, k_cache.size() * sizeof(float));
    std::memset(v_cache.data, 0, v_cache.size() * sizeof(float));
}

void Qwen3_5Attention::prefetch_weights() const {
    // q/k/v projections are read first (and concurrently, via matmul_batched).
    math::prefetch_weight_head(q_proj);
    math::prefetch_weight_head(k_proj);
    math::prefetch_weight_head(v_proj);
}

// --- Qwen3_5GatedDeltaNet (Linear Attention) Implementation ---

void Qwen3_5GatedDeltaNet::init(
    const Tensor& qkv, const Tensor& z, const Tensor& b, const Tensor& a,
    const Tensor& conv_w, const Tensor& norm_w, const Tensor& A_log_t, const Tensor& dt_bias_t,
    const Tensor& out, int n_heads, int h_dim, int max_seq_len
) {
    (void)max_seq_len;
    in_proj_qkv = qkv;
    in_proj_z = z;
    in_proj_b = b;
    in_proj_a = a;
    conv1d_weight = conv_w;
    norm_weight = norm_w;
    A_log = A_log_t;
    dt_bias = dt_bias_t;
    out_proj = out;

    num_heads = n_heads;
    head_dim = h_dim;
    hidden_size = num_heads * head_dim;

    // conv1d_weight shape is [conv_dim, 1, kernel_size] e.g. [6144, 1, 4].
    // The conv state is kept tap-major ([kernel_size-1, conv_dim] ring buffer)
    // and the kernel transposed to match, so the per-step depthwise conv is a
    // plain contiguous elementwise pass (see causal_conv1d_update_tapmajor).
    int conv_dim = conv1d_weight.shape[0];
    int kernel_size = conv1d_weight.shape.back();
    conv_state = Tensor({kernel_size - 1, conv_dim});
    conv_ring = 0;

    conv1d_weight_t = Tensor({kernel_size, conv_dim});
    for (int c = 0; c < conv_dim; ++c) {
        for (int k = 0; k < kernel_size; ++k) {
            conv1d_weight_t.data[(size_t)k * conv_dim + c] =
                conv1d_weight.data[(size_t)c * kernel_size + k];
        }
    }

    // Allocate Recurrent State: [num_heads, head_dim, head_dim]
    recurrent_state = Tensor({num_heads, head_dim, head_dim});
}

void Qwen3_5GatedDeltaNet::forward(const Tensor& input, Tensor& output, Context& ctx) {
    if (ctx.seq_len > 1) {
        forward_chunk(input, output, ctx);
        return;
    }
    // input shape: [hidden_size] (rank 1)

    // 1. Projections — qkv, z, b, a all read the same input; batch them under one
    //    thread-pool barrier to keep the memory bus saturated (bit-identical).
    Tensor qkv = scratch_view(s_qkv, {3 * hidden_size});
    Tensor z = scratch_view(s_z, {hidden_size});
    Tensor b_raw = scratch_view(s_b, {num_heads});
    Tensor a_raw = scratch_view(s_a, {num_heads});
    if (in_proj_qkv.is_bf16() && in_proj_z.is_bf16() &&
        in_proj_b.is_bf16() && in_proj_a.is_bf16()) {
        Tensor in_flat({1, in_proj_qkv.shape.back()}, input.data);
        math::matmul_batched({&in_flat, &in_flat, &in_flat, &in_flat},
                             {&in_proj_qkv, &in_proj_z, &in_proj_b, &in_proj_a},
                             {&qkv, &z, &b_raw, &a_raw});
    } else {
        math::matmul(input, in_proj_qkv, qkv, true);
        math::matmul(input, in_proj_z, z, true);
        math::matmul(input, in_proj_b, b_raw, true);
        math::matmul(input, in_proj_a, a_raw, true);
    }

    // 2. Causal depthwise conv1d state update on the full qkv vector
    // (tap-major ring layout; bit-identical per-channel arithmetic).
    Tensor qkv_conv = scratch_view(s_conv, {3 * hidden_size});
    int conv_kernel = conv1d_weight.shape.back();
    math::causal_conv1d_update_tapmajor(qkv.data, conv_state.data, conv_ring,
                                        conv1d_weight_t.data, qkv_conv.data,
                                        3 * hidden_size, conv_kernel);
    conv_ring = (conv_ring + 1) % (conv_kernel - 1);

    // 3. Split into q, k, v — each [num_heads, head_dim]
    // Layout: [q0..qN | k0..kN | v0..vN]
    float* q_ptr = qkv_conv.data;
    float* k_ptr = qkv_conv.data + hidden_size;
    float* v_ptr = qkv_conv.data + 2 * hidden_size;

    // 4. SiLU activation on the full post-conv q,k,v (the conv activation in HF:
    //    F.silu(conv1d(...))), applied before L2-norm of q/k.
    for (int i = 0; i < 3 * hidden_size; ++i) {
        float x = qkv_conv.data[i];
        qkv_conv.data[i] = x / (1.0f + std::exp(-x));
    }

    // 5. Normalize q and k (L2 per head) — required by Gated DeltaNet for stability
    for (int h = 0; h < num_heads; ++h) {
        float* q_h = q_ptr + h * head_dim;
        float* k_h = k_ptr + h * head_dim;
        float qn = std::sqrt(math::sumsq(q_h, head_dim) + 1e-6f);
        float kn = std::sqrt(math::sumsq(k_h, head_dim) + 1e-6f);
        math::vec_div_inplace(q_h, qn, head_dim);
        math::vec_div_inplace(k_h, kn, head_dim);
    }

    // 6. Compute per-head decay (alpha) and write gate (beta)
    std::vector<float> decay(num_heads);
    std::vector<float> write(num_heads);
    for (int h = 0; h < num_heads; ++h) {
        float dt_in = a_raw.data[h] + dt_bias.data[h];
        float sp = (dt_in > 20.0f) ? dt_in : std::log1p(std::exp(dt_in)); // softplus
        float alpha_h = -std::exp(A_log.data[h]) * sp;
        decay[h] = std::exp(alpha_h);

        // write gate = sigmoid(b)
        write[h] = 1.0f / (1.0f + std::exp(-b_raw.data[h]));
    }

    // 7. Gated Delta Rule recurrent state update + output
    //    S_h shape [head_dim, head_dim] — recurrent_state stores row-major
    //    Algorithm:
    //      S_t = decay * S_{t-1}
    //      kv_mem = k @ S_t   (1×d_k) @ (d_k×d_k) → (1×d_k)
    //      S_t += write * outer(k, v - kv_mem)
    //      y = q @ S_t        (1×d_k) @ (d_k×d_k) → (1×d_k)
    Tensor attn_out = scratch_view(s_attn, {num_heads, head_dim});

    float q_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Heads are independent (disjoint state/output slices) — split across the
    // thread pool. Per-head math unchanged, so results are bit-identical.
    parallel_heads(num_heads, 3LL * head_dim * head_dim, [&](int h) {
        float* state_h = recurrent_state.data + h * head_dim * head_dim;
        const float* q_h = q_ptr + h * head_dim;
        const float* k_h = k_ptr + h * head_dim;
        const float* v_h = v_ptr + h * head_dim;
        float* out_h = attn_out.data + h * head_dim;

        float dec = decay[h];
        float beta = write[h];

        // Decay state and compute kv_mem = k @ S (S is [d_k, d_k])
        // kv_mem[j] = sum_i k[i] * (S[i,j] * dec).  Row-wise over j → AVX2.
        std::vector<float> kv_mem(head_dim, 0.0f);
        {
            __m256 vdec = _mm256_set1_ps(dec);
            for (int i = 0; i < head_dim; ++i) {
                float* row = state_h + i * head_dim;
                __m256 vk = _mm256_set1_ps(k_h[i]);
                int j = 0;
                for (; j + 8 <= head_dim; j += 8) {
                    __m256 r = _mm256_mul_ps(_mm256_loadu_ps(row + j), vdec);
                    _mm256_storeu_ps(row + j, r);
                    __m256 acc = _mm256_loadu_ps(kv_mem.data() + j);
                    acc = _mm256_fmadd_ps(vk, r, acc);
                    _mm256_storeu_ps(kv_mem.data() + j, acc);
                }
                for (; j < head_dim; ++j) {
                    row[j] *= dec;
                    kv_mem[j] += k_h[i] * row[j];
                }
            }
        }

        // Update state: S[i,j] += (beta*k[i]) * (v[j] - kv_mem[j]).  Row-wise over j.
        {
            std::vector<float> delta(head_dim);
            for (int j = 0; j < head_dim; ++j) delta[j] = v_h[j] - kv_mem[j];
            for (int i = 0; i < head_dim; ++i) {
                float s = beta * k_h[i];
                __m256 vs = _mm256_set1_ps(s);
                float* row = state_h + i * head_dim;
                int j = 0;
                for (; j + 8 <= head_dim; j += 8) {
                    __m256 r = _mm256_fmadd_ps(vs, _mm256_loadu_ps(delta.data() + j),
                                               _mm256_loadu_ps(row + j));
                    _mm256_storeu_ps(row + j, r);
                }
                for (; j < head_dim; ++j) row[j] += s * delta[j];
            }
        }

        // Retrieve: y[j] = q_scale * sum_i q[i] * S[i,j].  Reorder to SAXPY over j
        // (accumulate y[j] += q[i]*S[i,j]) so the inner loop is contiguous → AVX2.
        {
            for (int j = 0; j < head_dim; ++j) out_h[j] = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                __m256 vq = _mm256_set1_ps(q_h[i]);
                const float* row = state_h + i * head_dim;
                int j = 0;
                for (; j + 8 <= head_dim; j += 8) {
                    __m256 acc = _mm256_fmadd_ps(vq, _mm256_loadu_ps(row + j),
                                                 _mm256_loadu_ps(out_h + j));
                    _mm256_storeu_ps(out_h + j, acc);
                }
                for (; j < head_dim; ++j) out_h[j] += q_h[i] * row[j];
            }
            __m256 vqs = _mm256_set1_ps(q_scale);
            int j = 0;
            for (; j + 8 <= head_dim; j += 8) {
                _mm256_storeu_ps(out_h + j, _mm256_mul_ps(_mm256_loadu_ps(out_h + j), vqs));
            }
            for (; j < head_dim; ++j) out_h[j] *= q_scale;
        }
    });

    // 8. Head-wise RMSNorm (with learned norm_weight, shape [head_dim])
    for (int h = 0; h < num_heads; ++h) {
        float* out_h = attn_out.data + h * head_dim;
        float scale = 1.0f / std::sqrt((math::sumsq(out_h, head_dim) / head_dim) + 1e-6f);
        math::norm_apply_gain(out_h, norm_weight.data, out_h, scale, head_dim);
    }

    // 9. Output gate: element-wise multiply by SiLU(z)
    //    z has shape {hidden_size}, attn_out is {num_heads*head_dim} = {hidden_size}
    for (int i = 0; i < hidden_size; ++i) {
        float zv = z.data[i];
        attn_out.data[i] *= zv / (1.0f + std::exp(-zv));  // SiLU gate
    }

    // 10. Output projection
    Tensor attn_flat({1, hidden_size}, attn_out.data);
    math::matmul(attn_flat, out_proj, output, true);
}

// Batched/chunked path used when ctx.seq_len > 1. The four input projections
// (qkv/z/b/a) batch across all K rows in one matmul call each — each weight
// matrix is read once instead of once per row. The causal conv1d state and
// the recurrent delta-rule state are genuine sequential recurrences (each
// row's update depends on the previous row's state), so those stay the
// existing single-step primitives, called once per row IN POSITION ORDER —
// identical math to the single-position path, just invoked K times instead
// of relying on K separate forward() calls to supply the row loop.
void Qwen3_5GatedDeltaNet::forward_chunk(const Tensor& input, Tensor& output, Context& ctx) {
    int K = ctx.seq_len;

    // 1. Batched projections for all K rows.
    Tensor qkv = scratch_view(s_qkv, {K, 3 * hidden_size});
    Tensor z = scratch_view(s_z, {K, hidden_size});
    Tensor b_raw = scratch_view(s_b, {K, num_heads});
    Tensor a_raw = scratch_view(s_a, {K, num_heads});
    math::matmul(input, in_proj_qkv, qkv, true);
    math::matmul(input, in_proj_z, z, true);
    math::matmul(input, in_proj_b, b_raw, true);
    math::matmul(input, in_proj_a, a_raw, true);

    Tensor attn_out = scratch_view(s_attn, {K, num_heads, head_dim});
    float q_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int r = 0; r < K; ++r) {
        float* qkv_row = qkv.data + (size_t)r * 3 * hidden_size;
        float* b_row = b_raw.data + (size_t)r * num_heads;
        float* a_row = a_raw.data + (size_t)r * num_heads;
        float* out_row = attn_out.data + (size_t)r * num_heads * head_dim;

        // 2. Causal depthwise conv1d state update — one sequential step
        // (tap-major ring layout; bit-identical per-channel arithmetic).
        Tensor qkv_conv = scratch_view(s_conv, {3 * hidden_size});
        int conv_kernel = conv1d_weight.shape.back();
        math::causal_conv1d_update_tapmajor(qkv_row, conv_state.data, conv_ring,
                                            conv1d_weight_t.data, qkv_conv.data,
                                            3 * hidden_size, conv_kernel);
        conv_ring = (conv_ring + 1) % (conv_kernel - 1);

        // 3. Split into q, k, v — each [num_heads, head_dim].
        float* q_ptr = qkv_conv.data;
        float* k_ptr = qkv_conv.data + hidden_size;
        float* v_ptr = qkv_conv.data + 2 * hidden_size;

        // 4. SiLU on the full post-conv q,k,v.
        for (int i = 0; i < 3 * hidden_size; ++i) {
            float x = qkv_conv.data[i];
            qkv_conv.data[i] = x / (1.0f + std::exp(-x));
        }

        // 5. L2-norm per head on q and k.
        for (int h = 0; h < num_heads; ++h) {
            float* q_h = q_ptr + h * head_dim;
            float* k_h = k_ptr + h * head_dim;
            float qn = std::sqrt(math::sumsq(q_h, head_dim) + 1e-6f);
            float kn = std::sqrt(math::sumsq(k_h, head_dim) + 1e-6f);
            math::vec_div_inplace(q_h, qn, head_dim);
            math::vec_div_inplace(k_h, kn, head_dim);
        }

        // 6. Per-head decay (alpha) and write gate (beta).
        std::vector<float> decay(num_heads);
        std::vector<float> write(num_heads);
        for (int h = 0; h < num_heads; ++h) {
            float dt_in = a_row[h] + dt_bias.data[h];
            float sp = (dt_in > 20.0f) ? dt_in : std::log1p(std::exp(dt_in));
            float alpha_h = -std::exp(A_log.data[h]) * sp;
            decay[h] = std::exp(alpha_h);
            write[h] = 1.0f / (1.0f + std::exp(-b_row[h]));
        }

        // 7. Gated Delta Rule recurrent state update + output — one
        // sequential step, mutates recurrent_state in place (same AVX2
        // kernel as the single-position path). Heads are independent →
        // split across the thread pool (bit-identical).
        parallel_heads(num_heads, 3LL * head_dim * head_dim, [&](int h) {
            float* state_h = recurrent_state.data + h * head_dim * head_dim;
            const float* q_h = q_ptr + h * head_dim;
            const float* k_h = k_ptr + h * head_dim;
            const float* v_h = v_ptr + h * head_dim;
            float* out_h = out_row + h * head_dim;

            float dec = decay[h];
            float beta = write[h];

            std::vector<float> kv_mem(head_dim, 0.0f);
            {
                __m256 vdec = _mm256_set1_ps(dec);
                for (int i = 0; i < head_dim; ++i) {
                    float* row = state_h + i * head_dim;
                    __m256 vk = _mm256_set1_ps(k_h[i]);
                    int j = 0;
                    for (; j + 8 <= head_dim; j += 8) {
                        __m256 rr = _mm256_mul_ps(_mm256_loadu_ps(row + j), vdec);
                        _mm256_storeu_ps(row + j, rr);
                        __m256 acc = _mm256_loadu_ps(kv_mem.data() + j);
                        acc = _mm256_fmadd_ps(vk, rr, acc);
                        _mm256_storeu_ps(kv_mem.data() + j, acc);
                    }
                    for (; j < head_dim; ++j) {
                        row[j] *= dec;
                        kv_mem[j] += k_h[i] * row[j];
                    }
                }
            }

            {
                std::vector<float> delta(head_dim);
                for (int j = 0; j < head_dim; ++j) delta[j] = v_h[j] - kv_mem[j];
                for (int i = 0; i < head_dim; ++i) {
                    float s = beta * k_h[i];
                    __m256 vs = _mm256_set1_ps(s);
                    float* row = state_h + i * head_dim;
                    int j = 0;
                    for (; j + 8 <= head_dim; j += 8) {
                        __m256 rr = _mm256_fmadd_ps(vs, _mm256_loadu_ps(delta.data() + j),
                                                    _mm256_loadu_ps(row + j));
                        _mm256_storeu_ps(row + j, rr);
                    }
                    for (; j < head_dim; ++j) row[j] += s * delta[j];
                }
            }

            {
                for (int j = 0; j < head_dim; ++j) out_h[j] = 0.0f;
                for (int i = 0; i < head_dim; ++i) {
                    __m256 vq = _mm256_set1_ps(q_h[i]);
                    const float* row = state_h + i * head_dim;
                    int j = 0;
                    for (; j + 8 <= head_dim; j += 8) {
                        __m256 acc = _mm256_fmadd_ps(vq, _mm256_loadu_ps(row + j),
                                                     _mm256_loadu_ps(out_h + j));
                        _mm256_storeu_ps(out_h + j, acc);
                    }
                    for (; j < head_dim; ++j) out_h[j] += q_h[i] * row[j];
                }
                __m256 vqs = _mm256_set1_ps(q_scale);
                int j = 0;
                for (; j + 8 <= head_dim; j += 8) {
                    _mm256_storeu_ps(out_h + j, _mm256_mul_ps(_mm256_loadu_ps(out_h + j), vqs));
                }
                for (; j < head_dim; ++j) out_h[j] *= q_scale;
            }
        });

        // 8. Head-wise RMSNorm.
        for (int h = 0; h < num_heads; ++h) {
            float* out_h = out_row + h * head_dim;
            float scale = 1.0f / std::sqrt((math::sumsq(out_h, head_dim) / head_dim) + 1e-6f);
            math::norm_apply_gain(out_h, norm_weight.data, out_h, scale, head_dim);
        }

        // 9. Output gate: element-wise multiply by SiLU(z) for this row.
        float* z_row = z.data + (size_t)r * hidden_size;
        for (int i = 0; i < hidden_size; ++i) {
            float zv = z_row[i];
            out_row[i] *= zv / (1.0f + std::exp(-zv));
        }
    }

    // 10. Output projection for all K rows in one batched matmul call.
    // matmul() only reads A.shape.back() as the contraction dim, so the 3D
    // [K, num_heads, head_dim] tensor must be flattened to 2D first (same
    // reason the single-position path above builds attn_flat).
    Tensor attn_out_flat({K, num_heads * head_dim}, attn_out.data);
    math::matmul(attn_out_flat, out_proj, output, true);
}

void Qwen3_5GatedDeltaNet::reset_states() {
    std::memset(conv_state.data, 0, conv_state.size() * sizeof(float));
    std::memset(recurrent_state.data, 0, recurrent_state.size() * sizeof(float));
    conv_ring = 0;
}

void Qwen3_5GatedDeltaNet::snapshot_states() {
    // Lazy allocation: non-speculative runs never call this, so they never pay
    // for the buffers (recurrent_state is the expensive one: dense, per-head).
    if (conv_state_snapshot.size() == 0) {
        conv_state_snapshot = Tensor(conv_state.shape);
        recurrent_state_snapshot = Tensor(recurrent_state.shape);
    }
    std::memcpy(conv_state_snapshot.data, conv_state.data,
                conv_state.size() * sizeof(float));
    std::memcpy(recurrent_state_snapshot.data, recurrent_state.data,
                recurrent_state.size() * sizeof(float));
    conv_ring_snapshot = conv_ring;
}

void Qwen3_5GatedDeltaNet::restore_states() {
    if (conv_state_snapshot.size() == 0) {
        throw std::runtime_error(
            "Qwen3_5GatedDeltaNet::restore_states() called before any snapshot_states()");
    }
    std::memcpy(conv_state.data, conv_state_snapshot.data,
                conv_state.size() * sizeof(float));
    std::memcpy(recurrent_state.data, recurrent_state_snapshot.data,
                recurrent_state.size() * sizeof(float));
    conv_ring = conv_ring_snapshot;
}

void Qwen3_5GatedDeltaNet::prefetch_weights() const {
    // in_proj_qkv is the largest and first-read projection (z/b/a follow under
    // the same matmul_batched barrier but are much smaller).
    math::prefetch_weight_head(in_proj_qkv);
}
