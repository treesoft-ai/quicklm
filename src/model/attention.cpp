#include "attention.hpp"
#include "math_ops.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <iostream>

// --- Qwen3_5Attention (Standard GQA) Implementation ---

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
    Tensor q_t({num_heads, q_head_dim});  // contains query+gate per head
    Tensor k_t({num_kv_heads, head_dim});
    Tensor v_t({num_kv_heads, head_dim});
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
            float ss = 0.0f;
            for (int d = 0; d < head_dim; ++d) ss += q_head[d] * q_head[d];
            float scale = 1.0f / std::sqrt((ss / head_dim) + 1e-6f);
            for (int d = 0; d < head_dim; ++d) q_head[d] = q_head[d] * scale * (1.0f + q_norm.data[d]);
        }
    }
    if (k_norm.data) {
        for (int h = 0; h < num_kv_heads; ++h) {
            float* k_head = k_t.data + h * head_dim;
            float ss = 0.0f;
            for (int d = 0; d < head_dim; ++d) ss += k_head[d] * k_head[d];
            float scale = 1.0f / std::sqrt((ss / head_dim) + 1e-6f);
            for (int d = 0; d < head_dim; ++d) k_head[d] = k_head[d] * scale * (1.0f + k_norm.data[d]);
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
    Tensor attn_out({num_heads, head_dim});
    int group_size = num_heads / num_kv_heads;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int h = 0; h < num_heads; ++h) {
        int kv_h = h / group_size;
        // Use query half of Q head for scoring
        float* q_head = q_t.data + h * q_head_dim;
        float* out_head = attn_out.data + h * head_dim;

        // Dot-product scores for positions 0..pos
        std::vector<float> scores(pos + 1, 0.0f);
        for (int p = 0; p <= pos; ++p) {
            float* k_cached = k_cache.data + p * kv_size + kv_h * head_dim;
            float sum = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                sum += q_head[d] * k_cached[d];
            }
            scores[p] = sum * scale;
        }

        // Softmax
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

        // Weighted sum of V
        std::memset(out_head, 0, head_dim * sizeof(float));
        for (int p = 0; p <= pos; ++p) {
            float* v_cached = v_cache.data + p * kv_size + kv_h * head_dim;
            float w = scores[p];
            for (int d = 0; d < head_dim; ++d) {
                out_head[d] += w * v_cached[d];
            }
        }
    }

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
    Tensor q_t({K, num_heads, q_head_dim});
    Tensor k_t({K, num_kv_heads, head_dim});
    Tensor v_t({K, num_kv_heads, head_dim});
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
                float ss = 0.0f;
                for (int d = 0; d < head_dim; ++d) ss += q_head[d] * q_head[d];
                float scale = 1.0f / std::sqrt((ss / head_dim) + 1e-6f);
                for (int d = 0; d < head_dim; ++d) q_head[d] = q_head[d] * scale * (1.0f + q_norm.data[d]);
            }
        }
        if (k_norm.data) {
            for (int h = 0; h < num_kv_heads; ++h) {
                float* k_head = k_row + h * head_dim;
                float ss = 0.0f;
                for (int d = 0; d < head_dim; ++d) ss += k_head[d] * k_head[d];
                float scale = 1.0f / std::sqrt((ss / head_dim) + 1e-6f);
                for (int d = 0; d < head_dim; ++d) k_head[d] = k_head[d] * scale * (1.0f + k_norm.data[d]);
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
    Tensor attn_out({K, num_heads, head_dim});
    int group_size = num_heads / num_kv_heads;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int r = 0; r < K; ++r) {
        int pos = base_pos + r;
        float* q_row = q_t.data + (size_t)r * q_row_size;
        float* out_row = attn_out.data + (size_t)r * num_heads * head_dim;

        for (int h = 0; h < num_heads; ++h) {
            int kv_h = h / group_size;
            float* q_head = q_row + h * q_head_dim;
            float* out_head = out_row + h * head_dim;

            std::vector<float> scores(pos + 1, 0.0f);
            for (int p = 0; p <= pos; ++p) {
                float* k_cached = k_cache.data + (size_t)p * kv_row_size + kv_h * head_dim;
                float sum = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    sum += q_head[d] * k_cached[d];
                }
                scores[p] = sum * scale;
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
                float* v_cached = v_cache.data + (size_t)p * kv_row_size + kv_h * head_dim;
                float w = scores[p];
                for (int d = 0; d < head_dim; ++d) {
                    out_head[d] += w * v_cached[d];
                }
            }
        }

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

    // conv1d_weight shape is [conv_dim, 1, kernel_size] e.g. [6144, 1, 4]
    int conv_dim = conv1d_weight.shape[0];
    int kernel_size = conv1d_weight.shape.back();
    conv_state = Tensor({conv_dim, kernel_size - 1});

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
    Tensor qkv({3 * hidden_size});
    Tensor z({hidden_size});
    Tensor b_raw({num_heads});
    Tensor a_raw({num_heads});
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
    Tensor qkv_conv({3 * hidden_size});
    math::causal_conv1d_update(qkv, conv_state, conv1d_weight, qkv_conv);

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

        float q_norm = 0.0f, k_norm = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            q_norm += q_h[i] * q_h[i];
            k_norm += k_h[i] * k_h[i];
        }
        q_norm = std::sqrt(q_norm + 1e-6f);
        k_norm = std::sqrt(k_norm + 1e-6f);
        for (int i = 0; i < head_dim; ++i) {
            q_h[i] /= q_norm;
            k_h[i] /= k_norm;
        }
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
    Tensor attn_out({num_heads, head_dim});

    float q_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int h = 0; h < num_heads; ++h) {
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
    }

    // 8. Head-wise RMSNorm (with learned norm_weight, shape [head_dim])
    for (int h = 0; h < num_heads; ++h) {
        float* out_h = attn_out.data + h * head_dim;

        float ss = 0.0f;
        for (int i = 0; i < head_dim; ++i) ss += out_h[i] * out_h[i];
        float scale = 1.0f / std::sqrt((ss / head_dim) + 1e-6f);
        for (int i = 0; i < head_dim; ++i) {
            out_h[i] = out_h[i] * scale * norm_weight.data[i];
        }
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
    Tensor qkv({K, 3 * hidden_size});
    Tensor z({K, hidden_size});
    Tensor b_raw({K, num_heads});
    Tensor a_raw({K, num_heads});
    math::matmul(input, in_proj_qkv, qkv, true);
    math::matmul(input, in_proj_z, z, true);
    math::matmul(input, in_proj_b, b_raw, true);
    math::matmul(input, in_proj_a, a_raw, true);

    Tensor attn_out({K, num_heads, head_dim});
    float q_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int r = 0; r < K; ++r) {
        float* qkv_row = qkv.data + (size_t)r * 3 * hidden_size;
        float* b_row = b_raw.data + (size_t)r * num_heads;
        float* a_row = a_raw.data + (size_t)r * num_heads;
        float* out_row = attn_out.data + (size_t)r * num_heads * head_dim;

        // 2. Causal depthwise conv1d state update — one sequential step.
        Tensor qkv_row_view({3 * hidden_size}, qkv_row);
        Tensor qkv_conv({3 * hidden_size});
        math::causal_conv1d_update(qkv_row_view, conv_state, conv1d_weight, qkv_conv);

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

            float qn = 0.0f, kn = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                qn += q_h[i] * q_h[i];
                kn += k_h[i] * k_h[i];
            }
            qn = std::sqrt(qn + 1e-6f);
            kn = std::sqrt(kn + 1e-6f);
            for (int i = 0; i < head_dim; ++i) {
                q_h[i] /= qn;
                k_h[i] /= kn;
            }
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
        // kernel as the single-position path).
        for (int h = 0; h < num_heads; ++h) {
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
        }

        // 8. Head-wise RMSNorm.
        for (int h = 0; h < num_heads; ++h) {
            float* out_h = out_row + h * head_dim;
            float ss = 0.0f;
            for (int i = 0; i < head_dim; ++i) ss += out_h[i] * out_h[i];
            float scale = 1.0f / std::sqrt((ss / head_dim) + 1e-6f);
            for (int i = 0; i < head_dim; ++i) {
                out_h[i] = out_h[i] * scale * norm_weight.data[i];
            }
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
}

void Qwen3_5GatedDeltaNet::prefetch_weights() const {
    // in_proj_qkv is the largest and first-read projection (z/b/a follow under
    // the same matmul_batched barrier but are much smaller).
    math::prefetch_weight_head(in_proj_qkv);
}
