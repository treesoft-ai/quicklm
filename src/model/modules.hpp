#pragma once

#include "tensor.hpp"
#include "math_ops.hpp"
#include <memory>
#include <string>
#include <vector>

// Context holds runtime state during text generation.
// `pos` is the absolute sequence position of the FIRST token in this
// forward() call. `seq_len` is how many new positions this call covers:
// 1 for the normal single-token path (DecoderModel::forward), >1 for a
// batched/chunked call (DecoderModel::forward_batch) where `input`/`output`
// are [seq_len, hidden_size] instead of [hidden_size]. Attention/DeltaNet
// modules use both together to know which absolute positions rows
// 0..seq_len-1 correspond to (pos, pos+1, ..., pos+seq_len-1).
struct Context {
    int pos = 0;
    int seq_len = 1;
};

// Base interface for all modular layers in the network
class IModule {
public:
    virtual ~IModule() = default;
    virtual void forward(const Tensor& input, Tensor& output, Context& ctx) = 0;
    virtual void reset_states() {}

    // Issue prefetch hints for this module's first-touched weight tensor(s).
    // Called on the NEXT layer while the current layer is still computing, so
    // the hardware has a full layer's worth of compute time to pull the next
    // layer's first weight rows into cache before they're actually needed.
    // Pure latency hiding — default no-op for modules with negligible weights.
    virtual void prefetch_weights() const {}

    // Toggle the QI --optimize=fusion method: merge adjacent elementwise
    // passes (e.g. SiLU+multiply, residual-add+RMSNorm) into a single loop so
    // intermediate values stay in registers instead of round-tripping through
    // memory. Same math, same rounding — pure memory-traffic reduction.
    // Default no-op for modules with no fusable elementwise ops.
    virtual void set_fusion_enabled(bool) {}
};

// Abstract interface for Normalization layers
class INormalization : public IModule {
public:
    virtual ~INormalization() = default;

    // Fused residual-add + normalize: residual[i] += delta[i], then normalize
    // the updated residual into output. Default falls back to a plain add
    // followed by forward(); concrete normalizations that can genuinely fuse
    // the two passes (e.g. RMSNorm) override this.
    virtual void forward_fused_add(Tensor& residual, const Tensor& delta,
                                    Tensor& output, Context& ctx) {
        math::elementwise_add(residual, delta);
        forward(residual, output, ctx);
    }
};

// Abstract interface for Attention mechanisms
class IAttention : public IModule {
public:
    virtual ~IAttention() = default;
};

// Abstract interface for MLP / Feed-Forward blocks
class IMLP : public IModule {
public:
    virtual ~IMLP() = default;
};

// Concrete implementation of RMSNorm
class RMSNorm : public INormalization {
public:
    RMSNorm() = default;
    virtual ~RMSNorm() = default;

    void init(const Tensor& weight_tensor, float epsilon) {
        weight = weight_tensor;
        eps = epsilon;
    }

    void forward(const Tensor& input, Tensor& output, Context& ctx) override {
        (void)ctx;
        math::rms_norm(input, weight, output, eps);
    }

    void forward_fused_add(Tensor& residual, const Tensor& delta,
                            Tensor& output, Context& ctx) override {
        (void)ctx;
        math::add_rms_norm(residual, delta, weight, output, eps);
    }

private:
    Tensor weight;
    float eps = 1e-6f;
};

// Concrete implementation of Qwen SwiGLU MLP:
// MLP(x) = DownProj( SiLU(GateProj(x)) * UpProj(x) )
class SwiGLU_MLP : public IMLP {
public:
    SwiGLU_MLP() = default;
    virtual ~SwiGLU_MLP() = default;

    void init(const Tensor& gate, const Tensor& up, const Tensor& down) {
        gate_proj = gate;
        up_proj = up;
        down_proj = down;
    }

    void forward(const Tensor& input, Tensor& output, Context& ctx) override {
        // input shape: [seq_len, hidden_size] or [hidden_size]
        int seq_len = (input.shape.size() > 1) ? input.shape[0] : 1;
        int hidden_size = input.shape.back();
        int intermediate_size = gate_proj.shape[0];

        // 1+3. Gate and Up projections share the same input — run them under one
        // thread-pool barrier so the memory bus stays fed (bit-identical).
        Tensor gate_out({seq_len, intermediate_size});
        Tensor up_out({seq_len, intermediate_size});
        if (seq_len == 1 && gate_proj.is_bf16() && up_proj.is_bf16()) {
            Tensor in_flat({1, hidden_size}, input.data);
            math::matmul_batched({&in_flat, &in_flat}, {&gate_proj, &up_proj},
                                 {&gate_out, &up_out});
        } else {
            math::matmul(input, gate_proj, gate_out, true);
            math::matmul(input, up_proj, up_out, true);
        }

        // 2+4. SiLU(GateProj(x)) * UpProj(x). Fused into one pass over
        // gate_out when --optimize fusion is on: the SiLU result never
        // leaves a register before being multiplied, instead of writing
        // gate_out then immediately reading it back for the multiply.
        if (fusion_enabled) {
            math::silu_mul(gate_out, up_out);
        } else {
            math::silu(gate_out, gate_out);
            math::elementwise_mul(gate_out, up_out);
        }

        // 5. Down projection: output = DownProj( gate_out )
        math::matmul(gate_out, down_proj, output, true);
    }

    void prefetch_weights() const override {
        // gate_proj/up_proj are read first (and concurrently); down_proj isn't
        // needed until after SiLU+mul, so it gets fewer lines.
        math::prefetch_weight_head(gate_proj);
        math::prefetch_weight_head(up_proj);
        math::prefetch_weight_head(down_proj, 4);
    }

    void set_fusion_enabled(bool enabled) override { fusion_enabled = enabled; }

private:
    Tensor gate_proj;
    Tensor up_proj;
    Tensor down_proj;
    bool fusion_enabled = false;
};

// Concrete implementation of a DecoderLayer linking Normalization, Attention, and MLP
class DecoderLayer : public IModule {
public:
    DecoderLayer() = default;
    virtual ~DecoderLayer() = default;

    void init(
        std::shared_ptr<INormalization> n1,
        std::shared_ptr<IAttention> a,
        std::shared_ptr<INormalization> n2,
        std::shared_ptr<IMLP> m
    ) {
        norm1 = n1;
        attn = a;
        norm2 = n2;
        mlp = m;
    }

    void forward(const Tensor& input, Tensor& output, Context& ctx) override {
        // input shape: [seq_len, hidden_size] or [hidden_size]
        // Copy input to output initially to accumulate residuals in-place
        int size = input.size();
        std::memcpy(output.data, input.data, size * sizeof(float));

        // --- Attention block ---
        Tensor norm1_out(input.shape);
        norm1->forward(output, norm1_out, ctx);

        Tensor attn_out(input.shape);
        attn->forward(norm1_out, attn_out, ctx);

        // --- MLP block ---
        Tensor norm2_out(input.shape);
        if (fusion_enabled) {
            // Fused: output += attn_out, then RMSNorm(output) -> norm2_out,
            // in one pass instead of a separate add followed by a full
            // re-read of output inside norm2->forward().
            norm2->forward_fused_add(output, attn_out, norm2_out, ctx);
        } else {
            // Residual: output = output + attn_out
            math::elementwise_add(output, attn_out);
            norm2->forward(output, norm2_out, ctx);
        }

        Tensor mlp_out(input.shape);
        mlp->forward(norm2_out, mlp_out, ctx);

        // Residual: output = output + mlp_out
        math::elementwise_add(output, mlp_out);
    }

    void reset_states() override {
        if (norm1) norm1->reset_states();
        if (attn) attn->reset_states();
        if (norm2) norm2->reset_states();
        if (mlp) mlp->reset_states();
    }

    // Fan out to the sub-module that will actually be touched first
    // (attention projections); the MLP and norms follow well after attention
    // in this layer's own forward(), so they don't need advance warming here.
    void prefetch_weights() const override {
        if (attn) attn->prefetch_weights();
    }

    void set_fusion_enabled(bool enabled) override {
        fusion_enabled = enabled;
        if (norm2) norm2->set_fusion_enabled(enabled);
        if (mlp) mlp->set_fusion_enabled(enabled);
    }

private:
    std::shared_ptr<INormalization> norm1 = nullptr;
    std::shared_ptr<IAttention> attn = nullptr;
    std::shared_ptr<INormalization> norm2 = nullptr;
    std::shared_ptr<IMLP> mlp = nullptr;
    bool fusion_enabled = false;
};
