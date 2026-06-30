#pragma once

#include "tensor.hpp"
#include "math_ops.hpp"
#include <memory>
#include <string>
#include <vector>

// Context holds runtime state during text generation
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
};

// Abstract interface for Normalization layers
class INormalization : public IModule {
public:
    virtual ~INormalization() = default;
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

        // 2. SiLU activation on gate projection
        math::silu(gate_out, gate_out);

        // 4. Elementwise product: gate_out = SiLU(GateProj(x)) * UpProj(x)
        math::elementwise_mul(gate_out, up_out);

        // 5. Down projection: output = DownProj( gate_out )
        math::matmul(gate_out, down_proj, output, true);
    }

private:
    Tensor gate_proj;
    Tensor up_proj;
    Tensor down_proj;
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

        // Residual: output = output + attn_out
        math::elementwise_add(output, attn_out);

        // --- MLP block ---
        Tensor norm2_out(input.shape);
        norm2->forward(output, norm2_out, ctx);

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

private:
    std::shared_ptr<INormalization> norm1 = nullptr;
    std::shared_ptr<IAttention> attn = nullptr;
    std::shared_ptr<INormalization> norm2 = nullptr;
    std::shared_ptr<IMLP> mlp = nullptr;
};
