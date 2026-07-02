#pragma once

#include "tensor.hpp"
#include "config.hpp"
#include "weights.hpp"
#include "modules.hpp"
#include "attention.hpp"
#include <string>
#include <vector>
#include <memory>

// Generic decoder runtime. Knows nothing model-specific: it selects an
// architecture descriptor (src/arch/) by model_type and delegates every
// architecture-dependent decision (config parse, per-layer module assembly,
// tensor names, name remap) to it.
class DecoderModel {
public:
    DecoderModel() = default;
    ~DecoderModel() = default;

    // Load model configuration and weight tensors from the model directory.
    // `precision` selects the in-RAM storage for matmul weight operands
    // ("bf16" default, or "int8" to quantize at load time); the on-disk
    // safetensors files are never modified.
    bool load(const std::string& model_dir, int max_seq_len = 2048,
              const std::string& precision = "bf16");

    // Reset recurrent/KV cache states for a new prompt
    void reset_states();

    // Gate the QI --optimize=prefetch method: warm layer i+1's first weight
    // matrix while layer i is still computing. Lossless (same math, pure
    // latency hiding); off by default until requested via --optimize.
    void set_prefetch_enabled(bool enabled) { prefetch_enabled = enabled; }

    // Gate the QI --optimize=fusion method: merge adjacent elementwise passes
    // (SiLU+multiply, residual-add+RMSNorm) into single loops per layer.
    // Lossless (same math, pure memory-traffic reduction); off by default
    // until requested via --optimize.
    void set_fusion_enabled(bool enabled) {
        for (auto& layer : layers) layer->set_fusion_enabled(enabled);
    }

    // Run forward pass for a single token ID at position pos.
    // Writes the vocabulary logits to the provided output tensor.
    void forward(int token_id, Tensor& logits, Context& ctx);

    // Greedy sampling with temperature and top-k filtering.
    // Runs on the final logits tensor.
    int sample(const Tensor& logits, float temperature, int top_k);

    const ModelConfig& get_config() const { return config; }

private:
    ModelConfig config;
    ModelWeights weights;

    // Word Embeddings
    Tensor embed_tokens;

    // Transformer layers
    std::vector<std::shared_ptr<DecoderLayer>> layers;

    // Final normalization
    RMSNorm final_norm;

    // LM Head (converts hidden state back to vocabulary logits)
    Tensor lm_head;

    bool prefetch_enabled = false;
};
