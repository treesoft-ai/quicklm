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

    // Load model configuration and weight tensors from the model directory
    bool load(const std::string& model_dir, int max_seq_len = 2048);

    // Reset recurrent/KV cache states for a new prompt
    void reset_states();

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
};
