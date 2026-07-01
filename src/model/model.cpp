#include "model.hpp"
#include "json_parser.hpp"
#include "architecture.hpp"
#include "registry.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <random>
#include <chrono>
#include <immintrin.h>

namespace fs = std::filesystem;

// Upcast a row of `n` bf16 values directly into a contiguous FP32 buffer.
static inline void upcast_bf16_row(const uint16_t* src, float* dst, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256 f = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(v), 16));
        _mm256_storeu_ps(dst + i, f);
    }
    for (; i < n; ++i) {
        uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&dst[i], &bits, sizeof(float));
    }
}

// --- DecoderModel Implementation ---

bool DecoderModel::load(const std::string& model_dir, int max_seq_len, const std::string& precision) {
    // 1. Parse config.json
    std::string config_path = model_dir + "/config.json";
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        std::cerr << "Failed to open config.json at: " << config_path << std::endl;
        return false;
    }

    std::stringstream buffer;
    buffer << config_file.rdbuf();
    config_file.close();

    const IArchitecture* arch = nullptr;
    try {
        JsonValue root = JsonParser::parse(buffer.str());

        // Multimodal checkpoints nest the text backbone under "text_config".
        JsonValue config_node = root;
        if (root.contains("text_config")) {
            config_node = root["text_config"];
        }

        // Detect model_type: prefer the (possibly unwrapped) node, then top-level,
        // then the first entry of "architectures".
        std::string model_type;
        if (config_node.contains("model_type")) {
            model_type = config_node["model_type"].str_val;
        } else if (root.contains("model_type")) {
            model_type = root["model_type"].str_val;
        } else if (root.contains("architectures") && root["architectures"].is_array() &&
                   !root["architectures"].arr_val.empty()) {
            model_type = root["architectures"][0].str_val;
        }

        // 2. Select the architecture descriptor by model_type.
        arch = arch::find_by_model_type(model_type);
        if (!arch) {
            std::cerr << "Unsupported model architecture: '" << model_type
                      << "'. No registered descriptor matches." << std::endl;
            return false;
        }

        // 3. Delegate config parsing to the descriptor.
        arch->parse_config(config_node, config);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing config.json: " << e.what() << std::endl;
        return false;
    }

    // 4. Load Safetensors weights, installing the architecture's name remap.
    try {
        std::cout << "  [Model] Loading safetensors files from dir..." << std::endl;
        weights.set_remap([arch](const std::string& name) { return arch->remap_tensor_name(name); });
        if (!weights.load_from_dir(model_dir)) {
            std::cerr << "Failed to find any .safetensors files in: " << model_dir << std::endl;
            return false;
        }
        weights.set_precision(precision == "int8" ? ModelWeights::Precision::Int8
                              : precision == "int4" ? ModelWeights::Precision::Int4
                                                     : ModelWeights::Precision::Default);

        std::cout << "  [Model] Loading embed_tokens..." << std::endl;
        // Zero-copy bf16 view (no full-table FP32 upcast at load time); only the
        // one row looked up per token gets upcast, in forward().
        embed_tokens = weights.get_tensor_bf16(arch->embed_name());

        std::cout << "  [Model] Setting up layers..." << std::endl;
        for (int i = 0; i < config.num_hidden_layers; ++i) {
            std::cout << "    - Layer " << i << "/" << config.num_hidden_layers << std::endl;
            layers.push_back(arch->build_layer(i, config, weights, max_seq_len));
        }

        std::cout << "  [Model] Loading final norm..." << std::endl;
        final_norm.init(weights.get_tensor(arch->final_norm_name()), config.rms_norm_eps);

        std::cout << "  [Model] Loading lm_head..." << std::endl;
        // lm_head is a matmul operand → load as bf16 to halve the dominant
        // single-token GEMV bandwidth. embed_tokens is also bf16 now (zero-copy
        // gather, upcast per-row in forward()). With tied weights these are the
        // same underlying tensor, loaded in both representations (both views
        // are zero-copy into the mmap).
        if (weights.has_tensor(arch->lm_head_name())) {
            lm_head = weights.get_weight(arch->lm_head_name());
        } else {
            lm_head = weights.get_weight(arch->tied_lm_head_name()); // tied weights
        }

        // The actual vocab size is the number of rows in lm_head (may differ from text_config.vocab_size)
        config.vocab_size = lm_head.shape[0];
        std::cout << "  [Model] Effective vocab_size: " << config.vocab_size << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model weights: " << e.what() << std::endl;
        return false;
    }

    return true;
}

void DecoderModel::reset_states() {
    for (auto& layer : layers) {
        layer->reset_states();
    }
}

void DecoderModel::forward(int token_id, Tensor& logits, Context& ctx) {
    auto start_fwd = std::chrono::high_resolution_clock::now();

    // Look up input embedding. Table is bf16 (zero-copy view; upcast just the one
    // row we need) unless the checkpoint stores it as F32, in which case
    // get_tensor_bf16() already fell back to a plain FP32 tensor.
    Tensor hidden_states({config.hidden_size});
    if (embed_tokens.is_bf16()) {
        const uint16_t* embed_ptr = embed_tokens.bf16_data + (size_t)token_id * config.hidden_size;
        upcast_bf16_row(embed_ptr, hidden_states.data, config.hidden_size);
    } else {
        const float* embed_ptr = embed_tokens.data + (size_t)token_id * config.hidden_size;
        std::memcpy(hidden_states.data, embed_ptr, config.hidden_size * sizeof(float));
    }

    auto start_layers = std::chrono::high_resolution_clock::now();
    // Execute transformer layers
    Tensor next_hidden_states({config.hidden_size});
    for (size_t i = 0; i < layers.size(); ++i) {
        auto start_layer = std::chrono::high_resolution_clock::now();
        layers[i]->forward(hidden_states, next_hidden_states, ctx);
        std::swap(hidden_states.data, next_hidden_states.data);
        std::swap(hidden_states.data_owner, next_hidden_states.data_owner);
        auto end_layer = std::chrono::high_resolution_clock::now();

        std::chrono::duration<float, std::milli> layer_dur = end_layer - start_layer;
        if (ctx.pos == 0) {
            std::cout << "      [Layer " << i << " (" << config.layer_types[i] << ")] took "
                      << layer_dur.count() << " ms" << std::endl;
            std::cout.flush();
        }
    }

    auto start_final = std::chrono::high_resolution_clock::now();
    // Final normalization
    Tensor normalized_states({config.hidden_size});
    final_norm.forward(hidden_states, normalized_states, ctx);

    auto start_lm = std::chrono::high_resolution_clock::now();
    // Compute vocabulary logits (matrix-vector multiply)
    Tensor normalized_states_flat({1, config.hidden_size}, normalized_states.data);
    math::matmul(normalized_states_flat, lm_head, logits, true);

    auto end_fwd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> layers_dur = start_final - start_layers;
    std::chrono::duration<float, std::milli> lm_dur = end_fwd - start_lm;
    std::chrono::duration<float, std::milli> fwd_dur = end_fwd - start_fwd;

    if (ctx.pos == 0) {
        std::cout << "    [Forward] layers: " << layers_dur.count() << " ms | lm_head: " << lm_dur.count()
                  << " ms | total: " << fwd_dur.count() << " ms" << std::endl;
        std::cout.flush();
    }
}

int DecoderModel::sample(const Tensor& logits, float temperature, int top_k) {
    int vocab_size = logits.shape.back();

    if (temperature <= 0.0f) {
        // Greedy sampling
        int best_id = 0;
        float max_logit = logits.data[0];
        for (int i = 1; i < vocab_size; ++i) {
            if (logits.data[i] > max_logit) {
                max_logit = logits.data[i];
                best_id = i;
            }
        }
        return best_id;
    }

    // Softmax with temperature scaling
    std::vector<float> probs(vocab_size);
    float max_logit = logits.data[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (logits.data[i] > max_logit) {
            max_logit = logits.data[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp((logits.data[i] - max_logit) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] /= sum;
    }

    // Top-k filtering
    if (top_k > 0 && top_k < vocab_size) {
        std::vector<std::pair<float, int>> idx_probs(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            idx_probs[i] = {probs[i], i};
        }
        std::partial_sort(
            idx_probs.begin(), idx_probs.begin() + top_k, idx_probs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; }
        );

        float top_sum = 0.0f;
        for (int i = 0; i < top_k; ++i) {
            top_sum += idx_probs[i].first;
        }

        std::vector<float> new_probs(vocab_size, 0.0f);
        for (int i = 0; i < top_k; ++i) {
            new_probs[idx_probs[i].second] = idx_probs[i].first / top_sum;
        }
        probs = std::move(new_probs);
    }

    // Categorical sampling
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    float r = dis(gen);

    float acc = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        acc += probs[i];
        if (r <= acc) {
            return i;
        }
    }

    return vocab_size - 1;
}
