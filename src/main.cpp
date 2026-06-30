#include "model.hpp"
#include "tokenizer.hpp"
#include "chat_template.hpp"
#include "registry.hpp"
#include "math_ops.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstdlib>

void print_usage() {
    std::cout << "Usage: quicklm.exe --path <model_directory> --prompt \"<prompt>\" [options]\n\n"
              << "Options:\n"
              << "  --temp <value>        Sampling temperature (default: 0.7, 0.0 for greedy)\n"
              << "  --top_k <value>       Top-k filtering (default: 40, 0 to disable)\n"
              << "  --max_tokens <value>  Maximum output tokens to generate (default: 256)\n"
              << "  --threads <value>     Number of threads (default: hardware threads)\n"
              << "  --version <value>     QI version (default: 1)\n"
              << "  --raw                 Disable the chat template (plain completion mode)\n"
              << "  --show_ids            Print generated token IDs to stderr (verification)\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    std::string model_path;
    std::string prompt;
    float temp = 0.7f;
    int top_k = 40;
    int max_tokens = 256;
    int num_threads = std::max(1u, std::thread::hardware_concurrency());
    int qi_version = 1;
    bool raw = false;  // --raw disables the chat template (plain completion mode)
    bool show_ids = false;  // --show_ids prints generated token IDs to stderr (verification)

    // Simple command line parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--path" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--temp" && i + 1 < argc) {
            temp = std::stof(argv[++i]);
        } else if (arg == "--top_k" && i + 1 < argc) {
            top_k = std::stoi(argv[++i]);
        } else if (arg == "--max_tokens" && i + 1 < argc) {
            max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "--version" && i + 1 < argc) {
            qi_version = std::stoi(argv[++i]);
        } else if (arg == "--raw") {
            raw = true;
        } else if (arg == "--show_ids") {
            show_ids = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }
    }

    if (qi_version != 1) {
        std::cerr << "Warning: QI version " << qi_version << " is invalid. Only version 1 is supported." << std::endl;
    }

    if (model_path.empty() || prompt.empty()) {
        std::cerr << "Error: --path and --prompt arguments are required." << std::endl;
        print_usage();
        return 1;
    }

    std::cout << "Initializing QI v" << qi_version << " (Quick Inference v" << qi_version << ") with " << num_threads << " thread" << (num_threads != 1 ? "s" : "") << "..." << std::endl;
    math::init_thread_pool(num_threads);

    // 1. Initialize tokenizer
    std::cout << "Loading Tokenizer..." << std::endl;
    Tokenizer tokenizer;
    if (!tokenizer.load(model_path)) {
        std::cerr << "Error: Failed to load tokenizer files (qwen.tiktoken or tokenizer.json) from: " 
                  << model_path << std::endl;
        return 1;
    }
    std::cout << "Vocabulary loaded with " << tokenizer.vocab_size() << " tokens." << std::endl;

    // 2. Load model config and weights
    std::cout << "Loading Model..." << std::endl;
    auto start_load = std::chrono::high_resolution_clock::now();
    DecoderModel model;
    if (!model.load(model_path)) {
        std::cerr << "Error: Failed to load model weights/configuration from: " << model_path << std::endl;
        return 1;
    }
    auto end_load = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> load_duration = end_load - start_load;
    std::cout << "Model loaded successfully in " << load_duration.count() << " seconds." << std::endl;

    // 3. Reset states & Tokenize prompt
    model.reset_states();

    // Wrap the prompt with the model's own chat template unless --raw is given.
    // The template is sourced from the model dir (chat_template.jinja /
    // tokenizer_config.json); if it uses constructs outside the renderer's
    // supported subset, it falls back to the architecture's built-in default,
    // which reproduces the canonical non-thinking assistant turn (an empty
    // <think></think> block primes generation).
    std::string full_prompt = prompt;
    if (!raw) {
        std::string fallback;
        const IArchitecture* arch = arch::find_by_model_type(model.get_config().model_type);
        if (arch) fallback = arch->default_chat_template();

        ChatTemplate chat_template;
        chat_template.load(model_path, fallback);

        std::vector<ChatMessage> messages = {{"user", prompt}};
        std::string rendered;
        if (chat_template.render(messages, /*add_generation_prompt=*/true,
                                 /*enable_thinking=*/false, rendered)) {
            full_prompt = rendered;
        } else {
            std::cerr << "Warning: chat template rendering failed; using raw prompt." << std::endl;
        }
    }
    std::vector<int> prompt_ids = tokenizer.encode(full_prompt);
    if (prompt_ids.empty()) {
        std::cerr << "Error: Tokenized prompt is empty." << std::endl;
        return 1;
    }

    std::cout << "\nPrompt Token IDs: ";
    for (int id : prompt_ids) {
        std::cout << id << " ";
    }
    std::cout << "\n\nGenerating response:\n" << std::endl;

    // 4. Prefill prompt tokens to build state caches
    Context ctx;
    ctx.seq_len = static_cast<int>(prompt_ids.size());
    Tensor logits({model.get_config().vocab_size});

    auto start_gen = std::chrono::high_resolution_clock::now();
    
    try {
        // Process all prompt tokens except the last one
        for (size_t i = 0; i < prompt_ids.size() - 1; ++i) {
            ctx.pos = static_cast<int>(i);
            model.forward(prompt_ids[i], logits, ctx);
        }

        // 5. Decode loop
        int next_token = prompt_ids.back();
        int generated_count = 0;
        std::vector<int> generated_ids;  // for --show_ids verification

        for (int i = 0; i < max_tokens; ++i) {
            ctx.pos = static_cast<int>(prompt_ids.size() - 1 + i);
            model.forward(next_token, logits, ctx);

            next_token = model.sample(logits, temp, top_k);
            if (next_token == tokenizer.eos_token_id() ||
                next_token == tokenizer.im_end_token_id()) {
                break;
            }

            if (show_ids) generated_ids.push_back(next_token);
            std::string token_str = tokenizer.decode({next_token});
            std::cout << token_str;
            std::cout.flush();
            generated_count++;
        }
        std::cout << std::endl;

        // Exact, decode-independent fingerprint of the greedy output. Used to
        // verify that optimizations leave the model's responses identical.
        if (show_ids) {
            std::cerr << "[token_ids]";
            for (int id : generated_ids) std::cerr << " " << id;
            std::cerr << std::endl;
        }

        auto end_gen = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> gen_duration = end_gen - start_gen;
        float tokens_per_sec = generated_count / gen_duration.count();

        std::cout << "\n---\n"
                  << "Generated " << generated_count << " tokens in " << gen_duration.count() << " seconds "
                  << "(" << tokens_per_sec << " tokens/sec)" << std::endl;
        if (std::getenv("QUICKLM_PROF")) {
            double mm = math::get_matmul_ms();
            double total = gen_duration.count() * 1000.0;
            std::cerr << "[prof] matmul: " << mm << " ms (" << (100.0 * mm / total)
                      << "% of wall), non-matmul: " << (total - mm) << " ms" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "\nRuntime Error during generation: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
