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
#include <sstream>
#include <set>

void print_usage() {
    std::cout << "Usage: quicklm.exe --path <model_directory> --prompt \"<prompt>\" [options]\n"
              << "       quicklm.exe --path <model_directory> --interactive [options]\n\n"
              << "Options:\n"
              << "  --temp <value>        Sampling temperature (default: 0.7, 0.0 for greedy)\n"
              << "  --top_k <value>       Top-k filtering (default: 40, 0 to disable)\n"
              << "  --max_tokens <value>  Maximum output tokens to generate per turn (default: 256)\n"
              << "  --threads <value>     Number of threads (default: hardware threads)\n"
              << "  --optimize <list>     Comma-separated optimization methods (default: speculative)\n"
              << "                        [PLACEHOLDER except 'prefetch'/'fusion'/'kv-reuse': the\n"
              << "                         others are reserved for future use and do not currently\n"
              << "                         affect inference]\n"
              << "                        Speculation & decoding:\n"
              << "                          speculative    Speculative decoding (needs --draft-model)\n"
              << "                        Memory / KV:\n"
              << "                          kv-reuse       Skip recomputing forward() for turns 2+ of\n"
              << "                                         a session whose token prefix is already in\n"
              << "                                         the KV-cache/DeltaNet state (lossless; only\n"
              << "                                         applies with --interactive or a \\np-delimited\n"
              << "                                         --prompt; falls back to a full reset+replay\n"
              << "                                         if the prefix ever diverges)\n"
              << "                          paged-kv       Paged attention for KV memory management\n"
              << "                        Attention & kernels:\n"
              << "                          flash-attn     Fused attention kernel\n"
              << "                          fusion         Fuse adjacent elementwise passes (SiLU+mul,\n"
              << "                                         residual-add+RMSNorm) to cut memory traffic\n"
              << "                                         (lossless)\n"
              << "                          cuda-graphs    Capture decode loop as CUDA graph\n"
              << "                        Memory movement:\n"
              << "                          prefetch       Warm next layer's weights while the\n"
              << "                                         current layer computes (lossless)\n"
              << "  --interactive         Read successive turns from stdin (one per line) after any\n"
              << "                        --prompt turns are exhausted. Ends on EOF or a line that is\n"
              << "                        exactly \"/exit\". Keeps the model and its caches loaded\n"
              << "                        across turns in one process, which is what --optimize\n"
              << "                        kv-reuse needs to have anything to reuse.\n"
              << "  --raw                 Disable the chat template (plain completion mode). With\n"
              << "                        multiple turns, each is an independent completion (no\n"
              << "                        shared history, no kv-reuse) since there is no chat\n"
              << "                        structure to extend.\n"
              << "  --show_ids            Print generated token IDs to stderr per turn (verification)\n"
              << "  --precision <value>   Weight precision: bf16 (default), int8, or int4\n"
              << "                        (quantized at load time; the model files on disk\n"
              << "                        are untouched)\n\n"
              << "Multi-turn: a single --prompt may contain several turns of the SAME conversation,\n"
              << "delimited by the literal 3-character sequence \\np (backslash, n, p), e.g.\n"
              << "  --prompt \"Hey\\npHow are you?\"\n"
              << "runs as two turns: \"Hey\", then \"How are you?\" with the first turn's user message\n"
              << "and the model's reply already in context.\n"
              << std::endl;
}

// Turn delimiter for a scripted multi-turn --prompt: the literal 3-character
// sequence \np (backslash, 'n', 'p'). Chosen over an actual embedded newline
// because it's trivially typable as a single command-line argument in any
// shell without needing real-newline quoting.
std::vector<std::string> split_turns(const std::string& s) {
    static const std::string delim = "\\np";
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        size_t pos = s.find(delim, start);
        if (pos == std::string::npos) {
            out.push_back(s.substr(start));
            break;
        }
        out.push_back(s.substr(start, pos - start));
        start = pos + delim.size();
    }
    return out;
}

int main(int argc, char* argv[]) {
    std::string model_path;
    std::string prompt;
    float temp = 0.7f;
    int top_k = 40;
    int max_tokens = 256;
    int num_threads = std::max(1u, std::thread::hardware_concurrency());
    std::string optimize;  // empty -> falls back to "speculative" below
    bool raw = false;  // --raw disables the chat template (plain completion mode)
    bool show_ids = false;  // --show_ids prints generated token IDs to stderr (verification)
    bool interactive = false;  // --interactive reads further turns from stdin
    std::string precision = "bf16";  // --precision bf16|int8|int4

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
        } else if (arg == "--optimize" && i + 1 < argc) {
            optimize = argv[++i];
        } else if (arg == "--raw") {
            raw = true;
        } else if (arg == "--show_ids") {
            show_ids = true;
        } else if (arg == "--interactive") {
            interactive = true;
        } else if (arg == "--precision" && i + 1 < argc) {
            precision = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }
    }

    // Validate optimization methods
    // TODO: Implement actual optimization logic for each method
    auto is_valid_optimize = [](const std::string& opt) {
        static const std::set<std::string> valid = {
            "speculative", "kv-reuse", "paged-kv", "flash-attn", "fusion",
            "cuda-graphs", "prefetch"
        };
        return valid.count(opt) > 0;
    };

    // Parse comma-separated optimizations
    std::vector<std::string> optimizations;
    std::stringstream ss(optimize);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // Trim whitespace
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        if (!item.empty()) {
            if (!is_valid_optimize(item)) {
                std::cerr << "Error: Unknown optimization method '" << item << "'." << std::endl;
                print_usage();
                return 1;
            }
            optimizations.push_back(item);
        }
    }

    if (optimizations.empty()) {
        optimizations.push_back("speculative");
    }

    if (model_path.empty() || (prompt.empty() && !interactive)) {
        std::cerr << "Error: --path is required, and either --prompt or --interactive." << std::endl;
        print_usage();
        return 1;
    }

    if (precision != "bf16" && precision != "int8" && precision != "int4") {
        std::cerr << "Error: --precision must be 'bf16', 'int8', or 'int4', got '" << precision << "'." << std::endl;
        print_usage();
        return 1;
    }

    bool kv_reuse_enabled =
        std::find(optimizations.begin(), optimizations.end(), "kv-reuse") != optimizations.end();

    std::cout << "Initializing QI (Quick Inference) with optimizations: ";
    for (size_t i = 0; i < optimizations.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << optimizations[i];
    }
    std::cout << " (all except 'prefetch'/'fusion'/'kv-reuse' are placeholders) using " << num_threads
              << " thread" << (num_threads != 1 ? "s" : "") << "..." << std::endl;
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
    const int max_seq_len = 2048;
    DecoderModel model;
    if (!model.load(model_path, max_seq_len, precision)) {
        std::cerr << "Error: Failed to load model weights/configuration from: " << model_path << std::endl;
        return 1;
    }
    model.set_prefetch_enabled(
        std::find(optimizations.begin(), optimizations.end(), "prefetch") != optimizations.end());
    model.set_fusion_enabled(
        std::find(optimizations.begin(), optimizations.end(), "fusion") != optimizations.end());
    auto end_load = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> load_duration = end_load - start_load;
    std::cout << "Model loaded successfully in " << load_duration.count() << " seconds." << std::endl;

    // Chat template (unused in --raw mode).
    std::string fallback;
    const IArchitecture* arch = arch::find_by_model_type(model.get_config().model_type);
    if (arch) fallback = arch->default_chat_template();
    ChatTemplate chat_template;
    if (!raw) chat_template.load(model_path, fallback);

    // 3. Build the queue of turns to run. --prompt may itself contain several
    // turns of one conversation (see split_turns); --interactive appends
    // further turns read from stdin, one per line, after those are exhausted.
    std::vector<std::string> pending_turns;
    if (!prompt.empty()) pending_turns = split_turns(prompt);

    // Session state, persists across turns so kv-reuse has something to reuse.
    model.reset_states();
    std::vector<ChatMessage> messages;   // conversation history (non-raw only)
    std::vector<int> cached_ids;         // tokens whose KV/state is currently live in the caches
    Context ctx;
    Tensor logits({model.get_config().vocab_size});

    int turn_num = 0;
    bool fatal_error = false;

    while (true) {
        std::string turn_text;
        if (!pending_turns.empty()) {
            turn_text = pending_turns.front();
            pending_turns.erase(pending_turns.begin());
        } else if (interactive) {
            std::cout << "\n> " << std::flush;
            if (!std::getline(std::cin, turn_text)) break;  // EOF
            if (turn_text == "/exit") break;
            if (turn_text.empty()) continue;
        } else {
            break;  // no more turns, not interactive
        }

        ++turn_num;
        std::cout << "\n--- Turn " << turn_num << " ---" << std::endl;

        std::string full_prompt;
        if (raw) {
            // No chat structure to extend: each raw turn is an independent
            // completion, so its cache/state is unrelated to any previous
            // turn's. Reset before every raw turn.
            if (!cached_ids.empty()) {
                model.reset_states();
                cached_ids.clear();
            }
            full_prompt = turn_text;
        } else {
            messages.push_back({"user", turn_text});
            if (!chat_template.render(messages, /*add_generation_prompt=*/true,
                                      /*enable_thinking=*/false, full_prompt)) {
                std::cerr << "Warning: chat template rendering failed; using raw turn text." << std::endl;
                full_prompt = turn_text;
            }
        }

        // Extend the live cache directly instead of re-tokenizing the whole
        // conversation from scratch. Comparing token IDs after an independent
        // full re-tokenization is not reliable: re-encoding a prior assistant
        // turn's decoded text can pick different BPE tokens than what the
        // model actually sampled, even though the text is byte-identical (a
        // classic decode-then-re-encode round trip is not guaranteed to be
        // stable). Decoding what's already forwarded and confirming it's a
        // textual prefix of this turn's render sidesteps that entirely: the
        // reused portion is never re-tokenized, so it can't drift.
        std::vector<int> new_ids;
        int reused = 0;
        bool incremental = false;
        if (kv_reuse_enabled && !cached_ids.empty()) {
            std::string cached_text = tokenizer.decode(cached_ids);
            if (full_prompt.size() >= cached_text.size() &&
                full_prompt.compare(0, cached_text.size(), cached_text) == 0) {
                std::string suffix_text = full_prompt.substr(cached_text.size());
                std::vector<int> suffix_ids = tokenizer.encode(suffix_text);
                new_ids = cached_ids;
                new_ids.insert(new_ids.end(), suffix_ids.begin(), suffix_ids.end());
                reused = static_cast<int>(cached_ids.size());
                incremental = true;
            } else if (show_ids) {
                std::cerr << "[kv-reuse] cached text is not a prefix of this turn's render; resetting caches"
                          << std::endl;
            }
        }
        if (!incremental) {
            new_ids = tokenizer.encode(full_prompt);
            if (!cached_ids.empty()) {
                // kv-reuse disabled, raw turn, or the cached text didn't line
                // up with this turn's render: none of the live state is known
                // to correspond to a prefix of new_ids, so the only correct
                // recovery is a full reset + replay from scratch.
                model.reset_states();
                cached_ids.clear();
            }
        }
        if (new_ids.empty()) {
            std::cerr << "Error: Tokenized turn is empty; skipping." << std::endl;
            if (!raw) messages.pop_back();
            --turn_num;
            continue;
        }

        auto do_forward = [&](int tok) {
            ctx.pos = static_cast<int>(cached_ids.size());
            if (ctx.pos >= max_seq_len) {
                throw std::runtime_error("context length exceeded max_seq_len (" +
                                         std::to_string(max_seq_len) + ")");
            }
            model.forward(tok, logits, ctx);
            cached_ids.push_back(tok);
        };

        auto start_gen = std::chrono::high_resolution_clock::now();
        std::string assistant_text;
        std::vector<int> generated_ids;  // for --show_ids verification
        int generated_count = 0;

        try {
            int prefill_end = static_cast<int>(new_ids.size()) - 2;  // inclusive; last token starts decode
            for (int i = reused; i <= prefill_end; ++i) {
                do_forward(new_ids[i]);
            }
            if (kv_reuse_enabled) {
                int recomputed = std::max(0, prefill_end - reused + 1);
                std::cout << "  [kv-reuse] reused " << reused << " / recomputed " << recomputed
                          << " prefill position" << (recomputed == 1 ? "" : "s") << std::endl;
            }

            auto after_prefill = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> prefill_duration = after_prefill - start_gen;
            std::cout << "  Prefill: " << prefill_duration.count() << " seconds" << std::endl;

            int next_token = new_ids.back();
            for (int i = 0; i < max_tokens; ++i) {
                do_forward(next_token);
                next_token = model.sample(logits, temp, top_k);
                if (next_token == tokenizer.eos_token_id() ||
                    next_token == tokenizer.im_end_token_id()) {
                    break;
                }
                generated_ids.push_back(next_token);
                std::string token_str = tokenizer.decode({next_token});
                assistant_text += token_str;
                std::cout << token_str;
                std::cout.flush();
                generated_count++;
            }
            std::cout << std::endl;

            // Exact, decode-independent fingerprint of this turn's greedy
            // output. Used to verify kv-reuse leaves responses identical.
            if (show_ids) {
                std::cerr << "[token_ids]";
                for (int id : generated_ids) std::cerr << " " << id;
                std::cerr << std::endl;
            }

            auto end_gen = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> gen_duration = end_gen - after_prefill;
            float tokens_per_sec = generated_count / gen_duration.count();
            std::cout << "  Generated " << generated_count << " tokens in " << gen_duration.count()
                      << " seconds (" << tokens_per_sec << " tokens/sec)" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "\nRuntime Error during generation: " << e.what() << std::endl;
            fatal_error = true;
            break;
        }

        if (!raw) messages.push_back({"assistant", assistant_text});
    }

    if (turn_num == 0 && !fatal_error) {
        std::cerr << "Error: no turns to run (empty --prompt and --interactive produced nothing)."
                  << std::endl;
        return 1;
    }

    if (std::getenv("QUICKLM_PROF")) {
        double mm = math::get_matmul_ms();
        std::cerr << "[prof] cumulative matmul: " << mm << " ms" << std::endl;
    }

    return fatal_error ? 1 : 0;
}
