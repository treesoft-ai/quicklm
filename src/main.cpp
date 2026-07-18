#include "model.hpp"
#include "speculative.hpp"
#include "tokenizer.hpp"
#include "chat_template.hpp"
#include "registry.hpp"
#include "math_ops.hpp"
#include "sieve_convert.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <set>
#include <thread>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// Number of physical cores (0 if detection fails). The default thread count
// uses this instead of hardware_concurrency(): the decode GEMVs are memory-
// bandwidth-bound and the fork-join pool spin-waits at barriers, so running
// one thread per SMT sibling (e.g. 12 on a 6-core part) just makes siblings
// steal each other's execution and cache resources. One thread per physical
// core is the sweet spot; --threads still overrides.
static int physical_core_count() {
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0) return 0;
    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len)) {
        return 0;
    }
    int cores = 0;
    char* p = buf.data();
    while (p < buf.data() + len) {
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(p);
        if (info->Relationship == RelationProcessorCore) ++cores;
        p += info->Size;
    }
    return cores;
#else
    return 0;
#endif
}

void print_usage() {
    std::cout << "Usage: quicklm.exe --path <model_directory> --prompt \"<prompt>\" [options]\n"
              << "       quicklm.exe --path <model_directory> --interactive [options]\n\n"
              << "Options:\n"
              << "  --temp <value>        Sampling temperature (default: 0.7, 0.0 for greedy)\n"
              << "  --top_k <value>       Top-k filtering (default: 40, 0 to disable)\n"
              << "  --max_tokens <value>  Maximum output tokens to generate per turn (default: 256)\n"
              << "  --threads <value>     Number of threads (default: physical cores)\n"
              << "  --optimize <list>     Comma-separated optimization methods (default: speculative)\n"
              << "                        [PLACEHOLDER except 'prefetch'/'fusion'/'kv-reuse'/\n"
              << "                         'speculative': the others are reserved for future use\n"
              << "                         and do not currently affect inference]\n"
              << "                        Speculation & decoding:\n"
              << "                          speculative    Speculative decoding: draft --draft-tokens\n"
              << "                                         tokens on a cheap draft model, verify them\n"
              << "                                         in one batched target pass, keep the\n"
              << "                                         accepted prefix. Greedy (--temp 0) only;\n"
              << "                                         lossless (token-for-token identical to\n"
              << "                                         non-speculative greedy decode). With\n"
              << "                                         --temp > 0 it is skipped with a notice.\n"
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
              << "                        are untouched)\n"
              << "  --draft-model <dir>   Checkpoint dir for the speculative draft model\n"
              << "                        (default: same as --path, i.e. self-speculative —\n"
              << "                        the same checkpoint loaded again at --draft-precision).\n"
              << "                        An independent draft model must have a tokenizer\n"
              << "                        byte-identical to the target's; mismatch is a hard\n"
              << "                        error, never silent fallback. Only used with\n"
              << "                        --optimize speculative.\n"
              << "  --draft-precision <v> Precision to load the draft model at: bf16|int8|int4\n"
              << "                        (default: int4; same set --precision accepts). Only\n"
              << "                        used with --optimize speculative.\n"
              << "  --draft-tokens <K>    Tokens to draft per round before verifying (default: 4).\n"
              << "                        Only used with --optimize speculative.\n\n"
              << "  --convert-sieve       Offline mode: build Sieve scout tables for --path's\n"
              << "                        checkpoint (see others/sieve_design.md §4.1) from its\n"
              << "                        original bf16 weights, written to <path>/sieve_scouts/.\n"
              << "                        Does not generate text; exits after conversion. This is\n"
              << "                        the Phase 0 offline step -- no --optimize mode reads\n"
              << "                        these scout tables yet.\n"
              << "  --sieve-rank <r>      SVD rank for scout tables (default: 16).\n"
              << "  --sieve-sketch-bits <b>  Residual sign-sketch width, must be a multiple of 64\n"
              << "                        (default: 128).\n"
              << "  --sieve-filter <sub>  Only convert tensors whose name contains this substring\n"
              << "                        (default: empty, converts every 2D weight matrix).\n\n"
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
    int phys_cores = physical_core_count();
    int num_threads = (phys_cores > 0)
        ? phys_cores
        : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    std::string optimize;  // empty -> falls back to "speculative" below
    bool raw = false;  // --raw disables the chat template (plain completion mode)
    bool show_ids = false;  // --show_ids prints generated token IDs to stderr (verification)
    bool interactive = false;  // --interactive reads further turns from stdin
    std::string precision = "bf16";  // --precision bf16|int8|int4
    std::string draft_model_path;    // --draft-model; empty -> same as --path (self-speculative)
    std::string draft_precision = "int4";  // --draft-precision bf16|int8|int4
    int draft_tokens = 4;            // --draft-tokens; K drafted per verify round
    bool convert_sieve = false;      // --convert-sieve: offline scout-table build, then exit
    int sieve_rank = 16;             // --sieve-rank
    int sieve_sketch_bits = 128;     // --sieve-sketch-bits
    std::string sieve_filter;        // --sieve-filter

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
        } else if (arg == "--draft-model" && i + 1 < argc) {
            draft_model_path = argv[++i];
        } else if (arg == "--draft-precision" && i + 1 < argc) {
            draft_precision = argv[++i];
        } else if (arg == "--draft-tokens" && i + 1 < argc) {
            draft_tokens = std::stoi(argv[++i]);
        } else if (arg == "--convert-sieve") {
            convert_sieve = true;
        } else if (arg == "--sieve-rank" && i + 1 < argc) {
            sieve_rank = std::stoi(argv[++i]);
        } else if (arg == "--sieve-sketch-bits" && i + 1 < argc) {
            sieve_sketch_bits = std::stoi(argv[++i]);
        } else if (arg == "--sieve-filter" && i + 1 < argc) {
            sieve_filter = argv[++i];
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

    if (model_path.empty()) {
        std::cerr << "Error: --path is required." << std::endl;
        print_usage();
        return 1;
    }

    // --convert-sieve is a standalone offline mode (Sieve Phase 0): build
    // scout tables and exit, never reaching the generation loop below. Runs
    // before the --prompt/--interactive requirement since it doesn't
    // generate text.
    if (convert_sieve) {
        if (sieve_sketch_bits <= 0 || sieve_sketch_bits % 64 != 0) {
            std::cerr << "Error: --sieve-sketch-bits must be a positive multiple of 64, got "
                      << sieve_sketch_bits << "." << std::endl;
            return 1;
        }
        if (sieve_rank <= 0) {
            std::cerr << "Error: --sieve-rank must be positive, got " << sieve_rank << "." << std::endl;
            return 1;
        }
        bool ok = run_sieve_convert(model_path, uint32_t(sieve_rank), uint32_t(sieve_sketch_bits), sieve_filter);
        return ok ? 0 : 1;
    }

    if (prompt.empty() && !interactive) {
        std::cerr << "Error: either --prompt or --interactive is required." << std::endl;
        print_usage();
        return 1;
    }

    if (precision != "bf16" && precision != "int8" && precision != "int4") {
        std::cerr << "Error: --precision must be 'bf16', 'int8', or 'int4', got '" << precision << "'." << std::endl;
        print_usage();
        return 1;
    }
    if (draft_precision != "bf16" && draft_precision != "int8" && draft_precision != "int4") {
        std::cerr << "Error: --draft-precision must be 'bf16', 'int8', or 'int4', got '"
                  << draft_precision << "'." << std::endl;
        print_usage();
        return 1;
    }
    if (draft_tokens < 1) {
        std::cerr << "Error: --draft-tokens must be >= 1, got " << draft_tokens << "." << std::endl;
        print_usage();
        return 1;
    }

    bool kv_reuse_enabled =
        std::find(optimizations.begin(), optimizations.end(), "kv-reuse") != optimizations.end();

    // Speculative decoding is greedy-only in this first cut (see
    // others/speculative_decoding_design.md §7): with --temp 0 it is provably
    // token-for-token identical to non-speculative greedy decode. Proper
    // distribution-preserving rejection sampling for temp > 0 is a separate
    // follow-up, so rather than silently changing the output distribution,
    // temp > 0 skips speculation with a notice ('speculative' is also the
    // --optimize default, so a hard error here would break plain runs).
    bool speculative_enabled =
        std::find(optimizations.begin(), optimizations.end(), "speculative") != optimizations.end();
    if (speculative_enabled && temp > 0.0f) {
        std::cout << "Note: --optimize speculative is greedy-only for now and --temp is "
                  << temp << " (> 0); decoding normally. Use --temp 0 to enable it."
                  << std::endl;
        speculative_enabled = false;
    }

    std::cout << "Initializing QI (Quick Inference) with optimizations: ";
    for (size_t i = 0; i < optimizations.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << optimizations[i];
    }
    std::cout << " (all except 'prefetch'/'fusion'/'kv-reuse'/'speculative' are placeholders) using " << num_threads
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

    // Speculative decoding: load the draft model. Default is self-speculative
    // (design doc §3.1): the SAME checkpoint loaded a second time at
    // --draft-precision (int4 by default), which needs no compatibility check
    // because both instances read the same tokenizer file. An independent
    // --draft-model (§3.2/§10) is gated behind a byte-level tokenizer
    // fingerprint comparison: same vocab SIZE is not sufficient (same size
    // doesn't mean same ID->token mapping), so any mismatch is a hard error —
    // a draft proposing IDs from a different vocabulary would verify against
    // the wrong tokens and produce silently wrong output.
    std::unique_ptr<DecoderModel> draft_model;
    if (speculative_enabled) {
        bool self_speculative = draft_model_path.empty() || draft_model_path == model_path;
        std::string draft_dir = self_speculative ? model_path : draft_model_path;
        if (!self_speculative) {
            Tokenizer draft_tokenizer;
            if (!draft_tokenizer.load(draft_dir)) {
                std::cerr << "Error: Failed to load draft model tokenizer from: " << draft_dir
                          << std::endl;
                return 1;
            }
            if (draft_tokenizer.fingerprint() != tokenizer.fingerprint()) {
                std::cerr << "Error: draft model tokenizer (" << draft_dir
                          << ") differs from target's (" << model_path
                          << "); refusing to run speculative decoding with mismatched "
                          << "vocabularies. No fallback — fix --draft-model or drop it "
                          << "to use self-speculative mode." << std::endl;
                return 1;
            }
        }
        std::cout << "Loading draft model (" << (self_speculative ? "self-speculative, " : "")
                  << draft_precision << ") from: " << draft_dir << std::endl;
        auto start_draft = std::chrono::high_resolution_clock::now();
        draft_model = std::make_unique<DecoderModel>();
        if (!draft_model->load(draft_dir, max_seq_len, draft_precision)) {
            std::cerr << "Error: Failed to load draft model from: " << draft_dir << std::endl;
            return 1;
        }
        std::chrono::duration<float> draft_load =
            std::chrono::high_resolution_clock::now() - start_draft;
        std::cout << "Draft model loaded in " << draft_load.count() << " seconds." << std::endl;
    }

    // Diagnostic (opt-in via env var, like QUICKLM_PROF): verify
    // DecoderModel::forward_batch() is bit-identical to calling forward()
    // once per token in order. This is the acceptance bar for the batched-
    // forward infra that speculative decoding's verify step will build on
    // (see QI_roadmap.md) — batching must never change a single value, only
    // how many times each layer's weights are read.
    if (std::getenv("QUICKLM_VERIFY_BATCH")) {
        std::vector<int> ids = tokenizer.encode(
            "The quick brown fox jumps over the lazy dog and runs to the old stone bridge near the river.");
        if (ids.size() > 16) ids.resize(16);
        int K = static_cast<int>(ids.size());
        int vocab_size = model.get_config().vocab_size;
        std::cout << "\n[verify-batch] chunk of " << K << " tokens" << std::endl;

        // Sequential path: one forward() call per token, in order.
        model.reset_states();
        Context seq_ctx;
        std::vector<std::vector<float>> seq_logits(K, std::vector<float>(vocab_size));
        for (int i = 0; i < K; ++i) {
            seq_ctx.pos = i;
            Tensor logits_i({vocab_size});
            model.forward(ids[i], logits_i, seq_ctx);
            std::memcpy(seq_logits[i].data(), logits_i.data, vocab_size * sizeof(float));
        }

        // Batched path: one forward_batch() call for the whole chunk.
        model.reset_states();
        Context batch_ctx;
        batch_ctx.pos = 0;
        Tensor batch_logits({K, vocab_size});
        model.forward_batch(ids, batch_logits, batch_ctx);

        bool identical = true;
        int first_mismatch_row = -1;
        for (int i = 0; i < K; ++i) {
            const float* batch_row = batch_logits.data + (size_t)i * vocab_size;
            if (std::memcmp(seq_logits[i].data(), batch_row, vocab_size * sizeof(float)) != 0) {
                identical = false;
                if (first_mismatch_row < 0) first_mismatch_row = i;
            }
        }

        if (identical) {
            std::cout << "[verify-batch] PASS: forward_batch is bit-identical to sequential forward() "
                      << "across all " << K << " positions." << std::endl;
        } else {
            std::cout << "[verify-batch] FAIL: mismatch starting at row " << first_mismatch_row
                      << std::endl;
        }
        return identical ? 0 : 1;
    }

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
    // The draft model (when present) is kept in lockstep with the target
    // everywhere: same resets, same prefill, and generate_turn() itself
    // advances both through every committed token — so cached_ids describes
    // BOTH models' live state at all times.
    model.reset_states();
    if (draft_model) draft_model->reset_states();
    std::vector<ChatMessage> messages;   // conversation history (non-raw only)
    std::vector<int> cached_ids;         // tokens whose KV/state is currently live in the caches
    Context ctx;
    Tensor logits({model.get_config().vocab_size});

    std::unique_ptr<SpeculativeDecoder> spec_decoder;
    if (draft_model) {
        try {
            spec_decoder = std::make_unique<SpeculativeDecoder>(
                model, *draft_model, draft_tokens, max_seq_len,
                tokenizer.eos_token_id(), tokenizer.im_end_token_id());
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    }

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
                if (draft_model) draft_model->reset_states();
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
                if (draft_model) draft_model->reset_states();
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
            int prefill_count = prefill_end - reused + 1;
            if (prefill_count > 0) {
                // Batch the whole (non-reused) prefill range into a single
                // forward_batch() call instead of looping single-token
                // forward(): each layer's weights are then read once for the
                // whole chunk instead of once per prompt token. Prefill is
                // weight-memory-bandwidth-bound exactly like decode (same
                // bottleneck prefetch/fusion target), so this is a direct win
                // on prompt-processing time, independent of speculative
                // decoding. Bit-identical to the old per-token loop by
                // construction (see QI_roadmap.md / QUICKLM_VERIFY_BATCH);
                // empirically verified identical token_ids and ~2x-2.2x
                // faster prefill in A/B testing.
                std::vector<int> prefill_ids(new_ids.begin() + reused, new_ids.begin() + prefill_end + 1);
                ctx.pos = static_cast<int>(cached_ids.size());
                if (ctx.pos + prefill_count > max_seq_len) {
                    throw std::runtime_error("context length exceeded max_seq_len (" +
                                             std::to_string(max_seq_len) + ")");
                }
                Tensor prefill_logits({prefill_count, model.get_config().vocab_size});
                model.forward_batch(prefill_ids, prefill_logits, ctx);
                if (draft_model) {
                    // Keep the draft model's state in lockstep: it must have
                    // consumed the same prompt before it can draft continuations
                    // of it. Cheap relative to the target's pass (int4 draft).
                    Context draft_ctx;
                    draft_ctx.pos = static_cast<int>(cached_ids.size());
                    Tensor draft_prefill_logits({prefill_count, draft_model->get_config().vocab_size});
                    draft_model->forward_batch(prefill_ids, draft_prefill_logits, draft_ctx);
                }
                cached_ids.insert(cached_ids.end(), prefill_ids.begin(), prefill_ids.end());
            }
            if (kv_reuse_enabled) {
                int recomputed = std::max(0, prefill_count);
                std::cout << "  [kv-reuse] reused " << reused << " / recomputed " << recomputed
                          << " prefill position" << (recomputed == 1 ? "" : "s") << std::endl;
            }

            auto after_prefill = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> prefill_duration = after_prefill - start_gen;
            std::cout << "  Prefill: " << prefill_duration.count() << " seconds" << std::endl;

            if (spec_decoder) {
                // Speculative decode loop (greedy-only; --temp > 0 disabled
                // speculation up front). Streams committed tokens as each
                // round resolves; on return, cached_ids exactly matches both
                // models' consumed state, same invariant the normal loop
                // maintains via do_forward().
                SpeculativeStats st = spec_decoder->generate_turn(
                    cached_ids, new_ids.back(), max_tokens, [&](int tok) {
                        generated_ids.push_back(tok);
                        std::string token_str = tokenizer.decode({tok});
                        assistant_text += token_str;
                        std::cout << token_str;
                        std::cout.flush();
                        generated_count++;
                    });
                std::cout << std::endl;
                // Acceptance-rate instrumentation (design doc §11): the whole
                // speedup is governed by this number, so report it per turn
                // instead of treating the draft as a black box.
                float rate = st.drafted > 0
                                 ? 100.0f * st.draft_accepted / st.drafted
                                 : 0.0f;
                std::cout << "  [speculative] " << st.rounds << " rounds, accepted "
                          << st.draft_accepted << "/" << st.drafted << " drafted tokens ("
                          << rate << "%), " << st.rejections << " rejection"
                          << (st.rejections == 1 ? "" : "s") << std::endl;
            } else {
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
            }

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
