#include "speculative.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>

// Implementation notes (see others/speculative_decoding_design.md §6, §9).
//
// The round loop maintains one invariant between rounds:
//
//   Both models have consumed exactly the committed tokens (cached_ids), and
//   target_logits / draft_logits hold each model's prediction for the next
//   position (produced by the round-closing single-token forward, or by the
//   pending-token forward before the first round).
//
// That held-over target_logits is what verifies the round's FIRST drafted
// token; the verify batch's row i then verifies drafted token i+1, and its
// last row supplies the bonus token on full accept. This is why every round
// ends with exactly one extra single-token forward() on both models (the
// correction or the bonus): it both advances state past the newly committed
// final token and produces the logits the next round starts from.

SpeculativeDecoder::SpeculativeDecoder(DecoderModel& target_model, DecoderModel& draft_model,
                                       int draft_tokens_, int max_seq_len_,
                                       int eos_id_, int im_end_id_)
    : target(target_model), draft(draft_model),
      draft_tokens(draft_tokens_), max_seq_len(max_seq_len_),
      eos_id(eos_id_), im_end_id(im_end_id_),
      vocab_size(target_model.get_config().vocab_size) {
    if (draft_tokens < 1) {
        throw std::runtime_error("--draft-tokens must be >= 1, got " +
                                 std::to_string(draft_tokens));
    }
    // The tokenizer fingerprint check (§10) guards the vocab MAPPING; this
    // guards the vocab SIZE the two lm_heads actually shipped with. Both must
    // hold for draft token IDs to be meaningful to the target.
    if (draft.get_config().vocab_size != vocab_size) {
        throw std::runtime_error(
            "speculative: draft model vocab_size (" +
            std::to_string(draft.get_config().vocab_size) +
            ") differs from target's (" + std::to_string(vocab_size) + ")");
    }
    target_logits = Tensor({vocab_size});
    draft_logits = Tensor({vocab_size});
    verify_logits = Tensor({draft_tokens, vocab_size});
    replay_logits = Tensor({draft_tokens, vocab_size});
}

int SpeculativeDecoder::greedy_pick(const float* logits_row) const {
    int best_id = 0;
    float max_logit = logits_row[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (logits_row[i] > max_logit) {
            max_logit = logits_row[i];
            best_id = i;
        }
    }
    return best_id;
}

SpeculativeStats SpeculativeDecoder::generate_turn(
    std::vector<int>& cached_ids, int pending, int max_tokens,
    const std::function<void(int)>& on_token) {
    SpeculativeStats stats;
    if (max_tokens <= 0) return stats;

    // Single-token forward on one model at an explicit absolute position.
    // Positions are passed explicitly because during a round both models run
    // ahead of cached_ids (which is only extended at commit time).
    auto step = [&](DecoderModel& m, int tok, int pos, Tensor& out) {
        if (pos >= max_seq_len) {
            throw std::runtime_error("context length exceeded max_seq_len (" +
                                     std::to_string(max_seq_len) + ")");
        }
        Context c;
        c.pos = pos;
        m.forward(tok, out, c);
    };

    // Batched forward over `toks` starting at absolute position `pos`.
    auto batch = [&](DecoderModel& m, const std::vector<int>& toks, int pos, Tensor& out) {
        if (pos + static_cast<int>(toks.size()) > max_seq_len) {
            throw std::runtime_error("context length exceeded max_seq_len (" +
                                     std::to_string(max_seq_len) + ")");
        }
        Context c;
        c.pos = pos;
        m.forward_batch(toks, out, c);
    };

    // Establish the between-rounds invariant: consume the prompt's final
    // token on both models (the same forward that starts decode in the
    // non-speculative loop) and hold on to both models' logits.
    {
        int pos = static_cast<int>(cached_ids.size());
        step(target, pending, pos, target_logits);
        step(draft, pending, pos, draft_logits);
        cached_ids.push_back(pending);
    }

    while (stats.emitted < max_tokens) {
        int base = static_cast<int>(cached_ids.size());
        // A round consumes up to K drafted tokens (positions base..base+K-1)
        // plus the round-closing forward at base+K, so shrink K near the
        // context cap. If not even a 1-token round fits, the context is
        // effectively full — same error the non-speculative loop raises (at
        // most K tokens earlier than it would have).
        int K = std::min(draft_tokens, max_seq_len - base - 1);
        if (K < 1) {
            throw std::runtime_error("context length exceeded max_seq_len (" +
                                     std::to_string(max_seq_len) + ")");
        }

        // Snapshot round-start DeltaNet state on both models (§9). The
        // attention K/V caches need no snapshot: they're position-indexed,
        // and every post-restore forward below overwrites or out-runs any
        // speculative rows (stale rows past the committed length are inert).
        target.snapshot_states();
        draft.snapshot_states();

        // --- Draft K tokens (greedy), consuming each on the draft model ---
        std::vector<int> d(K);
        d[0] = greedy_pick(draft_logits.data);
        for (int i = 0; i < K; ++i) {
            step(draft, d[i], base + i, draft_logits);
            if (i + 1 < K) d[i + 1] = greedy_pick(draft_logits.data);
        }
        stats.drafted += K;

        // --- Verify all K in ONE batched target pass (Phase 1 primitive) ---
        // v[i] is the target's greedy choice at drafted token i's position:
        // v[0] comes from the held-over target_logits, v[i>0] from verify
        // row i-1. v[K] (row K-1) is the bonus prediction after d[K-1].
        Tensor verify_view({K, vocab_size}, verify_logits.data);
        batch(target, d, base, verify_view);

        std::vector<int> v(K + 1);
        v[0] = greedy_pick(target_logits.data);
        for (int i = 1; i <= K; ++i) {
            v[i] = greedy_pick(verify_logits.data + (size_t)(i - 1) * vocab_size);
        }

        int accept = 0;
        while (accept < K && v[accept] == d[accept]) ++accept;
        ++stats.rounds;
        stats.draft_accepted += accept;

        // Tokens consumed by BOTH models this round, in order. Also leaves
        // target_logits/draft_logits holding next-position predictions,
        // re-establishing the invariant for the next round.
        std::vector<int> commit;
        if (accept == K) {
            // Full accept: no rollback — the verify pass advanced the target
            // through exactly the tokens being committed, and the bonus
            // token v[K] comes free from logits already computed.
            commit = d;
            commit.push_back(v[K]);
            step(target, v[K], base + K, target_logits);
            step(draft, v[K], base + K, draft_logits);
        } else {
            // Partial accept: both models consumed tokens past the accepted
            // prefix (target via the verify batch, draft via its own
            // drafting), and DeltaNet state can't be truncated — restore the
            // round-start snapshot and replay only the real prefix (§9),
            // then advance both by the target's correction token.
            ++stats.rejections;
            int correction = v[accept];
            target.restore_states();
            draft.restore_states();
            if (accept > 0) {
                std::vector<int> prefix(d.begin(), d.begin() + accept);
                Tensor replay_view({accept, vocab_size}, replay_logits.data);
                batch(target, prefix, base, replay_view);
                batch(draft, prefix, base, replay_view);  // logits discarded
            }
            step(target, correction, base + accept, target_logits);
            step(draft, correction, base + accept, draft_logits);
            commit.assign(d.begin(), d.begin() + accept);
            commit.push_back(correction);
        }

        // --- Trim at a terminator or the max_tokens cap ---
        // The terminator itself is never emitted (same as the non-speculative
        // loop). If fewer tokens are kept than the models consumed this
        // round, roll state back to exactly the kept prefix; the snapshot
        // still holds the round-start state, so restore + replay works the
        // same way the reject path does.
        size_t keep = commit.size();
        bool stop = false;
        for (size_t i = 0; i < commit.size(); ++i) {
            if (is_terminator(commit[i])) {
                keep = i;
                stop = true;
                break;
            }
        }
        size_t budget = static_cast<size_t>(max_tokens - stats.emitted);
        if (keep > budget) {
            keep = budget;
            stop = true;
        }

        if (stop && keep < commit.size()) {
            target.restore_states();
            draft.restore_states();
            if (keep > 0) {
                std::vector<int> prefix(commit.begin(), commit.begin() + keep);
                Tensor replay_view({static_cast<int>(keep), vocab_size}, replay_logits.data);
                batch(target, prefix, base, replay_view);
                batch(draft, prefix, base, replay_view);
            }
        }

        for (size_t i = 0; i < keep; ++i) {
            cached_ids.push_back(commit[i]);
            on_token(commit[i]);
            ++stats.emitted;
        }
        if (stop) break;
    }

    return stats;
}
