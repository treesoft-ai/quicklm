#pragma once

#include "model.hpp"
#include <functional>
#include <vector>

// Per-turn instrumentation for the speculative decode loop. The design doc
// (§11) calls for logging acceptance from day one: the whole win is governed
// by how often the draft agrees with the target, so a bad acceptance rate
// must be visible immediately instead of discovered as "why didn't
// tokens/sec move."
struct SpeculativeStats {
    int rounds = 0;          // draft/verify rounds run
    int drafted = 0;         // draft tokens proposed across all rounds
    int draft_accepted = 0;  // of those, accepted by the target
    int emitted = 0;         // tokens delivered to the caller (excludes EOS)
    int rejections = 0;      // rounds that exercised the restore/replay path
};

// Greedy-only speculative decoding driver (design doc §6/§7): draft a few
// tokens cheaply on `draft`, verify them all in ONE batched forward pass
// through `target` (the Phase 1 primitive), keep the accepted prefix, correct
// the first wrong token. "Lossless" here is the greedy sense: accept iff
// argmax(target_logits) == draft token, which makes the committed sequence
// provably identical to non-speculative greedy decode. Sampling-mode
// (temp > 0) rejection sampling is an explicit non-goal of this first cut.
//
// Both models advance in lockstep: every committed token is (eventually)
// consumed by both, so the caller can treat `cached_ids` as the single source
// of truth for both models' state (which is what kv-reuse needs).
class SpeculativeDecoder {
public:
    // `draft_tokens` is K from the design doc: tokens drafted per round
    // before verifying. eos/im_end terminate a turn exactly like the
    // non-speculative loop (the terminator itself is never emitted).
    SpeculativeDecoder(DecoderModel& target, DecoderModel& draft,
                       int draft_tokens, int max_seq_len,
                       int eos_id, int im_end_id);

    // Run one turn's decode loop.
    //
    // Entry invariant: both models have consumed exactly `cached_ids`, and
    // `pending` (the prompt's final token, whose forward starts decode — same
    // convention as the non-speculative loop) has NOT been forwarded on
    // either model.
    //
    // Exit invariant: `cached_ids` again equals exactly the tokens consumed
    // by both models' states (speculative tokens that ended up rejected, or
    // consumed past EOS / the max_tokens cap, are rolled back via
    // snapshot/restore + replay — see §9). This keeps kv-reuse working
    // unchanged across turns.
    //
    // Emits at most `max_tokens` tokens, streamed in order to `on_token`.
    SpeculativeStats generate_turn(std::vector<int>& cached_ids, int pending,
                                   int max_tokens,
                                   const std::function<void(int)>& on_token);

private:
    // argmax over the vocab — the greedy accept/sample rule (§7).
    int greedy_pick(const float* logits_row) const;

    bool is_terminator(int tok) const {
        return tok == eos_id || tok == im_end_id;
    }

    DecoderModel& target;
    DecoderModel& draft;
    int draft_tokens;
    int max_seq_len;
    int eos_id;
    int im_end_id;
    int vocab_size;

    // Persistent scratch, allocated once: per-round logits buffers. The
    // verify buffer is [draft_tokens, vocab] (a few MB); reallocating it
    // every round would put a large malloc on the hot loop for nothing.
    Tensor target_logits;   // [vocab] — target's next-position logits
    Tensor draft_logits;    // [vocab] — draft's next-position logits
    Tensor verify_logits;   // [draft_tokens, vocab] — one row per drafted token
    Tensor replay_logits;   // [draft_tokens, vocab] — discarded replay output
};
