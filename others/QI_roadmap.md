QI --optimize methods (all lossless, no INT4/8)
=================================================
Status: prefetch, fusion, and kv-reuse are implemented; the rest are placeholders
(parsed/validated by the CLI but have no effect on inference yet). speculative's
Phase 1 prerequisite (batched/chunked forward) is implemented AND wired into
prefill, giving a real ~2x prefill speedup today, independent of Phase 2. The
actual draft/verify/accept-reject decoding loop (Phase 2) is not yet
implemented — see below.


speculative     [needs --draft-model]  [PHASE 1 (batched forward): IMPLEMENTED]
  A small, fast "draft" model proposes several tokens ahead. The full model
  then verifies them in a SINGLE forward pass. Accepted tokens are kept; the
  first rejected one is corrected. Output is bit-for-bit identical to normal
  decoding — you just read the big weights once but commit multiple tokens
  per pass. This is how you beat the per-token bandwidth wall. Win scales
  with draft acceptance rate.

  --- Phase 1: batched/chunked forward (prerequisite) — IMPLEMENTED ---

  Discovery that scoped this phase: every forward() in the codebase (model,
  attention, DeltaNet) was strictly single-token, including prefill (a
  sequential loop of single-token forward() calls). Speculative decoding's
  entire value proposition — verify K drafted tokens for ~1 target forward
  pass, since decode is weight-memory-bandwidth-bound not compute-bound (the
  same bottleneck that scoped prefetch/fusion) — is unachievable without a
  real batched/chunked forward pass first. Building that was pulled forward
  as its own phase before any draft-model/accept-reject work.

  What was already batch-ready (no changes needed): math::matmul() already
  supports A.shape[0] > 1 (weights read once, reused across all rows in the
  inner loop); math::rms_norm/silu/elementwise_* are already row-shape-
  generic; SwiGLU_MLP::forward and DecoderLayer::forward already branch on
  input.shape.size() > 1 and pass shape through generically; Context already
  had an unused seq_len field. So only Qwen3_5Attention::forward_chunk,
  Qwen3_5GatedDeltaNet::forward_chunk, and a new
  DecoderModel::forward_batch() entry point needed real work.

  Implementation: DecoderModel::forward_batch(token_ids, logits_out, ctx)
  gathers K embedding rows into [K, hidden_size], sets ctx.seq_len = K, runs
  the layer stack once (each layer's forward() dispatches internally to
  forward_chunk when ctx.seq_len > 1), then final_norm + lm_head on all K
  rows at once, producing [K, vocab_size] logits in one pass. Restores
  ctx.seq_len = 1 on return so later single-token forward() calls are
  unaffected.

  Qwen3_5Attention::forward_chunk batches the QKV projections for all K rows
  in one matmul call each, then loops rows in position order applying
  bias/QK-norm/RoPE per row (identical per-row math to the single-position
  path), writing K/V into k_cache/v_cache at base_pos+r for all K rows before
  computing any attention output (so row r's causal attention correctly sees
  every prior position, including earlier rows in the same chunk).

  Qwen3_5GatedDeltaNet::forward_chunk batches the four input projections
  (qkv/z/b/a) for all K rows in one matmul call each, then loops rows in
  position order through causal_conv1d_update and the recurrent gated-delta-
  rule update — both genuine sequential recurrences (each row's update
  depends on the previous row's state) — reusing the identical single-step
  primitives and AVX2 kernel, just invoked K times instead of via K separate
  forward() calls.

  Bit-exactness: since per-row math in the batched path is byte-for-byte the
  same formulas as the sequential path, invoked in the same order, with no
  new cross-row reductions introduced anywhere, forward_batch() is bit-
  identical to K sequential forward() calls by construction. Verified via an
  opt-in diagnostic, QUICKLM_VERIFY_BATCH=1, which runs a 16-token chunk both
  ways (reset_states() between) and memcmp's the logits row-by-row:

    [verify-batch] PASS: forward_batch is bit-identical to sequential
    forward() across all 16 positions.

  Scope: this phase adds the batched forward primitive, its verification, AND
  one consumer of it: prefill. No CLI flags, no draft model, no accept/reject
  logic yet — those are Phase 2.

  --- Prefill now uses forward_batch() — IMPLEMENTED, real speedup today ---

  Independent of speculative decoding itself: main.cpp's prefill loop
  previously called forward() once per prompt token sequentially (rereading
  every layer's weights once per token, same as decode). It now gathers the
  whole (non-reused, see kv-reuse above) prefill range into one
  forward_batch() call, reading each layer's weights once for the entire
  chunk instead of once per prompt token. The decode loop is untouched (still
  single-token forward() per step, since each step needs the previous step's
  sampled token before it can run — genuinely sequential, unlike prefill).

  Verified via A/B binaries (batched prefill vs. the old sequential-loop
  prefill, same checkpoint, --show_ids): identical token_ids on every prompt
  tested (confirms losslessness beyond the QUICKLM_VERIFY_BATCH diagnostic —
  this is the real prefill path, wired through do_forward's caller, not a
  synthetic check), and:
    - ~30-token prompt: prefill 2.107s -> 1.024s (~2.1x)
    - ~90-token prompt: prefill 5.731s -> 2.605s (~2.2x)
  Win scales with prompt length, as expected from eliminating once-per-token
  weight rereads.

  --- Phase 2: draft model + verify/accept-reject loop — NOT YET IMPLEMENTED ---

  Design decided so far (not yet built):
  - Draft strategy: self-speculative decoding using the SAME Qwen3.5-0.8B
    checkpoint loaded twice via two DecoderModel instances at different
    precisions — bf16 target, int4 draft — rather than a separate smaller
    model. --precision is already a per-DecoderModel::load() runtime param,
    not baked into the checkpoint, so this needs zero new quantization code
    (int4/int8 already exist in ModelWeights/safetensors.cpp/math_ops.cpp,
    generic across architectures) and zero vocab-mismatch risk (same
    tokenizer file for both instances).
  - A genuinely different (smaller/other) model as the draft is also
    architecturally possible via the pluggable arch registry, but only if
    its tokenizer is bit-identical to the target's. Policy decided: HARD
    REJECT at load time on any tokenizer mismatch (fingerprint/hash check —
    not yet implemented). No silent fallback to full decoding.
  - New CLI surface (not yet added): --draft-model (defaults to the target
    checkpoint for self-speculative mode), --draft-precision (e.g. int4),
    --draft-tokens (how many tokens to draft per round).
  - Verify step: draft model proposes K tokens single-token-at-a-time (its
    own decode loop, cheap because int4), then the target model verifies all
    K in one forward_batch() call and the accept/reject + rejection-sampling
    logic walks the K rows, keeping the accepted prefix and resampling at
    the first rejection from the target's distribution.
  - Outstanding hard constraint: Qwen3_5GatedDeltaNet::recurrent_state is a
    single dense accumulator mutated in place — it cannot be rolled back
    once later positions are folded in (the same constraint that scoped
    kv-reuse to linear-extension-only, see below). Phase 2's accept/reject
    loop will need to snapshot conv_state/recurrent_state before each verify
    chunk and restore + replay the accepted prefix on any rejection. Not
    designed in detail yet — the next thing to scope when this phase is
    picked up.

kv-reuse        [IMPLEMENTED]
  Reuses the live KV-cache/DeltaNet state across turns of one conversation
  instead of recomputing it, when a later turn's prompt is a superset of an
  earlier one. Implemented alongside a minimal multi-turn session mode:
  --interactive (reads further turns from stdin until /exit or EOF) and a
  "\np"-delimited --prompt for scripted multi-turn runs, e.g.
  --prompt "Hey\npHow are you?" is 2 turns of one conversation.

  Scope: in-session, strictly linear reuse only (no cross-process/disk
  caching, no branching/multiple-conversation caching). This is a hard
  constraint, not a simplification: Qwen3.5's Gated DeltaNet layers keep a
  single dense recurrent_state mutated in place every forward() call, unlike
  plain GQA's per-position K/V cache, so once later tokens are folded into it
  there is no rolling it back or branching it. Any mismatch anywhere in the
  cached range means a full reset + replay from scratch, always correct,
  just not sped up that turn.

  Implementation lives entirely in main.cpp's turn loop (no changes to
  model.hpp/attention.cpp/modules.hpp): each turn, decode() the tokens
  already forwarded and check they're a literal text prefix of this turn's
  freshly-rendered prompt; if so, tokenize only the new suffix and extend
  cached_ids directly instead of re-tokenizing the whole conversation.
  Also fixed default_chat_template()'s assistant-role branch to re-emit the
  "<think>\n\n</think>\n\n" generation-priming block in history (it was being
  dropped, which guaranteed a cache miss right after every turn boundary).

  Verified: reuse actually fires and skips real work (e.g. a 2-turn
  conversation reused 63/91 cached tokens on turn 2, cutting that turn's
  prefill from 8.4s to 2.4s), and is deterministic (identical output across
  repeated runs of the same conversation).

  NOTE: comparing kv-reuse on vs off token-for-token beyond turn 1 is not a
  valid correctness bar and was dropped as a verification criterion. The
  "off" path (and turn 1, always) builds its prompt by decoding history text
  and re-tokenizing the whole conversation from scratch; re-encoding a prior
  turn's decoded text is not guaranteed to reproduce the exact tokens the
  model originally sampled, even though the text is byte-identical (BPE
  decode-then-re-encode is not a stable round trip in general). That is a
  property of any from-scratch multi-turn retokenization, not something
  kv-reuse introduces — kv-reuse's entire design point is to avoid it by
  extending cached_ids directly instead of re-tokenizing history.

paged-kv
  Stores the KV-cache in fixed-size "pages" (like OS virtual memory)
  instead of one contiguous block. Removes over-allocation and
  fragmentation, so you fit longer context / more streams in the same
  VRAM. Same math, better memory packing.

flash-attn
  Fused attention kernel: computes softmax(QK^T)V without ever writing the
  full attention matrix to global memory. Keeps intermediates in on-chip
  SRAM, so it moves far fewer bytes. Identical math, less memory traffic —
  a true free lunch. Especially helps long context.

fusion          [IMPLEMENTED]
  Merges several small elementwise ops into one loop so intermediate values
  stay in registers instead of round-tripping through memory. Same results,
  less memory traffic. Implemented as two fused kernels: silu_mul (SiLU +
  elementwise-multiply in the SwiGLU MLP) and add_rms_norm (residual-add +
  RMSNorm before the MLP block). Verified bit-for-bit lossless (identical
  greedy token IDs with the flag on vs. off). NOTE: measured effect on
  tokens/sec is negligible on this model/hardware — decode is bound by
  weight-matmul memory traffic (hundreds of MB/token), and the elementwise
  buffers fusion touches are only a few KB/token, several orders of
  magnitude smaller. Kept for correctness/cleanliness and as a building
  block, not as a throughput win.

cuda-graphs
  Captures the repetitive decode loop as a single CUDA graph and replays
  it, instead of re-issuing hundreds of individual kernel launches per
  token. Kills CPU-side launch overhead — pure efficiency, zero math change.

prefetch        [IMPLEMENTED]
  Streams/loads the next layer's weights while the current layer is still
  computing, hiding memory latency behind compute. Same weights, just
  fetched earlier so the pipe never stalls. Implemented as cross-layer
  software prefetch (_mm_prefetch, L2 hint): each layer warms the first
  weight tensor of the NEXT layer (attention Q/K/V, or the fused QKV
  projection for Gated DeltaNet layers) while it is still computing itself.
  Verified bit-for-bit lossless (identical greedy token IDs with the flag
  on vs. off). All other methods remain placeholders.
=================================================