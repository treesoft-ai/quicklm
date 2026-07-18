# Speculative Decoding — Phase 2 Design

Status: **NOT YET IMPLEMENTED.** This is a design document only, written after
Phase 1 (batched/chunked forward — see `QI_roadmap.md`) landed and was
verified. Phase 2 is the actual `speculative` optimization: a draft model,
a verify step, and an accept/reject loop that lets decode commit more than
one token per target-model weight read.

---

## 1. TL;DR

- Draft a few tokens cheaply, verify all of them in **one** batched forward
  pass through the target model (this is what Phase 1 made possible), keep
  the accepted prefix, correct the first wrong token.
- Default draft strategy: **self-speculative** — the same checkpoint loaded
  twice at different precisions (bf16 target, int4 draft). No second model
  file needed, no vocab-mismatch risk.
- Also supports a genuinely different (smaller) draft model, gated behind a
  **hard-reject tokenizer fingerprint check** — refuses to run rather than
  silently producing wrong output.
- The one genuinely hard problem is `Qwen3_5GatedDeltaNet`'s recurrent
  state: it's a dense accumulator, not a position-indexed cache, so it can't
  be "truncated" the way the attention K/V cache can. Section 9 below is the
  actual design for that; everything else is comparatively mechanical.

---

## 2. Background

Decode is weight-memory-bandwidth-bound, not compute-bound — this is the
same bottleneck `prefetch` and `fusion` targeted, and the reason Phase 1
(batched forward) was worth building before touching anything speculative:
verifying K drafted tokens in one target forward pass means the target's
weights are read once per K tokens instead of once per token, which is
exactly where the time goes.

Phase 1 shipped that primitive and is already wired into prefill (~2.1–2.2x
faster prefill, verified bit-for-bit lossless). Phase 2 is applying the same
primitive to the **decode** loop, where it actually pays off as tokens/sec,
via a draft/verify/accept-reject loop instead of just "give it a longer
known-in-advance chunk" (prefill's case).

---

## 3. Draft model strategy

### 3.1 Self-speculative (default)

Load the **same** checkpoint twice, as two independent `DecoderModel`
instances at different `--precision`:

- Target: `bf16` (or whatever `--precision` the user already picked)
- Draft: `int4`

This works with zero new machinery because:
- `--precision` is already a per-`DecoderModel::load()` runtime parameter,
  not baked into the checkpoint.
- int4/int8 quantization already lives in the generic
  `ModelWeights`/`safetensors.cpp`/`math_ops.cpp` path, not
  architecture-specific code — any registered architecture gets it for
  free.
- Same tokenizer file for both instances → zero vocab-mismatch risk,
  no fingerprint check needed for this mode.

### 3.2 Independent draft model

The pluggable arch registry means a genuinely different (and presumably
smaller/faster) model can act as draft. This is **only** safe if its
tokenizer is identical to the target's — a draft that proposes token IDs
from a different vocabulary produces silently wrong verification.

**Policy (decided): hard reject on any tokenizer mismatch.** No silent
fallback to full decoding, no best-effort remapping. If the fingerprints
don't match, refuse to start. See Section 11.

---

## 4. New CLI surface

None of this exists yet. Proposed flags:

| Flag | Default | Meaning |
|---|---|---|
| `--draft-model <path>` | same as `--path` | Checkpoint dir for the draft model. Omitting it means self-speculative mode. |
| `--draft-precision <fp32\|bf16\|int8\|int4>` | `int4` | Precision to load the draft model at. |
| `--draft-tokens <K>` | `4` | Tokens to draft per round before verifying. |

Gated the same way `prefetch`/`fusion`/`kv-reuse` are: only active when
`--optimize speculative` is passed. `--draft-model`/`--draft-precision`/
`--draft-tokens` are parsed but ignored unless `speculative` is enabled
(matches the existing "parsed/validated but no effect" convention for
placeholder flags).

---

## 5. Runtime architecture

Two `DecoderModel` instances live for the duration of the run:

- `target` — loaded at `--precision`, this is the model whose output
  distribution the run must match.
- `draft` — loaded at `--draft-precision` from `--draft-model` (or the same
  checkpoint dir as `target` for self-speculative mode).

Each has its own `Context` (own `pos`, own K/V cache, own DeltaNet
`conv_state`/`recurrent_state` per layer) — they are two fully independent
sequences of forward calls that happen to be kept in lockstep by the
accept/reject loop below. Memory cost is one extra model's weights (small
for an int4 draft of a 0.8B model) plus one extra set of per-layer states.

---

## 6. Per-round algorithm

```
loop each round:
    snapshot target's DeltaNet state (see §9)      # cheap, small buffers
    snapshot draft's DeltaNet state  (see §9)

    draft_ids = []
    for i in 0..K-1:
        tok = draft.forward(prev_tok) -> sample -> tok
        draft_ids.append(tok)
        prev_tok = tok

    target_logits[0..K-1] = target.forward_batch(draft_ids)   # ONE pass, Phase 1 primitive

    accept_count = 0
    while accept_count < K and accepts(target_logits[accept_count], draft_ids[accept_count]):
        accept_count += 1

    if accept_count < K:
        # target's state now reflects K rows, but only accept_count are real.
        restore target snapshot; restore draft snapshot
        if accept_count > 0:
            target.forward_batch(draft_ids[0..accept_count-1])   # replay only the real prefix
            draft.forward_batch(draft_ids[0..accept_count-1])
        correction = sample_from(target_logits[accept_count])     # target's actual choice here
        target.forward(correction)   # advance target state by exactly 1
        draft.forward(correction)    # keep draft state in lockstep
        commit(draft_ids[0..accept_count-1] + [correction])
    else:
        # full accept: target's state already correctly reflects all K tokens (no rollback needed)
        bonus = sample_from(target_logits[K-1])   # free extra token, logits already computed
        target.forward(bonus)
        draft.forward(bonus)
        commit(draft_ids + [bonus])
```

Two things worth calling out:
- **Every round ends with exactly one extra single-token `forward()` call**
  on both models — either the correction token (partial accept) or the
  bonus token (full accept) — because `forward_batch` only advances state
  through the tokens it was given, and the newly-committed final token was
  sampled *after* that call returned. This unifies the two branches.
- The replay on partial accept costs at most `K-1` batched-forward rows —
  small and bounded by `--draft-tokens`, not by anything unbounded.

---

## 7. `accepts()` — correctness policy

This determines what "lossless" even means here, and needs a decision before
implementation, not during it:

- **Greedy decoding (temp = 0):** simplest and matches the roadmap's
  existing claim ("bit-for-bit identical to normal decoding"). Accept iff
  `argmax(target_logits[i]) == draft_ids[i]`. If they match, the target
  would have deterministically produced that exact token anyway, so
  committing it is provably identical to non-speculative greedy decode.
- **Sampling (temp > 0):** true distribution-preserving speculative decoding
  needs the standard rejection-sampling rule (accept `x_i` with probability
  `min(1, p_target(x_i)/p_draft(x_i))`, resample from the residual
  `max(0, p_target - p_draft)` on rejection — see Leviathan et al.,
  "Fast Inference from Transformers via Speculative Decoding"). This is
  real additional work and its own correctness surface (needs both models'
  probabilities, not just logits/argmax).

**Recommendation:** ship greedy-only in the first cut, where "lossless" has
a clean, cheaply-verifiable meaning (`--show_ids` token-for-token match
against non-speculative greedy decode, same as every other optimization in
this project). Treat proper rejection sampling for `temp > 0` as a clearly
separate follow-up, not silently bolted on.

---

## 8. Verified this session, reusable for Phase 2

Not new work, just noting what Phase 2 gets for free from Phase 1:

- `DecoderModel::forward_batch()` — this *is* the verify step.
- Bit-exactness of batching itself is already proven
  (`QUICKLM_VERIFY_BATCH=1`, and the prefill A/B test). Phase 2 does not
  need to re-litigate whether batching changes results — it doesn't.

---

## 9. The hard part: DeltaNet state snapshot/restore

This is the one piece of infrastructure Phase 2 actually needs to build.

**Why attention's K/V cache does *not* need this:** it's indexed by
absolute position. If a round's verify pass writes speculative K/V entries
at positions that later turn out to be rejected, those entries are simply
never read again — the next forward call only ever advances
`ctx.pos = cached_ids.size()`, i.e. the committed length. Stale rows past
that point are inert. This is the same reasoning `kv-reuse` already
established: append-only, self-truncating for free.

**Why `Qwen3_5GatedDeltaNet` state does need this:** `conv_state` and
`recurrent_state` are not position-indexed — they're a single dense
accumulator mutated in place every step (`state_t = g(state_{t-1}, x_t)`).
Once a rejected draft token's contribution is folded into `recurrent_state`,
there is no way to subtract it back out. This is the exact same constraint
that scoped `kv-reuse` to linear-extension-only.

**Design:** snapshot-before, restore + replay-on-partial-reject (per layer,
per model instance that has DeltaNet layers — 18 of 24 layers):

- **Snapshot** = copy `conv_state` (small: `kernel_size - 1` rows per
  channel) and `recurrent_state` (the expensive one: dense, per-head) for
  every `Qwen3_5GatedDeltaNet` layer, taken once before each round's draft
  begins.
- **Restore** = copy back over the live state, only needed when
  `accept_count < K`.
- **Replay** = re-run `forward_batch` over just the accepted prefix
  (`accept_count` tokens) after restoring, so the state ends up exactly
  where it would be had only those tokens ever been forwarded. Bounded by
  `draft_tokens - 1` rows, cheap relative to the K-row verify pass already
  paid for.

Considered and rejected: snapshotting state after *every* row within a
round (to jump directly to the right intermediate state instead of
restore+replay). Would avoid the replay forward pass, but costs `K` full
`recurrent_state` copies per round instead of 1, for a saving that's already
small (`draft_tokens` is expected to be single digits). Not worth the
complexity for v1; worth revisiting only if profiling shows replay cost
matters.

Both `target` and `draft` need this — the draft model's own state also
advances past the eventual correction point during its own drafting loop
and must be rolled back the same way, so its next round starts from the
right place.

---

## 10. Tokenizer compatibility check (independent draft model only)

Not yet implemented. Proposed mechanism:

- At load time, hash each model's tokenizer artifact (vocab + merges +
  special-token table, whatever `Tokenizer::load()` reads off disk) with a
  simple fingerprint (e.g. FNV-1a or SHA-256 over the raw bytes it loads).
- Compare `target`'s fingerprint against `draft`'s before starting any
  generation.
- **Hard reject on mismatch**: print a clear error and exit non-zero. No
  fallback path, no partial-compatibility heuristics (e.g. "same vocab
  size" is not sufficient — same size doesn't mean same ID→token mapping).
- Self-speculative mode (§3.1) skips this check entirely — it's provably
  unnecessary when both instances load from the same tokenizer file.

---

## 11. Expected performance

Expected tokens committed per target-forward-pass, for per-token draft
acceptance probability `α` and draft length `K`, is the standard
speculative-decoding geometric series:

```
E[tokens per round] = (1 - α^(K+1)) / (1 - α)      (α < 1)
                     = K + 1                        (α = 1, degenerate)
```

This means the win is entirely governed by how often the int4 draft agrees
with the bf16 target — not something to promise a number for without
measuring on this model. Worth instrumenting from day one (log
`accept_count` per round, report a running average) rather than treating it
as a black box, so a bad acceptance rate is visible immediately instead of
discovered as "why didn't tokens/sec move."

---

## 12. Verification plan

Same bar every other optimization in this project has been held to:

- `--show_ids` token-for-token identical output vs. `--optimize` without
  `speculative`, on greedy decoding, across several prompts — this is the
  actual acceptance criterion, not a nice-to-have.
- A/B binaries + wall-clock comparison for the tokens/sec claim, same
  method used for the prefill result in Phase 1 (not just a diagnostic
  flag — real runs, real timing).
- Explicitly test a prompt long enough to trigger at least one rejection
  (not just runs that happen to fully agree every round) — the
  restore/replay path in §9 needs its own coverage, not just the happy
  path.

---

## 13. Out of scope for the first cut

- Sampling-mode (`temp > 0`) proper rejection sampling — see §7.
- Multiple concurrent speculative streams / batched requests.
- Interaction with `paged-kv` / `flash-attn` / `cuda-graphs` (all still
  placeholders themselves).
- Per-row DeltaNet state snapshots (the rejected alternative in §9).
- Anything about `--draft-model` pointing at an architecture different
  from the target's own `arch` registration — the tokenizer check in §10
  is necessary but not by itself sufficient for that; cross-architecture
  self-speculative isn't a real use case (self-speculative always shares
  the architecture by definition) and hasn't been thought through for the
  independent-draft-model case.
