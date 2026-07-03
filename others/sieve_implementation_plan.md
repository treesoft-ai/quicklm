# Sieve implementation plan

Implementation plan for the current Sieve design (`others/sieve_design.md`).
Older Sieve artifacts (a stray `dist/cache/sieve.obj` with no matching source,
and old-format `models/Qwen3.5-0.8B/vocab_sieve.bin` / `vocab_sieve_512.bin`)
were deleted as irrelevant leftovers from a prior version — nothing in `src/`
implements Sieve yet as of this plan.

Follows the design doc's own §8.3 validation order: cheapest-to-falsify
first, biggest-win-smallest-blast-radius first. Scoped to a subset of the
full four-sieve system (σ/V/KV/draft), not all of it at once.

## Phase 0 — Format decision for the scout substrate

No producer tool for the old `vocab_sieve*.bin` files exists in-repo, and
their header was only ever inferred from raw bytes. Decision: define a
clean, documented scout file format now rather than reverse-engineer the
deleted binaries, and write the offline converter as part of this phase.

- Format: magic + version, vocab_size, hidden_size, SVD rank r, sketch bit
  width, then `U_r` / `V_r` factors, per-row sketches, per-row norms.
- New files: `src/model/scout.hpp` / `.cpp` (load/parse the scout table).
- A conversion utility that builds scouts from the original fp16 checkpoint
  weights (never from int4 — quantization noise must not contaminate the
  ranking signal, per §4.1). Could be a `quicklm.exe --convert-sieve` mode
  or a standalone tool.

## Phase 1 — V-sieve (vocabulary sieve), greedy mode only

Highest value, smallest risk, exactly checkable token-for-token against the
dense baseline under greedy decoding (§4.4's exactness table).

- Load the scout table for `lm_head` at model load.
- Decode time: scout-rank all vocab rows cheaply, take a shortlist, compute
  exact int4/bf16 logits only for the shortlist, margin-check/widen until
  the top-1 gap exceeds the scout error bound before returning.
- CLI: add `sieve` to `--optimize`'s valid set in `main.cpp`. Auto-disable
  outside greedy decoding (same pattern as the existing `speculative`
  temp>0 handling) with a printed notice.
- Correctness gate: paired greedy generations (sieve on/off) over a fixed
  prompt set must produce identical token sequences before this phase is
  done.

## Phase 2 — σ-sieve for the FFN (SwiGLU gate/up pairing)

Per the admission model (§3.6), FFN is the clear "always admit" case at
this model size; attention projections stay dense (§3.6/§4.3 — Qwen3.5's
recurrent `qkvz` must stay dense per §4.3's error-compounding argument).

- Extend the scout format/loader to per-layer FFN matrices.
- Rank gate+up row pairs jointly (one scout, two matrices); survivor list +
  int4 GEMV over survivors only.
- Threshold τ hardcoded/config-driven for now — the full global KL-budget
  allocator (§6.2) is a calibration project of its own and is explicitly
  deferred, not silently dropped.
- Correctness gate: perplexity / logit-KL comparison vs. the dense int4
  baseline on a small eval set (this sieve is lossy by design, so token
  identity isn't the right bar here).

## Phase 3 — Wiring, benchmarking, docs

- `--optimize sieve` help text in `main.cpp`, matching the existing style
  for `speculative` / `fusion` / `prefetch`.
- Makefile: add new `.obj` build rules and `OBJECTS`/`HEADERS` entries for
  the new files.
- Benchmark against the one measured baseline number that exists today
  (~13.7 tok/s / ~20.6 GB/s effective) to get a first real datapoint —
  everything in §7 of the design doc is currently a projection.

## Explicitly out of scope for this pass

- **KV-sieve** — the design doc itself says the win is small on this
  hybrid model (only 6/24 layers have a KV cache at all).
- **draft-sieve** fusion into `speculative.cpp` — the doc's own honest
  number for the Qwen3.5 hybrid case is only ×1.2–1.6, and it depends on
  Phase 2 already existing to have something to fuse with.
- **Full per-tensor admission model / global divergence-budget allocator**
  (§3.6, §6.2) — real calibration infrastructure. Conservative thresholds
  are hardcoded first; this is revisited once Phase 1/2 numbers exist.

## Prerequisite note

Per `[[quicklm-baseline-broken]]`-type concerns (see project memory): if the
pristine baseline is decoding garbage on the current checkpoint at plan
time, quality gates for Phase 1/2 (token-identity check, perplexity/KL
comparison) are blocked until that's fixed — a lossy optimization can't be
validated against a broken reference.
