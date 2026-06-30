# DEVLOG_2 — Lossless performance optimization (2026-06-27)

**Goal.** Make decode faster **without changing the model's outputs at all** — no
quantization, no approximation that alters intelligence. Every stage is verified
against a golden greedy-token-ID fingerprint (`--show_ids`) on a fixed prompt set;
a stage only ships if all sequences are byte-for-byte identical to the pre-opt
baseline (which itself matches the DEVLOG_1 correct engine).

## The bottleneck (measured)

Hardware: Intel i5-10600K (6c/12t), **DDR4-2133 dual-channel** (~34 GB/s
theoretical, ~25–28 GB/s achievable).

Batch-1 decode is **memory-bandwidth bound**: each token streams the full weight
set from RAM once. Thread-scaling probe confirmed it — 4 threads → 8.5 tok/s,
12 threads → 8.85 tok/s (nearly flat: cores are idle, the bus is the wall).
A `QUICKLM_PROF` breakdown showed **~82% of wall time in matmul**, ~18% in the
rest (DeltaNet recurrence, RMSNorm, RoPE, softmax).

Consequence: the dominant lever is **bytes streamed per token**, and the hard
ceiling is `achievable_bandwidth / bytes_per_token`. At bf16 (the smallest lossless
weight format, since the checkpoint is natively bf16) that ceiling is ~16–19 tok/s
on this RAM. 50 tok/s is physically impossible here without quantization.

## What "lossless" means here

The checkpoint ships as **bf16**. The engine already upcast bf16→FP32 at load, and
that upcast (`f32_bits = u16 << 16`) is exact. So storing weights as bf16 in RAM and
upcasting in-register feeds the matmul the *same values* as before — bit-identical
inputs, no precision lost relative to the current engine or to HF (which also runs
this model in bf16). SIMD/FMA changes only reorder summation (lane reassociation),
which can alter the last FP bit but loses no precision; verified not to change any
sampled token.

## Stages (all verified output-identical)

1. **AVX2 FMA matmul + threading fix** (`math_ops.cpp`).
   - Replaced the scalar dot-product with an AVX2 + FMA kernel using 4 independent
     accumulators (`dot_avx2`).
   - Fixed the matmul to split work across the *actual* worker count (was hardcoded
     8) and to capture operands by pointer instead of copying `Tensor` (shared_ptr +
     vector copies) on every dispatch. Added a work threshold so tiny matmuls run
     single-threaded (avoids dispatch overhead).

2. **bf16 weight storage** (`tensor.hpp`, `safetensors.cpp`, `model.cpp`,
   `math_ops.cpp`).
   - `Tensor` gained an optional `bf16_data` backing store (zero-copy view into the
     mmap). `SafetensorsLoader::get_tensor_keep_bf16` and `ModelWeights::get_weight`
     return weights as bf16; all large matmul operands (q/k/v/o, gate/up/down,
     in_proj_*, out_proj, lm_head) use it. Norms/biases/conv/A_log stay FP32;
     `embed_tokens` stays FP32 for the lookup while lm_head loads bf16 for the GEMV.
   - `dot_avx2_bf16` upcasts bf16→FP32 in-register (`cvtepu16` + shift-left-16) and
     FMAs. Halves the weight bytes streamed per token — the single biggest win.

3. **Vectorized GatedDeltaNet recurrence** (`attention.cpp`).
   - The three `[128,128]` per-head passes (decay+kv_mem, state update, retrieve)
     were scalar; now AVX2/FMA, row-wise over the contiguous `j` axis. The retrieve
     loop was reordered to a SAXPY accumulation so its inner loop is contiguous.
     This was the bulk of the ~18% non-matmul time.

4. **RoPE inv_freq precompute** (`math_ops.cpp`).
   - Hoisted the per-token, per-head `std::pow(theta, 2i/rot)` out of the hot path
     into a cached `inv_freq` table (keyed by theta+rotary_dim). Same arithmetic,
     bit-identical.

5. **Weight-row prefetch** (`math_ops.cpp`).
   - The bf16 matmul prefetches the next weight row (`_mm_prefetch T0`) to hide
     memory latency. Pure hint, numerically identical.

## Build-system fix (root-caused a nasty crash)

Mid-session, editing the `Tensor` struct caused heap corruption (`bad array new
length` on a vector `push_back`, sometimes an access violation). Root cause: the
**Makefile didn't track header dependencies**, so NMake left stale `.obj` files
compiled against the *old* `Tensor` layout and linked them with freshly-compiled
ones → `sizeof(Tensor)` mismatch (ODR/ABI violation) → corruption. Not a bf16 logic
bug at all.
**Fix:** every object rule now depends on `$(HEADERS)` (all headers), forcing a full
recompile when any header changes. When in doubt, touch all `src/*.cpp` to force a
clean rebuild — NMake's incremental logic has bitten us here.

## Results

Baseline (correct engine, pre-opt): **3.83 tok/s**.

| Stage | tok/s | note |
|---|---|---|
| Baseline | 3.83 | scalar FP32 matmul, hardcoded 8-thread split |
| + AVX2 matmul/threading | 4.76 | +24% |
| + bf16 weights | ~9.2 | lm_head matmul 43 ms → 23 ms; ~2.4× over baseline |
| + recurrence AVX2 + RoPE precompute + prefetch | **10.5** | clean idle-machine measurement; ~2.74× over baseline |

**RAM speed dominates (bandwidth-bound confirmation).** Same binary, after enabling
XMP to raise RAM from 2133 → 2666 MT/s (+25% bandwidth): **10.5 → 12.85 tok/s
(+22%)**. Speed tracks memory clock almost 1:1 — definitive proof the engine is
memory-bandwidth bound, not compute bound. Outputs remain byte-for-byte identical
(XMP changes timing, not arithmetic). The kit (Corsair CMK16GX4M2A2666C16) is rated
2666 and now runs at 2666 — no further RAM headroom without manual overclocking.

**Final kernel work @ 2666 MT/s (all bit-identical):**
- 4-row then 8-row register-blocked bf16 GEMV (`dot4_avx2_bf16`, `dot8_avx2_bf16`):
  multiple independent weight-row streams per thread → better memory-level
  parallelism. 12.85 → ~13.7 (quiet machine).
- Row-group prefetch helped; in-kernel per-stream prefetch *hurt* (too many
  prefetch µops clog the load ports) — reverted.
- `matmul_batched`: runs same-input projection groups (MLP gate+up, attn q/k/v,
  linear-attn qkv/z/b/a) under one thread-pool barrier instead of one per matmul.
  Correct and bit-identical; speed effect within measurement noise.

**Diagnostic (`QUICKLM_BENCH`).** A cold large bf16 GEMV (390 MB, doesn't fit cache)
runs at **30.7 GB/s — ~95% of this machine's achievable bandwidth**, so the kernel
itself is essentially optimal. Real decode averages ~16–20 GB/s aggregate; the gap
is the unavoidable serial structure between matmuls (DeltaNet recurrence, norms,
sampling) and per-token overhead, not kernel inefficiency.

**Where it landed.** ~**13–13.7 tok/s on a quiet machine** (≈3.5× over the 3.83
baseline), all outputs byte-for-byte identical. Measurements are sensitive to
background load — a bandwidth-bound workload shares the bus with anything else
running, so numbers dip to ~11–12 under even ~18% CPU load. The lossless bf16 wall
at 2666 MT/s is ~20–22 tok/s; reaching 18+ would need a structural change with
diminishing, hard-to-verify returns, or faster RAM. Quantization (ruled out) or a
GPU are the only routes to materially more.

| Stage | tok/s | note |
|---|---|---|
| + register blocking + batched projections | **~13.7** | quiet-machine best; ~3.5× over baseline |

All stages verified byte-for-byte identical greedy output across the prompt set
(`hey`, `What is the capital of France?`, `Write a haiku about winter.`,
`Explain recursion in one sentence.`).

**Ceiling reminder.** On DDR4-2133 the lossless bf16 wall is ~16–19 tok/s; realistic
optimized max ~12–16. Beyond that requires quantization (ruled out) or faster RAM.

## Verification recipe

```
# golden fingerprint (must be identical before/after any change)
quicklm.exe --path Qwen3.5-0.8B --prompt "<p>" --temp 0 --max_tokens 80 --show_ids
# timing (measure on an idle machine — background load skews bandwidth-bound numbers)
quicklm.exe --path Qwen3.5-0.8B --prompt "<p>" --temp 0 --max_tokens 60
# matmul vs non-matmul split
QUICKLM_PROF=1 quicklm.exe ...   # prints [prof] matmul ms / % of wall
```
