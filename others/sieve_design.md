# Sieve — a bandwidth-first, admission-controlled, slightly-lossy CPU inference engine

**STATUS: DESIGN DOCUMENT ONLY.** Nothing in this document has been implemented or
benchmarked. Every performance figure is a paper estimate derived from the roofline
math shown alongside it. The single measured datum anywhere in this document is
QuickLM's existing bf16 baseline (~13.7 tok/s on the reference machine); everything
else is a target with stated assumptions. Section 8 defines the validation order
that would turn these estimates into numbers.

Reference hardware profile (one target tier among several, not a limit):
Intel i5-10600K (6C/12T, Comet Lake, AVX2, no AVX-512), 32 GB DDR4-2666
dual-channel (~42.7 GB/s theoretical).

---

## 1. Core thesis

Single-stream autoregressive decode on a CPU is governed by one number:

```
tok/s  ≈  effective_memory_bandwidth / bytes_read_per_token
```

FLOPs are nearly free at this operating point (§2 quantifies the ~35:1 idle-compute
ratio on the reference machine). Therefore the *only* route to a 10–30× speedup is
reading 10–30× fewer bytes per emitted token, and every byte-reduction technique
must itself cost almost no bytes.

Sieve's organizing principle: **never read a byte that has not been justified by a
cheaper byte.** Every weight matrix carries a compact "scout" summary (~3–6% of its
size, §4.1). Scouts are read every token; full weights are read only where a scout
predicts the result will matter. One scout substrate feeds four sieves:

| Sieve        | What it filters                | Bytes it protects            |
| ------------ | ------------------------------ | ---------------------------- |
| σ-sieve      | rows of weight matrices        | FFN + attention projections  |
| V-sieve      | vocabulary rows of the lm_head | the single largest matmul    |
| KV-sieve     | pages of the KV cache          | long-context attention reads |
| draft-sieve  | token futures (speculation)    | whole forward passes         |

**Scaling honesty, stated up front:** *"one substrate, four sieves" is Sieve's full
form, and the full form is a ≥7B-vanilla-transformer configuration.* On small
models (≤1B), the per-tensor admission model of §3.6 — the same cost model, run at
load time — selects a degenerate subset: V-sieve + FFN σ-sieve + draft-sieve, with
attention projections left dense because scout overhead exceeds the bytes saved on
small matrices. That subset is where the bytes actually are on small models
(lm_head ~20%, FFN ~40%), so this is the design working as intended, not a
fallback. Likewise, on hybrid-recurrent architectures (Qwen3.5's DeltaNet layers),
§4.3's conservative policy further narrows what is sieved. The headline scales
*with the model*, and §7's table shows each configuration separately rather than
quoting the best case as if it were general.

Sieve is deliberately lossy, but only slightly and only where a knob says so: a
global divergence budget (§6) is allocated across layers by measured
bytes-saved-per-unit-error, every preset is a hypothesis until it passes an eval
gate, and two components (speculation; V-sieve under greedy decoding) add zero
loss on top of the sieved target by construction.

---

## 2. Bottleneck analysis: why current CPU inference is slow

### 2.1 The roofline, with real numbers

Decode reads every active weight once per token (plus KV cache). On the reference
machine:

- **Theoretical bandwidth:** 2 channels × 8 B × 2666 MT/s = **42.7 GB/s**.
- **Achieved by QuickLM today:** ~13.7 tok/s at bf16 with ~1.5 GB of weights read
  per token → **~20.6 GB/s** effective. (The one measured number in this doc.)
- **Practical ceiling with tuned streaming kernels:** ~30–35 GB/s (NT loads,
  sequential access, all channels busy). Getting from 20 → 30 GB/s is ordinary
  kernel engineering, assumed but not yet done.

- **Compute available:** 6 cores × 2 FMA ports × 8 fp32 lanes × 2 FLOP × 4.1 GHz
  ≈ **~790 GFLOP/s** fp32 (more in int8). A dense 0.8B-parameter decode step is
  ~1.6 GFLOP. Compute alone could sustain ~490 tok/s; bandwidth at bf16 sustains
  13.7. **The machine idles ~35:1 on compute during decode.** Every design
  decision below spends that idle compute to avoid bytes.

The same shape holds across tiers — a 2016 i7-6700K (DDR4-2133, ~25–28 GB/s
practical) idles ~25:1; a modern AVX-512+VNNI chip on DDR5 idles even harder in
int8. CPU inference is a memory-bandwidth problem on every CPU made since 2016.

### 2.2 The three levers, and why they don't multiply cleanly

```
tok/s ≈ (BW × tokens_committed_per_pass) / (bytes_per_pass)
bytes_per_pass ≈ model_bytes × quant_ratio × sieve_survival_ratio + scout_bytes
```

1. **Fewer bytes per weight** — quantization (int4: 4× vs bf16).
2. **Fewer weights per token** — sieving (skip rows/pages that won't matter).
3. **More tokens per read** — speculation (verify K drafted tokens in one pass).

These interact destructively, and the estimates in §7 assume 30–50% mutual
erosion rather than clean multiplication:

- Quantization noise degrades scout ranking quality (mitigated in §4.1 by
  building scouts from the original fp16 weights, but the *computed* activations
  the scouts consume are still noisier).
- Sieving degrades draft quality, which degrades speculative acceptance rates.
- Speculation's batched verify makes each pass bigger, which dilutes the
  fixed-cost overheads but stresses the same bandwidth.

### 2.3 The second bottleneck, visible only after the first falls

Once bytes/token drops ~5–10×, previously invisible costs become first-order:
per-matmul thread synchronization (µs each × ~7 matmuls × 24 layers), scout
compute on the serial path, sampling, and gather latency for sieved (non-
sequential) row fetches. §4.2 and §7 carry these as an explicit **orchestration
tax** line item rather than letting them silently eat the projections. At the
extreme (server tier, §7.4), the ceiling is set by synchronization, not
bandwidth — this is why "thousands of tok/s" is an engineering claim about
latency floors, not just a bandwidth claim.

---

## 3. The stacked techniques (established, composed)

Each entry: what it is, why it works at the hardware level, what it costs in
quality.

### 3.1 Int4 groupwise quantization with int8 outlier channels

- **What:** 4-bit weights with per-group (g = 64–128) fp16 scales, AWQ-style
  activation-aware scaling at conversion. The ~0.5–1% of channels with outlier
  magnitudes, plus the first and last decoder layers and all norms, stay int8/fp32.
- **Hardware:** 4× fewer bytes than bf16 — the single biggest lever. Nibble
  unpacking via `_mm256_shuffle_epi8` LUT + `maddubs` int8 dot products is a
  handful of ops per 32 weights, invisible next to the DRAM stalls it eliminates.
  No VNNI required (Comet Lake and 2016 Skylake both qualify); VNNI/AMX tiers get
  it faster still.
- **Quality:** the best-characterized technique in the stack: typically +2–5%
  relative perplexity at 7B, +3–8% at 0.8B (small models are hit harder). This is
  the floor of Sieve's total loss — everything else stacks on top of it.

### 3.2 Batched verify forward (K tokens per weight pass)

- **What:** a `forward_batch()` primitive that runs K positions through the layer
  stack reading each weight once. Precedent exists in QuickLM, where wiring it
  into prefill measured 2.1–2.2× — evidence that the memory-bound model of §2 is
  correct on this codebase.
- **Hardware:** amortizes the per-pass weight read across K tokens; this is the
  mechanism that makes speculation (§4.6) pay.
- **Quality:** lossless (bit-exact batched math, already verified in QuickLM).

### 3.3 Int8 KV cache, paged layout, per-page metadata

- **What:** K/V stored int8 with per-head scales in fixed-size pages (e.g., 16–64
  positions); each page carries small metadata (per-dimension key min/max or a key
  centroid) used by the KV-sieve (§4.5).
- **Hardware:** halves KV bytes vs fp16; paging makes attention reads
  cache-line-aligned and gives the KV-sieve a skippable unit that maps 1:1 to
  DRAM bytes avoided.
- **Quality:** int8 KV with per-head scales is near-lossless in the literature;
  page *skipping* is where loss enters, budgeted in §6.

### 3.4 Survivor-list software prefetch

- **What:** sieved row fetches are gather-shaped — addresses known only after
  ranking. The sieve emits its survivor list a block ahead; a prefetch cursor runs
  `_mm_prefetch` 4–8 rows ahead of the compute cursor.
- **Hardware:** raw random 0.5–2 KB reads achieve maybe 60–80% of streaming
  bandwidth; with software prefetch at depth d, each row's compute (~100–600 ns)
  hides the ~90 ns DRAM latency as long as d ≥ 2–4. Documented temporal locality
  of activation sparsity (this token's survivors predict the next token's) also
  lets the *previous* token's survivor list seed prefetch before ranking finishes.
- **Quality:** none — pure latency hiding. **Risk:** if this fails to recover
  near-streaming bandwidth, it silently eats the σ-sieve's win; it is R3 in §8 and
  gets a dedicated microbenchmark before any kernel is built on it.

### 3.5 Persistent threads, lock-free handoff

- **What:** threads pin to cores once at startup and coordinate through spinning
  barriers / SPSC queues; no per-matmul fork-join.
- **Hardware:** at hundreds of tok/s the per-token time budget is 2–7 ms;
  OS-mediated wake-ups (tens of µs) × ~170 matmul boundaries per token would
  consume the entire budget. Spin barriers on 6 cores cost ~0.5–2 µs each; this
  is what makes the orchestration tax (§4.2) merely significant instead of fatal.
- **Quality:** none.

### 3.6 The per-tensor admission model

Sieving is not free: per sieved matmul it costs scout bytes + scout compute + one
extra synchronization point + ranking. So **every tensor is admitted to the sieve
individually, by a cost model evaluated at load time** against the machine's
measured bandwidth and barrier cost (microbenchmarked once at startup):

```
admit(T)  ⇔  E[bytes_saved(T, τ)] / BW_measured  >  scout_cost(T)
scout_cost(T) = scout_bytes(T)/BW + rank_cost(rows(T)) + barrier_cost + stall_bound
```

Worked example on the reference machine (h = 1024, 0.8B-class):

| Tensor            | Dense int4 read | Sieve saving (40–60% skip) | Scout cost   | Verdict      |
| ----------------- | --------------- | -------------------------- | ------------ | ------------ |
| QKV proj 1024²    | 512 KB ≈ 17 µs  | ~7–10 µs                   | ~4–6 µs      | marginal/NO  |
| FFN up+gate pair  | ~3 MB ≈ 100 µs  | ~45–60 µs                  | ~6–8 µs      | YES          |
| lm_head 150k×1024 | 77 MB ≈ 2.6 ms  | ~2.4 ms                    | ~80–100 µs   | YES (always) |

At h = 4096 (7B-class) the QKV verdict flips to YES (rows are 4× bigger, scout
cost grows sublinearly). This is the formal version of §1's scaling statement:
the admission model, not a human, decides which subset of Sieve a given model on
given hardware actually runs — and on small models that subset is V-sieve + FFN +
draft-sieve. The SVD scout rank r also scales with matrix size (r = 8 for small
matrices, up to 32 for lm_head-sized ones) to keep `rank_cost` sublinear.

---

## 4. The Sieve mechanisms

### 4.1 The scout substrate

One data structure per weight matrix, built **offline at model conversion from the
original fp16 weights** (never from the int4 — quantization noise must not
contaminate the ranking signal):

1. **Spectral scout:** the top-r SVD factors `U_r [h×r]`, `V_r [d_out×r]`
   (r = 8–32). At inference, `(x·U_r)·V_rᵀ` estimates every output row's
   pre-activation for `d_out × r` MACs — e.g., 176 K MACs for an 11k-row FFN
   matrix, single-digit µs across cores — and near-zero bytes (the factors are
   tiny and stay cache-hot).
2. **Per-row residual sketch:** a 64–128-bit sign sketch of each row's residual
   (after removing the rank-r component) under a shared random projection, plus an
   fp16 residual norm. Used as a safety band: rows whose spectral estimate is
   below threshold but whose residual norm is large enough to cross it are kept.
3. **Per-row norms** — for magnitude bounds and for the V-sieve's tail estimate.

**Why the hybrid, not sketches alone (the honest math):** a pure SimHash sketch
estimates the angle θ between row and activation with
`std(θ̂) = π·√(p(1−p)/n)` ≈ **0.139 rad at n = 128 bits** (p ≈ 0.5 since
high-dimensional vectors are near-orthogonal). But the rows that survive vs. die
in an FFN differ by |cos θ| gaps of only ~0.05–0.15 — signal-to-noise ≈ 1.
**1-bit sketches alone provably cannot rank FFN rows.** The spectral scout
captures the systematic component of ⟨w, x⟩ (which SVD concentrates by
construction); the sketch only has to bound the residual, a much easier job. This
split is also calibration-free — derived from the weights themselves, no training
data, unlike DejaVu/PowerInfer-style trained predictors.

Storage: sketches+norms ≈ 18 B/row ≈ 0.9–3.5% of int4 weight bytes; SVD factors
< 1%. Total substrate ≈ **3–6% of model size**, read (in part) every token.

### 4.2 σ-sieve: row selection for weight matmuls

Per admitted matmul: scout estimate → threshold τ with safety band → survivor
list → prefetched gather → int4 GEMV over survivors only.

- **SwiGLU pairing:** gate and up projections share the input and their rows pair
  1:1 through the elementwise `silu(gate)·up`; one scout ranks the *pairs* and
  both matrices skip together. One scout, two matrices' savings — this is why the
  FFN admission verdict is so favorable.
- **`down_proj` caveat:** its input (the post-SiLU intermediate) is exactly what
  the sieve just sparsified, so its *columns* can be skipped for free (a skipped
  intermediate value zeroes a column's contribution — no scout needed). Its
  low-rank structure for row-sieving the output side is unknown (open question
  O1, §8); the admission model leaves it column-sparse-only if SVD tail energy is
  poor.
- **Pipeline timing (the serial-path accounting):** the scout adds a genuinely
  serial phase per matmul — the next op's input doesn't exist until the previous
  op finishes, so scout latency sits on the critical path. Costs, on 6 cores at
  0.8B scale: scout compute+read ~3–8 µs, ranking (parallel partial-select +
  merge) ~2–4 µs, **one extra spin-barrier ~1–2 µs** (the scout→gather boundary,
  which dense fork-join never had), first-block gather stall ≤ ~1 µs (bounded by
  prefetch_depth × DRAM latency; mitigated by issuing prefetches for
  early-confirmed survivors *during* the rank-merge). Summed over ~5 admitted
  matmuls × 24 layers: **~0.3–0.5 ms/token of orchestration tax ≈ 10–15% of a
  300 tok/s budget.** Carried explicitly in §7's math. It grows with core count —
  the seed of the server-tier sync ceiling (§7.4).

### 4.3 Policy for recurrent (DeltaNet-style) layers

Qwen3.5 interleaves 18 Gated-DeltaNet linear-attention layers with 6 GQA layers.
The DeltaNet state update itself is not a bandwidth problem (the state is small
and cache-resident — it's compute, and sequential). The real issue is **recurrent
error accumulation**: a skipped row in the `qkvz` projection perturbs k/v/q,
which perturbs the state, which feeds forward into *every subsequent token*
through the recurrence — unlike GQA, where a perturbed KV entry stays local to
its position. Sieving error there compounds with sequence length.

**Policy:** on recurrent layers, `qkvz` and all state math stay dense; only the
MLP (fully) and the output projection (conservative τ) are sieved. Consequence,
carried into §7's table as its own row: the hybrid model caps at ~2× byte
reduction from sieving, vs ~2.4–2.8× for a vanilla transformer of the same size.
It also shrinks the draft-sieve's advantage (§4.6). Vanilla-attention models are
Sieve's best case; Qwen3.5 is a deliberately conservative one.

### 4.4 V-sieve: the vocabulary sieve — and its hard limitations

**Mechanism:** the lm_head (150k × h; ~77 MB int4 at h = 1024 — the single
largest per-token read on a small model, ~2.6 ms of the budget) is never read in
full. The scout ranks all 150k rows (~2.4 MB of scout reads + ~15–40 µs of
compute), a shortlist of ~1–2k candidates gets exact int4 logits, and a margin
check widens the shortlist adaptively when top scores are too close to the
excluded set's upper bound. Net: ~2.6 ms → ~0.2 ms, ~13–20×, on the matmul where
small models bleed most.

**Exactness taxonomy — this is a hard limitation, stated as such:**

| Decoding mode                  | Guarantee |
| ------------------------------ | --------- |
| Greedy / argmax                | **Exact** w.h.p. — margin-widening re-checks until the top-1 gap exceeds the scout error bound. |
| Top-k sampling (k ≤ shortlist) | **Conditionally exact**: exact iff shortlist ⊇ true top-k, enforced by the same margin machinery on the k-th gap. Weaker than greedy; checkable at runtime. |
| Top-p / plain temperature      | **Inherently lossy.** If the shortlist misses tail mass q, the total-variation distance of the renormalized shortlist distribution from the truth is *exactly q* (the ½·Σ\|p−p′\| algebra collapses to q). Runtime control: bound the excluded tail via scout scores, `q̂ = Σ_excluded exp((ŝᵢ+εᵢ)/T)/Ẑ`, widen until q̂ < δ. **q̂ is a high-probability bound, not a certificate** — the scout error terms εᵢ come from sketch concentration inequalities, so δ is a probabilistic target, and this remains true everywhere δ appears in this document (§6 included). |
| High temperature, no truncation | **Auto-disabled.** Fat tails by construction; above a temperature threshold the V-sieve turns itself off rather than pretend. |
| Log-prob / scoring APIs        | **Disabled.** The partition function Z is wrong even for shortlisted tokens; anything reading calibrated logprobs — including perplexity evaluation *of Sieve itself* — must run with V-sieve off. |

The adaptive widening loop is likewise best-effort-with-high-probability: it
shrinks q̂ below δ, it does not prove q < δ. Practical note, not an excuse: most
real deployments already truncate (top-k/top-p), which is the regime where the
V-sieve's guarantee is strongest; §7's sampling-mode numbers assume truncated
sampling and say so.

### 4.5 KV-sieve: page skipping for long-context attention

Per-page key metadata (min/max per dimension, or centroid + radius) yields a
cheap upper bound on any page's attention mass for the current query; pages whose
bound falls below ε of the running softmax denominator are never fetched. **Prior
art is acknowledged:** this is Quest/SparQ/H2O territory — Sieve's contribution
is the CPU-specific composition (a skipped page is *exactly* a DRAM read
avoided, and the page bound rides the same scout machinery and admission model as
everything else), not the idea. On Qwen3.5 specifically the win is small — only
6 of 24 layers have a KV cache at all — another entry on the hybrid-penalty list.
ε is charged against the global divergence budget (§6).

### 4.6 Fused draft-sieve: speculation where the draft is the sieve

Self-speculative decoding, greedy accept rule (`accept iff argmax_target ==
draft_token`), batched verify via §3.2, snapshot/restore of recurrent state
around each round (the mechanism QuickLM's speculative decoder already
implements). What's new is what the draft *is*:

**The draft model is the sieve machinery itself** — a forward pass using the
spectral scouts plus int4 on only the top few percent of rows. No second
checkpoint, no tokenizer-mismatch risk, and near-zero extra weight memory: every
byte the draft reads is a byte the target's own sieve needs anyway. The fusion is
two-way — the draft's row rankings at each position are byproducts the verify
pass reuses as its σ-sieve masks (same scouts, same positions), so drafting is
never wasted work, even on rejection.

**The speedup multiplier is architecture-dependent, not a constant.** Draft cost
fraction f (draft bytes / target bytes per token) sets the economics
(net ≈ E[committed]/(1 + K·f + overheads)):

- **Vanilla ≥7B:** everything sieves aggressively in the draft → f ≈ 0.05–0.10,
  the draft is **10–20× cheaper**, and net ×1.6–2.2 at plausible acceptance
  (α ≈ 0.6–0.75, K = 4–6).
- **Qwen3.5 hybrid:** the draft must keep `qkvz` dense too — sieving it collapses
  draft quality through the same recurrent compounding as §4.3, which kills
  acceptance — so f ≈ 0.2–0.35, the draft is only **~3–5× cheaper**, and net
  drops to **×1.2–1.6**. This, stacked on §4.3's byte cap, is why the Qwen3.5 row
  is the lowest in §7's table.

Wherever a ×1.5–2.2 speculation factor appears in this document, it is the
vanilla-transformer figure; the hybrid gets the smaller range. Acceptance rate
itself is unmeasured and is risk R2 (§8) — the whole mechanism degrades
gracefully to ~1× (minus draft overhead, bounded by adaptive K) if acceptance
disappoints, but the projected win evaporates with it.

---

## 5. Adaptation across hardware tiers

Sieve treats the machine as a parameter. At startup it microbenchmarks streaming
bandwidth, gather bandwidth (with prefetch), and barrier cost, then feeds those
into the admission model (§3.6) and the knob policy below. The same model file
runs everywhere; what changes is which sieves are admitted and how hard they bite.

### 5.1 Kernel dispatch

| Tier                       | Int4 dot path                              | Sketch path                | Notes |
| -------------------------- | ------------------------------------------ | -------------------------- | ----- |
| 2016 AVX2 (Skylake-class)  | `pshufb` nibble LUT + `maddubs` int8       | XOR + `popcnt` (64-bit)    | No VNNI; compute margin thinner but still ~25:1 idle |
| AVX2 (reference i5)        | same, higher clocks                        | same                       | Baseline profile for §7 math |
| AVX-512 + VNNI             | `vpdpbusd` fused int8 dot                  | `vpopcntdq`                | ~2–4× int throughput/core; sketches nearly free |
| AMX / server               | tile int8 GEMM for batched verify          | AVX-512 paths              | Verify pass (K rows) is where AMX tiles actually get fed |

### 5.2 Knob policy per tier

The compute:bandwidth ratio decides how aggressively to trade compute for bytes:

- **Low-bandwidth machines (2016, DDR4):** sieve hardest. τ aggressive, wider
  sketches (the compute to read them is idle anyway), longer draft rounds
  (K = 5–6) since each avoided target pass is worth more.
- **High-bandwidth consumer (DDR5):** lighter τ (bytes are cheaper, quality
  budget better spent elsewhere), speculation still fully on.
- **Many-core server:** the admission model starts *rejecting* sieves whose
  barrier cost exceeds their byte savings — on a 64-core part a spin barrier
  costs 2–5+ µs, so small-matrix sieving dies first, exactly as it should.

### 5.3 Threading and NUMA

6–16 cores: flat spin barriers, row-partitioned matmuls, dynamic work-stealing
over survivor lists (survivors are data-dependent, so static partitions skew;
stealing granularity = one prefetched row block). Past ~16 cores: hierarchical
barriers (per-CCX/per-socket then global). Multi-socket: int4 weights sharded by
rows across nodes, **scouts replicated per node** (they're 3–6% of the model —
replication is cheap and keeps every node's serial scout phase local), survivor
lists exchanged via the lock-free queues. The per-token sync floor this
establishes is the binding constraint of §7.4's stretch tier.

---

## 6. Expected accuracy loss

### 6.1 Per-technique budgets (targets, not measurements)

| Technique       | Expected loss                                   | Confidence |
| --------------- | ----------------------------------------------- | ---------- |
| int4 + outliers | +2–5% rel. ppl (7B), +3–8% (0.8B)               | High — best-characterized technique in the literature |
| σ-sieve (FFN)   | +1–5% at conservative τ                         | **Low — the core research bet (R1).** SwiGLU has no hard ReLU; 50–70% skippable is a hypothesis |
| σ-sieve (recurrent layers) | excluded by policy (§4.3)            | — |
| V-sieve         | 0 (greedy); TV ≤ δ per token under truncated sampling — **high-probability, never a certificate (§4.4)** | Medium |
| KV-sieve        | near-lossless at ε = 1–2% (per Quest-adjacent literature) | Medium |
| draft-sieve     | **Zero on top of the sieved target** — greedy speculation is exactly lossless w.r.t. the target policy; the target is already lossy, speculation adds nothing | High (by construction) |

### 6.2 Compounding — why the naive sum is optimistic

Per-layer errors do not add independently: they correlate through the residual
stream, int4 noise makes the activations the scouts consume noisier (degrading
ranking even with fp16-built scouts), and any error entering a recurrent state
compounds across the sequence (§4.3). A naive "+2–6% for q1" could plausibly be
2× worse before tuning. Two mechanisms replace hope with measurement:

1. **The global divergence budget allocator.** On a calibration set, measure each
   layer's logit-KL contribution vs. the dense int4 reference as a function of
   its τ. Then allocate a single global KL budget across layers by greedy
   knapsack on bytes-saved-per-unit-KL — sensitive layers (empirically: first,
   last, and recurrent-adjacent) get lax τ automatically; byte-heavy insensitive
   layers absorb the aggression. Uniform thresholds are never used.
2. **The eval gate as ship-blocker.** Per-sieve error attribution is instrumented
   (each sieve's contribution to logit KL measured independently), and no preset
   ships without passing task-level evals against the dense int4 baseline. Note
   the self-measurement trap from §4.4: perplexity evaluation must run with
   V-sieve disabled, since V-sieve corrupts the partition function.

### 6.3 Presets — hypotheses pending measurement

| Preset | Configuration                          | Target loss (hypothesis)      |
| ------ | -------------------------------------- | ----------------------------- |
| q0     | sieves off (int4 only, or bf16)        | quantization floor only       |
| q1     | conservative τ, δ = 0.01, ε = 1%       | +2–6% rel. ppl — *plausibly 2× that before allocator tuning* |
| q2     | aggressive τ, δ = 0.02, ε = 2%         | +5–12% rel. ppl               |

Small models are more fragile at every stage; a 0.8B model at q2 may be
noticeably degraded in ways perplexity understates. The gate, not this table,
decides what ships.

---

## 7. Projected performance — estimates with the math shown

**Nothing below is measured.** Method: per-token byte budget → passes/s at
assumed effective bandwidth → × speculation multiplier (architecture-dependent,
§4.6) → − orchestration tax (§4.2). Ranges span pessimistic (25 GB/s effective,
low acceptance, full tax) to optimistic (35 GB/s, good acceptance) on the
reference machine. All estimates assume 30–50% mutual erosion between levers
(§2.2) is already priced into the multipliers used.

### 7.1 Worked byte budgets (int4, per token)

**Qwen3.5-0.8B (hybrid, conservative policy §4.3):** total ~400 MB dense.
lm_head 77 → ~5 MB (V-sieve incl. scout reads); FFN ~180 → ~85 MB (45–50%
survivors incl. scouts); projections ~145 → ~110 MB (recurrent `qkvz` dense,
out-proj partial, 6 GQA layers full σ-sieve). **Sieved ≈ 200 MB/token (~2.0×).**

**0.8B-class vanilla transformer (hypothetical comparator):** all 24 layers
admit attention sieving at h ≥ 2048... at h = 1024 the admission model keeps QKV
dense (§3.6), but all layers' out-proj and FFN sieve → **≈ 150–165 MB/token
(~2.4–2.8×).**

**7–8B vanilla:** ~3.5 GB dense int4; FFN dominates and everything admits →
**≈ 1.5–1.8 GB/token.**

### 7.2 From bytes to tokens (reference i5, 25–35 GB/s effective)

| Config            | passes/s     | × speculation        | − tax  | **est. tok/s** |
| ----------------- | ------------ | -------------------- | ------ | -------------- |
| Qwen3.5-0.8B      | 125–175      | ×1.2–1.6 (hybrid)    | ~10%   | **~150–280**   |
| 0.8B vanilla      | 155–230      | ×1.6–2.2             | ~10%   | **~250–400**   |
| 7–8B vanilla      | 14–23        | ×1.6–2.0             | ~5%    | **~25–40**     |

The speculation column is deliberately split per §4.6: the hybrid's draft is only
3–5× cheaper than its target (dense `qkvz` in both), so its multiplier is
×1.2–1.6, **not** the ×1.6–2.2 a vanilla model earns. Reading the vanilla
multiplier as model-independent would overstate the Qwen3.5 row by ~35%.

### 7.3 The full table across tiers (all estimates)

| Config                     | 2016 (i7-6700K, ~25 GB/s) | i5-10600K (~30 GB/s) | Consumer DDR5 + AVX-512 (~80 GB/s) | Server (many-channel, 350+ GB/s) |
| -------------------------- | ------------------------- | -------------------- | ---------------------------------- | -------------------------------- |
| **Qwen3.5-0.8B (hybrid)**  | ~110–220                  | **~150–280**         | ~350–650                           | sync-capped ~800–1500            |
| 0.8B-class vanilla         | ~180–320                  | ~250–400             | ~550–1000                          | sync-capped ~1000–2000           |
| 7–8B vanilla               | ~20–35                    | ~25–40               | ~60–100                            | ~300–600                         |

Sampling-mode footnote: these figures assume greedy or truncated (top-k/top-p)
sampling; unbounded high-temperature sampling disables the V-sieve (§4.4) and
costs the 0.8B rows roughly their lm_head savings back (~25–40% slower).

### 7.4 The "hundreds" claim and the "1000s" claim, honestly

- **Hundreds on 2016 hardware: plausible for ≤1B models only.** Model bytes scale
  linearly and no sieve changes that; a 7B model on a 2016 desktop lands in the
  tens no matter what.
- **Thousands: conditional on all of** (a) a vanilla (non-recurrent) ≤1–2B model,
  (b) DDR5-consumer bandwidth at the low end or server bandwidth comfortably, and
  (c) beating the synchronization floor. At 1000 tok/s the whole token budget is
  1 ms; ~170 matmul boundaries × 1–3 µs of barrier on a many-core part is
  0.2–0.5 ms **before any byte is read** — bandwidth stops being the binding
  constraint and latency engineering (fused layers, fewer sync points, batched
  verify amortizing boundaries across K tokens) takes over. That is why the
  server column says "sync-capped": adding memory channels past that point buys
  nothing for single-stream decode. Physics pushes back through DRAM latency and
  core-to-core coherence, not GB/s.

---

## 8. Risks, open questions, and the validation plan

### 8.1 Ranked risks

- **R1 — Scout SNR on SwiGLU FFNs (the core research bet).** No hard ReLU means
  no clean zero-set; if the spectral+sketch ranking can't separate the bottom
  50–70% of rows at small loss, τ must loosen and the FFN win shrinks toward
  ~1.5×. Measurable offline (see 8.3) before writing a single kernel.
- **R2 — Draft acceptance rate.** An aggressive sieve-draft may agree with the
  target too rarely; speculation then degrades gracefully to ~1× (adaptive K
  bounds the waste) but the projected ×1.2–2.2 evaporates. Also offline-testable.
- **R3 — Gather bandwidth erosion.** If survivor-list prefetch can't recover
  ≥85–90% of streaming bandwidth on sieved row fetches, it silently eats the
  σ-sieve's entire margin. The #1 *systems* risk; pure microbenchmark.
- **R4 — Error compounding beyond the allocator's control** (§6.2), especially
  through the 18 recurrent layers of the primary target model.
- **R5 — Small-model fragility.** 0.8B at int4 is already degraded; sieving on
  top may cross usability thresholds that perplexity understates.
- **R6 — Sync scaling.** The orchestration tax grows with core count; the server
  tier's numbers assume hierarchical-barrier engineering that is itself unbuilt.

### 8.2 Open questions

- **O1:** Does `down_proj` have exploitable low-rank structure, or is it
  column-sparse-only (§4.2)?
- **O2:** Optimal split of scout bytes between SVD rank r and sketch width per
  matrix class — a pure Pareto sweep, needs data.
- **O3:** τ ↔ acceptance interplay: the σ-sieve setting that maximizes raw
  passes/s may not maximize end-to-end tok/s once it degrades the draft.
- **O4:** Can the draft sieve recurrent `qkvz` *mildly* without collapsing
  acceptance (contradicting §4.6's conservative assumption and recovering some
  hybrid speedup)? Cheap to test once R2's harness exists.
- **O5:** KV-sieve value on hybrids — only 6 of 24 Qwen3.5 layers have KV at
  all; possibly not worth its complexity there.

### 8.3 Validation order — cheapest falsification first

1. **Offline oracle studies (R1, R2) — no engine code at all.** In
   NumPy/PyTorch, on real checkpoints: (a) oracle FFN skip rates (drop true
   bottom-x% of rows, measure ppl) — the ceiling; (b) scout ranking recall vs.
   the oracle — the achievable fraction; (c) simulated greedy agreement between
   sieve-draft logits and target — the acceptance estimate. **If R1 or R2 fails
   here, Sieve dies for the price of a notebook.**
2. **Gather microbenchmark (R3)** on each hardware tier: prefetched
   0.5–2 KB row gathers vs. streaming, standalone C++.
3. **V-sieve alone in QuickLM** — biggest single win on the 0.8B target,
   smallest blast radius, and its greedy mode is exactly verifiable against the
   dense baseline token-for-token.
4. **FFN σ-sieve** behind the admission model, KL-allocator calibration, eval
   gate.
5. **Fused draft-sieve**, reusing QuickLM's existing snapshot/restore
   speculative machinery.
6. **Tier ports** (AVX-512/VNNI paths, hierarchical barriers) only after the
   reference machine validates the model of §7 — if the roofline math is wrong
   at 6 cores, it's wrong everywhere.

Each step gates the next; every number in §7 that survives becomes a measurement,
and every one that doesn't gets corrected in this document rather than defended.

