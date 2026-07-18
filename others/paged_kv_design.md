# Paged KV-Cache Design

Status: **NOT YET IMPLEMENTED.** This is a design document only. Unlike
`speculative_decoding_design.md`, this doc also records why the feature is
currently judged **not worth building for QuickLM as it exists today** — it
exists to scope the work precisely *if* that judgment changes (e.g. a future
multi-stream serving mode), not as a queued implementation plan.

---

## 1. TL;DR

- Paged-KV replaces one contiguous `Tensor({max_seq_len, num_kv_heads,
  head_dim})` per attention layer with fixed-size pages allocated on demand,
  addressed through a per-sequence page table, so unused capacity isn't
  reserved up front.
- Its value proposition is fundamentally about **multi-sequence serving**:
  packing many concurrent streams of unpredictable length tightly into a
  shared, fixed VRAM/RAM pool without fragmentation. vLLM invented it for
  exactly that setting.
- QuickLM has no concurrent-stream scheduler — `--interactive` and
  `--prompt` both run **one sequence at a time**. There is nothing to pack
  more tightly against, so the core motivation for paging doesn't apply yet.
- It only helps `Qwen3_5Attention`'s position-indexed K/V cache.
  `Qwen3_5GatedDeltaNet` (18 of 24 layers) has no such cache at all — it's a
  dense recurrent accumulator — so paging can never be more than a partial
  win here, capped by 6/24 layers' memory footprint.
- **Recommendation: do not implement now.** Section 8 below is the design to
  come back to if/when QuickLM grows a multi-stream serving mode.

---

## 2. Background

From `QI_roadmap.md`:

> Stores the KV-cache in fixed-size "pages" (like OS virtual memory) instead
> of one contiguous block. Removes over-allocation and fragmentation, so you
> fit longer context / more streams in the same VRAM. Same math, better
> memory packing.

The "same math" part is correct and important: paging is purely a memory
*layout* change. Every attention computation reads the identical K/V values
it would have read from a contiguous cache — this is not a lossy or
approximate technique, and if implemented it would be exactly as
bit-identical as `prefetch`/`fusion`/Phase 1 batching already are.

The "more streams" part is the entire reason paging exists as a technique,
and is where it stops applying to QuickLM's current shape (§3).

---

## 3. Why it doesn't pay off here today

### 3.1 No concurrent streams to pack

Contiguous over-allocation only wastes memory *relative to something else
that wants that memory*. In a multi-tenant server (vLLM's original setting),
dozens of sequences of different, unpredictable final lengths share one
GPU's VRAM; reserving `max_seq_len` per sequence means most of that
reservation sits empty most of the time, and that waste is multiplied by
however many concurrent requests are in flight. Paging turns that into
"page contents", used by whichever sequence currently needs it.

QuickLM's `--interactive` mode and `--prompt "\np"`-delimited scripted mode
(`kv-reuse`, see roadmap) are still exactly **one sequence, one
`Qwen3_5Attention` instance's cache, at a time**. There's no second sequence
competing for the unused tail of that allocation. Fragmentation, by
definition, requires multiple allocations of different lifetimes sharing an
arena — with one live sequence there is nothing to fragment against.

### 3.2 Absolute memory scale is small

`k_cache`/`v_cache` are `Tensor({max_seq_len, num_kv_heads, head_dim})` —
for the 0.8B checkpoint's dimensions, that's on the order of a few hundred
MB even at a generous `max_seq_len`, per direction, held once (not per
concurrent request). This is a CPU inference binary on one machine, not a
GPU serving multiple tenants against a hard VRAM ceiling — there's no
memory-pressure signal driving a paging investment the way there was for
vLLM's actual deployment target.

### 3.3 Partial coverage: DeltaNet layers are untouched by design

Only `Qwen3_5Attention::k_cache`/`v_cache` are position-indexed arrays that
paging could apply to at all. `Qwen3_5GatedDeltaNet::conv_state`/
`recurrent_state` (§9 of the speculative design doc) are dense accumulators
mutated in place every step — there is no "cache of positions" to page,
only a fixed small per-layer state that's already minimal. In this model,
18 of 24 layers are DeltaNet, so even a fully-realized paged-kv only ever
addresses the memory profile of 6 layers' caches.

### 3.4 Conflicts with kv-reuse's contiguity assumption

`kv-reuse` (**IMPLEMENTED**, see roadmap) and the planned speculative
decoding snapshot/restore path both currently rely on the cache being flat
and position-addressable via direct pointer arithmetic
(`k_cache.data + pos * kv_size`, `attention.cpp:99-151`). Paging replaces
that with a page-table lookup (`pos -> (page_id, offset)`) on every read/
write in the single hottest loop in the codebase. That's a real per-token
indirection cost paid on every decode step, for a memory-packing benefit
that (per §3.1–3.3) has no consumer today.

---

## 4. What would have to be true for this to be worth building

Revisit this doc when any of the following becomes an actual QuickLM goal:

- A serving mode that holds **more than one sequence's state concurrently**
  (e.g. an HTTP server handling several in-flight requests), where
  fragmentation/over-allocation across sequences becomes a real, measurable
  problem instead of a theoretical one.
- `max_seq_len` growing large enough (very long context) that the
  contiguous reservation itself — not multi-tenancy — becomes the memory
  bottleneck for a *single* sequence, e.g. wanting to support a context
  length where reserving the full worst case up front is wasteful even for
  one stream.
- A GPU port where VRAM is a hard, small ceiling shared across whatever else
  is resident, unlike the current CPU/RAM setting.

None of these are currently true of QuickLM.

---

## 5. Design sketch (if/when revisited)

Recorded here so the next attempt doesn't have to re-derive it from
scratch.

### 5.1 Page structure

- Fixed page size `P` positions (vLLM commonly uses 16; a power of 2 sized
  to amortize page-table lookup overhead against SIMD row width is the
  right way to pick it here — this needs benchmarking, not a guess).
- Each page: `Tensor({P, num_kv_heads, head_dim})` for K, same for V —
  i.e. the current single `k_cache`/`v_cache` allocation split into
  `ceil(max_seq_len / P)` chunks, allocated lazily as `pos` advances past
  each page boundary instead of all up front.
- Per-sequence page table: `std::vector<int>` (or similar) mapping logical
  page index -> physical page slot. For QuickLM's single-sequence case this
  table only ever grows (append-only), mirroring `kv-reuse`'s existing
  append-only assumption — no need for the reclaim/eviction machinery
  multi-tenant servers need, unless §4's multi-stream case actually lands.

### 5.2 Allocator

- A simple free-list of page-sized buffers, shared across attention layers
  (each layer still owns its own page table, since K/V are per-layer, but
  the underlying page buffers can come from one pool sized in page units
  rather than per-layer `max_seq_len` units).
- Only matters once there's more than one sequence — for one sequence, a
  pool is equivalent to just growing the existing contiguous tensor
  on demand, without introducing indirection at all (see §6 below for that
  cheaper alternative).

### 5.3 Read/write path changes

`attention.cpp`'s KV write (`k_cache.data + pos * kv_size`) and the
attention-score inner loop's read (`k_cache.data + p * kv_size + kv_h *
head_dim`) both become two-step: `page = table[p / P]; offset = (p % P) *
kv_size + kv_h * head_dim; ptr = page.data + offset`. This is the cost
called out in §3.4 — an extra integer div/mod and an extra pointer
indirection per position touched, on the hottest loop in the codebase.

### 5.4 Scope boundary

- Applies to `Qwen3_5Attention` only. `Qwen3_5GatedDeltaNet` is explicitly
  out of scope (§3.3) — its `IAttention::reset_states()` /
  `snapshot_states()` / `restore_states()` interface is unaffected.
- `kv-reuse`'s prefix-reuse logic (main.cpp turn loop) would need no
  changes beyond whatever accessor replaces raw pointer arithmetic — it
  already only deals in logical positions, not raw offsets.

---

## 6. Cheaper alternative that solves the *actual* current problem

If the real complaint is "`max_seq_len` reservation is wasteful for short
conversations," a simpler fix captures most of the benefit without paging's
indirection cost: allocate `k_cache`/`v_cache` lazily and let them **grow**
(e.g. geometric growth, like `std::vector`) instead of being fixed at
`max_seq_len` from `init()`. This keeps the cache contiguous — no page
table, no per-access indirection, `kv-reuse` and the speculative
snapshot/restore path need zero changes — and only pays a (rare, amortized)
reallocation-and-copy cost on growth instead of a lookup on every access.
This does not help multi-stream packing (§4's actual paging use case) but
directly addresses "why did we reserve memory nothing used" for the
single-sequence case QuickLM actually has today.

Worth doing (small, low-risk) independent of whether real paging is ever
built. Not yet implemented; also out of scope for this document's core
question (paged-kv proper) but noted since it's the more relevant next step
given QuickLM's current single-stream shape.

---

## 7. Verification plan (if built)

Same bar as every other optimization in this project:

- `--show_ids` token-for-token identical output vs. `--optimize` without
  `paged-kv`, on greedy decoding, across several prompts and context
  lengths that cross multiple page boundaries (not just single-page runs —
  the page-boundary crossing is the actual new code path being tested).
- A/B wall-clock comparison isolating the page-table indirection's cost on
  the single-sequence path (expected: a small regression, since §3 argues
  there's no offsetting benefit yet) versus its benefit under a synthetic
  multi-stream harness, if one exists by the time this is built.

---

## 8. Out of scope until §4's preconditions hold

- Multi-stream/multi-tenant scheduling itself — paging is a memory-layout
  primitive a scheduler would use, not a scheduler.
- Page eviction/reclaim policy — only needed once concurrent sequences
  compete for a bounded pool; single-sequence use is append-only forever.
- Cross-sequence page sharing (e.g. shared system-prompt prefixes across
  requests) — a real vLLM feature, but presupposes the multi-stream case
  this doc argues doesn't exist in QuickLM yet.
- Interaction with `flash-attn` / `cuda-graphs` / speculative decoding
  Phase 2 — all still placeholders or in-progress themselves; not
  designed against a feature that isn't built.
