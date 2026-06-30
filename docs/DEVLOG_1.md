# DEVLOG_1 — Engine reference & correctness work (2026-06-27)

QuickLM is a from-scratch, zero-dependency **C++17 CPU inference engine** for the
**Qwen3.5-0.8B** hybrid language model. It implements the entire stack — safetensors
loading, byte-level BPE tokenization, the hybrid (linear + full attention)
transformer, and sampling — in plain C++ with nothing but the standard library and
a hand-rolled thread pool. No PyTorch, no BLAS, no ONNX.

This file is the project reference (what each piece does, how the model maps onto
the code, how to build and run it) plus the first session's change history. Later
sessions are in `DEVLOG_2.md`, `DEVLOG_3.md`, … and `DEVLOG.md` is the index.

---

## 1. What it is, at a glance

- **Target model:** `Qwen3_5ForConditionalGeneration` (a.k.a. Qwen3-Next) — the
  text path of a multimodal checkpoint. Hybrid attention: **Gated DeltaNet linear
  attention** (18 layers) interleaved with **full gated attention** (6 layers) in a
  3:1 pattern.
- **Precision:** FP32 compute. Weights are stored on disk as bf16 and upcast to FP32
  at load time. (See DEVLOG_2 for the bf16-in-RAM optimization.)
- **Platform:** Windows / MSVC primary (memory-mapped via Win32; AVX2 build flags),
  with POSIX `mmap` and a generic compiler path also present.
- **Dependencies:** none beyond the C++17 standard library + `std::thread`.

```
prompt ──▶ Tokenizer ──▶ token IDs ──▶ QwenModel.forward (×N tokens) ──▶ logits ──▶ sample ──▶ token ──▶ decode ──▶ text
                                              │
                            embed → 24 × DecoderLayer → final RMSNorm → lm_head
```

---

## 2. Source map

| File | Responsibility |
|---|---|
| `src/main.cpp` | CLI parsing, chat-template wrapping, prefill + decode loop, timing/output. |
| `src/tensor.hpp` | `Tensor`: shape/strides, owned (`std::vector`) or external-view storage. |
| `src/json_parser.hpp` | Minimal zero-dependency JSON parser (config + tokenizer + safetensors header). |
| `src/safetensors.{hpp,cpp}` | Memory-mapped safetensors reader; bf16→FP32 / F32 tensor views. |
| `src/tokenizer.{hpp,cpp}` | GPT-2 byte-level BPE: byte↔unicode map, merges-rank BPE, encode/decode, special tokens. |
| `src/math_ops.{hpp,cpp}` | Thread pool + all numeric kernels: matmul, RMSNorm, SiLU, softmax, RoPE, causal conv1d, gated-delta update. |
| `src/modules.hpp` | `IModule` interface, `RMSNorm`, `SwiGLU_MLP`, `DecoderLayer` (residual wiring). |
| `src/attention.{hpp,cpp}` | `Qwen3_5Attention` (full GQA + gate) and `Qwen3_5GatedDeltaNet` (linear). |
| `src/model.{hpp,cpp}` | `ModelConfig`, `ModelWeights` (multi-file lookup), `QwenModel` (load/forward/sample). |
| `Makefile` | NMake build (MSVC). Produces `dist/quicklm.exe`. |
| `CMakeLists.txt` | Alternative CMake build. |

---

## 3. Component reference

### 3.1 Tensor (`tensor.hpp`)
A lightweight struct holding `shape`, row-major `strides`, a raw `float*`, and an
optional `shared_ptr<vector<float>>` owner. Two constructors: **owning** (allocates
and zero-fills) and **viewing** (wraps external memory, e.g. an mmap'd weight or a
slice of another tensor). This view capability is used heavily — e.g. treating one
attention head as a `[1, head_dim]` tensor without copying. (DEVLOG_2 adds an
optional bf16 backing store for weights.)

### 3.2 Safetensors loader (`safetensors.{hpp,cpp}`)
`MappedFile` wraps Win32 file mapping (or POSIX `mmap`). `SafetensorsLoader::open`
parses the JSON header (tensor name → dtype, shape, byte offsets) and records the
binary start offset. `get_tensor`:
- **F32** → returns a zero-copy `Tensor` view directly into the mapping.
- **BF16** → allocates an owned FP32 buffer and upcasts each `uint16` by
  `f32_bits = u16 << 16` (bf16 is the high 16 bits of FP32, so this is exact).
Unsupported dtypes throw.

### 3.3 Tokenizer (`tokenizer.{hpp,cpp}`)
GPT-2 / Qwen **byte-level BPE**:
- `build_byte_unicode_maps()` reconstructs GPT-2's `bytes_to_unicode` (256 reversible
  byte↔printable-codepoint mappings, each stored as a UTF-8 string).
- Loads `vocab` and `merges` from `tokenizer.json` (falls back to `merges.txt`),
  plus `added_tokens` as special tokens.
- **Encode:** `pre_tokenize` splits text into GPT-2-style pieces (contractions
  approximated; words / numbers / symbol-runs / whitespace; non-ASCII bytes treated
  as letters so multibyte UTF-8 stays grouped), byte-encodes each piece, then
  `bpe_encode` merges by lowest merge-rank.
- **Decode:** concatenates the byte-level token strings and maps each unicode
  "char" back to its original byte → raw UTF-8 out. Special tokens pass through
  verbatim.
- Special IDs: `<|endoftext|>`=248044, `<|im_start|>`=248045, `<|im_end|>`=248046.
  EOS defaults to `<|im_end|>` (chat turn end), falling back to `<|endoftext|>`.

### 3.4 Math kernels (`math_ops.{hpp,cpp}`)
- **ThreadPool:** a minimal `enqueue`→`future` pool; one global instance sized by
  `--threads`.
- **`matmul(A, B, C, transpose_B)`:** the workhorse. For decode, `A` is `[1,K]`
  (one token) and `B` is a `[N,K]` weight, so this is a GEMV. Parallelized by
  splitting the `N` output rows across worker tasks. (See DEVLOG_2 for the AVX2 /
  bf16 kernels.)
- **`rms_norm`:** `x / sqrt(mean(x²)+eps) * (1 + weight)` — **zero-centered** gain.
- **`silu`, `elementwise_mul/add`, `scale`, `softmax`.**
- **`apply_rope` / `apply_rope_neox`:** interleaved (legacy) and GPT-NeoX partial
  RoPE. The model uses `apply_rope_neox` (split-in-half pairing, first `rotary_dim`
  dims only).
- **`causal_conv1d_update`:** depthwise causal conv step for DeltaNet; shifts a
  per-channel state of `kernel_size-1` past inputs.
- **`recurrent_gated_delta_rule_update`:** reference helper for the linear-attention
  recurrence (the layer also inlines its own copy).

### 3.5 Modules (`modules.hpp`)
- **`RMSNorm`** wraps `math::rms_norm`.
- **`SwiGLU_MLP`:** `down( silu(gate(x)) * up(x) )`.
- **`DecoderLayer`:** the residual block —
  `h += attn(norm1(h)); h += mlp(norm2(h))` (pre-norm), via an `IAttention` and an
  `IMLP`. Holds no attention-type knowledge; the concrete attention is injected at
  load time.

### 3.6 Attention (`attention.{hpp,cpp}`)
**`Qwen3_5Attention` (full, gated GQA):** 8 query heads, 2 KV heads, head_dim 256.
`q_proj` is double-width `[query | gate]` per head. Pipeline: project Q/K/V →
per-head zero-centered RMSNorm on Q,K → partial NeoX RoPE (first 64 dims) → write
K,V to KV cache → scaled-dot-product GQA over positions `0..pos` → multiply output
by `sigmoid(gate)` → `o_proj`.

**`Qwen3_5GatedDeltaNet` (linear):** 16 heads, head_dim 128. Pipeline: project
`qkv`, `z`, `b`, `a` → causal depthwise conv1d on the concatenated qkv → SiLU on all
of q,k,v → L2-normalize q,k → `beta=sigmoid(b)`, `g=-exp(A_log)·softplus(a+dt_bias)`
→ recurrent gated delta-rule update of a `[head_dim,head_dim]` state (scaled query
read-out by `1/sqrt(head_dim)`) → gated RMSNorm (`norm` then `silu(z)`) → `out_proj`.
State (conv + recurrent) persists across tokens and is cleared by `reset_states`.

### 3.7 Model (`model.{hpp,cpp}`)
- **`ModelConfig`** parsed from `config.json` (handles the nested `text_config` and
  `rope_parameters`, including `partial_rotary_factor` and the 10M `rope_theta`).
- **`ModelWeights`** opens every `.safetensors` in the dir and resolves names,
  transparently rewriting `model.*` → `model.language_model.*` for this checkpoint's
  layout.
- **`QwenModel::load`** builds embeddings, 24 `DecoderLayer`s (each with the right
  attention type from `config.layer_types`), final norm, and lm_head
  (tied to embeddings — `tie_word_embeddings=true`). Effective vocab size is taken
  from the lm_head row count.
- **`forward`** does embedding lookup → layers (ping-pong buffers) → final norm →
  lm_head GEMV → logits. Prints per-layer timings at `pos==0`.
- **`sample`** supports greedy (`temp≤0`), and temperature + top-k categorical
  sampling.

---

## 4. Build & run

**Build (MSVC / NMake):**
```
nmake all            # → dist\quicklm.exe   (needs a VS dev environment: cl + nmake)
```
Flags: `/O2 /Oi /Ot /EHsc /arch:AVX2 /std:c++17 /DNOMINMAX`. A CMake build is also
provided.

**Run:**
```
dist\quicklm.exe --path Qwen3.5-0.8B --prompt "hey" [options]
  --temp <v>        sampling temperature (default 0.7; 0 = greedy)
  --top_k <v>       top-k filtering (default 40; 0 = off)
  --max_tokens <v>  max generated tokens (default 256)
  --threads <v>     worker threads (default = hardware concurrency)
  --raw             disable the chat template (plain completion)
  --show_ids        print generated token IDs to stderr (verification)
```
The model directory must contain `config.json`, `tokenizer.json`, and the
`*.safetensors` shard(s). By default the prompt is wrapped in the Qwen chat template.

---

## 5. Model architecture cheat-sheet

| Property | Value |
|---|---|
| hidden_size | 1024 |
| layers | 24 (3:1 linear:full, full at indices 3,7,11,15,19,23) |
| full attn | 8 Q heads / 2 KV heads, head_dim 256, **output gate**, q/k-norm |
| RoPE | partial (factor 0.25 → 64 dims), NeoX split-in-half, θ=10,000,000 |
| linear attn | Gated DeltaNet, 16 heads, head_dim 128, conv kernel 4 |
| MLP | SwiGLU, intermediate 3584 |
| norm | zero-centered RMSNorm `(1+w)`; gated RMSNorm in linear attn uses `w` |
| vocab | ~248k, tied embeddings, byte-level BPE |
| EOS | `<|im_end|>` (248046) / `<|endoftext|>` (248044) |

---

## 6. Known limitations (correctness-preserving)

- Decode is single-token recurrent; no chunked-parallel prefill like HF.
- The pre-tokenizer approximates the GPT-2 Unicode regex — fine for normal text,
  may mis-split exotic mixed-script input (emoji clusters, rare scripts).
- No vision tower, MoE, or multi-token-prediction (MTP) — dense text path only.

---

## 7. Session log — Make Qwen3.5-0.8B inference produce correct output

**Symptom.** The engine loaded and ran a full forward pass without crashing but
produced garbage — first CJK mojibake, then (after partial fixes) endless
whitespace — and the output was insensitive to the prompt.

**Root causes & fixes** (all cross-checked against HF
`transformers/models/qwen3_5/modeling_qwen3_5.py`):

1. **Zero-centered RMSNorm (decisive).** Standard RMSNorm scales by `(1 + weight)`,
   not `weight`. Using `weight` collapsed the residual stream (~0.02 RMS) and
   flattened the logits → whitespace-only output. Fixed in `math::rms_norm` and the
   full-attention q/k-norm. (The *gated* RMSNorm in linear attention correctly uses
   `weight` with no `+1`.) Found by tracing per-layer hidden RMS.
2. **Full-attention output gate** (`attn_output_gate=true`). `q_proj` is
   double-width `[query|gate]` per head; output must be `× sigmoid(gate)` before
   `o_proj`. Was missing.
3. **q_norm / k_norm.** Per-head RMSNorm over head_dim, before RoPE. Was missing.
4. **RoPE.** Switched to **partial** (factor 0.25 → 64 dims) **NeoX** rotation
   (split-in-half pairing) with θ=10M; the old code rotated all 256 dims with the
   interleaved convention. Text-only mrope reduces to standard RoPE.
5. **Linear attention.** SiLU now applied to all of conv(q,k,v) before q/k
   L2-norm (was v-only); added the missing `1/sqrt(head_dim)` query scale.
6. **Tokenizer rewrite.** Implemented true GPT-2 byte-level BPE (byte↔unicode map +
   merges-rank BPE + reverse decode). The old code did raw-byte BPE and never
   byte-encoded, so space-prefixed tokens (`Ġcapital`) never matched — destroying
   context — and decode emitted literal `Ġ`/`Ċ`.
7. **Chat template + EOS.** `main.cpp` wraps prompts as the Qwen
   `user`/`assistant` template (with an empty `<think>` block); `--raw` disables it.
   Generation stops on `<|im_end|>` or `<|endoftext|>`.

**Verification (greedy, `--temp 0`).**
- `"hey"` → `Hey! How can I help you today? 😊` (stops at EOS).
- `"What is the capital of France?"` → `The capital of France is **Paris**. …`
- `"Write a haiku about winter."` → a clean, correctly-terminated haiku.

**Also added.** `--show_ids` (exact greedy token-ID fingerprint, for verifying that
future optimizations leave responses identical).

**Next.** Lossless performance work — see `DEVLOG_2.md`.
