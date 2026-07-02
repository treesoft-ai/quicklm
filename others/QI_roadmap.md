QI --optimize methods (all lossless, no INT4/8)
=================================================
Status: prefetch, fusion, and kv-reuse are implemented; the rest are placeholders
(parsed/validated by the CLI but have no effect on inference yet).


speculative     [needs --draft-model]
  A small, fast "draft" model proposes several tokens ahead. The full
  FP32 model then verifies them in a SINGLE forward pass. Accepted tokens
  are kept; the first rejected one is corrected. Output is bit-for-bit
  identical to normal decoding — you just read the big weights once but
  commit multiple tokens per pass. This is how you beat the per-token
  bandwidth wall. Win scales with draft acceptance rate.

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