QI --optimize methods (all lossless, no INT4/8)
=================================================
Status: prefetch is implemented; the rest are placeholders (parsed/validated
by the CLI but have no effect on inference yet).


speculative     [needs --draft-model]
  A small, fast "draft" model proposes several tokens ahead. The full
  FP32 model then verifies them in a SINGLE forward pass. Accepted tokens
  are kept; the first rejected one is corrected. Output is bit-for-bit
  identical to normal decoding — you just read the big weights once but
  commit multiple tokens per pass. This is how you beat the per-token
  bandwidth wall. Win scales with draft acceptance rate.

kv-reuse
  The attention Key/Value cache for a given prefix is reused instead of
  recomputed. If a prompt shares a prefix (system prompt, doc, chat
  history) with a previous run, that portion is fetched, not recomputed.
  Same cached numbers → fully lossless. Big win only when context is shared.

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

fusion
  Merges several small ops (e.g. add + norm + activation) into one kernel.
  Fewer kernel launches and fewer round-trips of intermediate tensors to
  global memory. Same results, less overhead.

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
  on vs. off). All other methods below remain placeholders.
=================================================