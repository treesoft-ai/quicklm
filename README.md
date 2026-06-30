# QuickLM

> A zero-dependency C++17 CPU inference engine for the Qwen 3.5 family, optimized for memory bandwidth with lossless performance techniques and generic architecture support.

---

## Overview

QuickLM is a from-scratch implementation of a complete language model inference stack in plain C++, without external dependencies beyond the standard library. It targets the **Qwen 3.5 family** (hybrid transformer with 18 Gated DeltaNet linear attention layers interleaved with 6 full GQA attention layers across all model sizes) and achieves **~13–13.7 tokens/sec** on consumer DDR4 hardware through memory-bandwidth optimization. The inference engine, **Quick Inference (QI) v1**, implements byte-level BPE tokenization, chat template rendering, and temperature-controlled sampling entirely in-process. It preserves output correctness with bit-for-bit verification against HuggingFace while exposing a generic architecture interface for adding new model families without duplicating the runtime loop.

---

## Features

- **Zero Dependencies**: C++17 standard library only (plus std::thread); no PyTorch, BLAS, ONNX, or external libraries
- **Lossless Performance**: 3.5× speedup over baseline via AVX2 FMA, bf16 weight storage, vectorized recurrence, and memory-mapped I/O — all verified byte-for-byte identical to baseline outputs
- **Complete Inference Stack**: Safetensors loading, GPT-2 byte-level BPE tokenization, full hybrid transformer (linear + full attention), causal masking, temperature/top-k sampling, and Jinja-subset chat template rendering
- **Pluggable Architecture Registry**: Generic `IArchitecture` interface enables support for multiple model families (LLaMA, standard Qwen, etc.) without runtime loop duplication
- **Memory-Optimized**: bf16 weights stay in memory-mapped storage, upcasted in-register per operation; single-pass layer streaming avoids intermediate buffers
- **Correct Hybrid Attention**: Full GQA with output gate, per-head q/k normalization, zero-centered RMSNorm, partial NeoX RoPE; Gated DeltaNet with causal conv1d and recurrent state
- **Interactive CLI**: Chat template support (--raw for plain completion), temperature/top-k control, thread count tuning, token ID verification mode for reproducibility, version reporting

---

## Project Structure

```
QuickLM/
├── src/
│   ├── main.cpp              (CLI driver, inference loop)
│   ├── tensor/
│   │   └── tensor.hpp        (shape/strides/storage primitive)
│   ├── ops/
│   │   ├── math_ops.hpp
│   │   └── math_ops.cpp      (matmul, RMSNorm, RoPE, conv1d, thread pool)
│   ├── model/
│   │   ├── modules.hpp       (module interfaces)
│   │   ├── attention.hpp / attention.cpp  (GQA + GatedDeltaNet)
│   │   ├── model.hpp / model.cpp          (generic DecoderModel)
│   │   ├── config.hpp        (ModelConfig from config.json)
│   │   └── weights.hpp / weights.cpp      (safetensors + name remapping)
│   ├── arch/
│   │   ├── architecture.hpp  (IArchitecture interface)
│   │   ├── qwen3_5.hpp / qwen3_5.cpp      (Qwen3.5 descriptor)
│   │   └── registry.hpp / registry.cpp    (compile-time lookup)
│   └── io/
│       ├── json_parser.hpp   (zero-dep JSON)
│       ├── safetensors.hpp / safetensors.cpp (mmap reader)
│       ├── tokenizer.hpp / tokenizer.cpp     (byte-level BPE)
│       └── chat_template.hpp / chat_template.cpp (Jinja-subset)
├── models/
│   └── Qwen3.5-{size}/       (model checkpoint[s], git-LFS sourced)
├── docs/
│   ├── DEVLOG_1.md           (correctness work)
│   ├── DEVLOG_2.md           (lossless perf optimization)
│   ├── DEVLOG_3.md           (source tree reorg)
│   └── DEVLOG_4.md           (generic architecture + templates)
├── Makefile                  (NMake / MSVC build)
├── CMakeLists.txt            (CMake alternative)
├── .gitignore
└── LICENSE
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Maintainers

| Name        | Role              | GitHub                         |
| ----------- | ----------------- | ------------------------------ |
| Alexutzu    | Lead Engineer     | [@alexutzusoft](https://github.com/alexutzusoft) |

---
