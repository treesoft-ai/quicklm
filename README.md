# QuickLM

**Zero-dependency C++17 CPU inference engine for hybrid transformer LLMs.**

---

## Overview

QuickLM is a zero-dependency C++17 CPU inference engine designed to execute hybrid transformer models with high throughput. Utilizing AVX2 FMA vectorization and memory-mapped safetensors, it delivers 13+ tokens/sec on consumer CPU hardware.

## Features

- **Zero-Dependency C++17 Runtime** — Written entirely in ISO C++17 with std::thread; requires no PyTorch, BLAS, ONNX, or third-party libraries.
- **AVX2 Vectorized Computation** — Accelerated matmul, RMSNorm, RoPE, and Gated DeltaNet linear attention routines using AVX2 SIMD intrinsics.
- **Byte-Level BPE & Chat Templates** — Complete in-process GPT-2 byte-level BPE tokenizer and Jinja-subset chat template renderer.

## Maintainers

| Name | Role | GitHub |
|---|---|---|
| Alexutzu | Lead Engineer | [@alexutzusoft](https://github.com/alexutzusoft) |

## License

Open Source © TreeSoft [AI]
