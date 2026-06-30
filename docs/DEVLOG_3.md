# DEVLOG_3 — Source tree reorganization (2026-06-28)

**Goal.** Make the codebase easier to navigate and work in by grouping `src/` into
domain subfolders, **without changing a single line of engine logic**. This is a
move + build-config change only; the binary and its output must be byte-for-byte
identical to before.

Earlier sessions: correctness work in `DEVLOG_1.md`, lossless performance work in
`DEVLOG_2.md`.

## Motivation

`src/` had grown to 14 files (8 headers, 6 `.cpp`) in one flat directory — tensor
primitives, math kernels, the hybrid model, and I/O all mixed together. Finding the
right file and seeing the shape of the codebase at a glance had gotten harder.

## New layout

```
src/
  main.cpp                       (CLI driver — stays at root)
  tensor/
    tensor.hpp
  ops/
    math_ops.hpp  math_ops.cpp
  model/
    modules.hpp
    attention.hpp attention.cpp
    model.hpp     model.cpp
  io/
    json_parser.hpp
    safetensors.hpp safetensors.cpp
    tokenizer.hpp   tokenizer.cpp
```

- **`tensor/`** — the foundational data type (`Tensor`).
- **`ops/`** — the numeric kernels + thread pool (`math::matmul`, RMSNorm, RoPE,
  conv1d, gated-delta recurrence).
- **`model/`** — the network: module interfaces, both attention types, model assembly.
- **`io/`** — everything that reads bytes from disk (safetensors reader, byte-level BPE
  tokenizer, and the shared JSON parser they both use).

Files were moved with `git mv` so per-file history is preserved (git records them as
renames, not delete+add).

## Includes: unchanged, resolved via include path

Every `#include` in the project is a **bare filename** (`#include "tensor.hpp"`,
`#include "modules.hpp"`, …). Rather than rewrite them to relative paths
(`#include "../tensor/tensor.hpp"`) — which would couple each file to the folder
layout and add noise — the four subfolders are added to the compiler's **include
search path**. The preprocessor then finds each header regardless of which folder the
including file lives in, so **not a single `#include` line changed**. The
header-to-header graph (e.g. `model.hpp` → `tensor.hpp` / `modules.hpp` /
`attention.hpp` / `safetensors.hpp`) resolves exactly as before.

## Build-system changes

**`Makefile` (NMake / MSVC):**
- Added `INCLUDES = /Isrc\tensor /Isrc\ops /Isrc\model /Isrc\io` and appended it to
  `CXXFLAGS`.
- Updated each compile rule's **source path** and prerequisite
  (`src\math_ops.cpp` → `src\ops\math_ops.cpp`, `src\attention.cpp` →
  `src\model\attention.cpp`, etc.). `main.cpp` stayed at `src\main.cpp`.
- Updated the `HEADERS` list to the new paths, preserving the "any header changed →
  full recompile" safety net (the stale-object ABI-mismatch trap from DEVLOG_2 still
  guarded).
- **Object outputs stay flat** (`dist\cache\*.obj`), so the link rule, the `$(OBJECTS)`
  list, and `.gitignore` needed no change.

**`CMakeLists.txt`:**
- Prefixed each `SOURCES` / `HEADERS` entry with its new subfolder.
- Added `target_include_directories(quicklm PRIVATE src/tensor src/ops src/model src/io)`.

## Verification

The whole point is an identical binary, so the change was verified by a clean rebuild
plus the golden greedy fingerprint from DEVLOG_1/_2.

1. **Clean rebuild (MSVC / NMake)** under a VS dev environment — all six translation
   units compiled from their new subfolder paths (bare includes resolved via the new
   `/I` dirs) and linked `dist\quicklm.exe`. Only the pre-existing `getenv` C4996
   warnings appeared; no "cannot open include file" errors.
2. **Golden greedy check** (must match the known-good output):
   ```
   quicklm.exe --path models\Qwen3.5-0.8B --prompt "hey" --temp 0 --max_tokens 80 --show_ids
   ```
   → `Hey! How can I help you today? 😊`, EOS-terminated.
   token_ids fingerprint: `18103 0 2500 628 353 1438 488 3242 30 25677 232`.

Result: **byte-for-byte identical output** — the reorg is lossless, as intended.

## Side change — model checkpoint moved out of `dist/`

During verification, `nmake clean` (which runs `rmdir /s /q dist`) deleted the model
checkpoint, because the weights lived **inside** the gitignored `dist/Qwen3.5-0.8B/`.
`clean` wipes the entire `dist/` tree, not just build artifacts, and the checkpoint is
git-LFS-sourced (not in version control), so it was only recoverable by re-cloning.

**Fix.** The model now lives at top-level **`models/Qwen3.5-0.8B`** (added to
`.gitignore`), outside `dist/`, so `make clean` can never delete it again. Re-cloned
from `https://huggingface.co/Qwen/Qwen3.5-0.8B` (git-LFS). Run the engine with
`--path models/Qwen3.5-0.8B` going forward.

## Files touched

- **Moved** (via `git mv`): all of `src/*.hpp` and `src/*.cpp` except `main.cpp`.
- **Edited**: `Makefile`, `CMakeLists.txt`, `.gitignore`.
- **Not touched**: any `#include` line, `main.cpp`, `dist/` build outputs, engine logic.
