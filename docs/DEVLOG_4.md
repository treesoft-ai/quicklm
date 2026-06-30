# DEVLOG_4 — Generic architecture runtime & chat template engine (2026-06-28)

**Goal.** Decouple model-specific logic from the core decoder runtime to support running multiple model architectures via a pluggable descriptor registry, and render chat prompts dynamically using a built-in Jinja-subset interpreter.

Earlier sessions: correctness work in `DEVLOG_1.md`, lossless performance work in `DEVLOG_2.md`, and directory organization in `DEVLOG_3.md`.

## Motivation

Until now, the inference engine was hardcoded specifically for the **Qwen3.5-0.8B** hybrid network layout. The model loop in `QwenModel` directly assumed the Qwen3.5 3:1 layer pattern (18 linear attention layers interleaved with 6 full attention layers) and carried hardcoded tensor names (e.g. `model.layers.i.self_attn...`). In addition, the chat prompt wrapper was hardcoded in `src/main.cpp`.

To make QuickLM a reusable engine that can run other transformer models (e.g. LLaMA, standard Qwen models, etc.) without duplicating the runtime loop, we needed to:
1. Extract model-specific configuration and layer assembly into pluggable **architecture descriptors**.
2. Decouple weight loading and configure name-remapping hooks so that checkpoints with different naming conventions (e.g. multimodal `model.language_model.*` vs flat `model.*`) map to the same internal modules.
3. Parse and interpret the model's own `chat_template.jinja` or `tokenizer_config.json` rather than using a hardcoded CLI string.

## Architecture

```
                                  +-------------------+
                                  |   DecoderModel    | (Generic runtime loop)
                                  +---------+---------+
                                            |
                         +------------------+------------------+
                         | delegates model-specific tasks to   |
                         v                                     v
             +-----------------------+              +----------------------+
             |     IArchitecture     |              |     ModelWeights     |
             +-----------+-----------+              +----------+-----------+
                         |                                     |
                         | matches & constructs                | translates names
                         v                                     v
             +-----------+-----------+              +----------+-----------+
             |  Qwen3_5Architecture  |              |    Name Remap Fn     |
             +-----------------------+              +----------------------+
```

### 1. Pluggable Architecture Descriptors (`src/arch/`)

We introduced the `IArchitecture` interface (in `src/arch/architecture.hpp`). Adding a new architecture family requires implementing this interface and registering it with a single line in `src/arch/registry.cpp`. The core decoder runtime never has to change.

- **`matches(model_type)`**: Checks if the descriptor supports the given model type.
- **`parse_config(config_node, cfg)`**: Populates the generic `ModelConfig` from the model's specific `config.json`.
- **`build_layer(i, cfg, weights, max_seq_len)`**: Assembles and initializes decoder layer `i` (norms, attention type, and MLP) using the loaded weights.
- **`embed_name()`, `final_norm_name()`, `lm_head_name()`, `tied_lm_head_name()`**: Provide model-specific tensor naming.
- **`remap_tensor_name(name)`**: Installs a translation function to handle alternate weight keys (e.g., mapping canonical names to `model.language_model.*` for multimodal model backbones).
- **`default_chat_template()`**: Provides a fallback built-in template for this architecture.

Currently, `Qwen3_5Architecture` (in `src/arch/qwen3_5.hpp` and `src/arch/qwen3_5.cpp`) implements this interface, encapsulating the hybrid full-attention/Gated-DeltaNet layers, the 3:1 pattern, the specific config parameters, and the tensor name remapping.

### 2. Weight and Config Extraction (`src/model/`)

To support clean architecture boundaries:
- `ModelConfig` was extracted to a dedicated header `src/model/config.hpp`.
- `ModelWeights` was extracted to `src/model/weights.hpp` and `src/model/weights.cpp`. It now accepts a name remapping callback function (`ModelWeights::set_remap`) which it queries when standard lookups miss.

### 3. Minimal Jinja Template Interpreter (`src/io/`)

Rather than maintaining a hardcoded prompt template, we built a zero-dependency Jinja-subset interpreter in `src/io/chat_template.hpp` and `src/io/chat_template.cpp`.

- **Scope**: Parses and tokenizes the template string, executing loops (`{% for message in messages %}`), conditionals (`{% if/elif/else %}`), printing variables/attributes (`{{ message.role }}`, `{{ message.content }}`), string concatenation (`+` and `~`), and the `|trim` filter.
- **Fallback**: To ensure safety, `ChatTemplate::load` rejects templates using unsupported constructs (e.g. tools, macros, vision) before rendering. In those cases, it falls back to the architecture's `default_chat_template()`, preserving prompt correctness.

## Build-system changes

The includes search paths were updated to resolve bare headers inside `src/arch/` seamlessly.

**`Makefile` (NMake / MSVC):**
- Added `/Isrc\arch` to `INCLUDES`.
- Added object files `chat_template.obj`, `weights.obj`, `registry.obj`, and `qwen3_5.obj` to `OBJECTS`.
- Added compile rules and updated `HEADERS` list.

**`CMakeLists.txt`:**
- Appended the new source and header files to `SOURCES` and `HEADERS`.
- Added `src/arch` to `target_include_directories`.

## Verification

The refactored, generic codebase was compiled and verified using the golden fingerprint recipe on the native **Qwen3.5-0.8B** model checkpoint.

1. **Clean compilation**: Built successfully via MSVC `nmake` after updating the Makefile.
2. **Golden greedy check**:
   ```bash
   dist\quicklm.exe --path models\Qwen3.5-0.8B --prompt "hey" --temp 0 --max_tokens 80 --show_ids
   ```
   Output:
   ```
   Hey! How can I help you today? 😊
   [token_ids] 18103 0 2500 628 353 1438 488 3242 30 25677 232
   ```
   Result: **Byte-for-byte identical output** to the hardcoded baseline, confirming the generic runtime abstraction preserves exact mathematical and token output correctness.

## Files touched

- **Modified**:
  - [CMakeLists.txt](file:///s:/TreeSoft/QuickLM/CMakeLists.txt) (added new files & search path)
  - [Makefile](file:///s:/TreeSoft/QuickLM/Makefile) (added compiler rules & dependency tracking)
  - [src/main.cpp](file:///s:/TreeSoft/QuickLM/src/main.cpp) (updated to instantiate `DecoderModel` and run prompt templates)
  - [src/model/model.hpp](file:///s:/TreeSoft/QuickLM/src/model/model.hpp) (renamed `QwenModel` to `DecoderModel`, removed model-specific logic)
  - [src/model/model.cpp](file:///s:/TreeSoft/QuickLM/src/model/model.cpp) (delegated config loading and layer setup to `IArchitecture`)
- **New**:
  - [src/arch/architecture.hpp](file:///s:/TreeSoft/QuickLM/src/arch/architecture.hpp) (defined generic model architecture interface)
  - [src/arch/registry.hpp](file:///s:/TreeSoft/QuickLM/src/arch/registry.hpp) / [src/arch/registry.cpp](file:///s:/TreeSoft/QuickLM/src/arch/registry.cpp) (compile-time architecture lookup registry)
  - [src/arch/qwen3_5.hpp](file:///s:/TreeSoft/QuickLM/src/arch/qwen3_5.hpp) / [src/arch/qwen3_5.cpp](file:///s:/TreeSoft/QuickLM/src/arch/qwen3_5.cpp) (Qwen3.5 model architecture descriptor)
  - [src/model/config.hpp](file:///s:/TreeSoft/QuickLM/src/model/config.hpp) (isolated model configuration structure)
  - [src/model/weights.hpp](file:///s:/TreeSoft/QuickLM/src/model/weights.hpp) / [src/model/weights.cpp](file:///s:/TreeSoft/QuickLM/src/model/weights.cpp) (isolated weights loader with name remapping support)
  - [src/io/chat_template.hpp](file:///s:/TreeSoft/QuickLM/src/io/chat_template.hpp) / [src/io/chat_template.cpp](file:///s:/TreeSoft/QuickLM/src/io/chat_template.cpp) (lightweight Jinja-subset prompt template engine)
  - [docs/DEVLOG_4.md](file:///s:/TreeSoft/QuickLM/docs/DEVLOG_4.md) (this file)
