#pragma once

#include <string>

// Offline scout-table builder (Sieve Phase 0, others/sieve_implementation_plan.md).
// Reads a model's ORIGINAL bf16 safetensors weights directly (never a quantized
// copy — see others/sieve_design.md §4.1) and writes one .qlsc scout file per
// eligible weight matrix next to a manifest, under
// `<model_dir>/sieve_scouts/`.
//
// `tensor_filter`, if non-empty, restricts conversion to tensor names
// containing this substring (e.g. "lm_head" for a V-sieve-only build in
// Phase 1). Empty converts every 2D weight matrix in the checkpoint.
//
// Returns true on success (at least one scout written), false on any hard
// failure (bad model_dir, no safetensors found).
bool run_sieve_convert(const std::string& model_dir, uint32_t rank, uint32_t sketch_bits,
                        const std::string& tensor_filter);
