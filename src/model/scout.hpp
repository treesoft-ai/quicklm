#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Sieve scout table (Phase 0 of others/sieve_implementation_plan.md): a compact,
// offline-built summary of one weight matrix W [d_out, h] used to predict which
// output rows matter for a given activation, without reading the full matrix.
// See others/sieve_design.md §4.1 for the design rationale this implements.
//
// Two pieces of signal per row, combined at inference (Phase 1+ — this file only
// builds/(de)serializes the table, it does not yet rank anything at decode time):
//
//  1. Spectral estimate: a rank-r factorization W ≈ V_r * U_r^T (U_r [h,r] on the
//     input side, V_r [d_out,r] on the output side, singular values folded into
//     V_r). estimate_i = dot(x . U_r, V_r[i,:]) approximates row i's pre-activation
//     w_i . x in d_out*r multiply-adds instead of d_out*h.
//  2. Residual safety band: a per-row L2 norm of the leftover (W_row - low-rank
//     approx) bounds the spectral estimate's error via Cauchy-Schwarz
//     (|error| <= residual_norm * ||x||), and a per-row sign sketch (SimHash of
//     the residual under a shared random projection, regenerated deterministically
//     from `sketch_seed`) lets that bound be tightened using the current
//     activation's own sketch instead of the loose Cauchy-Schwarz worst case.
//
// On-disk format (little-endian, all offsets from file start):
//   [0:4)    magic "QLSC"
//   [4:8)    uint32 version (=1)
//   [8:12)   uint32 d_out          -- rows of the source weight matrix
//   [12:16)  uint32 h              -- cols of the source weight matrix (input dim)
//   [16:20)  uint32 rank           -- SVD rank r
//   [20:24)  uint32 sketch_bits    -- residual sketch width, multiple of 64
//   [24:32)  uint64 sketch_seed    -- seeds the shared random projection (deterministic
//                                     splitmix64 + Box-Muller; see scout.cpp)
//   [32:40)  uint64 name_hash      -- FNV-1a of the source tensor name
//   [40:44)  uint32 name_len
//   [44:44+name_len)       name bytes (not null-terminated)
//   -- padded to the next 8-byte boundary from file start --
//   U_r          : h * rank floats,      row-major [h, rank]
//   V_r          : d_out * rank floats,  row-major [d_out, rank] (singular values folded in)
//   row_norms    : d_out floats          -- residual L2 norm per row
//   row_sketches : d_out * (sketch_bits/64) uint64 words, sign bits of the
//                  residual's projection onto the shared random hyperplanes
//
// Built once offline from the ORIGINAL bf16/fp32 checkpoint weights (never from a
// quantized copy — quantization noise must not contaminate the ranking signal,
// design doc §4.1) via build_from_matrix() / build_from_bf16(). A future decode-time
// consumer (Phase 1: V-sieve) loads this file at model-load time and consults it
// every token while the full weight matrix is read only for surviving rows.
struct ScoutTable {
    uint32_t d_out = 0;
    uint32_t h = 0;
    uint32_t rank = 0;
    uint32_t sketch_bits = 0;
    uint64_t sketch_seed = 0;
    uint64_t name_hash = 0;
    std::string source_name;

    std::vector<float> U_r;              // [h, rank]
    std::vector<float> V_r;              // [d_out, rank]
    std::vector<float> row_norms;        // [d_out]
    std::vector<uint64_t> row_sketches;  // [d_out, sketch_bits/64]

    // Build a scout table for weight matrix `W` ([d_out, h], row-major, fp32 —
    // caller upcasts from bf16 first; see build_from_bf16 for a convenience
    // wrapper that does the upcast). `rank` is the target SVD rank (r); a small
    // oversampling margin is added internally during the randomized SVD and
    // dropped from the final output. `sketch_bits` must be a multiple of 64.
    // `seed` seeds the shared random projection deterministically — the same
    // seed regenerates the same projection at inference time without storing
    // the (h * sketch_bits)-sized matrix on disk.
    //
    // Memory note: this holds several [d_out, h]-sized fp32 buffers as working
    // memory (the source matrix, its low-rank reconstruction, and the residual);
    // for a vocab-sized matrix (d_out ~ 2.5e5, h ~ 1024) that's a few GB
    // transient — fine for an offline build step on the reference machine
    // (others/sieve_design.md's 32 GB target), not something to run per-token.
    static ScoutTable build_from_matrix(
        const float* W, uint32_t d_out, uint32_t h,
        uint32_t rank, uint32_t sketch_bits, uint64_t seed,
        const std::string& source_name);

    // Convenience: upcast a bf16 [d_out, h] matrix (raw FP32-high-half bits, same
    // convention as Tensor::bf16_data) to fp32 internally, then build.
    static ScoutTable build_from_bf16(
        const uint16_t* W_bf16, uint32_t d_out, uint32_t h,
        uint32_t rank, uint32_t sketch_bits, uint64_t seed,
        const std::string& source_name);

    bool save(const std::string& path) const;
    bool load(const std::string& path);

    // FNV-1a of `name`, matching the hash embedded in name_hash at build time.
    // A load-time caller can compare this against the checkpoint's own tensor
    // name to catch a scout file built for the wrong tensor.
    static uint64_t hash_name(const std::string& name);
};
