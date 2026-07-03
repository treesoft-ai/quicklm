#include "scout.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <thread>
#include <vector>

namespace {

// ---- deterministic RNG (splitmix64 + Box-Muller) -------------------------
// Used both to build the scout offline and to be regenerable, byte-for-byte,
// by a future decode-time consumer from just `sketch_seed` — the random
// projection matrix itself is never stored on disk.
uint64_t splitmix64_next(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

double next_uniform01(uint64_t& state) {
    // 53 bits of mantissa from the top of a 64-bit draw.
    uint64_t r = splitmix64_next(state);
    return double(r >> 11) * (1.0 / 9007199254740992.0);  // 2^-53
}

double next_gaussian(uint64_t& state) {
    double u1 = std::max(next_uniform01(state), 1e-300);
    double u2 = next_uniform01(state);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2);
}

// [rows, cols] row-major standard-normal matrix, deterministic from `seed`.
std::vector<double> gaussian_matrix(uint32_t rows, uint32_t cols, uint64_t seed) {
    std::vector<double> m(size_t(rows) * cols);
    uint64_t state = seed;
    for (size_t i = 0; i < m.size(); i++) m[i] = next_gaussian(state);
    return m;
}

// ---- tiny thread-based parallel-for over [0, n) --------------------------
// Kept self-contained rather than reusing math::ThreadPool: this file runs
// once, offline, per tensor -- not on the hot decode path -- so it doesn't
// need to share the engine's persistent pool.
void parallel_for(uint32_t n, const std::function<void(uint32_t)>& body) {
    if (n == 0) return;
    unsigned hw = std::thread::hardware_concurrency();
    unsigned nthreads = std::min<unsigned>(hw ? hw : 4, n);
    if (nthreads <= 1) {
        for (uint32_t i = 0; i < n; i++) body(i);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(nthreads);
    for (unsigned t = 0; t < nthreads; t++) {
        workers.emplace_back([&, t]() {
            for (uint32_t i = t; i < n; i += nthreads) body(i);
        });
    }
    for (auto& w : workers) w.join();
}

// ---- modified Gram-Schmidt: orthonormalize the columns of Q [rows,cols] --
// in place, row-major with stride `cols`.
void gram_schmidt_columns(std::vector<double>& Q, uint32_t rows, uint32_t cols) {
    for (uint32_t c = 0; c < cols; c++) {
        // subtract projection onto all earlier columns
        for (uint32_t p = 0; p < c; p++) {
            double dot = 0.0;
            for (uint32_t i = 0; i < rows; i++) dot += Q[size_t(i) * cols + c] * Q[size_t(i) * cols + p];
            if (dot != 0.0) {
                for (uint32_t i = 0; i < rows; i++) Q[size_t(i) * cols + c] -= dot * Q[size_t(i) * cols + p];
            }
        }
        double norm = 0.0;
        for (uint32_t i = 0; i < rows; i++) {
            double v = Q[size_t(i) * cols + c];
            norm += v * v;
        }
        norm = std::sqrt(norm);
        if (norm < 1e-12) norm = 1e-12;
        for (uint32_t i = 0; i < rows; i++) Q[size_t(i) * cols + c] /= norm;
    }
}

// ---- classic cyclic Jacobi eigenvalue algorithm for symmetric C [k,k] ----
// Produces eigvecs [k,k] (columns are eigenvectors) and eigvals [k], both
// unsorted. k is small here (rank + oversampling, ~tens), so O(k^3) per
// sweep over a handful of sweeps is negligible next to the O(d_out*h*rank)
// work elsewhere in this file.
void jacobi_eigen(std::vector<double> C, uint32_t k, std::vector<double>& eigvecs, std::vector<double>& eigvals) {
    eigvecs.assign(size_t(k) * k, 0.0);
    for (uint32_t i = 0; i < k; i++) eigvecs[size_t(i) * k + i] = 1.0;

    const int max_sweeps = 100;
    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        double off = 0.0;
        for (uint32_t p = 0; p < k; p++)
            for (uint32_t q = p + 1; q < k; q++) off += C[size_t(p) * k + q] * C[size_t(p) * k + q];
        if (off < 1e-24) break;

        for (uint32_t p = 0; p < k; p++) {
            for (uint32_t q = p + 1; q < k; q++) {
                double apq = C[size_t(p) * k + q];
                if (std::fabs(apq) < 1e-300) continue;
                double app = C[size_t(p) * k + p];
                double aqq = C[size_t(q) * k + q];
                double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
                double c = std::cos(phi), s = std::sin(phi);

                for (uint32_t i = 0; i < k; i++) {
                    double cip = C[size_t(i) * k + p];
                    double ciq = C[size_t(i) * k + q];
                    C[size_t(i) * k + p] = c * cip - s * ciq;
                    C[size_t(i) * k + q] = s * cip + c * ciq;
                }
                for (uint32_t j = 0; j < k; j++) {
                    double cpj = C[size_t(p) * k + j];
                    double cqj = C[size_t(q) * k + j];
                    C[size_t(p) * k + j] = c * cpj - s * cqj;
                    C[size_t(q) * k + j] = s * cpj + c * cqj;
                }
                for (uint32_t i = 0; i < k; i++) {
                    double vip = eigvecs[size_t(i) * k + p];
                    double viq = eigvecs[size_t(i) * k + q];
                    eigvecs[size_t(i) * k + p] = c * vip - s * viq;
                    eigvecs[size_t(i) * k + q] = s * vip + c * viq;
                }
            }
        }
    }
    eigvals.resize(k);
    for (uint32_t i = 0; i < k; i++) eigvals[i] = C[size_t(i) * k + i];
}

void write_u32(std::FILE* f, uint32_t v) { std::fwrite(&v, sizeof(v), 1, f); }
void write_u64(std::FILE* f, uint64_t v) { std::fwrite(&v, sizeof(v), 1, f); }
bool read_u32(std::FILE* f, uint32_t& v) { return std::fread(&v, sizeof(v), 1, f) == 1; }
bool read_u64(std::FILE* f, uint64_t& v) { return std::fread(&v, sizeof(v), 1, f) == 1; }

}  // namespace

uint64_t ScoutTable::hash_name(const std::string& name) {
    // FNV-1a 64-bit.
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : name) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

ScoutTable ScoutTable::build_from_matrix(const float* W, uint32_t d_out, uint32_t h,
                                          uint32_t rank, uint32_t sketch_bits, uint64_t seed,
                                          const std::string& source_name) {
    ScoutTable t;
    t.d_out = d_out;
    t.h = h;
    t.sketch_bits = sketch_bits;
    t.sketch_seed = seed;
    t.source_name = source_name;
    t.name_hash = hash_name(source_name);

    // Randomized SVD (Halko/Martinsson/Tropp), oversampled then truncated to
    // `rank`. See the header comment for the derivation of U_r/V_r from it.
    const uint32_t oversample = 10;
    const uint32_t k = std::min<uint32_t>(rank + oversample, h == 0 ? 1 : h);
    const uint32_t eff_rank = std::min(rank, k);

    // 1) Omega [h, k] gaussian.
    std::vector<double> Omega = gaussian_matrix(h, k, seed ^ 0x9E3779B97F4A7C15ULL);

    // 2) Y = W * Omega : [d_out, k]
    std::vector<double> Y(size_t(d_out) * k, 0.0);
    parallel_for(d_out, [&](uint32_t i) {
        const float* row = W + size_t(i) * h;
        double* yrow = &Y[size_t(i) * k];
        for (uint32_t c = 0; c < k; c++) {
            double acc = 0.0;
            for (uint32_t j = 0; j < h; j++) acc += double(row[j]) * Omega[size_t(j) * k + c];
            yrow[c] = acc;
        }
    });

    // 3) Q = orthonormal basis of Y's column space.
    std::vector<double> Q = Y;
    gram_schmidt_columns(Q, d_out, k);

    // 4) B = Q^T * W : [k, h]
    std::vector<double> B(size_t(k) * h, 0.0);
    parallel_for(k, [&](uint32_t c) {
        double* brow = &B[size_t(c) * h];
        for (uint32_t i = 0; i < d_out; i++) {
            double qc = Q[size_t(i) * k + c];
            if (qc == 0.0) continue;
            const float* row = W + size_t(i) * h;
            for (uint32_t j = 0; j < h; j++) brow[j] += qc * double(row[j]);
        }
    });

    // 5) C = B * B^T : [k, k], symmetric.
    std::vector<double> C(size_t(k) * k, 0.0);
    for (uint32_t a = 0; a < k; a++) {
        const double* ra = &B[size_t(a) * h];
        for (uint32_t b = a; b < k; b++) {
            const double* rb = &B[size_t(b) * h];
            double acc = 0.0;
            for (uint32_t j = 0; j < h; j++) acc += ra[j] * rb[j];
            C[size_t(a) * k + b] = acc;
            C[size_t(b) * k + a] = acc;
        }
    }

    // 6) Eigen-decompose C -> Uhat [k,k], eigvals [k]; sort descending.
    std::vector<double> Uhat, eigvals;
    jacobi_eigen(C, k, Uhat, eigvals);

    std::vector<uint32_t> order(k);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return eigvals[a] > eigvals[b]; });

    // W2[c,j] = S[j] * Uhat[c, order[j]]   -> V_r = Q * W2   [d_out, eff_rank]
    // W3[c,j] = Uhat[c, order[j]] / S[j]   -> U_r = B^T * W3 [h, eff_rank]
    std::vector<double> W2(size_t(k) * eff_rank, 0.0), W3(size_t(k) * eff_rank, 0.0);
    for (uint32_t j = 0; j < eff_rank; j++) {
        double sv = std::sqrt(std::max(eigvals[order[j]], 0.0));
        double inv_sv = sv > 1e-9 ? 1.0 / sv : 0.0;
        for (uint32_t c = 0; c < k; c++) {
            double uc = Uhat[size_t(c) * k + order[j]];
            W2[size_t(c) * eff_rank + j] = sv * uc;
            W3[size_t(c) * eff_rank + j] = inv_sv * uc;
        }
    }

    t.rank = eff_rank;
    t.U_r.assign(size_t(h) * eff_rank, 0.0f);
    t.V_r.assign(size_t(d_out) * eff_rank, 0.0f);

    // U_r = B^T * W3 : [h, eff_rank]
    parallel_for(h, [&](uint32_t row) {
        for (uint32_t j = 0; j < eff_rank; j++) {
            double acc = 0.0;
            for (uint32_t c = 0; c < k; c++) acc += B[size_t(c) * h + row] * W3[size_t(c) * eff_rank + j];
            t.U_r[size_t(row) * eff_rank + j] = float(acc);
        }
    });

    // V_r = Q * W2 : [d_out, eff_rank]
    parallel_for(d_out, [&](uint32_t row) {
        for (uint32_t j = 0; j < eff_rank; j++) {
            double acc = 0.0;
            for (uint32_t c = 0; c < k; c++) acc += Q[size_t(row) * k + c] * W2[size_t(c) * eff_rank + j];
            t.V_r[size_t(row) * eff_rank + j] = float(acc);
        }
    });

    // Residual safety band: per-row leftover after the low-rank approximation.
    // reconstruction[row][col] = sum_j V_r[row,j] * U_r[col,j]
    t.row_norms.assign(d_out, 0.0f);
    uint32_t words_per_row = sketch_bits / 64;
    t.row_sketches.assign(size_t(d_out) * words_per_row, 0ULL);

    std::vector<double> R;  // shared random projection [h, sketch_bits], regenerated deterministically
    if (sketch_bits > 0) R = gaussian_matrix(h, sketch_bits, seed);

    parallel_for(d_out, [&](uint32_t row) {
        std::vector<float> residual(h);
        const float* wrow = W + size_t(row) * h;
        const float* vr = &t.V_r[size_t(row) * eff_rank];
        for (uint32_t col = 0; col < h; col++) {
            double recon = 0.0;
            for (uint32_t j = 0; j < eff_rank; j++) recon += double(vr[j]) * double(t.U_r[size_t(col) * eff_rank + j]);
            residual[col] = float(double(wrow[col]) - recon);
        }
        double norm2 = 0.0;
        for (uint32_t col = 0; col < h; col++) norm2 += double(residual[col]) * double(residual[col]);
        t.row_norms[row] = float(std::sqrt(norm2));

        uint64_t* srow = &t.row_sketches[size_t(row) * words_per_row];
        for (uint32_t b = 0; b < sketch_bits; b++) {
            double proj = 0.0;
            for (uint32_t col = 0; col < h; col++) proj += double(residual[col]) * R[size_t(col) * sketch_bits + b];
            if (proj >= 0.0) srow[b / 64] |= (1ULL << (b % 64));
        }
    });

    return t;
}

ScoutTable ScoutTable::build_from_bf16(const uint16_t* W_bf16, uint32_t d_out, uint32_t h,
                                        uint32_t rank, uint32_t sketch_bits, uint64_t seed,
                                        const std::string& source_name) {
    std::vector<float> W(size_t(d_out) * h);
    parallel_for(d_out, [&](uint32_t row) {
        const uint16_t* src = W_bf16 + size_t(row) * h;
        float* dst = &W[size_t(row) * h];
        for (uint32_t j = 0; j < h; j++) {
            uint32_t bits = uint32_t(src[j]) << 16;  // bf16 -> fp32: high 16 bits
            std::memcpy(&dst[j], &bits, sizeof(float));
        }
    });
    return build_from_matrix(W.data(), d_out, h, rank, sketch_bits, seed, source_name);
}

bool ScoutTable::save(const std::string& path) const {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;

    std::fwrite("QLSC", 1, 4, f);
    write_u32(f, 1);  // version
    write_u32(f, d_out);
    write_u32(f, h);
    write_u32(f, rank);
    write_u32(f, sketch_bits);
    write_u64(f, sketch_seed);
    write_u64(f, name_hash);
    write_u32(f, uint32_t(source_name.size()));
    std::fwrite(source_name.data(), 1, source_name.size(), f);

    // pad to 8-byte boundary from file start
    long pos = std::ftell(f);
    long pad = (8 - (pos % 8)) % 8;
    static const char zeros[8] = {0};
    if (pad > 0) std::fwrite(zeros, 1, size_t(pad), f);

    bool ok = true;
    ok &= std::fwrite(U_r.data(), sizeof(float), U_r.size(), f) == U_r.size();
    ok &= std::fwrite(V_r.data(), sizeof(float), V_r.size(), f) == V_r.size();
    ok &= std::fwrite(row_norms.data(), sizeof(float), row_norms.size(), f) == row_norms.size();
    ok &= std::fwrite(row_sketches.data(), sizeof(uint64_t), row_sketches.size(), f) == row_sketches.size();

    std::fclose(f);
    return ok;
}

bool ScoutTable::load(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;

    char magic[4];
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "QLSC", 4) != 0) {
        std::fclose(f);
        return false;
    }
    uint32_t version = 0;
    if (!read_u32(f, version) || version != 1) {
        std::fclose(f);
        return false;
    }
    if (!read_u32(f, d_out) || !read_u32(f, h) || !read_u32(f, rank) || !read_u32(f, sketch_bits)) {
        std::fclose(f);
        return false;
    }
    if (!read_u64(f, sketch_seed) || !read_u64(f, name_hash)) {
        std::fclose(f);
        return false;
    }
    uint32_t name_len = 0;
    if (!read_u32(f, name_len)) {
        std::fclose(f);
        return false;
    }
    source_name.resize(name_len);
    if (name_len > 0 && std::fread(&source_name[0], 1, name_len, f) != name_len) {
        std::fclose(f);
        return false;
    }

    long pos = std::ftell(f);
    long pad = (8 - (pos % 8)) % 8;
    if (pad > 0) std::fseek(f, pad, SEEK_CUR);

    uint32_t words_per_row = sketch_bits / 64;
    U_r.assign(size_t(h) * rank, 0.0f);
    V_r.assign(size_t(d_out) * rank, 0.0f);
    row_norms.assign(d_out, 0.0f);
    row_sketches.assign(size_t(d_out) * words_per_row, 0ULL);

    bool ok = true;
    ok &= std::fread(U_r.data(), sizeof(float), U_r.size(), f) == U_r.size();
    ok &= std::fread(V_r.data(), sizeof(float), V_r.size(), f) == V_r.size();
    ok &= std::fread(row_norms.data(), sizeof(float), row_norms.size(), f) == row_norms.size();
    ok &= std::fread(row_sketches.data(), sizeof(uint64_t), row_sketches.size(), f) == row_sketches.size();

    std::fclose(f);
    return ok;
}
