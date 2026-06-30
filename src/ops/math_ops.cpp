#include "math_ops.hpp"
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <atomic>
#include <immintrin.h>

// --- ThreadPool Implementation ---

ThreadPool::ThreadPool(size_t threads) : stop(false) {
    for (size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this]() {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this]() {
                        return this->stop || !this->tasks.empty();
                    });
                    if (this->stop && this->tasks.empty()) {
                        return;
                    }
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

// --- Global ThreadPool Reference ---

namespace math {

static std::unique_ptr<ThreadPool> global_pool = nullptr;
static int global_pool_threads = 0;  // number of worker threads (0 if single-threaded)
static double g_matmul_ms = 0.0;     // accumulated matmul wall-time (QUICKLM_PROF)

double get_matmul_ms() { return g_matmul_ms; }
void reset_matmul_ms() { g_matmul_ms = 0.0; }

void init_thread_pool(size_t num_threads) {
    if (num_threads > 1) {
        global_pool = std::make_unique<ThreadPool>(num_threads);
        global_pool_threads = static_cast<int>(num_threads);
    } else {
        global_pool = nullptr;
        global_pool_threads = 0;
    }
}

ThreadPool* get_thread_pool() {
    return global_pool.get();
}

// --- AVX2 horizontal dot product ---
// Computes sum_k a[k]*b[k] over a contiguous run of length K using FMA with four
// independent accumulators (to hide FMA latency), with a scalar tail. Lane-wise
// summation reorders the additions vs a naive loop, but introduces no precision
// loss — results are numerically equivalent in FP32.
static inline float dot_avx2(const float* a, const float* b, int K) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    int k = 0;
    int limit = K - (K % 32);
    for (; k < limit; k += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k),      _mm256_loadu_ps(b + k),      acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 8),  _mm256_loadu_ps(b + k + 8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 16), _mm256_loadu_ps(b + k + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 24), _mm256_loadu_ps(b + k + 24), acc3);
    }
    // Remaining whole 8-lane chunks
    for (; k + 8 <= K; k += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc0);
    }

    // Reduce the four accumulators to a single scalar.
    __m256 sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float result = _mm_cvtss_f32(lo);

    // Scalar tail (K not a multiple of 8).
    for (; k < K; ++k) result += a[k] * b[k];
    return result;
}

// Dot product where `b` is stored as bf16 (raw FP32-high-half bits). Each bf16 is
// upcast to FP32 exactly (zero-extend to 32 bits, shift left 16). Same arithmetic
// as the FP32 kernel, just half the bytes streamed for the weight operand.
static inline float dot_avx2_bf16(const float* a, const uint16_t* b, int K) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    int k = 0;
    int limit = K - (K % 16);
    for (; k < limit; k += 16) {
        // Load 16 bf16 values, split into two groups of 8.
        __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + k));
        __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + k + 8));
        // Zero-extend uint16 -> uint32, shift into the high half to form FP32 bits.
        __m256 fb0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b0), 16));
        __m256 fb1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b1), 16));
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k),     fb0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 8), fb1, acc1);
    }
    for (; k + 8 <= K; k += 8) {
        __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + k));
        __m256 fb0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b0), 16));
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), fb0, acc0);
    }

    __m256 sum = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float result = _mm_cvtss_f32(lo);

    for (; k < K; ++k) {
        uint32_t bits = static_cast<uint32_t>(b[k]) << 16;
        float bf;
        std::memcpy(&bf, &bits, sizeof(float));
        result += a[k] * bf;
    }
    return result;
}

// Helper: upcast 8 bf16 (uint16) at ptr to a __m256 of FP32. Exact.
static inline __m256 load_bf16_8(const uint16_t* p) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(v), 16));
}

// Compute 8 bf16 dot products at once (8 contiguous weight rows). More independent
// memory streams than dot4 → better saturates the memory controller on a
// bandwidth-bound GEMV. Same per-row 2-accumulator/16-element structure as
// dot_avx2_bf16, so each row is bit-identical to the single-row kernel.
static inline void dot8_avx2_bf16(const float* a, const uint16_t* B, int row_stride,
                                  int K, float* out) {
    __m256 lo[8], hi[8];
    for (int r = 0; r < 8; ++r) { lo[r] = _mm256_setzero_ps(); hi[r] = _mm256_setzero_ps(); }
    const uint16_t* b[8];
    for (int r = 0; r < 8; ++r) b[r] = B + r * row_stride;

    int k = 0;
    int limit = K - (K % 16);
    for (; k < limit; k += 16) {
        __m256 av_lo = _mm256_loadu_ps(a + k);
        __m256 av_hi = _mm256_loadu_ps(a + k + 8);
        for (int r = 0; r < 8; ++r) {
            lo[r] = _mm256_fmadd_ps(av_lo, load_bf16_8(b[r] + k),     lo[r]);
            hi[r] = _mm256_fmadd_ps(av_hi, load_bf16_8(b[r] + k + 8), hi[r]);
        }
    }
    for (; k + 8 <= K; k += 8) {
        __m256 av = _mm256_loadu_ps(a + k);
        for (int r = 0; r < 8; ++r) lo[r] = _mm256_fmadd_ps(av, load_bf16_8(b[r] + k), lo[r]);
    }

    auto hsum = [](__m256 l, __m256 h) -> float {
        __m256 sum = _mm256_add_ps(l, h);
        __m128 a0 = _mm256_castps256_ps128(sum);
        __m128 a1 = _mm256_extractf128_ps(sum, 1);
        a0 = _mm_add_ps(a0, a1);
        a0 = _mm_hadd_ps(a0, a0);
        a0 = _mm_hadd_ps(a0, a0);
        return _mm_cvtss_f32(a0);
    };
    for (int r = 0; r < 8; ++r) out[r] = hsum(lo[r], hi[r]);

    if (k < K) {
        for (int r = 0; r < 8; ++r) {
            float acc = out[r];
            for (int kk = k; kk < K; ++kk) {
                uint32_t x = static_cast<uint32_t>(b[r][kk]) << 16;
                float bf; std::memcpy(&bf, &x, 4);
                acc += a[kk] * bf;
            }
            out[r] = acc;
        }
    }
}

// Compute 4 bf16 dot products at once (4 contiguous weight rows b0..b3 of length K
// against the shared input a). Register-blocking: the shared a[] vector is loaded
// once per step and reused across all 4 rows, and the 4 rows form 4 independent
// memory streams that the hardware prefetcher can run ahead on — this is what
// pushes a bandwidth-bound GEMV closer to peak. Uses the SAME two-accumulator,
// 16-element-per-iteration structure as dot_avx2_bf16, so each row's result is
// bit-identical to calling that kernel — blocking changes only memory scheduling.
static inline void dot4_avx2_bf16(const float* a,
                                  const uint16_t* b0, const uint16_t* b1,
                                  const uint16_t* b2, const uint16_t* b3,
                                  int K, float* out0, float* out1,
                                  float* out2, float* out3) {
    __m256 a0lo = _mm256_setzero_ps(), a0hi = _mm256_setzero_ps();
    __m256 a1lo = _mm256_setzero_ps(), a1hi = _mm256_setzero_ps();
    __m256 a2lo = _mm256_setzero_ps(), a2hi = _mm256_setzero_ps();
    __m256 a3lo = _mm256_setzero_ps(), a3hi = _mm256_setzero_ps();

    int k = 0;
    int limit = K - (K % 16);
    for (; k < limit; k += 16) {
        __m256 av_lo = _mm256_loadu_ps(a + k);
        __m256 av_hi = _mm256_loadu_ps(a + k + 8);
        a0lo = _mm256_fmadd_ps(av_lo, load_bf16_8(b0 + k),     a0lo);
        a0hi = _mm256_fmadd_ps(av_hi, load_bf16_8(b0 + k + 8), a0hi);
        a1lo = _mm256_fmadd_ps(av_lo, load_bf16_8(b1 + k),     a1lo);
        a1hi = _mm256_fmadd_ps(av_hi, load_bf16_8(b1 + k + 8), a1hi);
        a2lo = _mm256_fmadd_ps(av_lo, load_bf16_8(b2 + k),     a2lo);
        a2hi = _mm256_fmadd_ps(av_hi, load_bf16_8(b2 + k + 8), a2hi);
        a3lo = _mm256_fmadd_ps(av_lo, load_bf16_8(b3 + k),     a3lo);
        a3hi = _mm256_fmadd_ps(av_hi, load_bf16_8(b3 + k + 8), a3hi);
    }
    for (; k + 8 <= K; k += 8) {
        __m256 av = _mm256_loadu_ps(a + k);
        a0lo = _mm256_fmadd_ps(av, load_bf16_8(b0 + k), a0lo);
        a1lo = _mm256_fmadd_ps(av, load_bf16_8(b1 + k), a1lo);
        a2lo = _mm256_fmadd_ps(av, load_bf16_8(b2 + k), a2lo);
        a3lo = _mm256_fmadd_ps(av, load_bf16_8(b3 + k), a3lo);
    }

    auto hsum = [](__m256 lo, __m256 hi) -> float {
        __m256 sum = _mm256_add_ps(lo, hi);
        __m128 l = _mm256_castps256_ps128(sum);
        __m128 h = _mm256_extractf128_ps(sum, 1);
        l = _mm_add_ps(l, h);
        l = _mm_hadd_ps(l, l);
        l = _mm_hadd_ps(l, l);
        return _mm_cvtss_f32(l);
    };
    float r0 = hsum(a0lo, a0hi), r1 = hsum(a1lo, a1hi);
    float r2 = hsum(a2lo, a2hi), r3 = hsum(a3lo, a3hi);

    for (; k < K; ++k) {
        float av = a[k];
        uint32_t x;
        float bf;
        x = static_cast<uint32_t>(b0[k]) << 16; std::memcpy(&bf, &x, 4); r0 += av * bf;
        x = static_cast<uint32_t>(b1[k]) << 16; std::memcpy(&bf, &x, 4); r1 += av * bf;
        x = static_cast<uint32_t>(b2[k]) << 16; std::memcpy(&bf, &x, 4); r2 += av * bf;
        x = static_cast<uint32_t>(b3[k]) << 16; std::memcpy(&bf, &x, 4); r3 += av * bf;
    }
    *out0 = r0; *out1 = r1; *out2 = r2; *out3 = r3;
}

// --- Matrix Multiplication ---

// Compute output columns [start_n, end_n) of a single-row bf16 GEMV
// (c_row[n] = dot(a_row, B_bf16 row n)). Shared by matmul and matmul_batched.
static inline void bf16_gemv_range(const float* a_row, const uint16_t* B_bf16,
                                   float* c_row, int start_n, int end_n, int K) {
    int n = start_n;
    for (; n + 8 <= end_n; n += 8) {
        if (n + 16 <= end_n) {
            for (int pf = 0; pf < 8; pf += 2)
                _mm_prefetch(reinterpret_cast<const char*>(B_bf16 + (n + 8 + pf) * K), _MM_HINT_T0);
        }
        dot8_avx2_bf16(a_row, B_bf16 + n * K, K, K, &c_row[n]);
    }
    for (; n + 4 <= end_n; n += 4) {
        dot4_avx2_bf16(a_row, B_bf16 + (n + 0) * K, B_bf16 + (n + 1) * K,
                       B_bf16 + (n + 2) * K, B_bf16 + (n + 3) * K, K,
                       &c_row[n], &c_row[n + 1], &c_row[n + 2], &c_row[n + 3]);
    }
    for (; n < end_n; ++n) {
        c_row[n] = dot_avx2_bf16(a_row, B_bf16 + n * K, K);
    }
}

void matmul(const Tensor& A, const Tensor& B, Tensor& C, bool transpose_B) {
    static const bool prof = std::getenv("QUICKLM_PROF") != nullptr;
    std::chrono::high_resolution_clock::time_point _t0;
    if (prof) _t0 = std::chrono::high_resolution_clock::now();
    // A shape: [M, K]
    // B shape: transpose_B ? [N, K] : [K, N]
    // C shape: [M, N]
    int M = (A.shape.size() > 1) ? A.shape[0] : 1;
    int K = A.shape.back();
    int N = transpose_B ? B.shape[0] : B.shape[1];

    const float* A_data = A.data;
    const float* B_data = B.data;
    const uint16_t* B_bf16 = B.bf16_data;
    float* C_data = C.data;
    int B_stride0 = B.strides.empty() ? 0 : B.strides[0];
    int C_stride0 = C.strides.empty() ? N : C.strides[0];

    // Computes output columns [start_n, end_n) for all M rows.
    auto compute_range = [=](int start_n, int end_n) {
        for (int m = 0; m < M; ++m) {
            const float* a_row = A_data + m * K;
            float* c_row = C_data + m * C_stride0;
            if (B_bf16) {
                // bf16 weight operand (transpose_B layout): rows are contiguous.
                bf16_gemv_range(a_row, B_bf16, c_row, start_n, end_n, K);
            } else if (transpose_B) {
                for (int n = start_n; n < end_n; ++n) {
                    c_row[n] = dot_avx2(a_row, B_data + n * K, K);
                }
            } else {
                for (int n = start_n; n < end_n; ++n) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += a_row[k] * B_data[k * B_stride0 + n];
                    }
                    c_row[n] = sum;
                }
            }
        }
    };

    ThreadPool* pool = get_thread_pool();
    // Only parallelize when there is enough work to amortize dispatch overhead.
    long long work = static_cast<long long>(M) * N * K;
    if (pool && global_pool_threads > 1 && work >= (1LL << 16)) {
        int nthreads = global_pool_threads;
        int chunk_size = (N + nthreads - 1) / nthreads;
        std::vector<std::future<void>> futures;
        futures.reserve(nthreads);
        for (int t = 0; t < nthreads; ++t) {
            int start_n = t * chunk_size;
            int end_n = std::min(start_n + chunk_size, N);
            if (start_n >= end_n) break;
            futures.push_back(pool->enqueue([compute_range, start_n, end_n]() {
                compute_range(start_n, end_n);
            }));
        }
        for (auto& f : futures) f.get();
    } else {
        compute_range(0, N);
    }

    if (prof) {
        auto _t1 = std::chrono::high_resolution_clock::now();
        g_matmul_ms += std::chrono::duration<double, std::milli>(_t1 - _t0).count();
    }
}

void matmul_batched(const std::vector<const Tensor*>& As,
                    const std::vector<const Tensor*>& Bs,
                    const std::vector<Tensor*>& Cs) {
    // Runs several independent bf16 GEMVs (each A=[1,K] · B=[N,K]^T → C=[1,N]) under
    // a SINGLE thread-pool barrier, instead of one dispatch+join per matmul. Keeps
    // the memory bus continuously fed across the whole projection group (fewer
    // pipeline drains between matmuls → higher effective bandwidth). Each output row
    // is computed by exactly one thread with the same kernel as matmul(), so results
    // are bit-identical. All B must be bf16 with M==1.
    static const bool prof = std::getenv("QUICKLM_PROF") != nullptr;
    std::chrono::high_resolution_clock::time_point _t0;
    if (prof) _t0 = std::chrono::high_resolution_clock::now();

    int G = (int)Bs.size();
    // Flattened global row index → (which matmul g, local row n).
    long long total_rows = 0;
    std::vector<long long> offset(G + 1, 0);
    for (int g = 0; g < G; ++g) { total_rows += Bs[g]->shape[0]; offset[g + 1] = total_rows; }

    auto run_global = [&](long long gr0, long long gr1) {
        for (int g = 0; g < G; ++g) {
            long long lo = std::max(gr0, offset[g]);
            long long hi = std::min(gr1, offset[g + 1]);
            if (lo >= hi) continue;
            int K = As[g]->shape.back();
            bf16_gemv_range(As[g]->data, Bs[g]->bf16_data, Cs[g]->data,
                            (int)(lo - offset[g]), (int)(hi - offset[g]), K);
        }
    };

    ThreadPool* pool = get_thread_pool();
    if (pool && global_pool_threads > 1 && total_rows >= 64) {
        int nthreads = global_pool_threads;
        long long chunk = (total_rows + nthreads - 1) / nthreads;
        std::vector<std::future<void>> futures;
        futures.reserve(nthreads);
        for (int t = 0; t < nthreads; ++t) {
            long long gr0 = (long long)t * chunk;
            long long gr1 = std::min(gr0 + chunk, total_rows);
            if (gr0 >= gr1) break;
            futures.push_back(pool->enqueue([run_global, gr0, gr1]() { run_global(gr0, gr1); }));
        }
        for (auto& f : futures) f.get();
    } else {
        run_global(0, total_rows);
    }

    if (prof) {
        auto _t1 = std::chrono::high_resolution_clock::now();
        g_matmul_ms += std::chrono::duration<double, std::milli>(_t1 - _t0).count();
    }
}


// --- RMSNorm ---

void rms_norm(const Tensor& input, const Tensor& weight, Tensor& output, float eps) {
    // input shape: [size] or [seq_len, size]
    // weight shape: [size]
    int size = weight.shape[0];
    int num_rows = input.size() / size;

    for (int r = 0; r < num_rows; ++r) {
        const float* in_row = input.data + r * size;
        float* out_row = output.data + r * size;

        // Compute mean of squared values
        float ss = 0.0f;
        for (int i = 0; i < size; ++i) {
            ss += in_row[i] * in_row[i];
        }
        float scale = 1.0f / std::sqrt((ss / size) + eps);

        // Normalize and scale by (1 + weight). Qwen3.5 uses zero-centered RMSNorm:
        // weights are stored centered at 0, so the effective gain is (1 + weight).
        for (int i = 0; i < size; ++i) {
            out_row[i] = in_row[i] * scale * (1.0f + weight.data[i]);
        }
    }
}

// --- SiLU Activation ---

void silu(const Tensor& input, Tensor& output) {
    int len = input.size();
    for (int i = 0; i < len; ++i) {
        float x = input.data[i];
        output.data[i] = x / (1.0f + std::exp(-x));
    }
}

// --- Elementwise Multiplication ---

void elementwise_mul(Tensor& A, const Tensor& B) {
    int len = A.size();
    for (int i = 0; i < len; ++i) {
        A.data[i] *= B.data[i];
    }
}

// --- Elementwise Addition ---

void elementwise_add(Tensor& A, const Tensor& B) {
    int len = A.size();
    for (int i = 0; i < len; ++i) {
        A.data[i] += B.data[i];
    }
}

// --- Scale ---

void scale(Tensor& A, float s) {
    int len = A.size();
    for (int i = 0; i < len; ++i) {
        A.data[i] *= s;
    }
}

// --- Softmax ---

void softmax(Tensor& logits) {
    // logits is typically [num_heads, seq_len] or similar. We softmax the last dimension.
    int size = logits.shape.back();
    int num_rows = logits.size() / size;

    for (int r = 0; r < num_rows; ++r) {
        float* row = logits.data + r * size;

        // Find max element for numerical stability
        float max_val = row[0];
        for (int i = 1; i < size; ++i) {
            if (row[i] > max_val) {
                max_val = row[i];
            }
        }

        // Exponential sum
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            row[i] = std::exp(row[i] - max_val);
            sum += row[i];
        }

        // Divide by sum
        for (int i = 0; i < size; ++i) {
            row[i] /= sum;
        }
    }
}

// --- RoPE (Rotary Position Embeddings) ---

void apply_rope(Tensor& t, int pos, float rope_theta, int head_dim) {
    // t shape: [num_heads, head_dim] or [seq_len, num_heads, head_dim]
    int num_heads = t.shape[t.shape.size() - 2];
    int num_tokens = t.size() / (num_heads * head_dim);

    for (int tok = 0; tok < num_tokens; ++tok) {
        int current_pos = pos + tok;
        float* tok_data = t.data + tok * num_heads * head_dim;

        for (int h = 0; h < num_heads; ++h) {
            float* head_data = tok_data + h * head_dim;
            for (int i = 0; i < head_dim / 2; ++i) {
                float theta_i = 1.0f / std::pow(rope_theta, (2.0f * i) / head_dim);
                float angle = current_pos * theta_i;
                float c = std::cos(angle);
                float s = std::sin(angle);

                float v1 = head_data[2 * i];
                float v2 = head_data[2 * i + 1];

                head_data[2 * i] = v1 * c - v2 * s;
                head_data[2 * i + 1] = v1 * s + v2 * c;
            }
        }
    }
}

// --- Partial NeoX RoPE (HF rotate_half convention) ---

void apply_rope_neox(Tensor& t, int pos, float rope_theta, int head_dim, int rotary_dim) {
    // t shape: [num_heads, head_dim]. Only the first `rotary_dim` dims per head are
    // rotated; dims [rotary_dim, head_dim) pass through unchanged.
    // HF convention: the rotary block is split in half; element i pairs with
    // element i + rotary_dim/2 (NOT adjacent interleaving).
    int num_heads = t.shape[t.shape.size() - 2];
    int half = rotary_dim / 2;

    // Precompute the position-independent inverse frequencies once (cached by
    // theta+rotary_dim). Same value as 1/pow(theta, 2i/rotary_dim) — hoisting the
    // expensive pow() out of the per-token hot path. Bit-identical arithmetic.
    static float cached_theta = 0.0f;
    static int cached_rotary_dim = 0;
    static std::vector<float> inv_freq;
    if (cached_theta != rope_theta || cached_rotary_dim != rotary_dim) {
        inv_freq.resize(half);
        for (int i = 0; i < half; ++i) {
            inv_freq[i] = 1.0f / std::pow(rope_theta, (2.0f * i) / rotary_dim);
        }
        cached_theta = rope_theta;
        cached_rotary_dim = rotary_dim;
    }

    for (int h = 0; h < num_heads; ++h) {
        float* head_data = t.data + h * head_dim;
        for (int i = 0; i < half; ++i) {
            float angle = pos * inv_freq[i];
            float c = std::cos(angle);
            float s = std::sin(angle);

            float x1 = head_data[i];          // first half
            float x2 = head_data[i + half];   // second half
            // q_embed = q*cos + rotate_half(q)*sin, rotate_half = [-x2, x1]
            head_data[i]        = x1 * c - x2 * s;
            head_data[i + half] = x2 * c + x1 * s;
        }
    }
}

// --- Causal Depthwise Conv1D Update ---

void causal_conv1d_update(const Tensor& input, Tensor& conv_state, const Tensor& weight, Tensor& output) {
    // input shape: [conv_dim]
    // conv_state shape: [conv_dim, kernel_size - 1] (e.g. kernel_size = 4, so state has 3 steps)
    // weight shape: [conv_dim, kernel_size] (groups = conv_dim depthwise)
    // output shape: [conv_dim]
    int conv_dim = input.shape[0];
    int kernel_size = weight.shape.back(); // weight shape [conv_dim, 1, kernel_size] - last dim is kernel size

    for (int c = 0; c < conv_dim; ++c) {
        float* state_row = conv_state.data + c * (kernel_size - 1);
        const float* weight_row = weight.data + c * kernel_size;

        // Compute convolution output
        float sum = 0.0f;
        for (int k = 0; k < kernel_size - 1; ++k) {
            sum += state_row[k] * weight_row[k];
        }
        sum += input.data[c] * weight_row[kernel_size - 1];
        output.data[c] = sum;

        // Shift state in-place
        for (int k = 0; k < kernel_size - 2; ++k) {
            state_row[k] = state_row[k + 1];
        }
        state_row[kernel_size - 2] = input.data[c];
    }
}

// --- Gated Delta Rule Recurrent Update ---

void recurrent_gated_delta_rule_update(
    const Tensor& q, const Tensor& k, const Tensor& v, 
    const Tensor& g, const Tensor& beta, Tensor& state, Tensor& output
) {
    // q shape: [num_heads, key_dim]
    // k shape: [num_heads, key_dim]
    // v shape: [num_heads, value_dim]
    // g shape: [num_heads, key_dim] (raw log decay gate)
    // beta shape: [num_heads, key_dim] (write gate)
    // state shape: [num_heads, key_dim, value_dim]
    // output shape: [num_heads, value_dim]

    int num_heads = q.shape[0];
    int key_dim = q.shape[1];
    int value_dim = v.shape[1];

    for (int h = 0; h < num_heads; ++h) {
        const float* q_h = q.data + h * key_dim;
        const float* k_h = k.data + h * key_dim;
        const float* v_h = v.data + h * value_dim;
        const float* g_h = g.data + h * key_dim;
        const float* beta_h = beta.data + h * key_dim;
        float* state_h = state.data + h * key_dim * value_dim;
        float* out_h = output.data + h * value_dim;

        // 1. Decay the old state and compute key-value memory retrieval
        // Decay gate is exp(g_h). 
        // Also retrieve memory: kv_mem = k_h * state_decayed
        std::vector<float> kv_mem(value_dim, 0.0f);
        for (int i = 0; i < key_dim; ++i) {
            float decay = std::exp(g_h[i]);
            float* row = state_h + i * value_dim;

            // In-place decay
            for (int j = 0; j < value_dim; ++j) {
                row[j] *= decay;
                kv_mem[j] += k_h[i] * row[j];
            }
        }

        // 2. Compute error delta: delta_j = (v_j - kv_mem_j)
        // 3. Write update: state_i_j = state_i_j + (beta_i * k_i) * delta_j
        for (int i = 0; i < key_dim; ++i) {
            float scale = beta_h[i] * k_h[i];
            float* row = state_h + i * value_dim;
            for (int j = 0; j < value_dim; ++j) {
                float delta = v_h[j] - kv_mem[j];
                row[j] += scale * delta;
            }
        }

        // 4. Compute output: out = q_h * state_updated
        for (int j = 0; j < value_dim; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < key_dim; ++i) {
                sum += q_h[i] * state_h[i * value_dim + j];
            }
            out_h[j] = sum;
        }
    }
}

} // namespace math
