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

int get_thread_pool_size() {
    return global_pool_threads;
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

// Helper: sign-extend 8 int8 at ptr to a __m256 of FP32 (unscaled — the caller
// applies the row's scale once to the finished dot product, not per element).
static inline __m256 load_int8_8(const int8_t* p) {
    __m128i v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
}

// Dot product where `b` is stored as int8 with one scale per
// Tensor::INT8_BLOCK_SIZE-element block (Q8_0-style). Each block's raw partial
// sum is scaled and folded into a running scaled total immediately, since the
// scale now varies across the row and can't be factored out to the end.
static inline float dot_avx2_int8(const float* a, const int8_t* b, const float* scales, int K) {
    const int BLOCK = Tensor::INT8_BLOCK_SIZE;
    __m256 total = _mm256_setzero_ps();

    int k = 0;
    int blk = 0;
    for (; k + BLOCK <= K; k += BLOCK, ++blk) {
        __m256 acc0 = _mm256_mul_ps(_mm256_loadu_ps(a + k),      load_int8_8(b + k));
        __m256 acc1 = _mm256_mul_ps(_mm256_loadu_ps(a + k + 8),  load_int8_8(b + k + 8));
        __m256 acc2 = _mm256_mul_ps(_mm256_loadu_ps(a + k + 16), load_int8_8(b + k + 16));
        __m256 acc3 = _mm256_mul_ps(_mm256_loadu_ps(a + k + 24), load_int8_8(b + k + 24));
        __m256 block_sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        total = _mm256_fmadd_ps(block_sum, _mm256_set1_ps(scales[blk]), total);
    }

    __m128 lo = _mm256_castps256_ps128(total);
    __m128 hi = _mm256_extractf128_ps(total, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float result = _mm_cvtss_f32(lo);

    if (k < K) {
        // Partial trailing block (K not a multiple of BLOCK): scalar tail,
        // scaled by that block's own scale.
        float tail = 0.0f;
        for (int kk = k; kk < K; ++kk) tail += a[kk] * static_cast<float>(b[kk]);
        result += tail * scales[blk];
    }
    return result;
}

// 8-row int8 dot product (register-blocked, same structure as dot8_avx2_bf16),
// applying each row's own run of block scales.
static inline void dot8_avx2_int8(const float* a, const int8_t* B, int row_stride,
                                  int K, float* out, const float* scales, int num_blocks_per_row) {
    const int BLOCK = Tensor::INT8_BLOCK_SIZE;
    __m256 total[8];
    for (int r = 0; r < 8; ++r) total[r] = _mm256_setzero_ps();
    const int8_t* b[8];
    const float* s[8];
    for (int r = 0; r < 8; ++r) {
        b[r] = B + r * row_stride;
        s[r] = scales + (size_t)r * num_blocks_per_row;
    }

    int k = 0;
    int blk = 0;
    for (; k + BLOCK <= K; k += BLOCK, ++blk) {
        __m256 av0 = _mm256_loadu_ps(a + k);
        __m256 av1 = _mm256_loadu_ps(a + k + 8);
        __m256 av2 = _mm256_loadu_ps(a + k + 16);
        __m256 av3 = _mm256_loadu_ps(a + k + 24);
        for (int r = 0; r < 8; ++r) {
            __m256 acc0 = _mm256_mul_ps(av0, load_int8_8(b[r] + k));
            __m256 acc1 = _mm256_mul_ps(av1, load_int8_8(b[r] + k + 8));
            __m256 acc2 = _mm256_mul_ps(av2, load_int8_8(b[r] + k + 16));
            __m256 acc3 = _mm256_mul_ps(av3, load_int8_8(b[r] + k + 24));
            __m256 block_sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            total[r] = _mm256_fmadd_ps(block_sum, _mm256_set1_ps(s[r][blk]), total[r]);
        }
    }

    auto hsum = [](__m256 v) -> float {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        return _mm_cvtss_f32(lo);
    };
    for (int r = 0; r < 8; ++r) out[r] = hsum(total[r]);

    if (k < K) {
        for (int r = 0; r < 8; ++r) {
            float acc = 0.0f;
            for (int kk = k; kk < K; ++kk) acc += a[kk] * static_cast<float>(b[r][kk]);
            out[r] += acc * s[r][blk];
        }
    }
}

// 4-row int8 dot product (register-blocked, same structure as dot4_avx2_bf16),
// applying each row's own run of block scales.
static inline void dot4_avx2_int8(const float* a,
                                  const int8_t* b0, const int8_t* b1,
                                  const int8_t* b2, const int8_t* b3,
                                  const float* s0, const float* s1,
                                  const float* s2, const float* s3,
                                  int K, float* out0, float* out1,
                                  float* out2, float* out3) {
    const int BLOCK = Tensor::INT8_BLOCK_SIZE;
    __m256 t0 = _mm256_setzero_ps(), t1 = _mm256_setzero_ps();
    __m256 t2 = _mm256_setzero_ps(), t3 = _mm256_setzero_ps();

    int k = 0;
    int blk = 0;
    for (; k + BLOCK <= K; k += BLOCK, ++blk) {
        __m256 av0 = _mm256_loadu_ps(a + k);
        __m256 av1 = _mm256_loadu_ps(a + k + 8);
        __m256 av2 = _mm256_loadu_ps(a + k + 16);
        __m256 av3 = _mm256_loadu_ps(a + k + 24);

        auto block_dot = [&](const int8_t* b) -> __m256 {
            __m256 acc0 = _mm256_mul_ps(av0, load_int8_8(b + k));
            __m256 acc1 = _mm256_mul_ps(av1, load_int8_8(b + k + 8));
            __m256 acc2 = _mm256_mul_ps(av2, load_int8_8(b + k + 16));
            __m256 acc3 = _mm256_mul_ps(av3, load_int8_8(b + k + 24));
            return _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        };

        t0 = _mm256_fmadd_ps(block_dot(b0), _mm256_set1_ps(s0[blk]), t0);
        t1 = _mm256_fmadd_ps(block_dot(b1), _mm256_set1_ps(s1[blk]), t1);
        t2 = _mm256_fmadd_ps(block_dot(b2), _mm256_set1_ps(s2[blk]), t2);
        t3 = _mm256_fmadd_ps(block_dot(b3), _mm256_set1_ps(s3[blk]), t3);
    }

    auto hsum = [](__m256 v) -> float {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        return _mm_cvtss_f32(lo);
    };
    float r0 = hsum(t0), r1 = hsum(t1), r2 = hsum(t2), r3 = hsum(t3);

    if (k < K) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int kk = k; kk < K; ++kk) {
            float av = a[kk];
            a0 += av * static_cast<float>(b0[kk]);
            a1 += av * static_cast<float>(b1[kk]);
            a2 += av * static_cast<float>(b2[kk]);
            a3 += av * static_cast<float>(b3[kk]);
        }
        r0 += a0 * s0[blk]; r1 += a1 * s1[blk]; r2 += a2 * s2[blk]; r3 += a3 * s3[blk];
    }
    *out0 = r0; *out1 = r1; *out2 = r2; *out3 = r3;
}

// Compute output columns [start_n, end_n) of a single-row int8 GEMV
// (c_row[n] = dot(a_row, B_int8 row n) with per-block scales folded in).
// Shared by matmul.
static inline void int8_gemv_range(const float* a_row, const int8_t* B_int8,
                                   const float* scales, float* c_row,
                                   int start_n, int end_n, int K, int num_blocks_per_row) {
    int n = start_n;
    for (; n + 8 <= end_n; n += 8) {
        dot8_avx2_int8(a_row, B_int8 + (size_t)n * K, K, K, &c_row[n],
                       scales + (size_t)n * num_blocks_per_row, num_blocks_per_row);
    }
    for (; n + 4 <= end_n; n += 4) {
        dot4_avx2_int8(a_row, B_int8 + (size_t)(n + 0) * K, B_int8 + (size_t)(n + 1) * K,
                       B_int8 + (size_t)(n + 2) * K, B_int8 + (size_t)(n + 3) * K,
                       scales + (size_t)(n + 0) * num_blocks_per_row,
                       scales + (size_t)(n + 1) * num_blocks_per_row,
                       scales + (size_t)(n + 2) * num_blocks_per_row,
                       scales + (size_t)(n + 3) * num_blocks_per_row,
                       K, &c_row[n], &c_row[n + 1], &c_row[n + 2], &c_row[n + 3]);
    }
    for (; n < end_n; ++n) {
        c_row[n] = dot_avx2_int8(a_row, B_int8 + (size_t)n * K,
                                 scales + (size_t)n * num_blocks_per_row, K);
    }
}

// Unpack a 32-element int4 (Q4_0-style) block: `packed` points to 16 bytes
// (2 values/byte, low nibble = even column, high nibble = odd column). Returns
// the block's 32 signed values (still in [-8,7], unscaled) as two 16-lane int8
// halves in original column order: out_lo = columns 0-15, out_hi = columns 16-31.
static inline void unpack_int4_block(const uint8_t* packed, __m128i& out_lo, __m128i& out_hi) {
    __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed));
    __m128i mask = _mm_set1_epi8(0x0F);
    __m128i lo_nibbles = _mm_and_si128(raw, mask);                    // columns 0,2,4,...,30
    __m128i hi_nibbles = _mm_and_si128(_mm_srli_epi16(raw, 4), mask); // columns 1,3,5,...,31
    __m128i bias = _mm_set1_epi8(8);
    out_lo = _mm_sub_epi8(_mm_unpacklo_epi8(lo_nibbles, hi_nibbles), bias); // columns 0-15
    out_hi = _mm_sub_epi8(_mm_unpackhi_epi8(lo_nibbles, hi_nibbles), bias); // columns 16-31
}

// Sign-extend the low 8 int8 lanes (already in [-8,7]) of `v` to FP32.
static inline __m256 int4_lane_to_ps(__m128i v) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
}

// Dot product where `b` is packed int4 (Q4_0-style) with one scale per
// Tensor::INT4_BLOCK_SIZE-element block. Same per-block-fmadd structure as
// dot_avx2_int8, just sourcing each block's 32 values via unpack_int4_block
// instead of a direct int8 load.
static inline float dot_avx2_int4(const float* a, const uint8_t* b_packed, const float* scales, int K) {
    const int BLOCK = Tensor::INT4_BLOCK_SIZE;
    __m256 total = _mm256_setzero_ps();

    int k = 0;
    int blk = 0;
    for (; k + BLOCK <= K; k += BLOCK, ++blk) {
        __m128i lo16, hi16;
        unpack_int4_block(b_packed + (size_t)blk * (BLOCK / 2), lo16, hi16);

        __m256 acc0 = _mm256_mul_ps(_mm256_loadu_ps(a + k),      int4_lane_to_ps(lo16));
        __m256 acc1 = _mm256_mul_ps(_mm256_loadu_ps(a + k + 8),  int4_lane_to_ps(_mm_srli_si128(lo16, 8)));
        __m256 acc2 = _mm256_mul_ps(_mm256_loadu_ps(a + k + 16), int4_lane_to_ps(hi16));
        __m256 acc3 = _mm256_mul_ps(_mm256_loadu_ps(a + k + 24), int4_lane_to_ps(_mm_srli_si128(hi16, 8)));
        __m256 block_sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        total = _mm256_fmadd_ps(block_sum, _mm256_set1_ps(scales[blk]), total);
    }

    __m128 lo = _mm256_castps256_ps128(total);
    __m128 hi = _mm256_extractf128_ps(total, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float result = _mm_cvtss_f32(lo);

    if (k < K) {
        // Partial trailing block (K not a multiple of BLOCK): scalar tail,
        // scaled by that block's own scale.
        float tail = 0.0f;
        for (int kk = k; kk < K; ++kk) {
            uint8_t byte = b_packed[kk / 2];
            uint8_t nibble = (kk % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
            tail += a[kk] * static_cast<float>(static_cast<int>(nibble) - 8);
        }
        result += tail * scales[blk];
    }
    return result;
}

// 8-row int4 dot product (register-blocked, same structure as dot8_avx2_int8),
// applying each row's own run of block scales. `row_stride_bytes` is the packed
// byte stride between rows (K/2 for a fully-packed row).
static inline void dot8_avx2_int4(const float* a, const uint8_t* B, int row_stride_bytes,
                                  int K, float* out, const float* scales, int num_blocks_per_row) {
    const int BLOCK = Tensor::INT4_BLOCK_SIZE;
    __m256 total[8];
    for (int r = 0; r < 8; ++r) total[r] = _mm256_setzero_ps();
    const uint8_t* b[8];
    const float* s[8];
    for (int r = 0; r < 8; ++r) {
        b[r] = B + (size_t)r * row_stride_bytes;
        s[r] = scales + (size_t)r * num_blocks_per_row;
    }

    int k = 0;
    int blk = 0;
    for (; k + BLOCK <= K; k += BLOCK, ++blk) {
        __m256 av0 = _mm256_loadu_ps(a + k);
        __m256 av1 = _mm256_loadu_ps(a + k + 8);
        __m256 av2 = _mm256_loadu_ps(a + k + 16);
        __m256 av3 = _mm256_loadu_ps(a + k + 24);
        size_t byte_off = (size_t)blk * (BLOCK / 2);
        for (int r = 0; r < 8; ++r) {
            __m128i lo16, hi16;
            unpack_int4_block(b[r] + byte_off, lo16, hi16);
            __m256 acc0 = _mm256_mul_ps(av0, int4_lane_to_ps(lo16));
            __m256 acc1 = _mm256_mul_ps(av1, int4_lane_to_ps(_mm_srli_si128(lo16, 8)));
            __m256 acc2 = _mm256_mul_ps(av2, int4_lane_to_ps(hi16));
            __m256 acc3 = _mm256_mul_ps(av3, int4_lane_to_ps(_mm_srli_si128(hi16, 8)));
            __m256 block_sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            total[r] = _mm256_fmadd_ps(block_sum, _mm256_set1_ps(s[r][blk]), total[r]);
        }
    }

    auto hsum = [](__m256 v) -> float {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        return _mm_cvtss_f32(lo);
    };
    for (int r = 0; r < 8; ++r) out[r] = hsum(total[r]);

    if (k < K) {
        for (int r = 0; r < 8; ++r) {
            float acc = 0.0f;
            for (int kk = k; kk < K; ++kk) {
                uint8_t byte = b[r][kk / 2];
                uint8_t nibble = (kk % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                acc += a[kk] * static_cast<float>(static_cast<int>(nibble) - 8);
            }
            out[r] += acc * s[r][blk];
        }
    }
}

// 4-row int4 dot product (register-blocked, same structure as dot4_avx2_int8),
// applying each row's own run of block scales.
static inline void dot4_avx2_int4(const float* a,
                                  const uint8_t* b0, const uint8_t* b1,
                                  const uint8_t* b2, const uint8_t* b3,
                                  const float* s0, const float* s1,
                                  const float* s2, const float* s3,
                                  int K, float* out0, float* out1,
                                  float* out2, float* out3) {
    const int BLOCK = Tensor::INT4_BLOCK_SIZE;
    __m256 t0 = _mm256_setzero_ps(), t1 = _mm256_setzero_ps();
    __m256 t2 = _mm256_setzero_ps(), t3 = _mm256_setzero_ps();

    int k = 0;
    int blk = 0;
    for (; k + BLOCK <= K; k += BLOCK, ++blk) {
        __m256 av0 = _mm256_loadu_ps(a + k);
        __m256 av1 = _mm256_loadu_ps(a + k + 8);
        __m256 av2 = _mm256_loadu_ps(a + k + 16);
        __m256 av3 = _mm256_loadu_ps(a + k + 24);
        size_t byte_off = (size_t)blk * (BLOCK / 2);

        auto block_dot = [&](const uint8_t* b) -> __m256 {
            __m128i lo16, hi16;
            unpack_int4_block(b + byte_off, lo16, hi16);
            __m256 acc0 = _mm256_mul_ps(av0, int4_lane_to_ps(lo16));
            __m256 acc1 = _mm256_mul_ps(av1, int4_lane_to_ps(_mm_srli_si128(lo16, 8)));
            __m256 acc2 = _mm256_mul_ps(av2, int4_lane_to_ps(hi16));
            __m256 acc3 = _mm256_mul_ps(av3, int4_lane_to_ps(_mm_srli_si128(hi16, 8)));
            return _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        };

        t0 = _mm256_fmadd_ps(block_dot(b0), _mm256_set1_ps(s0[blk]), t0);
        t1 = _mm256_fmadd_ps(block_dot(b1), _mm256_set1_ps(s1[blk]), t1);
        t2 = _mm256_fmadd_ps(block_dot(b2), _mm256_set1_ps(s2[blk]), t2);
        t3 = _mm256_fmadd_ps(block_dot(b3), _mm256_set1_ps(s3[blk]), t3);
    }

    auto hsum = [](__m256 v) -> float {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        return _mm_cvtss_f32(lo);
    };
    float r0 = hsum(t0), r1 = hsum(t1), r2 = hsum(t2), r3 = hsum(t3);

    if (k < K) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int kk = k; kk < K; ++kk) {
            auto nib = [kk](const uint8_t* b) -> int {
                uint8_t byte = b[kk / 2];
                uint8_t nibble = (kk % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                return static_cast<int>(nibble) - 8;
            };
            float av = a[kk];
            a0 += av * static_cast<float>(nib(b0));
            a1 += av * static_cast<float>(nib(b1));
            a2 += av * static_cast<float>(nib(b2));
            a3 += av * static_cast<float>(nib(b3));
        }
        r0 += a0 * s0[blk]; r1 += a1 * s1[blk]; r2 += a2 * s2[blk]; r3 += a3 * s3[blk];
    }
    *out0 = r0; *out1 = r1; *out2 = r2; *out3 = r3;
}

// Compute output columns [start_n, end_n) of a single-row int4 GEMV
// (c_row[n] = dot(a_row, B_int4 row n) with per-block scales folded in).
// Shared by matmul. Row byte stride is ceil(K/2) (2 packed values per byte).
static inline void int4_gemv_range(const float* a_row, const uint8_t* B_int4,
                                   const float* scales, float* c_row,
                                   int start_n, int end_n, int K, int num_blocks_per_row) {
    int row_stride_bytes = (K + 1) / 2;
    int n = start_n;
    for (; n + 8 <= end_n; n += 8) {
        dot8_avx2_int4(a_row, B_int4 + (size_t)n * row_stride_bytes, row_stride_bytes, K, &c_row[n],
                       scales + (size_t)n * num_blocks_per_row, num_blocks_per_row);
    }
    for (; n + 4 <= end_n; n += 4) {
        dot4_avx2_int4(a_row,
                       B_int4 + (size_t)(n + 0) * row_stride_bytes, B_int4 + (size_t)(n + 1) * row_stride_bytes,
                       B_int4 + (size_t)(n + 2) * row_stride_bytes, B_int4 + (size_t)(n + 3) * row_stride_bytes,
                       scales + (size_t)(n + 0) * num_blocks_per_row,
                       scales + (size_t)(n + 1) * num_blocks_per_row,
                       scales + (size_t)(n + 2) * num_blocks_per_row,
                       scales + (size_t)(n + 3) * num_blocks_per_row,
                       K, &c_row[n], &c_row[n + 1], &c_row[n + 2], &c_row[n + 3]);
    }
    for (; n < end_n; ++n) {
        c_row[n] = dot_avx2_int4(a_row, B_int4 + (size_t)n * row_stride_bytes,
                                 scales + (size_t)n * num_blocks_per_row, K);
    }
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
    const int8_t* B_int8 = B.int8_data;
    const float* B_int8_scales = B.int8_scales ? B.int8_scales->data() : nullptr;
    int B_int8_blocks_per_row = (K + Tensor::INT8_BLOCK_SIZE - 1) / Tensor::INT8_BLOCK_SIZE;
    const uint8_t* B_int4 = B.int4_data;
    const float* B_int4_scales = B.int4_scales ? B.int4_scales->data() : nullptr;
    int B_int4_blocks_per_row = (K + Tensor::INT4_BLOCK_SIZE - 1) / Tensor::INT4_BLOCK_SIZE;
    float* C_data = C.data;
    int B_stride0 = B.strides.empty() ? 0 : B.strides[0];
    int C_stride0 = C.strides.empty() ? N : C.strides[0];

    // Computes output columns [start_n, end_n) for all M rows.
    auto compute_range = [=](int start_n, int end_n) {
        for (int m = 0; m < M; ++m) {
            const float* a_row = A_data + m * K;
            float* c_row = C_data + m * C_stride0;
            if (B_int8) {
                // int8 weight operand (transpose_B layout): rows are contiguous.
                int8_gemv_range(a_row, B_int8, B_int8_scales, c_row, start_n, end_n, K, B_int8_blocks_per_row);
            } else if (B_int4) {
                // int4 weight operand (transpose_B layout): rows are contiguous.
                int4_gemv_range(a_row, B_int4, B_int4_scales, c_row, start_n, end_n, K, B_int4_blocks_per_row);
            } else if (B_bf16) {
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
