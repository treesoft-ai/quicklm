#include "math_ops.hpp"
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <atomic>
#include <algorithm>
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

// --- ForkJoinPool: low-latency fork-join for the hot decode path ---
//
// Persistent workers spin on a generation counter between dispatches (parking
// on a condvar after ~100µs idle), so a steady-state dispatch is one atomic
// increment + one uncontended mutex touch instead of a mutex/condvar round
// trip per task. Every worker participates in every dispatch and checks in on
// a done counter; the caller runs part 0 itself and spin-waits for the
// check-ins. Because the caller cannot start dispatch G+1 until all workers
// checked in for G, the job fields are never written while any worker might
// still read them — no per-worker mailboxes needed.
class ForkJoinPool {
public:
    explicit ForkJoinPool(int num_workers) : nworkers(num_workers) {
        workers.reserve(nworkers);
        for (int i = 0; i < nworkers; ++i) {
            workers.emplace_back([this, i]() { worker_loop(i); });
        }
    }

    ~ForkJoinPool() {
        {
            std::lock_guard<std::mutex> lk(park_mutex);
            stop.store(true, std::memory_order_release);
            gen.fetch_add(1, std::memory_order_release);
            park_cv.notify_all();
        }
        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }
    }

    void run(void (*fn)(void*, int, int), void* ctx) {
        job_fn = fn;
        job_ctx = ctx;
        done.store(0, std::memory_order_relaxed);
        gen.fetch_add(1, std::memory_order_release);
        {
            // Always take the (uncontended) mutex: a worker increments `parked`
            // under it and re-checks gen under it before sleeping, so this
            // lock+check can never miss a sleeper.
            std::lock_guard<std::mutex> lk(park_mutex);
            if (parked > 0) park_cv.notify_all();
        }
        fn(ctx, 0, nworkers + 1);
        while (done.load(std::memory_order_acquire) < nworkers) {
            _mm_pause();
        }
    }

    int width() const { return nworkers + 1; }

private:
    void worker_loop(int idx) {
        uint64_t last = 0;  // gen starts at 0; first dispatch is 1
        for (;;) {
            uint64_t g;
            int spins = 0;
            while ((g = gen.load(std::memory_order_acquire)) == last) {
                if (++spins >= SPIN_LIMIT) {
                    std::unique_lock<std::mutex> lk(park_mutex);
                    ++parked;
                    park_cv.wait(lk, [&]() {
                        return gen.load(std::memory_order_acquire) != last;
                    });
                    --parked;
                    g = gen.load(std::memory_order_acquire);
                    break;
                }
                _mm_pause();
            }
            last = g;
            if (stop.load(std::memory_order_acquire)) return;
            job_fn(job_ctx, idx + 1, nworkers + 1);
            done.fetch_add(1, std::memory_order_release);
        }
    }

    // ~100µs of _mm_pause before a worker parks. Decode dispatches arrive every
    // few hundred µs, so workers almost never park mid-generation.
    static constexpr int SPIN_LIMIT = 4096;

    int nworkers;
    std::vector<std::thread> workers;
    std::atomic<uint64_t> gen{0};
    std::atomic<int> done{0};
    std::atomic<bool> stop{false};
    std::mutex park_mutex;
    std::condition_variable park_cv;
    int parked = 0;  // guarded by park_mutex

    // Job descriptor — written by run() strictly before the gen release-store,
    // read by workers strictly after their gen acquire-load.
    void (*job_fn)(void*, int, int) = nullptr;
    void* job_ctx = nullptr;
};

// --- Global ThreadPool Reference ---

namespace math {

static std::unique_ptr<ThreadPool> global_pool = nullptr;
static std::unique_ptr<ForkJoinPool> fj_pool = nullptr;
static int global_pool_threads = 0;  // number of worker threads (0 if single-threaded)
static double g_matmul_ms = 0.0;     // accumulated matmul wall-time (QUICKLM_PROF)

// Re-entrancy guard: parallel_invoke from inside a parallel region would
// deadlock (all workers are busy with the outer job), so nested calls run
// serially instead.
static thread_local bool in_parallel_region = false;

double get_matmul_ms() { return g_matmul_ms; }
void reset_matmul_ms() { g_matmul_ms = 0.0; }

void init_thread_pool(size_t num_threads) {
    if (num_threads > 1) {
        global_pool = std::make_unique<ThreadPool>(num_threads);
        fj_pool = std::make_unique<ForkJoinPool>(static_cast<int>(num_threads) - 1);
        global_pool_threads = static_cast<int>(num_threads);
    } else {
        global_pool = nullptr;
        fj_pool = nullptr;
        global_pool_threads = 0;
    }
}

ThreadPool* get_thread_pool() {
    return global_pool.get();
}

int get_thread_pool_size() {
    return global_pool_threads;
}

int parallel_width() {
    return (fj_pool && !in_parallel_region) ? fj_pool->width() : 1;
}

void parallel_invoke_raw(void (*fn)(void*, int part, int nparts), void* ctx) {
    if (!fj_pool || in_parallel_region) {
        fn(ctx, 0, 1);
        return;
    }
    struct Guarded {
        void (*fn)(void*, int, int);
        void* ctx;
    } g{fn, ctx};
    fj_pool->run(
        [](void* p, int part, int nparts) {
            Guarded* gp = static_cast<Guarded*>(p);
            in_parallel_region = true;
            gp->fn(gp->ctx, part, nparts);
            in_parallel_region = false;
        },
        &g);
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

    // Only parallelize when there is enough work to amortize dispatch overhead.
    long long work = static_cast<long long>(M) * N * K;
    if (parallel_width() > 1 && work >= (1LL << 16)) {
        // Dynamic tiling: participants grab 64-output-row tiles off a shared
        // counter, so a thread that gets descheduled doesn't leave a fixed
        // chunk stranded. Each output element is still computed by exactly one
        // thread with the same kernel — bit-identical to the serial loop.
        constexpr int TILE = 64;
        int ntiles = (N + TILE - 1) / TILE;
        std::atomic<int> next_tile{0};
        parallel_invoke([&](int, int) {
            for (;;) {
                int t = next_tile.fetch_add(1, std::memory_order_relaxed);
                if (t >= ntiles) break;
                int s = t * TILE;
                compute_range(s, std::min(s + TILE, N));
            }
        });
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

    if (parallel_width() > 1 && total_rows >= 64) {
        // Same dynamic 64-row tiling as matmul(), over the flattened global
        // row space spanning all G matmuls.
        constexpr long long TILE = 64;
        long long ntiles = (total_rows + TILE - 1) / TILE;
        std::atomic<long long> next_tile{0};
        parallel_invoke([&](int, int) {
            for (;;) {
                long long t = next_tile.fetch_add(1, std::memory_order_relaxed);
                if (t >= ntiles) break;
                long long gr0 = t * TILE;
                run_global(gr0, std::min(gr0 + TILE, total_rows));
            }
        });
    } else {
        run_global(0, total_rows);
    }

    if (prof) {
        auto _t1 = std::chrono::high_resolution_clock::now();
        g_matmul_ms += std::chrono::duration<double, std::milli>(_t1 - _t0).count();
    }
}

// --- Cross-layer weight prefetch ---

void prefetch_weight_head(const Tensor& t, int num_lines) {
    constexpr int LINE = 64;  // bytes per cache line

    const char* base = nullptr;
    long long byte_len = 0;

    if (t.is_bf16()) {
        base = reinterpret_cast<const char*>(t.bf16_data);
        byte_len = (long long)t.size() * sizeof(uint16_t);
    } else if (t.is_int8()) {
        base = reinterpret_cast<const char*>(t.int8_data);
        byte_len = (long long)t.size();
    } else if (t.is_int4()) {
        base = reinterpret_cast<const char*>(t.int4_data);
        byte_len = ((long long)t.size() + 1) / 2;
    } else if (t.data) {
        base = reinterpret_cast<const char*>(t.data);
        byte_len = (long long)t.size() * sizeof(float);
    }
    if (!base || byte_len <= 0) return;

    long long max_lines = (byte_len + LINE - 1) / LINE;
    int lines = (int)std::min<long long>(num_lines, max_lines);
    for (int i = 0; i < lines; ++i) {
        _mm_prefetch(base + (size_t)i * LINE, _MM_HINT_T1);
    }
}

// --- RMSNorm ---

// AVX2 sum of squares over a contiguous run (4 independent accumulators,
// scalar tail). Reduction order differs from the naive loop but stays FP32.
float sumsq(const float* x, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int i = 0;
    int limit = n - (n % 32);
    for (; i < limit; i += 32) {
        __m256 v0 = _mm256_loadu_ps(x + i);
        __m256 v1 = _mm256_loadu_ps(x + i + 8);
        __m256 v2 = _mm256_loadu_ps(x + i + 16);
        __m256 v3 = _mm256_loadu_ps(x + i + 24);
        a0 = _mm256_fmadd_ps(v0, v0, a0);
        a1 = _mm256_fmadd_ps(v1, v1, a1);
        a2 = _mm256_fmadd_ps(v2, v2, a2);
        a3 = _mm256_fmadd_ps(v3, v3, a3);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        a0 = _mm256_fmadd_ps(v, v, a0);
    }
    __m256 sum = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float r = _mm_cvtss_f32(lo);
    for (; i < n; ++i) r += x[i] * x[i];
    return r;
}

// out[i] = in[i] * s * (1 + w[i]) over a contiguous run (AVX2 + scalar tail).
void norm_apply_zc(const float* in, const float* w, float* out, float s, int n) {
    __m256 vs = _mm256_set1_ps(s);
    __m256 one = _mm256_set1_ps(1.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 g = _mm256_add_ps(one, _mm256_loadu_ps(w + i));
        __m256 v = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(in + i), vs), g);
        _mm256_storeu_ps(out + i, v);
    }
    for (; i < n; ++i) out[i] = in[i] * s * (1.0f + w[i]);
}

// out[i] = in[i] * s * w[i] over a contiguous run (AVX2 + scalar tail).
void norm_apply_gain(const float* in, const float* w, float* out, float s, int n) {
    __m256 vs = _mm256_set1_ps(s);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(in + i), vs),
                                 _mm256_loadu_ps(w + i));
        _mm256_storeu_ps(out + i, v);
    }
    for (; i < n; ++i) out[i] = in[i] * s * w[i];
}

// x[i] /= d in place. Elementwise IEEE division, bit-identical to the scalar loop.
void vec_div_inplace(float* x, float d, int n) {
    __m256 vd = _mm256_set1_ps(d);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(x + i, _mm256_div_ps(_mm256_loadu_ps(x + i), vd));
    }
    for (; i < n; ++i) x[i] /= d;
}

void rms_norm(const Tensor& input, const Tensor& weight, Tensor& output, float eps) {
    // input shape: [size] or [seq_len, size]
    // weight shape: [size]
    int size = weight.shape[0];
    int num_rows = input.size() / size;

    for (int r = 0; r < num_rows; ++r) {
        const float* in_row = input.data + r * size;
        float* out_row = output.data + r * size;

        float ss = sumsq(in_row, size);
        float scale = 1.0f / std::sqrt((ss / size) + eps);

        // Normalize and scale by (1 + weight). Qwen3.5 uses zero-centered RMSNorm:
        // weights are stored centered at 0, so the effective gain is (1 + weight).
        norm_apply_zc(in_row, weight.data, out_row, scale, size);
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
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        _mm256_storeu_ps(A.data + i,
            _mm256_mul_ps(_mm256_loadu_ps(A.data + i), _mm256_loadu_ps(B.data + i)));
    }
    for (; i < len; ++i) A.data[i] *= B.data[i];
}

// --- Elementwise Addition ---

void elementwise_add(Tensor& A, const Tensor& B) {
    int len = A.size();
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        _mm256_storeu_ps(A.data + i,
            _mm256_add_ps(_mm256_loadu_ps(A.data + i), _mm256_loadu_ps(B.data + i)));
    }
    for (; i < len; ++i) A.data[i] += B.data[i];
}

// --- Scale ---

void scale(Tensor& A, float s) {
    int len = A.size();
    __m256 vs = _mm256_set1_ps(s);
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        _mm256_storeu_ps(A.data + i, _mm256_mul_ps(_mm256_loadu_ps(A.data + i), vs));
    }
    for (; i < len; ++i) A.data[i] *= s;
}

// --- Fused SiLU + elementwise multiply ---

void silu_mul(Tensor& A, const Tensor& B) {
    int len = A.size();
    for (int i = 0; i < len; ++i) {
        float x = A.data[i];
        float silu_x = x / (1.0f + std::exp(-x));
        A.data[i] = silu_x * B.data[i];
    }
}

// --- Fused residual add + RMSNorm ---

void add_rms_norm(Tensor& residual, const Tensor& delta, const Tensor& weight,
                   Tensor& output, float eps) {
    // residual/delta/output shape: [size] or [seq_len, size]
    // weight shape: [size]
    int size = weight.shape[0];
    int num_rows = residual.size() / size;

    for (int r = 0; r < num_rows; ++r) {
        float* res_row = residual.data + r * size;
        const float* delta_row = delta.data + r * size;
        float* out_row = output.data + r * size;

        // Fused pass: add the residual and accumulate sum-of-squares in one
        // sweep, instead of writing res_row then re-reading it for rms_norm.
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= size; i += 8) {
            __m256 v = _mm256_add_ps(_mm256_loadu_ps(res_row + i),
                                     _mm256_loadu_ps(delta_row + i));
            _mm256_storeu_ps(res_row + i, v);
            acc = _mm256_fmadd_ps(v, v, acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        float ss = _mm_cvtss_f32(lo);
        for (; i < size; ++i) {
            float v = res_row[i] + delta_row[i];
            res_row[i] = v;
            ss += v * v;
        }
        float s = 1.0f / std::sqrt((ss / size) + eps);

        // Normalize and scale by (1 + weight), same zero-centered convention
        // as rms_norm().
        norm_apply_zc(res_row, weight.data, out_row, s, size);
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
    // Position-keyed cos/sin cache: within one token every layer and every head
    // calls this with the SAME pos, so the trig only needs computing on the
    // first call per position. Values are identical to computing them in place.
    static int cached_pos = -1;
    static std::vector<float> cos_cache, sin_cache;
    if (cached_theta != rope_theta || cached_rotary_dim != rotary_dim) {
        inv_freq.resize(half);
        for (int i = 0; i < half; ++i) {
            inv_freq[i] = 1.0f / std::pow(rope_theta, (2.0f * i) / rotary_dim);
        }
        cached_theta = rope_theta;
        cached_rotary_dim = rotary_dim;
        cached_pos = -1;
    }
    if (cached_pos != pos) {
        cos_cache.resize(half);
        sin_cache.resize(half);
        for (int i = 0; i < half; ++i) {
            float angle = pos * inv_freq[i];
            cos_cache[i] = std::cos(angle);
            sin_cache[i] = std::sin(angle);
        }
        cached_pos = pos;
    }

    for (int h = 0; h < num_heads; ++h) {
        float* head_data = t.data + h * head_dim;
        for (int i = 0; i < half; ++i) {
            float c = cos_cache[i];
            float s = sin_cache[i];

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

void causal_conv1d_update_tapmajor(const float* input, float* state, int oldest_tap,
                                   const float* weight_t, float* output,
                                   int conv_dim, int kernel_size) {
    int ntaps = kernel_size - 1;
    // Chronological tap rows (oldest -> newest) resolved through the ring.
    // kernel_size is tiny (4), so a small fixed cap is plenty.
    const float* tap_rows[8];
    for (int j = 0; j < ntaps; ++j) {
        tap_rows[j] = state + (size_t)((oldest_tap + j) % ntaps) * conv_dim;
    }
    float* oldest_row = state + (size_t)oldest_tap * conv_dim;

    int c = 0;
    for (; c + 8 <= conv_dim; c += 8) {
        // Exact same per-channel arithmetic sequence as the scalar op:
        // acc starts at 0, then += tap_j*w_j in chronological order, then
        // += input*w_last. Separate mul and add (no FMA contraction) so each
        // intermediate is rounded exactly like the scalar loop.
        __m256 acc = _mm256_setzero_ps();
        for (int j = 0; j < ntaps; ++j) {
            acc = _mm256_add_ps(acc,
                _mm256_mul_ps(_mm256_loadu_ps(tap_rows[j] + c),
                              _mm256_loadu_ps(weight_t + (size_t)j * conv_dim + c)));
        }
        __m256 in_v = _mm256_loadu_ps(input + c);
        acc = _mm256_add_ps(acc,
            _mm256_mul_ps(in_v,
                          _mm256_loadu_ps(weight_t + (size_t)ntaps * conv_dim + c)));
        _mm256_storeu_ps(output + c, acc);
        // Ring update: the oldest row is overwritten with the new input. Safe
        // even though oldest_row aliases tap_rows[0] — its values for this
        // channel block were already consumed into acc above.
        _mm256_storeu_ps(oldest_row + c, in_v);
    }
    for (; c < conv_dim; ++c) {
        float sum = 0.0f;
        for (int j = 0; j < ntaps; ++j) {
            sum += tap_rows[j][c] * weight_t[(size_t)j * conv_dim + c];
        }
        sum += input[c] * weight_t[(size_t)ntaps * conv_dim + c];
        output[c] = sum;
        oldest_row[c] = input[c];
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
