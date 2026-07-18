#pragma once

#include "tensor.hpp"
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <type_traits>
#include <cmath>

// A simple, zero-dependency ThreadPool for parallel matrix multiplication.
class ThreadPool {
public:
    explicit ThreadPool(size_t threads);
    ~ThreadPool();

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// Global math operations for QI (Quick Inference)
namespace math {

// Initialize the global thread pool
void init_thread_pool(size_t num_threads);
ThreadPool* get_thread_pool();
// Number of worker threads in the global pool (0 if single-threaded / uninitialized).
int get_thread_pool_size();

// --- Fast fork-join parallelism (hot decode path) ---
//
// The classic ThreadPool above pays a mutex+condvar round trip per dispatch,
// which adds up over the ~hundreds of pool barriers per decoded token. The
// fork-join pool keeps persistent workers spinning briefly between dispatches
// (parking on a condvar only after ~100µs idle), so steady-state dispatch cost
// is a single atomic increment. Work partitioning is up to the caller (static
// ranges or an atomic tile counter), so any output element is still computed
// by exactly one thread with the same kernel — results are bit-identical to
// the serial loop.
//
// parallel_invoke runs f(part_index, num_parts) on num_parts = parallel_width()
// participants (the caller is participant 0). Falls back to f(0, 1) when the
// pool is absent, single-threaded, or when called from inside another
// parallel_invoke region (no nested parallelism).
void parallel_invoke_raw(void (*fn)(void*, int part, int nparts), void* ctx);
int parallel_width();  // total participants per dispatch (1 if serial)

template <class F>
inline void parallel_invoke(F&& f) {
    parallel_invoke_raw(
        [](void* p, int part, int nparts) {
            (*static_cast<typename std::remove_reference<F>::type*>(p))(part, nparts);
        },
        &f);
}

// Profiling: accumulated wall-time spent inside matmul (ms), gated by QUICKLM_PROF.
double get_matmul_ms();
void reset_matmul_ms();

// Perform Matrix Multiplication: C = A * B^T or C = A * B.
// Usually in LLM inference, we do y = x * W^T, where x is (1, K) and W is (N, K).
// For batch size 1 (decoding), this is a matrix-vector product.
void matmul(const Tensor& A, const Tensor& B, Tensor& C, bool transpose_B = true);

// Run several independent bf16 GEMVs (M==1, B in bf16) under one thread-pool
// barrier instead of one per call. Bit-identical to calling matmul() on each;
// reduces per-token dispatch/sync overhead so the memory bus stays saturated.
void matmul_batched(const std::vector<const Tensor*>& As,
                    const std::vector<const Tensor*>& Bs,
                    const std::vector<Tensor*>& Cs);

// RMSNorm operation: rms_norm(x) = x / sqrt(mean(x^2) + eps) * weight
void rms_norm(const Tensor& input, const Tensor& weight, Tensor& output, float eps);

// AVX2 sum of squares over a contiguous run. Lane-parallel reduction reorders
// the additions vs a naive loop (same acceptance class as the matmul kernels).
float sumsq(const float* x, int n);

// out[i] = in[i] * s * (1 + w[i])  (zero-centered RMSNorm gain). in may alias out.
void norm_apply_zc(const float* in, const float* w, float* out, float s, int n);

// out[i] = in[i] * s * w[i]  (plain gain, e.g. DeltaNet head RMSNorm). in may alias out.
void norm_apply_gain(const float* in, const float* w, float* out, float s, int n);

// x[i] /= d in place. Elementwise IEEE division — bit-identical to the scalar loop.
void vec_div_inplace(float* x, float d, int n);

// SiLU activation: silu(x) = x * sigmoid(x)
void silu(const Tensor& input, Tensor& output);

// In-place element-wise product: A = A * B
void elementwise_mul(Tensor& A, const Tensor& B);

// In-place element-wise addition: A = A + B
void elementwise_add(Tensor& A, const Tensor& B);

// In-place element-wise scale: A = A * scale
void scale(Tensor& A, float s);

// Fused: A[i] = silu(A[i]) * B[i]. Same arithmetic as calling silu(A, A)
// followed by elementwise_mul(A, B), but the SiLU result stays in a register
// instead of round-tripping through memory between the two steps.
void silu_mul(Tensor& A, const Tensor& B);

// Fused: residual[i] += delta[i], then RMSNorm(residual) written to output.
// Same arithmetic as elementwise_add(residual, delta) followed by
// rms_norm(residual, weight, output, eps), but residual's post-add
// sum-of-squares is accumulated in the same pass as the add instead of
// re-reading residual from memory afterward.
void add_rms_norm(Tensor& residual, const Tensor& delta, const Tensor& weight,
                   Tensor& output, float eps);

// Softmax operation along the last dimension
void softmax(Tensor& logits);

// Apply RoPE (Rotary Position Embeddings) to query or key tensors.
// Qwen3.5 uses standard RoPE when in text-only mode.
void apply_rope(Tensor& t, int pos, float rope_theta, int head_dim);

// Apply GPT-NeoX-style partial RoPE (HF convention) to a [num_heads, head_dim] tensor.
// Only the first `rotary_dim` dims of each head are rotated; the remainder pass through.
// Pairs are (x[i], x[i + rotary_dim/2]) per HF rotate_half, NOT interleaved.
void apply_rope_neox(Tensor& t, int pos, float rope_theta, int head_dim, int rotary_dim);

// Causal depthwise 1D convolution update step for Gated DeltaNet layers.
// Processes 1 new step for input (shape: [conv_dim]), updates conv_state (shape: [conv_dim, kernel_size - 1]),
// and computes the convolved output (shape: [conv_dim]) using weight (shape: [conv_dim, kernel_size]).
void causal_conv1d_update(const Tensor& input, Tensor& conv_state, const Tensor& weight, Tensor& output);

// Vectorizable variant of the causal conv1d step using a tap-major state layout.
// state is [kernel_size-1, conv_dim] (one row per time tap, contiguous across
// channels) used as a ring buffer: `oldest_tap` is the row index of the OLDEST
// tap; chronological tap j lives in row (oldest_tap + j) % (kernel_size-1).
// weight_t is the transposed kernel [kernel_size, conv_dim] (weight_t[k][c] ==
// weight[c][k]). Computes output[c] = sum_j tap_j[c]*w_t[j][c] + input[c]*
// w_t[K-1][c] with the exact same per-channel add/mul sequence as
// causal_conv1d_update (separate mul then add, accumulating from 0 in
// chronological order) — bit-identical, just 8 channels per instruction — then
// overwrites the oldest row with input. The caller advances its ring index.
void causal_conv1d_update_tapmajor(const float* input, float* state, int oldest_tap,
                                   const float* weight_t, float* output,
                                   int conv_dim, int kernel_size);

// Gated Delta Rule recurrent update step for Gated DeltaNet layers.
// Updates the state S_t based on q_t, k_t, v_t, g_t, beta_t.
// S_t shape: [num_heads, key_dim, value_dim]
// q_t shape: [num_heads, key_dim]
// k_t shape: [num_heads, key_dim]
// v_t shape: [num_heads, value_dim]
// g_t shape: [num_heads, key_dim] (forget gate)
// beta_t shape: [num_heads, key_dim] (write gate)
// output shape: [num_heads, value_dim]
void recurrent_gated_delta_rule_update(
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& g, const Tensor& beta, Tensor& state, Tensor& output
);

// Issue software prefetch hints (L2, non-blocking) for the first `num_lines`
// cache lines of a weight tensor's active backing store (bf16/int8/int4/fp32,
// whichever is populated). Pure latency hiding — does not read or alter any
// value. Intended to warm the next transformer layer's first-touched weight
// matrix while the current layer is still computing.
void prefetch_weight_head(const Tensor& t, int num_lines = 8);

} // namespace math
