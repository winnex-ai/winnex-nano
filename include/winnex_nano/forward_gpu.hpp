// forward_gpu.hpp — OpenCL dense projections for the native forward pass.
//
// Moves the DENSE MATRIX WORK of the transformer layers to the GPU while the
// control flow (RMSNorm, RoPE, causal mask, softmax, residuals, KV cache)
// stays on the CPU. The projections are ~95% of the FLOPs and are
// embarrassingly parallel; the attention path is the part whose numerical
// correctness was verified most carefully and is deliberately left untouched
// so that perplexity parity survives the move.
//
// Runtime-loaded OpenCL (dlopen / LoadLibrary) — no CUDA, no link-time
// dependency, portable across Linux/macOS/Windows. Same pattern as
// winnex-xfactor-c's attention_forward.cpp.
//
// BSL 1.1 | pay@winnex.ai
#ifndef WINNEX_NANO_FORWARD_GPU_HPP
#define WINNEX_NANO_FORWARD_GPU_HPP

#include <string>

namespace winnex_nano {

/**
 * ForwardGpu — the four attention projections of one layer, on the device.
 *
 *   Q = hn · Wqᵀ + bq        [seq, heads*head_dim]
 *   K = hn · Wkᵀ + bk        [seq, kv_heads*head_dim]
 *   V = hn · Wvᵀ + bv        [seq, kv_heads*head_dim]
 *   O = attn · Woᵀ           [seq, hidden]
 *
 * Each call uploads its weight matrix. Weights are constant per layer, so a
 * caller that runs the same layer repeatedly can keep them resident — see
 * `upload_weights` / `matmul_cached`. The simple `matmul` path re-uploads,
 * which is correct but pays the transfer per call.
 */
class ForwardGpu {
public:
    explicit ForwardGpu(const std::string& opencl_lib = "");
    ~ForwardGpu();
    ForwardGpu(const ForwardGpu&) = delete;
    ForwardGpu& operator=(const ForwardGpu&) = delete;

    /** True when a device was found and the kernel compiled. */
    bool ready() const;
    /** Device name reported by the driver. */
    std::string device_name() const;
    /** Human-readable reason for ready() == false (loader tried, build log). */
    static std::string reason();

    /**
     * C[seq, out_dim] = A[seq, in_dim] · W[out_dim, in_dim]ᵀ + bias[out_dim]
     * W is row-major [out, in] — the safetensors layout. `bias` may be null.
     */
    void matmul(const float* A, int seq,
                const float* W, int out_dim, int in_dim,
                const float* bias, float* C);

private:
    struct Impl;
    Impl* impl_;
};

} // namespace winnex_nano

#endif // WINNEX_NANO_FORWARD_GPU_HPP
