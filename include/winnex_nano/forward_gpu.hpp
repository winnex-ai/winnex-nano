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
     *
     * `tag` names the weight for the device cache. Weights are immutable, so a
     * stable name (e.g. "model.layers.3.mlp.gate_proj.weight") lets the device
     * keep them across forwards even when the host copy is reloaded per layer
     * and its address is reused — which is exactly what streaming does.
     */
    void matmul(const float* A, int seq,
                const float* W, int out_dim, int in_dim,
                const float* bias, float* C,
                const char* tag = nullptr);

    /**
     * Device-only projection. `tag` must already be resident (uploaded by a
     * previous tagged matmul); the host pointer is never read. This is the
     * path used once ForwardEngine has moved every weight to the device — the
     * host no longer holds a copy of any projection matrix.
     */
    void matmul_device(const char* tag, const float* A, int seq,
                       int out_dim, int in_dim, const float* bias, float* C);

    /**
     * Tell the device how many bytes of f32 weights the engine intends to
     * upload IN TOTAL. The bf16-vs-f32 decision is made once from this number;
     * deciding per matrix mixes formats on the device and corrupts every
     * launch after the first switch.
     */
    void declare_weight_bytes(size_t total_f32_bytes);

    /** True when `tag` is resident on the device. */
    bool has_weight(const char* tag) const;

    /**
     * Self-attention causal completa no device.
     *
     *   O[t,h,:] = Σ_{u≤cache_pos+t} softmax(u; Q[t,h]·K[u,kvh]) · V[u,kvh,:]
     *
     * Q [seq, nh, hd], K/V [ctx, nkv, hd], O [seq, nh, hd]. Faz o softmax
     * exato — não top-k — porque é a função com que os pesos foram treinados.
     */
    void attention(const float* Q, const float* K, const float* V, float* O,
                   int seq, int ctx, int nh, int nkv, int hd, int cache_pos);

    /** True quando o kernel de atenção compilou. */
    bool has_attention() const;

    /** Device memory total and free, in bytes (0 when the driver will not say). */
    size_t device_memory_total() const;
    size_t device_memory_free() const;

    /**
     * Decide, before a single byte moves, whether the whole model fits.
     * Any byte the device will hand out (95% of what it reports free) is
     * available; a model that does not fit must NOT be uploaded, because a
     * half-uploaded model cannot be run and the driver may not fail cleanly.
     */
    bool weights_fit(size_t total_f32_bytes) const;

    /**
     * True when the weights MUST be uploaded as bf16: the f32 form exceeds the
     * 85% fit budget but the bf16 form fits inside it.
     *
     * This exists so weights_fit() and should_use_bf16() cannot disagree. They
     * use different budgets (85% and 90%), and with a model whose f32 form sits
     * between them the pair would answer "only bf16 fits" and "use f32" at the
     * same time — and the upload would fail.
     */
    bool should_force_bf16(size_t total_f32_bytes) const;

    /** Drop every weight from the device and return to the streaming path. */
    void release_weights();

    /** Byte size of the weights currently resident on the device. */
    size_t device_weight_bytes() const;
    /** True when the device copy is bf16 (weights did not fit f32). */
    bool weights_are_bf16() const;

private:
    struct Impl;
    Impl* impl_;
};

} // namespace winnex_nano

#endif // WINNEX_NANO_FORWARD_GPU_HPP
