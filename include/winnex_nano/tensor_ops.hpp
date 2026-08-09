/**
 * tensor_ops.hpp — native tensor operations for the winnex-nano engine.
 *
 * These ops are model-agnostic (any transformer arch) and reuse the
 * winnex-madhava kernels where they exist (QK^T matmul, top-k, quantization).
 *
 * BSL 1.1 | pay@winnex.ai
 */
#ifndef WINNEX_NANO_TENSOR_OPS_HPP
#define WINNEX_NANO_TENSOR_OPS_HPP

#include <cstdint>
#include <vector>

namespace winnex_nano {

// RMSNorm: y = x / sqrt(mean(x²) + eps) * weight   (per-row, dim d).
void rms_norm(float* y, const float* x, const float* weight, int n, int d,
              float eps);

// RoPE (rotary position embedding), applied in place to a [seq, n_heads, head_dim]
// buffer. theta is the rope_theta (Qwen2: 10k, Qwen3: 1M).
void rope_apply(float* x, int seq_len, int n_heads, int head_dim, float rope_theta);

// SiLU (silu(x) = x·sigmoid(x)), in place.
void silu_inplace(float* x, int n);

// GPTQ int4 dequant + matmul: y[out] = x[in] · W (W int4-packed, group_size).
// qweight shape [in/8, out] int32-packed; qzeros [out/gs, in/8]; scales [out/gs, out].
void gptq_matmul(float* y, const float* x, const std::uint8_t* qweight,
                 const std::uint8_t* qzeros, const float* scales,
                 const std::int32_t* g_idx, int in_dim, int out_dim, int group_size);

} // namespace winnex_nano

#endif // WINNEX_NANO_TENSOR_OPS_HPP
