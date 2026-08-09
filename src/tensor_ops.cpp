// tensor_ops.cpp — native tensor operations (model-agnostic).
#include "winnex_nano/tensor_ops.hpp"

#include <cmath>
#include <cstddef>

namespace winnex_nano {

void rms_norm(float* y, const float* x, const float* weight, int n, int d,
              float eps) {
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        const float* row = x + (size_t)i * d;
        float* yrow = y + (size_t)i * d;
        float ss = 0.0f;
        for (int j = 0; j < d; ++j) ss += row[j] * row[j];
        float inv = 1.0f / std::sqrt(ss / d + eps);
        for (int j = 0; j < d; ++j) yrow[j] = row[j] * inv * (weight ? weight[j] : 1.0f);
    }
}

void rope_apply(float* x, int seq_len, int n_heads, int head_dim, float rope_theta) {
    for (int t = 0; t < seq_len; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            float* head = x + ((size_t)t * n_heads + h) * head_dim;
            for (int j = 0; j < head_dim; j += 2) {
                float inv_freq = std::pow(rope_theta, -2.0f * j / head_dim);
                float angle = t * inv_freq;
                float c = std::cos(angle), s = std::sin(angle);
                float a = head[j], b = head[j + 1];
                head[j] = a * c - b * s;
                head[j + 1] = a * s + b * c;
            }
        }
    }
}

void silu_inplace(float* x, int n) {
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        x[i] = v / (1.0f + std::exp(-v));
    }
}

void gptq_matmul(float* y, const float* x, const std::uint8_t* qweight,
                 const std::uint8_t* qzeros, const float* scales,
                 const std::int32_t* g_idx, int in_dim, int out_dim, int group_size) {
    // GPTQ int4 packed: qweight[row][out], each int32 holds 8 int4 values.
    // row = in/8, so qweight shape is [in_dim/8, out_dim].
    const int packed = in_dim / 8;
    const int groups = out_dim / group_size;

    for (int o = 0; o < out_dim; ++o) {
        float acc = 0.0f;
        // Determine the scale group for this output column.
        int grp = (g_idx && g_idx[o] >= 0) ? g_idx[o] : o / group_size;
        float scale = scales[(size_t)grp * out_dim + o];

        for (int r = 0; r < packed; ++r) {
            // qzeros per group: qzeros[grp][packed] int32-packed.
            std::int32_t zraw = reinterpret_cast<const std::int32_t*>(
                qzeros)[(size_t)grp * packed + r];
            std::int32_t wraw = reinterpret_cast<const std::int32_t*>(
                qweight)[(size_t)r * out_dim + o];

            for (int k = 0; k < 8; ++k) {
                int shift = k * 4;
                int wi = (wraw >> shift) & 0xF;
                int zi = (zraw >> shift) & 0xF;
                int v = wi - zi;
                acc += x[r * 8 + k] * (v * scale);
            }
        }
        y[o] = acc;
    }
}

} // namespace winnex_nano
