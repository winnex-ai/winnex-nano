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
    // GPTQ int4 packed, desc_act=false layout (the Qwen/Winnex models):
    //   qweight[out/8][in]      row = output/8, each int32 holds 8 int4
    //                           weights along the INPUT dim
    //   qzeros[out/gs][in/8]    zero-points packed per (group, input-block)
    //   scales[out/gs][in]      scale per (group, input)
    //   g_idx[in]               per-input group (desc_act=true only; ignored)
    // For output o, the weight row is o/8 and the nibble within each int32 is
    // o%8. Each int32 at [row][i] holds the 8 outputs {o/8*8 .. o/8*8+7} for
    // input i. So for output o and input i:
    //   w[o][i] = nibble(o%8) of qweight[o/8][i]
    //   z[grp][i] = nibble(o%8) of qzeros[grp][i/8]
    //   scale[grp][i]
    //   acc[o] += x[i] * (w - z) * scale
    const int packed_in = in_dim / 8;
    const int groups = out_dim / group_size;
    const int shift = 0;  // nibble offset varies by output

    for (int o = 0; o < out_dim; ++o) {
        const int row = o >> 3;
        const int nib = o & 7;
        const int sh = nib * 4;
        int grp = o / group_size;  // desc_act=false: group by output position
        const std::int32_t* wrow = reinterpret_cast<const std::int32_t*>(qweight)
                                   + (size_t)row * in_dim;
        const std::int32_t* zbase = reinterpret_cast<const std::int32_t*>(qzeros)
                                    + (size_t)grp * packed_in;
        const float* srow = scales + (size_t)grp * in_dim;

        float acc = 0.0f;
        for (int i = 0; i < in_dim; ++i) {
            int wi = (wrow[i] >> sh) & 0xF;
            int zi = (zbase[i >> 3] >> sh) & 0xF;
            int v = wi - zi;
            acc += x[i] * (v * srow[i]);
        }
        y[o] = acc;
    }
}

} // namespace winnex_nano
