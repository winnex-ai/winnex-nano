// tensor_ops.cpp — native tensor operations (model-agnostic).
#include "winnex_nano/tensor_ops.hpp"

#include <cmath>
#include <cstddef>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

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

void dense_matmul(float* y, const float* x, const float* W,
                  int in_dim, int out_dim) {
    // Row-major: y[o] = Σ_i x[i] · W[o*in_dim + i], for o in [0, out_dim).
    // AVX2/FMA: process 8 in_dim elements per iteration (FMA), OpenMP over
    // the output rows (each row is independent).
#pragma omp parallel for
    for (int o = 0; o < out_dim; ++o) {
        const float* wrow = W + (size_t)o * in_dim;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= in_dim; i += 8) {
            __m256 xv = _mm256_loadu_ps(x + i);
            __m256 wv = _mm256_loadu_ps(wrow + i);
            acc = _mm256_fmadd_ps(xv, wv, acc);
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, acc);
        float s = tmp[0] + tmp[1] + tmp[2] + tmp[3]
                + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        for (; i < in_dim; ++i) s += x[i] * wrow[i];
        y[o] = s;
#else
        float s = 0.0f;
        for (int i = 0; i < in_dim; ++i) s += x[i] * wrow[i];
        y[o] = s;
#endif
    }
}

} // namespace winnex_nano
