/**
 * spectral_tokenizer.hpp — Winnex-derived spectral tokenizer (deterministic).
 *
 * Replaces BPE with a character → quaternion spectral representation that is
 * PURE arithmetic (no vocabulary, no training, no external tokenizer files):
 *
 *   per character ascii_val, at sequence position pos, per mode j in [0, d):
 *       phase     = (ascii_val + pos + j) · 2π / 256
 *       amplitude = (ascii_val / 127.0) · (j / d)
 *       ψ₀ = amplitude·cos(phase)             (real, w)
 *       ψ₁ = amplitude·sin(phase)             (i,    x)
 *       ψ₂ = amplitude·cos(phase + π/4)       (j,    y)
 *       ψ₃ = amplitude·sin(phase + π/4)       (k,    z)
 *
 * Decoding is the conjugate probe:
 *       P_i  = |⟨mode_i, ψ⟩|²          (inner product over the spectral modes)
 *       P    = exp(P·gain)             (moderate amplification)
 *       P    = P / ΣP                  (safe normalization)
 *       char = argmax(P)
 *
 * This is the "text → quaternion spectral representation → text" round-trip
 * from the ΨQRH framework (Zenodo 17171112), ported to portable C++.
 * Each character has a unique spectral signature, so the mapping is
 * invertible and deterministic — the Madhava way (no black boxes).
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
 */
#ifndef WINNEX_NANO_SPECTRAL_TOKENIZER_HPP
#define WINNEX_NANO_SPECTRAL_TOKENIZER_HPP

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace winnex_nano {

// A quaternion [w, x, y, z].
struct Quat {
    float w = 0, x = 0, y = 0, z = 0;
};

// Quaternion Hamilton product: q1 * q2.
inline Quat quat_mul(const Quat& a, const Quat& b) {
    return {
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
    };
}

// Unit quaternion from rotation parameters (θ, ω, φ) — the Winnex "control knob".
inline Quat unit_quaternion(float theta, float omega, float phi) {
    float th = theta * 0.5f;
    float st = std::sin(th), ct = std::cos(th);
    return {
        ct,
        st * std::cos(omega),
        st * std::sin(omega) * std::cos(phi),
        st * std::sin(omega) * std::sin(phi),
    };
}

// Conjugate (inverse for unit quaternions).
inline Quat quat_conj(const Quat& q) { return {q.w, -q.x, -q.y, -q.z}; }

/**
 * SpectralTokenizer — deterministic character ↔ quaternion-spectrum mapping.
 *
 * embed_dim: number of spectral modes per character (default 64).
 * ascii_range: printable ASCII [32, 126] by default.
 */
class SpectralTokenizer {
public:
    explicit SpectralTokenizer(int embed_dim = 64)
        : embed_dim_(embed_dim) {}

    // Encodes a text into a sequence of quaternion states [seq_len, embed_dim].
    // Each character produces embed_dim quaternions (the spectral modes).
    std::vector<Quat> encode(const std::string& text) const;

    // Decodes a sequence of quaternion states back to text via the probe
    // (inner product + exp + normalize + argmax). Returns the best-guess text.
    std::string decode(const std::vector<Quat>& states) const;

    int embed_dim() const { return embed_dim_; }

private:
    int embed_dim_;
};

} // namespace winnex_nano

#endif // WINNEX_NANO_SPECTRAL_TOKENIZER_HPP
