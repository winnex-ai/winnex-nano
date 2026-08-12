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
 * SpectralTokenizer — deterministic byte ↔ unit-circle isomorphism.
 *
 * The mathematical structure is the group isomorphism
 *     f : ℤ/256ℤ → μ₂₅₆        (integers mod 256 → 256th roots of unity)
 *     f(byte) = e^{i·byte·2π/256}
 *
 * Encode:  each UTF-8 byte b at position pos maps to the unit complex number
 *          ψ = e^{i·(byte+pos+j)·2π/256}   (j = spectral mode, amp = 1.0)
 * Decode:  the inverse homomorphism, evaluated analytically in O(1) per byte:
 *          byte = arg(ψ) · 256/(2π) − pos − j   (mod 256)
 *
 * No FFT, no 256-ref correlation scan, no probe. The decode is the direct
 * inverse application of the group isomorphism — the fundamental algebra of
 * the problem, not an engineering acceleration.
 *
 * Auto-synchronizing (UTF-8 is byte-oriented), lossless round-trip for ANY
 * language, and deterministic (closed-form, no training).
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
 */
class SpectralTokenizer {
public:
    explicit SpectralTokenizer(int embed_dim = 64)
        : embed_dim_(embed_dim) {}

    // Encodes a text into a sequence of quaternion states [seq_len, embed_dim].
    // Each UTF-8 byte (0-255) at position pos produces embed_dim pure-phase
    // modes:  ψ[j] = e^{i·(byte+pos+j)·2π/256}  (amplitude 1.0 — phase only).
    std::vector<Quat> encode(const std::string& text) const;

    // Encodes a text into a DISCRIMINATIVE character histogram (normalized).
    // Empirically validated: the raw spectral average collapses (cos ~0.99),
    // while the character histogram discriminates well (cos 0.13-0.60, std 0.16).
    // This is the text representation projected onto the model manifold (X factor).
    std::vector<float> encode_histogram(const std::string& text, int bins = 256) const;

    // Decodes a sequence of quaternion states back to text via the INVERSE
    // GROUP HOMOMORPHISM:  byte = arg(ψ)·256/(2π) − pos − j (mod 256).
    // O(1) per byte — no probe, no correlation, no FFT.
    std::string decode(const std::vector<Quat>& states) const;

    // Batch encode into a CONTIGUOUS float buffer [n_bytes, embed_dim, 4].
    // Writes w,x,y,z for every mode. The caller owns the buffer (size
    // n_bytes*embed_dim*4). Used by the Python binding to return a numpy
    // array without per-Quat Python object overhead.
    void encode_into_buffer(const std::string& text, float* out) const;

    // Compatibility alias: the decode is already the analytic O(1) inverse,
    // so this is identical to decode() (kept so callers can migrate).
    std::string decode_fft(const std::vector<Quat>& states) const;

    int embed_dim() const { return embed_dim_; }
    // Number of byte references (0-255).
    static constexpr int kByteRange = 256;

private:
    int embed_dim_;
};

} // namespace winnex_nano

#endif // WINNEX_NANO_SPECTRAL_TOKENIZER_HPP
