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

    // COMPACT representation (64 bits/char): 1 complex number per byte.
    // The spectral signal is a pure tone; by the sampling theorem only ONE
    // sample (j=0) is needed to recover the phase — no embed_dim redundancy.
    // encode_compact: [cos((byte+pos)·2π/256), sin(...)] per byte → 2 floats.
    std::vector<Quat> encode_compact(const std::string& text) const;

    // decode_compact: analytic inverse from the single sample (O(1)/byte).
    std::string decode_compact(const std::vector<Quat>& states) const;

    // Compatibility alias: the decode is already the analytic O(1) inverse,
    // so this is identical to decode() (kept so callers can migrate).
    std::string decode_fft(const std::vector<Quat>& states) const;

    // ------------------------------------------------------------------
    // MODO HRR — agregação com preservação de norma (VSA / HRR)
    // ------------------------------------------------------------------
    //
    // O modo acima (encode/encode_compact) é um ISOMORFISMO: 1 byte ↔ 1 fase.
    // É exato para round-trip, mas NÃO agrega — a fase de cada byte vive em
    // posições distintas, e uma média simples sofre cancelamento destrutivo
    // (‖média‖ → 0 com fases aleatórias), violando a conservação de norma.
    //
    // O modo HRR (Holographic Reduced Representations, Plate 1995) resolve os
    // dois problemas:
    //
    //   símbolo do byte  :  s(b)   = e^{i·j·b·2π/256}      (j = frequência)
    //   binding posicional:  p(pos) = e^{i·j·pos·2π/n}
    //   vínculo           :  psi    = s(b) ⊙ p(pos)
    //   agregação         :  Z      = Σ_pos psi            (soma COERENTE)
    //
    // A diferença para o modo atual é UM caractere na fórmula: no modo atual
    // o modo espectral j é um OFFSET, `(b + pos + j)`; no HRR, j é FREQUÊNCIA,
    // `j·(b + pos)`. Com j como offset, todos os modos são rotações constantes
    // uns dos outros (redundância de D×); com j como frequência, cada modo
    // amostra o byte de forma distinta e a posição entra com gradiente real.
    //
    // Propriedades:
    //   - Parseval: ‖Z‖² ≤ n²·D, com igualdade para entradas alinhadas. A soma
    //     coerente conserva a energia; a média a destrói.
    //   - Ordem preservada: anagramas produzem Z distintos (cos < 0.99).
    //
    // ADIÇÃO PURA: nenhum método existente muda de comportamento.
    std::vector<float> encode_hrr(const std::string& text) const;

    // Idem, normalizado a norma unitária (contrato de cosseno do Madhava).
    // Devolve `embed_dim()` floats (a parte real do vetor complexo).
    std::vector<float> encode_hrr_unit(const std::string& text) const;

    // ── CORREÇÃO 2026-10-23: HRR complexo e com denominador fixo ──────────
    //
    // Dois defeitos medidos na versão em `encode_hrr`:
    //
    // 1. DESCARTA A PARTE IMAGINÁRIA.
    //    Z = Σ e^{i·φ} é complexo. A versão antiga devolvia `re[j]` e jogava
    //    `im[j]` fora: metade da informação, e a energia de uma senoide some
    //    junto. Medido em 11 KB de texto: norma do vetor real = 0,2 contra 2,0
    //    do complexo — 10× menos sinal por componente, antes de qualquer
    //    normalização. `encode_hrr_complex` devolve a intercalação
    //    [re0, im0, re1, im1, ...] com 2·embed_dim floats.
    //
    // 2. usa `pos/n` como denominador, o que acopla o binding posicional ao
    //    comprimento do texto: o mesmo prefixo em dois documentos de tamanhos
    //    diferentes recebe fases diferentes, e uma frase não é comparável
    //    consigo mesma em contextos distintos. `encode_hrr_fixed` usa uma
    //    taxa fixa por posição, então o vetor de "a+b" é estável.
    //
    // `encode_hrr_complex(text, pos_stride)` faz as duas coisas. A assinatura
    // antiga fica intocada para não quebrar chamadores.
    std::vector<float> encode_hrr_complex(const std::string& text,
                                          double pos_stride = 1.0 / 64.0) const;

    // Matriz [n_bytes, 2·embed_dim] em float32, para alimentar o ForwardEngine
    // diretamente. É a representação usada pelo benchmark de tokenizadores.
    void encode_hrr_into_buffer(const std::string& text, float* out,
                                double pos_stride = 1.0 / 64.0) const;

    int embed_dim() const { return embed_dim_; }
    // Number of byte references (0-255).
    static constexpr int kByteRange = 256;

private:
    int embed_dim_;
};

} // namespace winnex_nano

#endif // WINNEX_NANO_SPECTRAL_TOKENIZER_HPP
