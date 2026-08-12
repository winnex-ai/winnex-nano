// spectral_tokenizer.cpp — Winnex spectral tokenizer (deterministic, UTF-8).
//
// THE MATHEMATICS — the group isomorphism ℤ/256ℤ → μ₂₅₆.
//
//   Encode:  f : byte ↦ e^{i·(byte+pos+j)·2π/256}     (phase-only, amp = 1.0)
//   Decode:  f⁻¹ : ψ ↦ arg(ψ)·256/(2π) − pos − j  (mod 256)
//
// This is NOT a DFT to be accelerated by FFT, and NOT a correlation to be
// scanned. The byte is an element of ℤ/256ℤ; its image is a 256th root of
// unity. Decoding is the direct application of the inverse homomorphism —
// O(1) per byte, no probe, no FFT, no 256-ref scan. This is the fundamental
// algebraic structure of the problem, used as engineering optimization.
//
//   f(a+b) = e^{i(a+b)2π/256} = e^{ia2π/256}·e^{ib2π/256} = f(a)·f(b)  (homomorphism)
//   f(a) = f(b) ⟹ a−b ≡ 0 (mod 256)                                    (injective)
//   every μ₂₅₆ element is e^{ik2π/256}, k ∈ {0..255}                     (surjective)
//
// Auto-synchronizing (UTF-8 is byte-oriented), lossless for ANY language.
//
// BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
#include "winnex_nano/spectral_tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace winnex_nano {

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr int kByteRange = SpectralTokenizer::kByteRange;
} // namespace

std::vector<Quat> SpectralTokenizer::encode(const std::string& text) const {
    std::vector<Quat> states;
    states.reserve(text.size() * static_cast<size_t>(embed_dim_));

    // Phase-only: amplitude 1.0 (unit circle). The byte + pos define the phase;
    // the mode j is an additional constant phase offset, preserved analytically.
    for (size_t pos = 0; pos < text.size(); ++pos) {
        const int bpos = static_cast<int>(static_cast<unsigned char>(text[pos]) + pos);
        for (int j = 0; j < embed_dim_; ++j) {
            const float phase = (static_cast<float>(bpos + j)) * 2.0f * kPi / kByteRange;
            Quat q;
            q.w = std::cos(phase);
            q.x = std::sin(phase);
            // The (y,z) components are a fixed π/4-rotated copy of (w,x):
            // phase + π/4. They carry no independent information for the
            // analytic decode (which reads arg from (w,x)), but preserve the
            // quaternion-spectrum convention for downstream consumers.
            q.y = std::cos(phase + kPi / 4.0f);
            q.z = std::sin(phase + kPi / 4.0f);
            states.push_back(q);
        }
    }
    return states;
}

void SpectralTokenizer::encode_into_buffer(const std::string& text, float* out) const {
    const float kPi2 = 2.0f * kPi;
    for (size_t pos = 0; pos < text.size(); ++pos) {
        const int bpos = static_cast<int>(static_cast<unsigned char>(text[pos]) + pos);
        float* row = out + pos * static_cast<size_t>(embed_dim_) * 4u;
        for (int j = 0; j < embed_dim_; ++j) {
            const float phase = (static_cast<float>(bpos + j)) * kPi2 / kByteRange;
            const float c = std::cos(phase), s = std::sin(phase);
            const float cp = std::cos(phase + kPi / 4.0f), sp = std::sin(phase + kPi / 4.0f);
            float* q = row + static_cast<size_t>(j) * 4u;
            q[0] = c; q[1] = s; q[2] = cp; q[3] = sp;
        }
    }
}

std::vector<float> SpectralTokenizer::encode_histogram(const std::string& text, int bins) const {
    std::vector<float> hist(bins, 0.0f);
    for (char ch : text) {
        int idx = static_cast<unsigned char>(ch) % bins;
        hist[idx] += 1.0f;
    }
    float norm = 0.0f;
    for (auto v : hist) norm += v * v;
    norm = std::sqrt(norm) + 1e-9f;
    for (auto& v : hist) v /= norm;
    return hist;
}

std::string SpectralTokenizer::decode(const std::vector<Quat>& states) const {
    if (states.empty()) return "";
    const size_t n_bytes = states.size() / static_cast<size_t>(embed_dim_);
    if (n_bytes * static_cast<size_t>(embed_dim_) != states.size()) {
        throw std::runtime_error(
            "SpectralTokenizer::decode: state count not a multiple of embed_dim");
    }

    std::string out;
    out.reserve(n_bytes);

    // Analytic inverse homomorphism: O(1) per byte.
    //   phase = (byte + pos + j)·2π/256  ⟹  byte = phase·256/(2π) − pos − j
    // We read the phase from the FIRST mode (j = 0) — the (w,x) pair.
    // All other modes are redundant for decoding (each is a constant phase
    // rotation of mode 0), so reading j=0 is exact.
    for (size_t b = 0; b < n_bytes; ++b) {
        const Quat& q = states[b * static_cast<size_t>(embed_dim_)];
        float phase = std::atan2(q.x, q.w);
        if (phase < 0) phase += 2.0f * kPi;
        // byte = round(phase·256/(2π)) − pos
        float fp = phase * kByteRange / (2.0f * kPi);
        int byte = static_cast<int>(std::lround(fp)) - static_cast<int>(b);
        byte = ((byte % kByteRange) + kByteRange) % kByteRange;
        out.push_back(static_cast<char>(byte));
    }
    return out;
}

std::string SpectralTokenizer::decode_fft(const std::vector<Quat>& states) const {
    // The analytic decode IS the O(1) inverse of the isomorphism — there is
    // no FFT needed. Kept as an alias so existing callers can migrate.
    return decode(states);
}

} // namespace winnex_nano
