// spectral_tokenizer.cpp — Winnex spectral tokenizer (deterministic).
#include "winnex_nano/spectral_tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace winnex_nano {

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr int kAsciiMin = 32;
constexpr int kAsciiMax = 126;
constexpr float kProbeGain = 10.0f;   // moderate amplification (safe_wave_to_text)
constexpr float kEps = 1e-8f;
} // namespace

std::vector<Quat> SpectralTokenizer::encode(const std::string& text) const {
    std::vector<Quat> states;
    states.reserve(text.size() * static_cast<size_t>(embed_dim_));

    for (size_t pos = 0; pos < text.size(); ++pos) {
        int ascii_val = static_cast<unsigned char>(text[pos]);
        if (ascii_val < kAsciiMin) ascii_val = kAsciiMin;      // clamp control chars
        if (ascii_val > kAsciiMax) ascii_val = kAsciiMax;

        for (int j = 0; j < embed_dim_; ++j) {
            float phase = (ascii_val + static_cast<int>(pos) + j) * 2.0f * kPi / 256.0f;
            float amp = (ascii_val / 127.0f) * (static_cast<float>(j) / embed_dim_);

            Quat q;
            q.w = amp * std::cos(phase);
            q.x = amp * std::sin(phase);
            q.y = amp * std::cos(phase + kPi / 4.0f);
            q.z = amp * std::sin(phase + kPi / 4.0f);
            states.push_back(q);
        }
    }
    return states;
}

std::string SpectralTokenizer::decode(const std::vector<Quat>& states) const {
    if (states.empty()) return "";
    const size_t n_chars = states.size() / static_cast<size_t>(embed_dim_);
    const size_t total = n_chars * static_cast<size_t>(embed_dim_);
    if (total != states.size()) {
        throw std::runtime_error(
            "SpectralTokenizer::decode: state count not a multiple of embed_dim");
    }

    // Reference states for each ASCII char (position-independent; the probe
    // applies the position phase to the state side, matching encode).
    auto ref_state = [&](int ascii_val, size_t pos) {
        std::vector<Quat> ref;
        ref.reserve(embed_dim_);
        for (int j = 0; j < embed_dim_; ++j) {
            float phase = (ascii_val + static_cast<int>(pos) + j) * 2.0f * kPi / 256.0f;
            float amp = (ascii_val / 127.0f) * (static_cast<float>(j) / embed_dim_);
            ref.push_back({
                amp * std::cos(phase),
                amp * std::sin(phase),
                amp * std::cos(phase + kPi / 4.0f),
                amp * std::sin(phase + kPi / 4.0f),
            });
        }
        return ref;
    };

    std::string out;
    out.reserve(n_chars);

    for (size_t c = 0; c < n_chars; ++c) {
        // Quaternion cosine similarity against every printable ASCII char.
        // This is the exact scan (the decode ceiling). Complexity O(95 · d · 4).
        std::vector<float> sims;
        sims.reserve(kAsciiMax - kAsciiMin + 1);
        for (int ascii_val = kAsciiMin; ascii_val <= kAsciiMax; ++ascii_val) {
            auto ref = ref_state(ascii_val, c);
            float ip = 0.0f, n1 = 0.0f, n2 = 0.0f;
            for (int j = 0; j < embed_dim_; ++j) {
                const Quat& st = states[c * static_cast<size_t>(embed_dim_) + j];
                const Quat& m  = ref[j];
                ip += st.w*m.w + st.x*m.x + st.y*m.y + st.z*m.z;
                n1 += st.w*st.w + st.x*st.x + st.y*st.y + st.z*st.z;
                n2 += m.w*m.w + m.x*m.x + m.y*m.y + m.z*m.z;
            }
            float denom = std::sqrt(n1 * n2) + kEps;
            sims.push_back(ip / denom);
        }

        // Madhava principle: select the TOP-K by bound, re-score exactly only
        // the survivors. NO exp/sum (no softmax of any form). The decode is a
        // deterministic top-1 selection by cosine similarity.
        const int k = 1;  // hard decode: top-1 (the reference char)
        std::vector<int> idx(95);
        for (int i = 0; i < 95; ++i) idx[i] = i;
        // Partial selection of the top-k (exact scan is only 95 — negligible).
        for (int i = 0; i < k; ++i) {
            for (int j = i + 1; j < 95; ++j) {
                if (sims[j] > sims[idx[i]]) idx[i] = j;
            }
        }
        out.push_back(static_cast<char>(kAsciiMin + idx[0]));
    }
    return out;
}

} // namespace winnex_nano
