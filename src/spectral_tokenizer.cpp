// spectral_tokenizer.cpp — PsiQRH spectral tokenizer (deterministic).
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

    std::string out;
    out.reserve(n_chars);

    for (size_t c = 0; c < n_chars; ++c) {
        // Inner products of this char's spectral state against every char's
        // reference pattern.
        std::vector<float> probs;
        probs.reserve(kAsciiMax - kAsciiMin + 1);

        // Reference states for each ASCII char, computed with the SAME encode
        // formula INCLUDING the sequence position c (the phase shift (ascii+c+j)
        // is what distinguishes chars at a given position). Comparing full
        // quaternion states preserves the phase pattern.
        auto ref_state = [&](int ascii_val) {
            std::vector<Quat> ref;
            ref.reserve(embed_dim_);
            for (int j = 0; j < embed_dim_; ++j) {
                float phase = (ascii_val + static_cast<int>(c) + j) * 2.0f * kPi / 256.0f;
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

        for (int ascii_val = kAsciiMin; ascii_val <= kAsciiMax; ++ascii_val) {
            auto ref = ref_state(ascii_val);
            // Quaternion cosine similarity over the full [embed_dim, 4] state,
            // with the position-dependent phase already baked into the state.
            float ip = 0.0f, n1 = 0.0f, n2 = 0.0f;
            for (int j = 0; j < embed_dim_; ++j) {
                const Quat& st = states[c * static_cast<size_t>(embed_dim_) + j];
                const Quat& m  = ref[j];
                ip += st.w*m.w + st.x*m.x + st.y*m.y + st.z*m.z;
                n1 += st.w*st.w + st.x*st.x + st.y*st.y + st.z*st.z;
                n2 += m.w*m.w + m.x*m.x + m.y*m.y + m.z*m.z;
            }
            float denom = std::sqrt(n1 * n2) + kEps;
            probs.push_back(ip / denom);
        }

        // Moderate amplification + safe normalization (not softmax).
        float max_p = *std::max_element(probs.begin(), probs.end());
        if (max_p > 1.0f) {
            for (auto& v : probs) v /= max_p;
        }
        float sum = 0.0f;
        for (auto& v : probs) {
            v = std::exp(v * kProbeGain);
            sum += v;
        }
        if (sum > kEps) {
            for (auto& v : probs) v /= sum;
        }

        // argmax.
        size_t best = 0;
        float best_v = probs[0];
        for (size_t i = 1; i < probs.size(); ++i) {
            if (probs[i] > best_v) { best_v = probs[i]; best = i; }
        }
        out.push_back(static_cast<char>(kAsciiMin + static_cast<int>(best)));
    }
    return out;
}

} // namespace winnex_nano
