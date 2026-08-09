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

    // Precompute the position-independent reference states ONCE (pos = 0):
    //   phase_0 = (ascii + 0 + j)·2π/256
    // The position shift pos·2π/256 is a CONSTANT phase rotation across all
    // modes, so it can be applied as a rotation factor instead of recomputing
    // sin/cos for every character (the previous version recomputed 95·d·4 trig
    // calls per character — the decode bottleneck).
    struct RefSet {
        std::vector<std::vector<Quat>> refs;  // indexed by (ascii - kAsciiMin)
    };
    RefSet base_refs;
    base_refs.refs.reserve(kAsciiMax - kAsciiMin + 1);
    for (int ascii_val = kAsciiMin; ascii_val <= kAsciiMax; ++ascii_val) {
        std::vector<Quat> ref;
        ref.reserve(embed_dim_);
        for (int j = 0; j < embed_dim_; ++j) {
            float phase = (ascii_val + j) * 2.0f * kPi / 256.0f;
            float amp = (ascii_val / 127.0f) * (static_cast<float>(j) / embed_dim_);
            ref.push_back({
                amp * std::cos(phase),
                amp * std::sin(phase),
                amp * std::cos(phase + kPi / 4.0f),
                amp * std::sin(phase + kPi / 4.0f),
            });
        }
        base_refs.refs.push_back(std::move(ref));
    }

    std::string out;
    out.reserve(n_chars);

    for (size_t c = 0; c < n_chars; ++c) {
        // Quaternion cosine similarity against every printable ASCII char.
        // This is the exact scan (the decode ceiling). Complexity O(95 · d · 4).
        std::vector<float> sims;
        sims.reserve(kAsciiMax - kAsciiMin + 1);

        // The state's norm is constant across the 95 candidates — compute it
        // ONCE per character.
        const Quat* st_base = &states[c * static_cast<size_t>(embed_dim_)];
        float n1 = 0.0f;
        for (int j = 0; j < embed_dim_; ++j) {
            const Quat& st = st_base[j];
            n1 += st.w*st.w + st.x*st.x + st.y*st.y + st.z*st.z;
        }
        const float sqrt_n1 = std::sqrt(n1);

        // Position phase shift: pos·2π/256 (a constant rotation factor).
        const float pos_phase = static_cast<float>(c) * 2.0f * kPi / 256.0f;
        const float cp = std::cos(pos_phase), sp = std::sin(pos_phase);

        for (int ascii_val = kAsciiMin; ascii_val <= kAsciiMax; ++ascii_val) {
            const auto& ref = base_refs.refs[ascii_val - kAsciiMin];
            float ip = 0.0f, n2 = 0.0f;
            for (int j = 0; j < embed_dim_; ++j) {
                const Quat& st = st_base[j];
                const Quat& m  = ref[j];
                // Rotate the reference phase by the position shift: the (w,x)
                // AND (y,z) complex pairs both rotate by pos_phase (the π/4 in
                // y/z is a fixed offset preserved by the rotation).
                float mw = m.w * cp - m.x * sp;
                float mx = m.w * sp + m.x * cp;
                float my = m.y * cp - m.z * sp;
                float mz = m.y * sp + m.z * cp;
                ip += st.w*mw + st.x*mx + st.y*my + st.z*mz;
                n2 += mw*mw + mx*mx + my*my + mz*mz;
            }
            float denom = sqrt_n1 * std::sqrt(n2) + kEps;
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
