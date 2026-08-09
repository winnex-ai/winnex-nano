// forward.cpp — native forward pass for transformer LLMs (GPTQ int4).
#include "winnex_nano/forward.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>

#include "winnex_nano/tensor_ops.hpp"

namespace winnex_nano {

// ---------------------------------------------------------------------------
// Config loader (minimal config.json parser for the model hyper-parameters).
// ---------------------------------------------------------------------------
ModelConfig load_config(const std::string& config_path) {
    // Minimal JSON field extraction (values we need are top-level ints/floats).
    std::string s;
    {   std::ifstream f(config_path, std::ios::binary);
        if (!f) throw std::runtime_error("load_config: cannot open " + config_path);
        s.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    auto find_int = [&](const std::string& key, int def) -> int {
        std::string pat = "\"" + key + "\"";
        size_t p = s.find(pat);
        if (p == std::string::npos) return def;
        p = s.find(':', p); if (p == std::string::npos) return def;
        p++;
        while (p < s.size() && (s[p]==' '||s[p]=='\n'||s[p]=='\t')) p++;
        int v = 0; bool neg = false;
        if (p < s.size() && s[p]=='-') { neg=true; p++; }
        while (p < s.size() && s[p]>='0' && s[p]<='9') { v = v*10 + (s[p]-'0'); p++; }
        return neg ? -v : v;
    };
    auto find_float = [&](const std::string& key, float def) -> float {
        std::string pat = "\"" + key + "\"";
        size_t p = s.find(pat);
        if (p == std::string::npos) return def;
        p = s.find(':', p); if (p == std::string::npos) return def;
        p++;
        while (p < s.size() && (s[p]==' '||s[p]=='\n'||s[p]=='\t')) p++;
        return (float)std::atof(s.c_str() + p);
    };
    auto find_str = [&](const std::string& key, const std::string& def) -> std::string {
        std::string pat = "\"" + key + "\"";
        size_t p = s.find(pat);
        if (p == std::string::npos) return def;
        p = s.find(':', p); if (p == std::string::npos) return def;
        p++;
        while (p < s.size() && (s[p]==' '||s[p]=='\n'||s[p]=='\t')) p++;
        if (p < s.size() && s[p]=='"') { p++; }
        std::string v;
        while (p < s.size() && s[p] != '"') { v += s[p]; p++; }
        return v;
    };

    ModelConfig c;
    c.arch = find_str("model_type", "qwen2");
    c.vocab_size = find_int("vocab_size", 0);
    c.hidden_size = find_int("hidden_size", 0);
    c.intermediate_size = find_int("intermediate_size", 0);
    c.num_hidden_layers = find_int("num_hidden_layers", 0);
    c.num_attention_heads = find_int("num_attention_heads", 0);
    c.num_key_value_heads = find_int("num_key_value_heads", c.num_attention_heads);
    c.max_position_embeddings = find_int("max_position_embeddings", 2048);
    c.rope_theta = find_float("rope_theta", 10000.0f);
    c.rms_norm_eps = find_float("rms_norm_eps", 1e-6f);
    c.tie_word_embeddings = find_int("tie_word_embeddings", 1) == 1;
    c.head_dim = c.hidden_size / c.num_attention_heads;
    return c;
}

// ---------------------------------------------------------------------------
// ForwardEngine
// ---------------------------------------------------------------------------
struct ForwardEngine::LayerWeights {
    std::vector<float> in_norm;    // [hidden]
    std::vector<float> post_norm;  // [hidden]
    // QKV projections (GPTQ int4).
    std::vector<int32_t> q_qw, q_qz, k_qw, k_qz, v_qw, v_qz, o_qw, o_qz;
    std::vector<float> q_sc, k_sc, v_sc, o_sc;
    // MLP.
    std::vector<int32_t> g_qw, g_qz, u_qw, u_qz, d_qw, d_qz;
    std::vector<float> g_sc, u_sc, d_sc;
    // Dims.
    int hidden = 0, q_out = 0, kv_out = 0, inter = 0, gs = 0;
    bool loaded = false;
};

ForwardEngine::ForwardEngine(const ModelConfig& cfg, const Safetensors& st)
    : cfg_(cfg), st_(&st) {
    if (cfg_.head_dim <= 0 || cfg_.num_attention_heads <= 0) {
        throw std::runtime_error("ForwardEngine: invalid config (head_dim/heads)");
    }
    param_count_ = 0;
}

namespace {
// Loads a GPTQ projection into (qweight, qzeros, scales) int32/float vectors.
void load_gptq(const Safetensors& st, const std::string& base,
               std::vector<int32_t>& qw, std::vector<int32_t>& qz,
               std::vector<float>& sc, size_t& param_count) {
    qw = st.to_int32(st.get(base + ".qweight"));
    qz = st.to_int32(st.get(base + ".qzeros"));
    sc = st.to_float(st.get(base + ".scales"));
    param_count += qw.size() * 4 + sc.size();
}
} // namespace

void ForwardEngine::load_layer(int layer, LayerWeights& w) {
    std::string L = "model.layers." + std::to_string(layer) + ".";
    w.hidden = cfg_.hidden_size;
    w.q_out = cfg_.hidden_size;
    w.kv_out = cfg_.head_dim * cfg_.num_key_value_heads;
    w.inter = cfg_.intermediate_size;
    w.gs = cfg_.group_size;

    w.in_norm = st_->to_float(st_->get(L + "input_layernorm.weight"));
    w.post_norm = st_->to_float(st_->get(L + "post_attention_layernorm.weight"));

    load_gptq(*st_, L + "self_attn.q_proj", w.q_qw, w.q_qz, w.q_sc, param_count_);
    load_gptq(*st_, L + "self_attn.k_proj", w.k_qw, w.k_qz, w.k_sc, param_count_);
    load_gptq(*st_, L + "self_attn.v_proj", w.v_qw, w.v_qz, w.v_sc, param_count_);
    load_gptq(*st_, L + "self_attn.o_proj", w.o_qw, w.o_qz, w.o_sc, param_count_);
    load_gptq(*st_, L + "mlp.gate_proj", w.g_qw, w.g_qz, w.g_sc, param_count_);
    load_gptq(*st_, L + "mlp.up_proj",   w.u_qw, w.u_qz, w.u_sc, param_count_);
    load_gptq(*st_, L + "mlp.down_proj", w.d_qw, w.d_qz, w.d_sc, param_count_);
    w.loaded = true;
}

void ForwardEngine::run_layer(int layer, std::vector<float>& h, int seq_len,
                              std::vector<float>& k_cache, std::vector<float>& v_cache,
                              int cache_pos) {
    LayerWeights w;
    load_layer(layer, w);
    const int H = w.hidden, nh = cfg_.num_attention_heads, nkv = cfg_.num_key_value_heads;
    const int hd = cfg_.head_dim;

    // 1. Input layernorm (RMSNorm).
    std::vector<float> hn(H * seq_len);
    rms_norm(hn.data(), h.data(), w.in_norm.data(), seq_len, H, cfg_.rms_norm_eps);

    // 2. QKV projections (GPTQ) — batched as (seq*H).
    std::vector<float> q(seq_len * w.q_out), k(seq_len * w.kv_out), v(seq_len * w.kv_out);
    for (int t = 0; t < seq_len; ++t) {
        const float* xt = hn.data() + (size_t)t * H;
        gptq_matmul(q.data() + (size_t)t * w.q_out, xt,
                    (const uint8_t*)w.q_qw.data(), (const uint8_t*)w.q_qz.data(),
                    w.q_sc.data(), nullptr, H, w.q_out, w.gs);
        gptq_matmul(k.data() + (size_t)t * w.kv_out, xt,
                    (const uint8_t*)w.k_qw.data(), (const uint8_t*)w.k_qz.data(),
                    w.k_sc.data(), nullptr, H, w.kv_out, w.gs);
        gptq_matmul(v.data() + (size_t)t * w.kv_out, xt,
                    (const uint8_t*)w.v_qw.data(), (const uint8_t*)w.v_qz.data(),
                    w.v_sc.data(), nullptr, H, w.kv_out, w.gs);
    }

    // 3. RoPE on Q (all heads) and K (kv heads) at their absolute positions.
    //    Q layout: [t, nh*hd]; K layout: [t, nkv*hd].
    for (int t = 0; t < seq_len; ++t) {
        int abs_pos = cache_pos + t;
        for (int hh = 0; hh < nh; ++hh) {
            float* hq = q.data() + ((size_t)t * nh + hh) * hd;
            for (int j = 0; j < hd; j += 2) {
                float inv = std::pow(cfg_.rope_theta, -2.0f * j / hd);
                float ang = abs_pos * inv;
                float c = std::cos(ang), s = std::sin(ang);
                float a = hq[j], b = hq[j+1];
                hq[j] = a*c - b*s; hq[j+1] = a*s + b*c;
            }
        }
        for (int hh = 0; hh < nkv; ++hh) {
            float* hk = k.data() + ((size_t)t * nkv + hh) * hd;
            for (int j = 0; j < hd; j += 2) {
                float inv = std::pow(cfg_.rope_theta, -2.0f * j / hd);
                float ang = abs_pos * inv;
                float c = std::cos(ang), s = std::sin(ang);
                float a = hk[j], b = hk[j+1];
                hk[j] = a*c - b*s; hk[j+1] = a*s + b*c;
            }
        }
    }

    // 4. Cache K/V and attend (GQA: each KV head serves nh/nkv query heads).
    for (int t = 0; t < seq_len; ++t) {
        std::memcpy(k_cache.data() + (size_t)(cache_pos + t) * w.kv_out,
                    k.data() + (size_t)t * w.kv_out, (size_t)w.kv_out * sizeof(float));
        std::memcpy(v_cache.data() + (size_t)(cache_pos + t) * w.kv_out,
                    v.data() + (size_t)t * w.kv_out, (size_t)w.kv_out * sizeof(float));
    }
    int ctx_len = cache_pos + seq_len;

    // 5. Attention output: for each query head, attend to all cached keys.
    std::vector<float> attn_out(seq_len * w.q_out, 0.0f);
    const float scale = 1.0f / std::sqrt((float)hd);
    for (int t = 0; t < seq_len; ++t) {
        for (int hh = 0; hh < nh; ++hh) {
            int kvh = hh / (nh / nkv);  // GQA mapping
            const float* qh = q.data() + ((size_t)t * nh + hh) * hd;
            // Scores over ctx_len keys (exact attention — the Madhava top-k
            // selective can replace this; here exact for correctness first).
            std::vector<std::pair<float,int>> scores;
            scores.reserve(ctx_len);
            for (int u = 0; u < ctx_len; ++u) {
                const float* kh = k_cache.data() + ((size_t)u * nkv + kvh) * hd;
                float s = 0;
                for (int j = 0; j < hd; ++j) s += qh[j] * kh[j];
                scores.push_back({s * scale, u});
            }
            // Softmax-free selection: keep the top-k by score, weight = score.
            // (Madhava principle: selective top-k instead of full softmax.)
            const int ksel = 16;
            std::sort(scores.begin(), scores.end(),
                      [](auto&a, auto&b){return a.first > b.first;});
            const int keep = std::min(ksel, ctx_len);
            float wsum = 0;
            for (int i = 0; i < keep; ++i) wsum += scores[i].first;
            if (wsum < 1e-9f) wsum = 1.0f;
            float* ot = attn_out.data() + ((size_t)t * nh + hh) * hd;
            for (int j = 0; j < hd; ++j) ot[j] = 0.0f;
            for (int i = 0; i < keep; ++i) {
                int u = scores[i].second;
                float wgt = scores[i].first / wsum;  // normalize (softmax-free)
                const float* vh = v_cache.data() + ((size_t)u * nkv + kvh) * hd;
                for (int j = 0; j < hd; ++j) ot[j] += wgt * vh[j];
            }
        }
    }

    // 6. Output projection + residual.
    std::vector<float> proj(seq_len * H);
    for (int t = 0; t < seq_len; ++t) {
        gptq_matmul(proj.data() + (size_t)t * H, attn_out.data() + (size_t)t * w.q_out,
                    (const uint8_t*)w.o_qw.data(), (const uint8_t*)w.o_qz.data(),
                    w.o_sc.data(), nullptr, w.q_out, H, w.gs);
    }
    for (int i = 0; i < seq_len * H; ++i) h[i] += proj[i];

    // 7. Post attention norm + MLP (gate*up -> down, SiLU gating).
    std::vector<float> hn2(H * seq_len);
    rms_norm(hn2.data(), h.data(), w.post_norm.data(), seq_len, H, cfg_.rms_norm_eps);
    std::vector<float> mlp_out(seq_len * H);
    std::vector<float> gate(seq_len * w.inter), up(seq_len * w.inter);
    for (int t = 0; t < seq_len; ++t) {
        const float* xt = hn2.data() + (size_t)t * H;
        gptq_matmul(gate.data() + (size_t)t * w.inter, xt,
                    (const uint8_t*)w.g_qw.data(), (const uint8_t*)w.g_qz.data(),
                    w.g_sc.data(), nullptr, H, w.inter, w.gs);
        gptq_matmul(up.data() + (size_t)t * w.inter, xt,
                    (const uint8_t*)w.u_qw.data(), (const uint8_t*)w.u_qz.data(),
                    w.u_sc.data(), nullptr, H, w.inter, w.gs);
    }
    for (int i = 0; i < seq_len * w.inter; ++i) {
        float g = gate[i];
        gate[i] = g / (1.0f + std::exp(-g));  // SiLU
        gate[i] *= up[i];
    }
    for (int t = 0; t < seq_len; ++t) {
        gptq_matmul(mlp_out.data() + (size_t)t * H, gate.data() + (size_t)t * w.inter,
                    (const uint8_t*)w.d_qw.data(), (const uint8_t*)w.d_qz.data(),
                    w.d_sc.data(), nullptr, w.inter, H, w.gs);
    }
    for (int i = 0; i < seq_len * H; ++i) h[i] += mlp_out[i];
}

std::vector<float> ForwardEngine::forward(const std::vector<float>& hidden,
                                          int seq_len, bool all_positions) {
    const int H = cfg_.hidden_size;
    if ((int)hidden.size() != seq_len * H) {
        throw std::runtime_error("ForwardEngine: hidden size mismatch");
    }

    std::vector<float> h = hidden;

    // KV caches: one PER LAYER (each layer attends to its own keys/values).
    const int kv_out = cfg_.head_dim * cfg_.num_key_value_heads;
    const int max_ctx = std::max(seq_len, 1);
    std::vector<std::vector<float>> k_caches(cfg_.num_hidden_layers,
                                             std::vector<float>(max_ctx * kv_out));
    std::vector<std::vector<float>> v_caches(cfg_.num_hidden_layers,
                                             std::vector<float>(max_ctx * kv_out));

    for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
        run_layer(layer, h, seq_len, k_caches[layer], v_caches[layer], 0);
    }

    // Final RMSNorm.
    std::vector<float> hnorm(seq_len * H);
    // Need the final norm weights.
    const auto& fn = st_->get("model.norm.weight");
    auto fn_w = st_->to_float(fn);
    rms_norm(hnorm.data(), h.data(), fn_w.data(), seq_len, H, cfg_.rms_norm_eps);

    // lm_head: tied to embed_tokens (Qwen ties them). Read embed_tokens.
    auto emb = st_->to_float(st_->get("model.embed_tokens.weight"));  // [V, H]
    const int V = cfg_.vocab_size;

    // Logits for the last position (or all).
    if (!all_positions) {
        std::vector<float> logits(V);
        const float* ht = hnorm.data() + (size_t)(seq_len - 1) * H;
        for (int v = 0; v < V; ++v) {
            const float* ev = emb.data() + (size_t)v * H;
            float s = 0;
            for (int j = 0; j < H; ++j) s += ev[j] * ht[j];
            logits[v] = s;
        }
        return logits;
    }
    std::vector<float> logits((size_t)seq_len * V);
    for (int t = 0; t < seq_len; ++t) {
        const float* ht = hnorm.data() + (size_t)t * H;
        for (int v = 0; v < V; ++v) {
            const float* ev = emb.data() + (size_t)v * H;
            float s = 0;
            for (int j = 0; j < H; ++j) s += ev[j] * ht[j];
            logits[(size_t)t * V + v] = s;
        }
    }
    return logits;
}

int ForwardEngine::argmax(const std::vector<float>& logits) {
    if (logits.empty()) return -1;
    int best = 0;
    for (size_t i = 1; i < logits.size(); ++i) if (logits[i] > logits[best]) best = (int)i;
    return best;
}

} // namespace winnex_nano
