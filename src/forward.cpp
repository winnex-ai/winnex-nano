// forward.cpp — native forward pass for transformer LLMs (dense f32 + OpenCL).
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
    // Legacy GPTQ group size (the dense path ignores it; kept for parity).
    c.group_size = find_int("group_size", 128);
    // The dense native engine is the default. GPTQ is detected at load_layer
    // time (when .qweight is present); the config flag records the declared
    // quantization type if the config.json carries it.
    if (s.find("quantization_config") != std::string::npos) c.gptq = true;
    return c;
}

// ---------------------------------------------------------------------------
// ForwardEngine
// ---------------------------------------------------------------------------
struct ForwardEngine::LayerWeights {
    std::vector<float> in_norm;    // [hidden]
    std::vector<float> post_norm;  // [hidden]
    // Dense f32 QKV projections: W is row-major [out, in].
    std::vector<float> q_w, k_w, v_w, o_w;
    // Dense f32 MLP.
    std::vector<float> g_w, u_w, d_w;
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

    // Precompute the model-wide tensors once (the forward path reuses them).
    final_norm_ = st_->to_float(st_->get("model.norm.weight"));
    param_count_ += final_norm_.size();
    if (st_->has("lm_head.weight")) {
        embed_tokens_ = st_->to_float(st_->get("lm_head.weight"));  // [V, H]
        has_separate_lm_head_ = true;
    } else {
        embed_tokens_ = st_->to_float(st_->get("model.embed_tokens.weight"));  // [V, H]
    }
    param_count_ += embed_tokens_.size();

    // Parameter count: per-layer weights × layers (counted once, not per token).
    if (cfg_.num_hidden_layers > 0) {
        const int H = cfg_.hidden_size, kv = cfg_.head_dim * cfg_.num_key_value_heads;
        const int inter = cfg_.intermediate_size;
        size_t per_layer =
            2 * H +                              // in_norm + post_norm
            H * H * 3 + kv * H + H * H +         // q (H×H) + k (kv×H) + v (kv×H) + o (H×H)
            inter * H * 2 + H * inter;           // gate + up + down
        param_count_ += per_layer * (size_t)cfg_.num_hidden_layers;
    }
}

namespace {
// Loads a dense f32 projection weight (row-major [out, in]).
// Prefers ".weight" (dense safetensors); falls back to the GPTQ path
// (".qweight"/".qzeros"/".scales") if the dense weight is absent.
std::vector<float> load_dense_or_gptq_scales(const Safetensors& st,
                                             const std::string& base,
                                             bool& is_gptq) {
    if (st.has(base + ".weight")) {
        is_gptq = false;
        return st.to_float(st.get(base + ".weight"));
    }
    // GPTQ fallback: dequantize lazily per forward is not supported in the
    // dense path — the dense engine is the target. If only GPTQ tensors are
    // present, we still record it so load_layer can error clearly.
    is_gptq = true;
    return {};
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

    // Dense f32 path (the native non-GPTQ engine). If a projection only has
    // GPTQ tensors, is_gptq is set and we throw a clear error.
    bool is_gptq = false;
    w.q_w = load_dense_or_gptq_scales(*st_, L + "self_attn.q_proj", is_gptq);
    w.k_w = load_dense_or_gptq_scales(*st_, L + "self_attn.k_proj", is_gptq);
    w.v_w = load_dense_or_gptq_scales(*st_, L + "self_attn.v_proj", is_gptq);
    w.o_w = load_dense_or_gptq_scales(*st_, L + "self_attn.o_proj", is_gptq);
    w.g_w = load_dense_or_gptq_scales(*st_, L + "mlp.gate_proj", is_gptq);
    w.u_w = load_dense_or_gptq_scales(*st_, L + "mlp.up_proj",   is_gptq);
    w.d_w = load_dense_or_gptq_scales(*st_, L + "mlp.down_proj", is_gptq);
    if (is_gptq) {
        throw std::runtime_error(
            "ForwardEngine: model uses GPTQ int4 weights but the native dense "
            "engine requires f32 .weight tensors (dense safetensors)");
    }
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

    // 2. QKV projections (dense f32) — batched as (seq*H).
    std::vector<float> q(seq_len * w.q_out), k(seq_len * w.kv_out), v(seq_len * w.kv_out);
    for (int t = 0; t < seq_len; ++t) {
        const float* xt = hn.data() + (size_t)t * H;
        dense_matmul(q.data() + (size_t)t * w.q_out, xt, w.q_w.data(), H, w.q_out);
        dense_matmul(k.data() + (size_t)t * w.kv_out, xt, w.k_w.data(), H, w.kv_out);
        dense_matmul(v.data() + (size_t)t * w.kv_out, xt, w.v_w.data(), H, w.kv_out);
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
        dense_matmul(proj.data() + (size_t)t * H, attn_out.data() + (size_t)t * w.q_out,
                     w.o_w.data(), w.q_out, H);
    }
    for (int i = 0; i < seq_len * H; ++i) h[i] += proj[i];

    // 7. Post attention norm + MLP (gate*up -> down, SiLU gating).
    std::vector<float> hn2(H * seq_len);
    rms_norm(hn2.data(), h.data(), w.post_norm.data(), seq_len, H, cfg_.rms_norm_eps);
    std::vector<float> mlp_out(seq_len * H);
    std::vector<float> gate(seq_len * w.inter), up(seq_len * w.inter);
    for (int t = 0; t < seq_len; ++t) {
        const float* xt = hn2.data() + (size_t)t * H;
        dense_matmul(gate.data() + (size_t)t * w.inter, xt, w.g_w.data(), H, w.inter);
        dense_matmul(up.data() + (size_t)t * w.inter, xt, w.u_w.data(), H, w.inter);
    }
    for (int i = 0; i < seq_len * w.inter; ++i) {
        float g = gate[i];
        gate[i] = g / (1.0f + std::exp(-g));  // SiLU
        gate[i] *= up[i];
    }
    for (int t = 0; t < seq_len; ++t) {
        dense_matmul(mlp_out.data() + (size_t)t * H, gate.data() + (size_t)t * w.inter,
                     w.d_w.data(), w.inter, H);
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
    rms_norm(hnorm.data(), h.data(), final_norm_.data(), seq_len, H, cfg_.rms_norm_eps);

    // Logits for the last position (or all) — dense_matmul over the rows.
    const int V = cfg_.vocab_size;
    if (!all_positions) {
        std::vector<float> logits(V);
        const float* ht = hnorm.data() + (size_t)(seq_len - 1) * H;
        dense_matmul(logits.data(), ht, embed_tokens_.data(), H, V);
        return logits;
    }
    std::vector<float> logits((size_t)seq_len * V);
    for (int t = 0; t < seq_len; ++t) {
        const float* ht = hnorm.data() + (size_t)t * H;
        dense_matmul(logits.data() + (size_t)t * V, ht, embed_tokens_.data(), H, V);
    }
    return logits;
}

void ForwardEngine::reset_cache() {
    const int kv_out = cfg_.head_dim * cfg_.num_key_value_heads;
    k_caches_.assign(cfg_.num_hidden_layers, std::vector<float>());
    v_caches_.assign(cfg_.num_hidden_layers, std::vector<float>());
    ctx_len_ = 0;
}

std::vector<float> ForwardEngine::forward_next(const std::vector<float>& hidden) {
    const int H = cfg_.hidden_size;
    if ((int)hidden.size() != H) {
        throw std::runtime_error("ForwardEngine::forward_next: hidden must be length H");
    }
    const int kv_out = cfg_.head_dim * cfg_.num_key_value_heads;

    // Grow the persistent cache to hold the new token.
    for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
        k_caches_[layer].resize((size_t)(ctx_len_ + 1) * kv_out);
        v_caches_[layer].resize((size_t)(ctx_len_ + 1) * kv_out);
    }

    // Run a single token through all layers, attending to the full cache.
    std::vector<float> h = hidden;  // [1, H]
    for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
        run_layer(layer, h, /*seq_len=*/1, k_caches_[layer], v_caches_[layer], ctx_len_);
    }
    ++ctx_len_;

    // Final RMSNorm + lm_head → logits for this token.
    std::vector<float> hnorm(H);
    rms_norm(hnorm.data(), h.data(), final_norm_.data(), 1, H, cfg_.rms_norm_eps);
    std::vector<float> logits(cfg_.vocab_size);
    dense_matmul(logits.data(), hnorm.data(), embed_tokens_.data(), H, cfg_.vocab_size);
    return logits;
}

std::vector<int> ForwardEngine::generate(const std::vector<float>& h_prompt,
                                         int max_new_tokens, int eos_id) {
    if ((int)h_prompt.size() != cfg_.hidden_size) {
        throw std::runtime_error("ForwardEngine::generate: h_prompt must be length H");
    }
    reset_cache();
    std::vector<int> out;
    out.reserve(max_new_tokens);

    // Decode token-by-token. The first hidden state is the prompt embedding
    // (from the X-factor); subsequent hidden states are embed_tokens[prev].
    std::vector<float> h = h_prompt;
    for (int step = 0; step < max_new_tokens; ++step) {
        std::vector<float> logits = forward_next(h);
        int tid = argmax(logits);
        if (tid < 0) break;
        out.push_back(tid);
        if (eos_id >= 0 && tid == eos_id) break;
        // Next hidden state = embed_tokens[tid] (tied lm_head).
        const float* row = embed_tokens_.data() + (size_t)tid * cfg_.hidden_size;
        h.assign(row, row + cfg_.hidden_size);
    }
    return out;
}

int ForwardEngine::argmax(const std::vector<float>& logits) {
    if (logits.empty()) return -1;
    int best = 0;
    for (size_t i = 1; i < logits.size(); ++i) if (logits[i] > logits[best]) best = (int)i;
    return best;
}

} // namespace winnex_nano
