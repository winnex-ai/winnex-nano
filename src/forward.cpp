// forward.cpp — native forward pass for transformer LLMs (dense f32 + OpenCL).
#include "winnex_nano/forward.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>

#include "winnex_nano/tensor_ops.hpp"
#include "winnex_nano/forward_gpu.hpp"

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
    //
    // ARCHITECTURE GATE. This engine implements one family: RMSNorm, RoPE,
    // SwiGLU, grouped-query attention — Qwen2/Qwen2.5/Qwen3, Llama, Mistral.
    //
    // Everything else must be REFUSED here, not loaded and run wrong. The
    // tensors of GPT-2/BERT are named differently and simply fail to load
    // (loud, fine). The dangerous case is a model whose tensors DO match but
    // whose maths differs:
    //
    //   gemma2  — GeGLU instead of SwiGLU (gelu(gate)*up), logit softcapping,
    //             alternating local/global attention with a sliding window.
    //   gemma3  — same, plus per-head QK-norm already folded differently.
    //
    // Those load cleanly and produce plausible-looking perplexity that is
    // simply wrong. An earlier version of the Kaggle benchmark listed "gemma"
    // and "gemma2" as supported and would have reported a number for them.
    //
    // A LISTA É O CONTRATO. Ela só pode conter o que o código de fato executa.
    //
    // Medido em 2026-10-23: a lista tinha nove entradas, e quatro delas eram
    // MoE (qwen2_moe, qwen3_moe, mixtral) ou atenção latente (deepseek_v2/v3).
    // O motor não tem uma linha de nenhuma das duas:
    //
    //     $ grep -niE "router|expert|moe|num_experts" src/forward.cpp
    //     82:            "qwen2_moe", "qwen3_moe",      <- só a linha da lista
    //
    // Um Qwen2-MoE guarda mlp.experts.{i}.* e mlp.gate; o loader pede
    // mlp.gate_proj/up_proj/down_proj, que não existem nesses checkpoints. A
    // carga falhava — mas falhava com "tensor não encontrado", não com
    // "arquitetura não suportada", enquanto a lista dizia ao usuário que o
    // modelo era suportado. Um checkpoint MoE densificado ou híbrido passaria e
    // rodaria errado.
    //
    // O gate existe para impedir exatamente isso. Uma lista que anuncia o que o
    // código não faz é pior que não ter lista: transfere ao usuário a tarefa de
    // descobrir que o anúncio era falso.
    {
        static const char* kSupported[] = {
            "qwen2", "qwen3", "llama", "mistral",
        };
        bool ok = false;
        for (const char* a : kSupported) if (c.arch == a) { ok = true; break; }
        if (!ok) {
            throw std::runtime_error(
                "ForwardEngine: architecture '" + c.arch + "' is not implemented. "
                "Supported: qwen2, qwen3, llama, mistral (dense: RMSNorm, RoPE, "
                "SwiGLU, GQA). Deliberately REFUSED rather than run wrong: MoE "
                "(qwen2_moe, qwen3_moe, mixtral) needs a top-k router over N "
                "experts, which this engine does not have; deepseek_v2/v3 need "
                "MLA; gemma needs GeGLU + logit softcapping + sliding window. "
                "All of them would load and produce a plausible-looking "
                "perplexity that is simply wrong.");
        }
    }
    c.vocab_size = find_int("vocab_size", 0);
    c.hidden_size = find_int("hidden_size", 0);
    c.intermediate_size = find_int("intermediate_size", 0);
    c.num_hidden_layers = find_int("num_hidden_layers", 0);
    c.num_attention_heads = find_int("num_attention_heads", 0);
    c.num_key_value_heads = find_int("num_key_value_heads", c.num_attention_heads);
    c.max_position_embeddings = find_int("max_position_embeddings", 0);
    c.rope_theta = find_float("rope_theta", 0.0f);
    c.rms_norm_eps = find_float("rms_norm_eps", -1.0f);
    c.tie_word_embeddings = find_int("tie_word_embeddings", 1) == 1;
    c.sliding_window = find_int("sliding_window", 0);
    c.has_sliding_window = c.sliding_window > 0;

    // rope_scaling: o parser é de chave plana e `rope_scaling` é um objeto
    // aninhado. Procura-se dentro DELE, não no arquivo inteiro -- procurar
    // "factor" no config todo casaria `rope_scaling.factor` e também qualquer
    // outro `factor` que o checkpoint traga (vision_config, quant config).
    c.rope_type = "";
    c.rope_factor = 1.0f;
    {
        size_t rs = s.find("\"rope_scaling\"");
        if (rs != std::string::npos) {
            size_t open = s.find('{', rs);
            size_t close = (open == std::string::npos) ? std::string::npos
                                                       : s.find('}', open);
            if (open != std::string::npos && close != std::string::npos) {
                std::string body = s.substr(open, close - open + 1);
                auto sub_float = [&](const char* key, float def) -> float {
                    std::string pat = std::string("\"") + key + "\"";
                    size_t p = body.find(pat);
                    if (p == std::string::npos) return def;
                    p = body.find(':', p);
                    if (p == std::string::npos) return def;
                    return (float)std::atof(body.c_str() + p + 1);
                };
                auto sub_str = [&](const char* key) -> std::string {
                    std::string pat = std::string("\"") + key + "\"";
                    size_t p = body.find(pat);
                    if (p == std::string::npos) return "";
                    p = body.find(':', p);
                    if (p == std::string::npos) return "";
                    size_t q = body.find('"', p);
                    if (q == std::string::npos) return "";
                    size_t e = body.find('"', q + 1);
                    if (e == std::string::npos) return "";
                    return body.substr(q + 1, e - q - 1);
                };
                // "type" é o nome antigo do campo em alguns checkpoints.
                c.rope_type = sub_str("rope_type");
                if (c.rope_type.empty()) c.rope_type = sub_str("type");
                c.rope_factor = sub_float("factor", 1.0f);
                c.rope_low_freq_factor = sub_float("low_freq_factor", 1.0f);
                c.rope_high_freq_factor = sub_float("high_freq_factor", 4.0f);
                c.rope_original_max_pos =
                    find_int("original_max_position_embeddings", 0);
            }
        }
    }

    // ── DEFAULTS QUE INVENTAM VALORES SÃO FALSOS NEGATIVOS ────────────────
    //
    // These fields used to have plausible defaults: max_position_embeddings
    // 2048, rope_theta 10000, rms_norm_eps 1e-6. A config that omitted them
    // loaded "fine" and ran with a WRONG RoPE base or a WRONG context ceiling —
    // the engine silently behaved like a different model. Found in practice: a
    // checkpoint with a 32k window and no max_position_embeddings was cut at
    // 2048 positions, and the only symptom was an "out of range" error twenty
    // thousand tokens later.
    //
    // The rule now: a REQUIRED value that is absent is an error, not a guess.
    // The legitimate derivations are num_key_value_heads = num_attention_heads
    // (that IS what MHA means when the key is absent) and head_dim =
    // hidden/heads (what Qwen2 and Llama actually use).
    {
        std::string missing;
        auto need = [&](double v, const char* name) {
            if (v <= 0.0) missing += std::string(" ") + name;
        };
        need(c.vocab_size, "vocab_size");
        need(c.hidden_size, "hidden_size");
        need(c.intermediate_size, "intermediate_size");
        need(c.num_hidden_layers, "num_hidden_layers");
        need(c.num_attention_heads, "num_attention_heads");
        need(c.max_position_embeddings, "max_position_embeddings");
        need(c.rope_theta, "rope_theta");
        need(c.rms_norm_eps, "rms_norm_eps");
        if (!missing.empty()) {
            throw std::runtime_error(
                "ForwardEngine: config.json is missing required field(s):" + missing +
                ". These have no safe default — guessing them (2048 positions, "
                "rope_theta 10000) makes the engine run a different model than "
                "the checkpoint describes. Fix the config instead of letting it "
                "load.");
        }
    }

    // A sliding window means the model attends to a truncated history — a
    // different function, and this engine has no windowed attention. Same
    // contract as the architecture gate: refuse rather than run wrong.
    //
    // BUT a window that is >= max_position_embeddings is INERT: it never binds,
    // so the model attends to the full causal history anyway and full attention
    // is exactly right. Qwen2.5-0.5B declares sliding_window=32768 with a 32768
    // context; Qwen2.5-7B declares 131072 with 32768. Neither truncates.
    //
    // This distinction is the whole point of "no false negatives": refusing on
    // the mere PRESENCE of the key would have refused every Qwen2.5 checkpoint —
    // the exact models this engine runs and was validated against. Caught by
    // running it: the first version of this guard rejected Qwen2.5-0.5B.
    if (c.has_sliding_window &&
        c.sliding_window < c.max_position_embeddings) {
        throw std::runtime_error(
            "ForwardEngine: config declares sliding_window=" +
            std::to_string(c.sliding_window) + " against " +
            std::to_string(c.max_position_embeddings) + " positions, so the "
            "window BINDS and this engine cannot honour it: it implements full "
            "causal attention. Running it would attend to history the model was "
            "trained to ignore. (A window >= max_position_embeddings is inert "
            "and is accepted.)");
    }

    // ── rope_scaling: implementado até onde é implementado ────────────────
    //
    // "llama3" é o caso real (Llama-3.1/3.2): as frequências BAIXAS são
    // divididas por `factor`, as altas ficam, e há uma transição suave entre
    // elas. É implementado abaixo em run_layer.
    //
    // Qualquer outro tipo ("linear", "dynamic", "yarn", "longrope", ...) muda
    // a função posicional de uma forma que este código não calcula. Ignorá-lo
    // é a mesma classe de defeito que o motor já corrigiu em `rope_theta` e em
    // `sliding_window`: rodar um modelo diferente sem avisar. Recusa.
    if (!c.rope_type.empty() && c.rope_type != "llama3") {
        throw std::runtime_error(
            "ForwardEngine: rope_scaling.rope_type='" + c.rope_type +
            "' is not implemented. Supported: llama3 (and no scaling at all). "
            "Other types change the positional function in a way this engine "
            "does not compute — running them would produce fluent, wrong text "
            "with no indication of the difference.");
    }
    // HEAD_DIM: prefer the explicit value from config.json, fall back to
    // hidden/heads only when it is absent.
    //
    // BUG FIX (2026-10-21): this used to ALWAYS derive hidden/heads. Qwen3
    // decouples the two — Qwen3-4B has hidden=2560, heads=32 (so 2560/32 = 80)
    // but an explicit head_dim=128, which means q_proj outputs 32*128 = 4096
    // rows, not 2560. Deriving 80 made every projection reshape wrong, the
    // attention garbage, and the forward produced perplexity 1.1e7 on a model
    // that works fine — the logit correlation against transformers was -0.216
    // on a single token, where attention is trivial and only the dimensions
    // matter.
    c.head_dim = find_int("head_dim", 0);
    if (c.head_dim <= 0) c.head_dim = c.hidden_size / c.num_attention_heads;
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
    // QKV BIASES (see the BUG FIX note in load_layer). Qwen2/Qwen2.5 carry
    // attention biases; Llama does not. Empty when the checkpoint has none.
    std::vector<float> q_b, k_b, v_b;
    // The three above concatenated [q_out | kv_out | kv_out] for the fused QKV
    // GEMM. Empty when the checkpoint has no biases at all, which is how the
    // fused path decides whether to pass one to the kernel.
    std::vector<float> qkv_b;
    // PER-HEAD Q/K RMSNorm (Qwen3). Applied over head_dim on each head of Q
    // and K BEFORE RoPE. Empty for architectures that do not use it.
    std::vector<float> q_norm, k_norm;
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
    //
    // lm_head is NOT materialised. For the 7B it is 152064x3584 = 2.18 GB in
    // f32, and the forward only ever needs one row of it at a time (the
    // embedding for a token id) plus the final logits projection — both of
    // which read_rows() serves without a full copy. Materialising it put the
    // constructor 2.2 GB over budget and made the 7B fail before any
    // computation started. has_separate_lm_head_ records that the tensor
    // exists; embed_tokens_ stays empty and the callers use st_ directly.
    has_separate_lm_head_ = st_->has("lm_head.weight");
    if (st_->has("model.embed_tokens.weight")) {
        embed_tokens_shape_h_ = (int)st_->get("model.embed_tokens.weight").shape[1];
        param_count_ += (size_t)st_->get("model.embed_tokens.weight").shape[0] * embed_tokens_shape_h_;
    }

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

    // Decide the CPU weight strategy NOW, not when (or if) a GPU is enabled.
    // This is the fix for the 0.27 tok/s CPU path: the decision was correct and
    // lived in a function a CPU-only caller never calls.
    decide_cpu_residency();
}

namespace {
// Opt-in switch for the Madhava selective top-k attention path. Default OFF:
// softmax is what the pretrained weights expect (see the BUG FIX note in
// run_layer). Set MADHAVA_TOPK_ATTN=1 to measure the delta on a real model.
bool MadhavaTopKAttn() {
    static const bool on = [] {
        const char* v = std::getenv("MADHAVA_TOPK_ATTN");
        return v != nullptr && v[0] != '0';
    }();
    return on;
}

// ── Selective attention with a CONTROLLED LOST-MASS guarantee ───────────────
//
// MADHAVA_TOPK_ATTN=1 enables pruning. Two knobs:
//
//   MADHAVA_EPSILON_MASS  (default 1e-4) — the fraction of softmax mass the
//        pruned keys may account for, at most. This is the quantity that
//        actually bounds the attention error:
//
//            ‖attn_full − attn_pruned‖₁ ≤ 2·(mass of pruned keys)
//
//        (each dropped weight is ≤ its own mass, and both weight vectors sum
//        to 1). So a value ε here is a rigorous bound on how far the layer's
//        output can move — NOT a heuristic score cut.
//
//   MADHAVA_MIN_KEEP      (default 64) — a floor on the number of keys kept,
//        so short sequences behave like the exact softmax.
//
// HOW THE STOPPING TEST WORKS (and an important correction):
//
// The natural first instinct is to bound the score with Cauchy-Schwarz —
// s_i = q·k_i ≤ ‖q‖‖k_i‖ — and stop when exp(‖q‖‖k_i‖ − max)/Z_lower < ε.
// That bound is SOUND but far too loose to prune anything: measured on
// Qwen2.5-0.5B layer 0, real scores sit at ~80 while the CS bound is ~1974,
// a gap of ~1895. The test never fires; every key is kept.
//
// It is also unnecessary. Softmax mass is EXACTLY known once the scores are
// computed, so the pruning decision is made on the true remaining mass rather
// than on a bound of it. The guarantee is therefore exact, not conservative:
// the keys we drop, dropped by construction, sum to < ε of the mass.
//
// Cost note: computing all scores is O(n·d) — the same work as the full
// softmax. What this saves is the VALUE accumulation and the weight writes
// over the selected set, which dominates only when n is large.
float EpsilonMass() {
    static const float e = [] {
        const char* v = std::getenv("MADHAVA_EPSILON_MASS");
        return (v != nullptr && v[0] != '\0') ? std::atof(v) : 1e-4f;
    }();
    return e;
}

int MinKeep() {
    static const int k = [] {
        const char* v = std::getenv("MADHAVA_MIN_KEEP");
        return (v != nullptr && v[0] != '\0') ? std::atoi(v) : 64;
    }();
    return k;
}

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
    // q_proj is [num_attention_heads * head_dim, hidden]. With Qwen3's
    // decoupled head_dim (32*128 = 4096) this is NOT hidden_size (2560):
    // using hidden_size here under-allocated q and the o_proj matmul read
    // past the buffer (measured: malloc(): invalid size, SIGABRT).
    w.q_out = cfg_.num_attention_heads * cfg_.head_dim;
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
    // BUG FIX (attention bias): Qwen2/Qwen2.5 store a bias on q_proj, k_proj
    // and v_proj. This loader read only ".weight", so every projection ran
    // without its bias — and the error compounds over 24 layers. Measured on
    // Qwen2.5-0.5B: v_proj output differed from the reference by 0.098 for a
    // single token, and the forward produced perplexity 9.9e6 against 37.4.
    // Llama has no such bias; the check keeps both architectures correct.
    auto load_bias = [&](const std::string& base, std::vector<float>& dst) {
        if (st_->has(base + ".bias")) dst = st_->to_float(st_->get(base + ".bias"));
        else dst.clear();
    };
    auto load_opt = [&](const std::string& base, std::vector<float>& dst) {
        if (st_->has(base + ".weight")) dst = st_->to_float(st_->get(base + ".weight"));
        else dst.clear();
    };
    load_opt(L + "self_attn.q_norm", w.q_norm);
    load_opt(L + "self_attn.k_norm", w.k_norm);

    load_bias(L + "self_attn.q_proj", w.q_b);
    load_bias(L + "self_attn.k_proj", w.k_b);
    load_bias(L + "self_attn.v_proj", w.v_b);
    // Fused QKV bias, same rule as load_layer_shallow: only when all three are
    // present and sized as expected; otherwise qkv_proj is simply not pushed
    // and run_layer takes the three-kernel path.
    w.qkv_b.clear();
    if (w.q_b.size() == (size_t)w.q_out && w.k_b.size() == (size_t)w.kv_out &&
        w.v_b.size() == (size_t)w.kv_out) {
        w.qkv_b.reserve(w.q_out + 2 * w.kv_out);
        w.qkv_b.insert(w.qkv_b.end(), w.q_b.begin(), w.q_b.end());
        w.qkv_b.insert(w.qkv_b.end(), w.k_b.begin(), w.k_b.end());
        w.qkv_b.insert(w.qkv_b.end(), w.v_b.begin(), w.v_b.end());
    }

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

// Norms and biases only — everything the CPU still needs when the projections
// are resident on the device. Sizes: O(H) per layer, so 28 layers of the 7B
// cost a few MB, against 14 GB for the full layer.
void ForwardEngine::load_layer_shallow(int layer, LayerWeights& w) {
    std::string L = "model.layers." + std::to_string(layer) + ".";
    w.hidden = cfg_.hidden_size;
    w.q_out = cfg_.num_attention_heads * cfg_.head_dim;
    w.kv_out = cfg_.head_dim * cfg_.num_key_value_heads;
    w.inter = cfg_.intermediate_size;
    w.gs = cfg_.group_size;

    w.in_norm = st_->to_float(st_->get(L + "input_layernorm.weight"));
    w.post_norm = st_->to_float(st_->get(L + "post_attention_layernorm.weight"));
    auto load_bias = [&](const std::string& base, std::vector<float>& dst) {
        if (st_->has(base + ".bias")) dst = st_->to_float(st_->get(base + ".bias"));
        else dst.clear();
    };
    auto load_opt = [&](const std::string& base, std::vector<float>& dst) {
        if (st_->has(base + ".weight")) dst = st_->to_float(st_->get(base + ".weight"));
        else dst.clear();
    };
    load_opt(L + "self_attn.q_norm", w.q_norm);
    load_opt(L + "self_attn.k_norm", w.k_norm);
    load_bias(L + "self_attn.q_proj", w.q_b);
    load_bias(L + "self_attn.k_proj", w.k_b);
    load_bias(L + "self_attn.v_proj", w.v_b);
    // Fused QKV bias: only when all three are present and sized as expected.
    // A partial set cannot be folded into the single [out_dim] kernel arg, and
    // the run_layer fallback handles the three-kernel case.
    w.qkv_b.clear();
    if (w.q_b.size() == (size_t)w.q_out && w.k_b.size() == (size_t)w.kv_out &&
        w.v_b.size() == (size_t)w.kv_out) {
        w.qkv_b.reserve(w.q_out + 2 * w.kv_out);
        w.qkv_b.insert(w.qkv_b.end(), w.q_b.begin(), w.q_b.end());
        w.qkv_b.insert(w.qkv_b.end(), w.k_b.begin(), w.k_b.end());
        w.qkv_b.insert(w.qkv_b.end(), w.v_b.begin(), w.v_b.end());
    }
    w.loaded = true;
}

// ── OpenMP thread cap ──────────────────────────────────────────────────────
//
// dense_matmul opens an omp parallel region PER CALL, and the engine makes
// hundreds of calls per token (the lm_head alone used to make 297). On a host
// with other work, the default "use every core" turns into oversubscription and
// the engine collapses. Measured, Qwen2.5-0.5B, 32-token prefill, this build:
//
//     OMP=  1 :  1.40 s  ( 22.8 tok/s)
//     OMP=  2 :  0.86 s  ( 37.3 tok/s)
//     OMP=  4 :  0.61 s  ( 52.6 tok/s)   <- optimum
//     OMP=  8 :  0.83 s  ( 38.6 tok/s)
//     OMP= 16 :  0.79 s  ( 40.8 tok/s)
//     OMP= 28 : 90.35 s  (  0.4 tok/s)   <- 148x collapse
//
// Four threads beating twenty-eight is not a typo: each region launch costs a
// barrier, and with 28 threads the barriers dominate the 4864-wide rows.
//
// So: cap at the measured optimum unless the caller asks otherwise. An explicit
// OMP_NUM_THREADS is always honoured — the sweep above is reproducible that way,
// and a caller with a dedicated machine may legitimately want more.
namespace {
struct ThreadCap {
    ThreadCap() {
        if (std::getenv("OMP_NUM_THREADS") != nullptr) return;  // caller decides
#ifdef _OPENMP
        // `omp_set_num_threads` só existe quando OpenMP foi encontrado. O
        // CMakeLists trata OpenMP como OPCIONAL (o Apple Clang não traz
        // libomp, e exigi-lo quebrava o runner do macOS), mas esta chamada
        // não estava guardada: numa build sem OpenMP o símbolo não resolve e
        // o LINK falha — que é exatamente o que aconteceu no macos-14 da
        // 0.1.35, enquanto Windows e Linux passavam.
        //
        // Sem OpenMP a build é single-thread, então o teto não tem o que
        // limitar: não fazer nada é o comportamento correto, não uma perda.
        omp_set_num_threads(4);
#endif
    }
};
ThreadCap g_thread_cap;
}  // namespace

// ── Phase timers ───────────────────────────────────────────────────────────
// Decode was measured at 84 ms/token with only 22% of it in GPU kernels. The
// remaining 78% is host work, and guessing which part (idle hypotheses tested
// and refuted: KV-cache resize 0.08 ms, shallow reload 0.09 ms, dense_matmul
// 6.6 ms) is how a week disappears. Enabled by WINNEX_PHASE_TIMERS=1.
namespace {
struct PhaseTimers {
    bool on = std::getenv("WINNEX_PHASE_TIMERS") != nullptr;
    double rms = 0, rope = 0, attn = 0, attn_gpu = 0, resid = 0, mlp_epi = 0, lmh = 0;
    int n = 0;
    ~PhaseTimers() {
        if (!on || n == 0) return;
        std::fprintf(stderr,
            "[phase] n=%d  rms=%.2f rope=%.2f attn=%.2f (gpu=%.2f) resid=%.2f "
            "mlp_epi=%.2f lm_head=%.2f  (total=%.2f ms/token)\n",
            n, rms / n, rope / n, attn / n, attn_gpu / n, resid / n, mlp_epi / n, lmh / n,
            (rms + rope + attn + resid + mlp_epi + lmh) / n);
    }
};
PhaseTimers g_phase;
struct ScopedTimer {
    PhaseTimers& t; double& slot; bool on; std::chrono::steady_clock::time_point a;
    ScopedTimer(PhaseTimers& tt, double& s, bool o)
      : t(tt), slot(s), on(o), a(std::chrono::steady_clock::now()) {}
    ~ScopedTimer() {
        if (on) slot += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - a).count();
    }
};
}  // namespace

float ForwardEngine::RoPEInvFreq(int j) const {
    const int hd = cfg_.head_dim;
    const float base = std::pow(cfg_.rope_theta, -2.0f * j / (float)hd);
    if (cfg_.rope_type != "llama3") return base;

    // llama3 RoPE. Porta fiel de modeling_rope_utils._compute_llama3_parameters:
    // as duas metades (frequência alta e baixa) usam `inv_freq` DIFERENTES, e a
    // transição é uma interpolação, não um corte. A versão anterior deste
    // código não existia — `rope_scaling` era ignorado — e no Llama-3.2 isso
    // deixava 15 das 32 dimensões com a invariante errada por um fator de 32.
    const double theta = (double)cfg_.rope_theta;
    const double inv = std::pow(theta, -2.0 * j / (double)hd);
    const double factor = (double)cfg_.rope_factor;
    const double lo_f = (double)cfg_.rope_low_freq_factor;
    const double hi_f = (double)cfg_.rope_high_freq_factor;
    const double old_ctx = (double)(cfg_.rope_original_max_pos > 0
                                    ? cfg_.rope_original_max_pos
                                    : cfg_.max_position_embeddings);
    const double low_wavelen = old_ctx / lo_f;
    const double high_wavelen = old_ctx / hi_f;
    // pi como literal, não M_PI: `M_PI` não existe no MSVC sem
    // `_USE_MATH_DEFINES`, e o padrão C++ não a exige em <cmath>. O build
    // quebrou no runner do Windows por causa desta linha — o Linux passou e
    // o macOS/Windows não, que é a assinatura de uma macro POSIX usada como
    // se fosse padrão.
    constexpr double kPi = 3.14159265358979323846;
    const double wavelen = 2.0 * kPi / inv;

    double inv_llama = (wavelen > low_wavelen) ? inv / factor : inv;
    if (!(wavelen < high_wavelen) && !(wavelen > low_wavelen)) {
        // Faixa de transição. O sinal do smooth_factor é o do código oficial:
        // (old_ctx/wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor).
        const double smooth = (old_ctx / wavelen - lo_f) / (hi_f - lo_f);
        inv_llama = (1.0 - smooth) * (inv_llama / factor) + smooth * inv_llama;
    }
    return (float)inv_llama;
}

void ForwardEngine::run_layer(int layer, std::vector<float>& h, int seq_len,
                              std::vector<float>& k_cache, std::vector<float>& v_cache,
                              int cache_pos) {
    // Both paths prefer the RESIDENT copy when there is one.
    //
    // The comment here used to say the CPU path "keeps the historical per-call
    // load (it costs nothing measurable)". That was wrong, and it was the whole
    // CPU bottleneck: load_layer() re-reads seven matrices from the checkpoint
    // per layer, per token. Measured on the published 0.1.34 wheel: 0.25 tok/s
    // on a 32-token prefill.
    LayerWeights w_scratch;
    const bool can_resident = layer < (int)resident_.size() &&
                              resident_[(size_t)layer] != nullptr;
    LayerWeights& w = can_resident ? *resident_[(size_t)layer] : w_scratch;
    if (&w == &w_scratch) {
        if (device_weights_) load_layer_shallow(layer, w);
        else                 load_layer(layer, w);
    }
    const int H = w.hidden, nh = cfg_.num_attention_heads, nkv = cfg_.num_key_value_heads;
    const int hd = cfg_.head_dim;

    // 1. Input layernorm (RMSNorm).
    std::vector<float> hn(H * seq_len);
    { ScopedTimer _t(g_phase, g_phase.rms, g_phase.on && seq_len == 1);
      rms_norm(hn.data(), h.data(), w.in_norm.data(), seq_len, H, cfg_.rms_norm_eps); }

    // 2. QKV projections (dense f32) — batched as (seq*H).
    std::vector<float> q(seq_len * w.q_out), k(seq_len * w.kv_out), v(seq_len * w.kv_out);
    if (gpu_ != nullptr) {
        // GPU path: one batched call per projection instead of seq_len calls.
        // The bias is fused into the kernel, so the CPU-side add loops below
        // are not executed here.
        //
        // Q/K/V have different output widths (q_out = heads*head_dim,
        // kv_out = kv_heads*head_dim) so they cannot be a single GEMM without
        // padding buffers to the larger width and slicing after — the padding
        // costs more bandwidth than the three separate launches.
        //
        // The `tag` is the safetensors tensor name. The device cache keys on
        // it, so the weights stay resident on the GPU even when the host copy
        // is streamed in per layer through a reused scratch buffer — which is
        // the only way the 7B gets GPU speed at all (13.1 GB bf16 of device
        // weights, host RAM stays at ~2.3 GB anonymous).
        if (device_weights_) {
        // Q/K/V FUSED — same shape of win as the gate+up fusion below, and the
        // previous comment here ("they cannot be a single GEMM without padding
        // buffers") was true only for a PADDED concat. With a ragged one it is
        // exact: the three output widths are just stacked, [q_out | kv_out |
        // kv_out], and the kernel is a plain row-major [sum_out, H] GEMM. No
        // padding, no slicing, no extra kernel.
        //
        // This is the lever the repo's own measurement pointed at: "the cost is
        // not the transfer lock, it is one clFinish per projection (7 x 28 =
        // 196 round-trips per token)" (29e784c). Fusing QKV takes the layer
        // from 7 round-trips to 5. The weights are concatenated ONCE, at
        // upload, by warm_device_weights(), and cached under one tag.
        //
        // The fused path needs a single bias vector folded into the kernel. The
        // three biases have different lengths, so they are concatenated the same
        // way; when the checkpoint has none (Qwen3, Llama) we pass no bias and
        // the three separate kernels below keep handling the all-zero case.
        if (gpu_->has_weight(("L" + std::to_string(layer) + ".qkv_proj").c_str())) {
            const int qkv_out = w.q_out + 2 * w.kv_out;
            std::vector<float> qkv((size_t)seq_len * qkv_out);
            gpu_->matmul_device(("L" + std::to_string(layer) + ".qkv_proj").c_str(),
                                hn.data(), seq_len, qkv_out, H,
                                (w.q_b.size() + w.k_b.size() + w.v_b.size())
                                    ? w.qkv_b.data() : nullptr,
                                qkv.data());
            for (int t = 0; t < seq_len; ++t) {
                const float* row = qkv.data() + (size_t)t * qkv_out;
                std::memcpy(q.data() + (size_t)t * w.q_out, row,
                            (size_t)w.q_out * sizeof(float));
                std::memcpy(k.data() + (size_t)t * w.kv_out, row + w.q_out,
                            (size_t)w.kv_out * sizeof(float));
                std::memcpy(v.data() + (size_t)t * w.kv_out, row + w.q_out + w.kv_out,
                            (size_t)w.kv_out * sizeof(float));
            }
        } else {
            gpu_->matmul_device(("L" + std::to_string(layer) + ".q_proj").c_str(),
                                hn.data(), seq_len, w.q_out, H,
                                w.q_b.empty() ? nullptr : w.q_b.data(), q.data());
            gpu_->matmul_device(("L" + std::to_string(layer) + ".k_proj").c_str(),
                                hn.data(), seq_len, w.kv_out, H,
                                w.k_b.empty() ? nullptr : w.k_b.data(), k.data());
            gpu_->matmul_device(("L" + std::to_string(layer) + ".v_proj").c_str(),
                                hn.data(), seq_len, w.kv_out, H,
                                w.v_b.empty() ? nullptr : w.v_b.data(), v.data());
        }
        } else {
        gpu_->matmul(hn.data(), seq_len, w.q_w.data(), w.q_out, H,
                     w.q_b.empty() ? nullptr : w.q_b.data(), q.data(),
                     ("L" + std::to_string(layer) + ".q_proj").c_str());
        gpu_->matmul(hn.data(), seq_len, w.k_w.data(), w.kv_out, H,
                     w.k_b.empty() ? nullptr : w.k_b.data(), k.data(),
                     ("L" + std::to_string(layer) + ".k_proj").c_str());
        gpu_->matmul(hn.data(), seq_len, w.v_w.data(), w.kv_out, H,
                     w.v_b.empty() ? nullptr : w.v_b.data(), v.data(),
                     ("L" + std::to_string(layer) + ".v_proj").c_str());
        }
    } else {
        for (int t = 0; t < seq_len; ++t) {
            const float* xt = hn.data() + (size_t)t * H;
            float* qt = q.data() + (size_t)t * w.q_out;
            float* kt = k.data() + (size_t)t * w.kv_out;
            float* vt = v.data() + (size_t)t * w.kv_out;
            dense_matmul(qt, xt, w.q_w.data(), H, w.q_out);
            dense_matmul(kt, xt, w.k_w.data(), H, w.kv_out);
            dense_matmul(vt, xt, w.v_w.data(), H, w.kv_out);
            // Attention biases (Qwen2/Qwen2.5). No-op when the checkpoint has
            // none. The GPU path fuses this into the kernel.
            if (!w.q_b.empty()) for (int o = 0; o < w.q_out; ++o) qt[o] += w.q_b[o];
            if (!w.k_b.empty()) for (int o = 0; o < w.kv_out; ++o) kt[o] += w.k_b[o];
            if (!w.v_b.empty()) for (int o = 0; o < w.kv_out; ++o) vt[o] += w.v_b[o];
        }
    }

    // 3. RoPE on Q (all heads) and K (kv heads) at their absolute positions.
    //    Q layout: [t, nh*hd]; K layout: [t, nkv*hd].
    //
    //    CONVENTION (Qwen2 / Llama / HF "rotate_half"), which is NOT the naive
    //    interleaved-pair rotation:
    //
    //      inv_freq[j] = theta^(-2j/hd)        for j = 0 .. hd/2-1
    //      out[j]         = x[j]   *cos(p·inv_freq[j]) - x[j+hd/2]*sin(...)
    //      out[j + hd/2]  = x[j+hd/2]*cos(...)         + x[j]       *sin(...)
    //
    //    Two errors this replaces (measured: perplexity 2.5e7 vs 37.4 against
    //    transformers on Qwen2.5-0.5B — correlation of logits ≈ 0 at every
    //    position, i.e. the forward did not reproduce the model at all):
    //
    //      (a) the exponent used -2j/hd with j stepping by 2 over [0, hd),
    //          which yields 34 distinct frequencies instead of hd/2 = 32, and
    //          shifts every frequency by a factor of two (j=2 gave 0.4217
    //          where the model uses 0.6494);
    //      (b) the pairs were rotated as (j, j+1) — INTERLEAVED — whereas the
    //          model pairs element j with element j+hd/2 ("rotate_half").
    //          Interleaved and rotate_half are different bases of the same
    //          dimension, so mixing them scrambles the rotation entirely.
    //
    //    The precomputed table below is theta-independent per position and is
    //    cached for the sequence, so the pow()/cos()/sin() are evaluated once
    //    per (position, frequency) instead of once per (position, head, freq).
    // PER-HEAD Q/K RMSNorm (Qwen3) — applied over head_dim, per head, BEFORE
    // RoPE. Without it the attention scores are scaled differently from the
    // reference: measured on Qwen3-4B, logit correlation was 1.000000 for a
    // single token (attention trivial, norms irrelevant) and ~0.47 for n>=2 —
    // exactly the signature of a missing attention-time normalisation.
    if (!w.q_norm.empty() || !w.k_norm.empty()) {
        for (int t = 0; t < seq_len; ++t) {
            for (int hh = 0; hh < nh; ++hh) {
                float* hq = q.data() + ((size_t)t * nh + hh) * hd;
                float ss = 0.0f;
                for (int j = 0; j < hd; ++j) ss += hq[j] * hq[j];
                const float inv = 1.0f / std::sqrt(ss / hd + cfg_.rms_norm_eps);
                if (!w.q_norm.empty()) for (int j = 0; j < hd; ++j) hq[j] = hq[j] * inv * w.q_norm[j];
                else                   for (int j = 0; j < hd; ++j) hq[j] = hq[j] * inv;
            }
            for (int hh = 0; hh < nkv; ++hh) {
                float* hk = k.data() + ((size_t)t * nkv + hh) * hd;
                float ss = 0.0f;
                for (int j = 0; j < hd; ++j) ss += hk[j] * hk[j];
                const float inv = 1.0f / std::sqrt(ss / hd + cfg_.rms_norm_eps);
                if (!w.k_norm.empty()) for (int j = 0; j < hd; ++j) hk[j] = hk[j] * inv * w.k_norm[j];
                else                   for (int j = 0; j < hd; ++j) hk[j] = hk[j] * inv;
            }
        }
    }

    { ScopedTimer _t(g_phase, g_phase.rope, g_phase.on && seq_len == 1);
        const int half = hd / 2;
        std::vector<float> cosT((size_t)seq_len * half), sinT((size_t)seq_len * half);
        for (int t = 0; t < seq_len; ++t) {
            const int abs_pos = cache_pos + t;
            for (int j = 0; j < half; ++j) {
                const float inv = RoPEInvFreq(j);
                const float ang = (float)abs_pos * inv;
                cosT[(size_t)t * half + j] = std::cos(ang);
                sinT[(size_t)t * half + j] = std::sin(ang);
            }
        }
        auto rope_apply = [&](float* row, int t) {
            const float* c = cosT.data() + (size_t)t * half;
            const float* s = sinT.data() + (size_t)t * half;
            for (int j = 0; j < half; ++j) {
                const float x1 = row[j], x2 = row[j + half];
                row[j]        = x1 * c[j] - x2 * s[j];
                row[j + half] = x2 * c[j] + x1 * s[j];
            }
        };
        for (int t = 0; t < seq_len; ++t) {
            for (int hh = 0; hh < nh; ++hh)
                rope_apply(q.data() + ((size_t)t * nh + hh) * hd, t);
            for (int hh = 0; hh < nkv; ++hh)
                rope_apply(k.data() + ((size_t)t * nkv + hh) * hd, t);
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

    // ── ATENÇÃO NA GPU ───────────────────────────────────────────────────
    //
    // O caminho CPU abaixo foi, medido, a causa de 29x de desvantagem no
    // prefill: para cada (token, cabeça) um laço serial de hd sobre todas as
    // chaves. Aqui o mesmo cálculo — softmax causal exato, não top-k — vai
    // para o device num único launch por camada.
    //
    // Paridade verificada nos buffers reais: 1,9e-06 contra a referência em
    // numpy, e o estado escondido das camadas 0..2 difere em menos de 6e-06
    // entre os dois caminhos.
    { ScopedTimer _t(g_phase, g_phase.attn, g_phase.on && seq_len == 1);
    const bool attn_gpu = gpu_ != nullptr && gpu_->has_attention();
    // The resident path is preferred whenever the buffers exist: it ships only
    // the new rows instead of the whole cache. Both paths produce the same
    // numbers (parity verified against transformers), so this is a pure
    // bandwidth decision.
    // ── KV cache residente no device: implementado, exato, e MAIS LENTO ───
    //
    // A hipótese era que os 48 reenvios bloqueantes do cache por token fossem o
    // gargalo. Medido A/B no MESMO binário, ctx=512, 24 tokens por braço:
    //
    //   residente  192 ms/token   (157 -> 192, +22%)
    //   reenvio    157 ms/token   (baseline)
    //
    // Paridade exata nos dois (cos=1.0 contra transformers, PPL 48,0367 vs
    // 48,0341 em ambos). Ou seja: a correção está certa e não paga.
    //
    // A leitura que sobrou dos números: o tempo medido por chamada de atenção
    // (10,8 ms) NÃO é transferência — é o host esperando trabalho já enfileirado
    // dos matmuls. Por isso a soma das 24 chamadas (259 ms) excede o token
    // inteiro (111 ms). O número sólido é o tempo de kernel, medido por evento:
    // ~55 ms de 111 (49%) é compute de device, e o resto é host + sync.
    //
    // Fica opt-in (WINNEX_KV_RESIDENT=1) porque o código é correto e serve de
    // base para a versão que reduzir o número de launches, não só os bytes.
    static const bool kv_resident_ok = [] {
        const char* v = std::getenv("WINNEX_KV_RESIDENT");
        return v && std::string(v) == "1";
    }();
    const bool attn_res = attn_gpu && kv_resident_ok && gpu_->has_kv(layer);
    if (g_phase.on && seq_len == 1) {
        // Split the attention block: the KV-cache memcpy below is separate
        // from the device call, and they were being charged together.
        auto _a = std::chrono::steady_clock::now();
        if (attn_res) {
            // k_cache/v_cache hold the WHOLE cache; the device needs only the
            // rows being appended, or position cache_pos would receive row 0.
            // (This was the first version's bug: prefill was exact because
            // cache_pos == 0, decode diverged silently.)
            const size_t newrow = (size_t)cache_pos * (size_t)w.kv_out;
            gpu_->attention_resident(layer, q.data(), k_cache.data() + newrow,
                                     v_cache.data() + newrow,
                                     attn_out.data(), seq_len, nh, nkv, hd, cache_pos);
            g_phase.attn_gpu += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - _a).count();
        } else if (attn_gpu) {
            gpu_->attention(q.data(), k_cache.data(), v_cache.data(), attn_out.data(),
                            seq_len, cache_pos + seq_len, nh, nkv, hd, cache_pos);
            g_phase.attn_gpu += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - _a).count();
        }
    } else if (attn_res) {
        const size_t newrow = (size_t)cache_pos * (size_t)w.kv_out;
        gpu_->attention_resident(layer, q.data(), k_cache.data() + newrow,
                                 v_cache.data() + newrow,
                                 attn_out.data(), seq_len, nh, nkv, hd, cache_pos);
    } else if (attn_gpu) {
        gpu_->attention(q.data(), k_cache.data(), v_cache.data(), attn_out.data(),
                        seq_len, cache_pos + seq_len, nh, nkv, hd, cache_pos);
    } else {
    const float scale = 1.0f / std::sqrt((float)hd);
    for (int t = 0; t < seq_len; ++t) {
        for (int hh = 0; hh < nh; ++hh) {
            int kvh = hh / (nh / nkv);  // GQA mapping
            const float* qh = q.data() + ((size_t)t * nh + hh) * hd;
            // Scores over ctx_len keys.
            //
            // BUG FIX (see perplexity note below): this block used to run the
            // "Madhava principle" — keep only the top-16 keys by raw score and
            // weight them as score/Σscore, with NO softmax.
            //
            // That is not the attention the pretrained weights were trained
            // with, and two of its properties are fatal on real models:
            //
            //   (a) TRUNCATION.  Qwen2.5-0.5B attends over every visible key.
            //       Keeping 16 silently discards context as soon as the
            //       sequence is longer than 16 tokens; the model then sees a
            //       different function than the one whose weights were loaded.
            //
            //   (b) NOT A DISTRIBUTION.  Attention scores are signed.  With
            //       w_i = s_i / Σ s, weights are unbounded, can be negative,
            //       and wsum can pass through 0 — at which point the result is
            //       noise amplified by 1e9.  Softmax is exponentiation for a
            //       reason: it maps ℝ → (0,1) and always sums to 1.
            //
            // Measured consequence on Qwen2.5-0.5B (3 texts, real weights):
            // forward perplexity 2.0e7 vs 37.4 for the reference — and the
            // logit correlation against transformers was ≈ 0 at every
            // position, i.e. the forward did not reproduce the model at all.
            //
            // The stack's own claim is that selective top-k *can* replace this
            // "for correctness first".  It cannot — not until the model is
            // trained with it.  Softmax is restored as the default; the
            // top-k path stays available behind MADHAVA_TOPK_ATTN for the
            // experiment that wants to measure the delta honestly.
            // CAUSAL MASK. Token t may attend to keys 0..(cache_pos + t) only —
            // never to itself-in-the-future. The loop used to run to `ctx_len`
            // (= cache_pos + seq_len, the END of the chunk) for every t, so in
            // a prefill of n > 1 tokens each position saw the positions that
            // come after it. The model is trained autoregressive; leaking the
            // future makes the attention a different function.
            //
            // Measured (Qwen2.5-0.5B, last-position logits vs transformers):
            //   n=1 corr 1.000000 | n=2 corr 0.970 | n=3 corr 0.369 | n=4 0.609
            // — exactly the signature of future leakage, and it vanished once
            // `lim` was introduced. With the mask, correlation is 1.0 at every
            // length and the perplexity matches the reference.
            const int lim = cache_pos + t + 1;
            std::vector<float> sc(lim);
            for (int u = 0; u < lim; ++u) {
                const float* kh = k_cache.data() + ((size_t)u * nkv + kvh) * hd;
                float s = 0;
                for (int j = 0; j < hd; ++j) s += qh[j] * kh[j];
                sc[u] = s * scale;
            }
            float* ot = attn_out.data() + ((size_t)t * nh + hh) * hd;
            for (int j = 0; j < hd; ++j) ot[j] = 0.0f;

            // Max for numerical stability (shared by both paths).
            float m = sc[0];
            for (int u = 1; u < lim; ++u) if (sc[u] > m) m = sc[u];

            // Select the surviving key set. Full softmax by default; the
            // controlled-lost-mass rule when MADHAVA_TOPK_ATTN=1.
            //
            // The rule: sort by score, keep growing the surviving set until the
            // mass it accounts for is within ε of the full mass — i.e. until
            // (1 − Z_kept/Z) < ε. That is an EXACT statement about the mass
            // actually dropped, so the attention error is bounded by 2ε (see
            // EpsilonMass above). No bound-of-the-score is involved, because
            // the true mass is available for free once the scores exist.
            int keep = lim;
            std::vector<std::pair<float,int>> ranked;   // (score, key) desc
            if (MadhavaTopKAttn() && lim > MinKeep()) {
                ranked.resize(lim);
                for (int u = 0; u < lim; ++u) ranked[u] = {sc[u], u};
                std::sort(ranked.begin(), ranked.end(),
                          [](auto&a, auto&b){return a.first > b.first;});
                // Z over the whole window, and the target kept-mass.
                double z_all = 0.0;
                for (int u = 0; u < lim; ++u) z_all += std::exp((double)sc[u] - m);
                const double z_target = (1.0 - (double)EpsilonMass()) * z_all;
                double z_kept = 0.0;
                int k = 0;
                for (; k < lim; ++k) {
                    z_kept += std::exp((double)ranked[k].first - m);
                    if (k + 1 >= MinKeep() && z_kept >= z_target) break;
                }
                keep = std::min(lim, k + 1);
            }

            if (keep == lim) {
                // Standard causal softmax attention (the model's own function).
                float den = 0;
                for (int u = 0; u < lim; ++u) { sc[u] = std::exp(sc[u] - m); den += sc[u]; }
                const float inv = den > 0 ? 1.0f / den : 0.0f;
                for (int u = 0; u < lim; ++u) {
                    const float wgt = sc[u] * inv;
                    if (wgt == 0.0f) continue;
                    const float* vh = v_cache.data() + ((size_t)u * nkv + kvh) * hd;
                    for (int j = 0; j < hd; ++j) ot[j] += wgt * vh[j];
                }
            } else {
                // Selective path: weights are renormalised over the KEPT set, so
                // they still form a distribution that sums to 1.
                double den = 0.0;
                for (int i = 0; i < keep; ++i)
                    den += std::exp((double)ranked[i].first - m);
                const float inv = den > 0 ? (float)(1.0 / den) : 0.0f;
                for (int i = 0; i < keep; ++i) {
                    const float wgt = std::exp(ranked[i].first - m) * inv;
                    const int u = ranked[i].second;
                    const float* vh = v_cache.data() + ((size_t)u * nkv + kvh) * hd;
                    for (int j = 0; j < hd; ++j) ot[j] += wgt * vh[j];
                }
            }
        }
    }
    }  // fim do caminho CPU de atenção

    }  // ScopedTimer(attn)

    // 6. Output projection + residual.
    std::vector<float> proj(seq_len * H);
    if (gpu_ != nullptr) {
        // Fourth GPU projection of the layer. o_proj has no bias in any of the
        // supported architectures (Qwen2/2.5/3, Llama), so the fused-bias path
        // is not exercised here.
        if (device_weights_)
            gpu_->matmul_device(("L" + std::to_string(layer) + ".o_proj").c_str(),
                                attn_out.data(), seq_len, H, w.q_out,
                                nullptr, proj.data());
        else
        gpu_->matmul(attn_out.data(), seq_len, w.o_w.data(), H, w.q_out,
                     nullptr, proj.data(),
                     ("L" + std::to_string(layer) + ".o_proj").c_str());
    } else {
        for (int t = 0; t < seq_len; ++t) {
            dense_matmul(proj.data() + (size_t)t * H, attn_out.data() + (size_t)t * w.q_out,
                         w.o_w.data(), w.q_out, H);
        }
    }
    { ScopedTimer _t(g_phase, g_phase.resid, g_phase.on && seq_len == 1);
      for (int i = 0; i < seq_len * H; ++i) h[i] += proj[i]; }

    // 7. Post attention norm + MLP (gate*up -> down, SiLU gating).
    std::vector<float> hn2(H * seq_len);
    rms_norm(hn2.data(), h.data(), w.post_norm.data(), seq_len, H, cfg_.rms_norm_eps);
    std::vector<float> mlp_out(seq_len * H);
    std::vector<float> gate(seq_len * w.inter), up(seq_len * w.inter);
    if (gpu_ != nullptr) {
        // gate and up share the same input A and have the same output width.
        //
        // FUSED (2026-10-24): they were two matmul_device calls, which sent A to
        // the device twice per layer and launched two kernels. Since
        // warm_device_weights() now uploads a single [2·inter, H] buffer under
        // the tag "gate_up_proj", this is ONE call: A crosses the bus once, one
        // kernel runs, and the result comes back as [seq, 2·inter] with gate in
        // columns [0, inter) and up in [inter, 2·inter).
        //
        // This is the fusion the Madhava engine already demonstrates in
        // qkt_fused_topk: compute both stages against the same loaded operand
        // in one pass. Gain is ~2% of the forward (one call out of 7 per layer)
        // — real, small, and it belongs behind the big items, not in front.
        if (device_weights_ && gpu_->has_weight(("L" + std::to_string(layer) + ".gate_up_proj").c_str())) {
            std::vector<float> gu((size_t)seq_len * 2 * w.inter);
            gpu_->matmul_device(("L" + std::to_string(layer) + ".gate_up_proj").c_str(),
                                hn2.data(), seq_len, 2 * w.inter, H, nullptr, gu.data());
            for (int t = 0; t < seq_len; ++t) {
                std::memcpy(gate.data() + (size_t)t * w.inter,
                            gu.data() + (size_t)t * 2 * w.inter,
                            (size_t)w.inter * sizeof(float));
                std::memcpy(up.data() + (size_t)t * w.inter,
                            gu.data() + (size_t)t * 2 * w.inter + w.inter,
                            (size_t)w.inter * sizeof(float));
            }
        } else if (device_weights_) {
            gpu_->matmul_device(("L" + std::to_string(layer) + ".gate_proj").c_str(),
                                hn2.data(), seq_len, w.inter, H, nullptr, gate.data());
            gpu_->matmul_device(("L" + std::to_string(layer) + ".up_proj").c_str(),
                                hn2.data(), seq_len, w.inter, H, nullptr, up.data());
        } else {
        gpu_->matmul(hn2.data(), seq_len, w.g_w.data(), w.inter, H, nullptr, gate.data(),
                     ("L" + std::to_string(layer) + ".gate_proj").c_str());
        gpu_->matmul(hn2.data(), seq_len, w.u_w.data(), w.inter, H, nullptr, up.data(),
                     ("L" + std::to_string(layer) + ".up_proj").c_str());
        }
    } else {
        for (int t = 0; t < seq_len; ++t) {
            const float* xt = hn2.data() + (size_t)t * H;
            dense_matmul(gate.data() + (size_t)t * w.inter, xt, w.g_w.data(), H, w.inter);
            dense_matmul(up.data() + (size_t)t * w.inter, xt, w.u_w.data(), H, w.inter);
        }
    }
    { ScopedTimer _t(g_phase, g_phase.mlp_epi, g_phase.on && seq_len == 1);
      for (int i = 0; i < seq_len * w.inter; ++i) {
        float g = gate[i];
        gate[i] = g / (1.0f + std::exp(-g));  // SiLU (CPU, elementwise)
        gate[i] *= up[i];
      } }
    if (gpu_ != nullptr) {
        if (device_weights_)
            gpu_->matmul_device(("L" + std::to_string(layer) + ".down_proj").c_str(),
                                gate.data(), seq_len, H, w.inter, nullptr, mlp_out.data());
        else
        gpu_->matmul(gate.data(), seq_len, w.d_w.data(), H, w.inter, nullptr, mlp_out.data(),
                     ("L" + std::to_string(layer) + ".down_proj").c_str());
    } else {
        for (int t = 0; t < seq_len; ++t) {
            dense_matmul(mlp_out.data() + (size_t)t * H, gate.data() + (size_t)t * w.inter,
                         w.d_w.data(), w.inter, H);
        }
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
    std::vector<float> logits((size_t)seq_len * V);
    if (!all_positions) {
        // Only the last position is asked for, but lm_head_logits fills a
        // [n_positions, V] block — pass just that row.
        const float* ht = hnorm.data() + (size_t)(seq_len - 1) * H;
        lm_head_logits(ht, logits.data(), 1, H);
        logits.resize((size_t)V);
        return logits;
    }
    lm_head_logits(hnorm.data(), logits.data(), seq_len, H);
    return logits;
}


// ── lm_head, streamed ──────────────────────────────────────────────────────
//
// The embedding table is the single largest tensor in the checkpoint (152064
// x 3584 = 2.18 GB f32 for the 7B) and the forward needs it in two places:
// one row per decoded token, and the full logits projection once per forward.
// Materialising it costs 2.2 GB of anonymous RAM for the whole process life.
//
// Both uses go through the offset table instead: a row read touches one row of
// bytes, and the logits projection converts BLOCK rows at a time. Peak RSS for
// the table drops from 2.18 GB to one block (512 rows, 7.3 MB).
static const char* kLmHeadName = "lm_head.weight";
static const char* kEmbedName  = "model.embed_tokens.weight";

const SafeTensor& ForwardEngine::lm_head_tensor() const {
    return st_->has(kLmHeadName) ? st_->get(kLmHeadName) : st_->get(kEmbedName);
}

void ForwardEngine::lm_head_row(int token_id, float* out) const {
    const SafeTensor& t = lm_head_tensor();
    const int H = cfg_.hidden_size;
    std::vector<float> row = st_->read_rows(t, token_id, 1);
    std::memcpy(out, row.data(), (size_t)H * sizeof(float));
}

void ForwardEngine::lm_head_logits(const float* hnorm, float* logits,
                                   int n_positions, int stride) const {
    const SafeTensor& t = lm_head_tensor();
    const int H = cfg_.hidden_size;
    const int V = (int)t.shape[0];
    // Block size is a throughput/memory trade, and it depends on how many
    // positions are being projected — which is why a single constant was wrong
    // in both directions:
    //
    //   n_positions == 1 (decode, the hot path): the block is walked once per
    //   token. Too big and the table streams through the cache every token:
    //   measured on Qwen2.5-0.5B, lm_head went from 47 ms to 160.9 ms/token
    //   when the block was raised to ~14k rows. Small blocks win here because
    //   the working set stays hot and the per-block OpenMP launch is amortised
    //   over the next token's identical block.
    //
    //   n_positions > 1 (prefill, and the CPU quality harness): every block is
    //   reused n_positions times, so the launch overhead dominates and big
    //   blocks win — measured whole-table on CPU: BLOCK=512 took 7.91 s against
    //   0.93 s at BLOCK=4096 for the same logits.
    //
    // So: small and cache-friendly for decode, large and launch-efficient for
    // multi-position calls. Overridable for measurement.
    int BLOCK;
    if (const char* env = std::getenv("WINNEX_LMHEAD_BLOCK")) {
        BLOCK = std::max(64, std::atoi(env));
    } else if (n_positions <= 1) {
        BLOCK = 512;
    } else {
        BLOCK = (int)std::min<size_t>(16384, std::max<size_t>(
            2048, (48u * 1024u * 1024u) / ((size_t)H * sizeof(float))));
    }
    for (int v0 = 0; v0 < V; v0 += BLOCK) {
        const int n = std::min(BLOCK, V - v0);
        std::vector<float> block = st_->read_rows(t, v0, n);   // no second copy
        // dense_matmul (AVX2/FMA + OpenMP over rows), not a hand-written loop:
        // the scalar version measured 30.5 s against 9.1 s for the whole
        // forward, because the logits projection is 152064x3584 and a scalar
        // inner loop spends the entire budget waiting on memory.
        for (int p = 0; p < n_positions; ++p) {
            dense_matmul(logits + (size_t)p * V + v0,
                         hnorm + (size_t)p * stride,
                         block.data(), H, n);
        }
    }
}

// ── Device timing (forwarded to the OpenCL backend) ────────────────────────
void ForwardEngine::profiling_enable(bool on) { if (gpu_) gpu_->profiling_enable(on); }
double ForwardEngine::profiling_kernel_ms() const {
    return gpu_ ? gpu_->profiling_kernel_ms() : 0.0; }
double ForwardEngine::profiling_wall_ms() const {
    return gpu_ ? gpu_->profiling_wall_ms() : 0.0; }
double ForwardEngine::profiling_attn_ms() const {
    return gpu_ ? gpu_->profiling_attn_ms() : 0.0; }
int ForwardEngine::profiling_attn_calls() const {
    return gpu_ ? gpu_->profiling_attn_calls() : 0; }
int ForwardEngine::profiling_kernels() const {
    return gpu_ ? gpu_->profiling_kernels() : 0; }
std::vector<std::pair<std::string, std::pair<int, double>>>
ForwardEngine::profiling_by_tag() const {
    return gpu_ ? gpu_->profiling_by_tag() : decltype(gpu_->profiling_by_tag())(); }

ForwardEngine::~ForwardEngine() {
    delete gpu_;      // releases the OpenCL context, program and kernel
    gpu_ = nullptr;
}

// ── GPU projections ────────────────────────────────────────────────────────
// Enables OpenCL for the four attention projections of every layer. Throws
// when the device cannot be brought up — there is no CPU fallback, by design
// (see the header comment in forward_gpu.cpp).
// ── CPU residency: decided once, at construction ───────────────────────────
//
// Holding every layer resident is what makes the CPU path usable. Without it,
// run_layer() calls load_layer() per layer, which re-reads the checkpoint (7
// matrices per layer) — per layer, per token. The cost is not marginal:
// measured 0.27 tok/s on the 0.5B, 1,250x slower than the GPU path.
//
// This lived inside enable_gpu() until 0.1.35, behind `if (gpu_ == nullptr)`.
// The condition was correct; the placement was not. A caller that never calls
// enable_gpu() — the whole point of the CPU path — never reached it.
//
// Fails soft: if the bytes do not fit the budget, leave resident_ empty and
// report "stream". Streaming is slower but correct, and the caller can see it
// in weight_mode().
void ForwardEngine::decide_cpu_residency() {
    const double bytes_resident = estimate_resident_bytes();
    const double budget = available_ram_bytes() * 0.60;
    if (bytes_resident > budget || cfg_.num_hidden_layers <= 0) {
        resident_.clear();
        layers_resident_ = false;
        weight_mode_ = "stream";
        return;
    }
    if (!layers_resident_) {
        resident_.resize((size_t)cfg_.num_hidden_layers);
        for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
            resident_[(size_t)l] = std::make_unique<LayerWeights>();
            load_layer(l, *resident_[(size_t)l]);
        }
        layers_resident_ = true;
    }
    weight_mode_ = "resident";
}

void ForwardEngine::enable_gpu(const std::string& opencl_lib) {
    if (gpu_ != nullptr) return;
    gpu_ = new ForwardGpu(opencl_lib);

    //
    // RESIDENT vs STREAM — decided by whether the weights FIT.
    //
    // Residency exists for one reason: run_layer() builds a fresh LayerWeights
    // per call, which gives every weight tensor a new host address and defeats
    // the device-side weight cache. Measured on Qwen2.5-0.5B, uploading all 168
    // weight matrices per forward made the GPU path 0.80x the CPU — SLOWER.
    // Holding them resident gave 2.47x.
    //
    // But residency costs RAM equal to the whole model in float32. For
    // Qwen2.5-7B that is 14.9 GB, on top of the 14.2 GB the Safetensors copy
    // already holds — 29.1 GB against 16 GB on Kaggle. Measured: "Kernel died".
    //
    // So: estimate the cost and only go resident when it fits. Otherwise fall
    // back to STREAMING, which is what the CPU path already does via
    // w_scratch: load one layer, use it, free it. Streaming is slower (the
    // device weight cache misses once per layer per forward) but it RUNS.
    //
    // The threshold uses 60% of available RAM, leaving room for the activations
    // and the OpenCL driver's own allocations.
    // ── DEVICE MODE (2026-10-23) ──────────────────────────────────────────
    //
    // The measurement that forced this: on Qwen2.5-7B the GPU path in stream
    // mode ran at 90.8 s against the CPU's 67.0 s. Re-uploading 14.2 GB of
    // weights per forward costs ~1 s more than the CPU spends computing the
    // whole thing — so "GPU + stream" was strictly a loss.
    //
    // But the upload was only happening because the device cache is keyed by
    // host pointer and streaming reuses one scratch buffer. Key the cache by
    // TENSOR NAME instead (ForwardEngine passes `L3.gate_proj` etc.) and the
    // device keeps every matrix it has ever seen, for free, across forwards.
    //
    // Once the weights live on the device they are never read from the host
    // again. So in device mode run_layer loads nothing per layer: no 14.2 GB
    // memcpy per forward, no f32 host copy of any projection, and host RAM
    // drops to norms + biases + activations (~200 MB for the 7B). The weights
    // are bf16 on the device (13.1 GB) because the f32 form (26.1 GB) does not
    // fit the 16 GB card — see ForwardGpu::should_use_bf16.
    const double bytes_resident = estimate_resident_bytes();
    const double ram_avail = available_ram_bytes();
    const double budget = ram_avail * 0.60;

    // Residency was decided at construction (decide_cpu_residency). If the GPU
    // is up, the weights go to the device instead and resident_ is not used;
    // if it is not up, residency is already in place. Nothing to decide here.
    (void)bytes_resident;
    (void)budget;
    // Warm the device copy once. Done here rather than lazily inside the first
    // forward so the upload cost does not land inside a timed region, and so a
    // device that cannot hold the weights fails at enable_gpu() rather than
    // halfway through a benchmark.
    if (gpu_ != nullptr) {
        // Declare the total BEFORE the first upload so bf16-vs-f32 is decided
        // once for the whole model, not per matrix.
        gpu_->declare_weight_bytes((size_t)estimate_resident_bytes());
        const bool fit = gpu_->weights_fit((size_t)estimate_resident_bytes())
                         && device_mode_viable();
        const bool warmed = fit && warm_device_weights();
        if (const char* dbg = std::getenv("WINNEX_GPU_DEBUG_MODE"))
            std::fprintf(stderr, "[gpu] fit=%d warmed=%d total=%.2fGB dev_total=%.2fGB "
                                 "dev_free=%.2fGB devbytes=%.2fGB\n",
                         (int)fit, (int)warmed, estimate_resident_bytes() / 1073741824.0,
                         gpu_->device_memory_total() / 1073741824.0,
                         gpu_->device_memory_free() / 1073741824.0,
                         gpu_->device_weight_bytes() / 1073741824.0);
        if (warmed) {
            device_weights_ = true;
            weight_mode_ = "device";
        } else {
            // O modelo não cabe no device. Warmup pode já ter subido parte;
            // liberar garante que cada peso viva em um lugar só.
            gpu_->release_weights();
            //
            // `stream` FOI REMOVIDO DO CAMINHO DE FALLBACK (2026-10-24).
            //
            // Ele não é um fallback: é um modo que o próprio README documenta
            // como MAIS LENTO QUE A CPU (90,8 s contra 67,0 s no 7B), porque
            // reenvia o modelo inteiro a cada forward. Escolhê-lo quando o
            // modelo não cabe no device é responder "não cabe" degradando o
            // desempenho sem avisar — um falso positivo de capacidade.
            //
            // O caminho honesto é CPU, com o motivo registrado em
            // gpu_refusal(). Nada se perde: `stream` continua existindo e
            // selecionável por quem souber que quer isso (WINNEX_WEIGHT_MODE=
            // stream), mas não é mais o que o motor escolhe sozinho quando falha.
            const char* want = std::getenv("WINNEX_WEIGHT_MODE");
            const bool user_wants_stream = want && std::string(want) == "stream";
            if (user_wants_stream && device_mode_viable()) {
                weight_mode_ = "stream";
            } else {
                const double need_gb = estimate_resident_bytes() / 2.0 / 1073741824.0;
                const double have_gb = gpu_->device_memory_total() / 1073741824.0;
                delete gpu_;
                gpu_ = nullptr;
                weight_mode_ = "cpu";
                char buf[512];
                std::snprintf(buf, sizeof buf,
                    "model does not fit the device: needs %.1f GB at bf16, "
                    "device has %.1f GB. Running on the CPU, which is FASTER "
                    "than the stream fallback for an oversized model (measured "
                    "90.8 s against 67.0 s on the 7B). Set WINNEX_WEIGHT_MODE="
                    "stream to force the stream path.", need_gb, have_gb);
                gpu_refusal_ = buf;
            }
        }
    }
}

// Upload every projection to the device once and drop the host copy.
//
// After this returns, run_host_weights_ is false and run_layer reads only the
// norms and biases from the host — it never touches the 14 GB of projections
// again. On a machine where the device cannot hold them, this throws and the
// engine falls back to resident/stream automatically.
bool ForwardEngine::warm_device_weights() {
    const int nl = cfg_.num_hidden_layers;
    const int H = cfg_.hidden_size;
    const int inter = cfg_.intermediate_size;
    std::vector<float> dummy((size_t)H, 0.0f);
    std::vector<float> out;

    // One projection at a time. Holding a whole LayerWeights costs 3.4 GB for a
    // 7B layer (gate+up+down are 1.1 GB each) on top of the 12 GB already on
    // the device, and the upload then fails with CL_MEM_OBJECT_ALLOCATION_
    // FAILURE (-4) even though the model itself fits. Each vector is released
    // as soon as its cl_mem exists, so the host peak is one matrix (~1.1 GB).
    try {
    for (int l = 0; l < nl; ++l) {
        LayerWeights w;
        const std::string L = "model.layers." + std::to_string(l) + ".";
        w.hidden = H;
        w.q_out = cfg_.num_attention_heads * cfg_.head_dim;
        w.kv_out = cfg_.head_dim * cfg_.num_key_value_heads;
        w.inter = inter;
        w.in_norm = st_->to_float(st_->get(L + "input_layernorm.weight"));
        w.post_norm = st_->to_float(st_->get(L + "post_attention_layernorm.weight"));
        auto bias_of = [&](const std::string& b) {
            return st_->has(b + ".bias") ? st_->to_float(st_->get(b + ".bias"))
                                          : std::vector<float>();
        };
        w.q_b = bias_of(L + "self_attn.q_proj");
        w.k_b = bias_of(L + "self_attn.k_proj");
        w.v_b = bias_of(L + "self_attn.v_proj");

        // The tag MUST match the one run_layer() asks for, or the device keeps
        // two copies of every matrix and device mode never finds its weights.
        const std::string T = "L" + std::to_string(l) + ".";
        auto push = [&](const char* tag, const std::string& base,
                        int out_dim, int in_dim, const std::vector<float>& bias) {
            std::vector<float> Wt = st_->to_float(st_->get(base + ".weight"));
            out.assign((size_t)out_dim, 0.0f);
            gpu_->matmul(dummy.data(), 1, Wt.data(), out_dim, in_dim,
                         bias.empty() ? nullptr : bias.data(), out.data(),
                         (T + tag).c_str());
            std::vector<float>().swap(Wt);      // release before the next one
        };
        // Q/K/V FUSED into one [q_out + 2·kv_out, H] buffer, under one tag.
        //
        // run_layer() asks for "qkv_proj" and gets all three from a single
        // kernel: A crosses the bus once and, more importantly, the layer pays
        // ONE round-trip instead of three. That is the number the repo already
        // measured as the cost ("7 x 28 = 196 round-trips per token", 29e784c),
        // and the fusion is exact — the output widths are stacked, not padded,
        // so the kernel is a plain row-major GEMM with out_dim = q_out+2·kv_out.
        //
        // The individual tags are still pushed when the fused upload is not
        // possible, so a checkpoint whose biases do not line up degrades to the
        // three-kernel path instead of failing.
        {
            std::vector<float> qw = st_->to_float(st_->get(L + "self_attn.q_proj.weight"));
            std::vector<float> kw = st_->to_float(st_->get(L + "self_attn.k_proj.weight"));
            std::vector<float> vw = st_->to_float(st_->get(L + "self_attn.v_proj.weight"));
            const bool sizes_ok = (int)qw.size() == w.q_out * H &&
                                  (int)kw.size() == w.kv_out * H &&
                                  (int)vw.size() == w.kv_out * H;
            if (sizes_ok) {
                std::vector<float> qkv;
                qkv.reserve(qw.size() + kw.size() + vw.size());
                qkv.insert(qkv.end(), qw.begin(), qw.end());
                qkv.insert(qkv.end(), kw.begin(), kw.end());
                qkv.insert(qkv.end(), vw.begin(), vw.end());
                // Bias: concatenated in the same order, or none at all. The
                // kernel takes a single [out_dim] vector, so a partial set of
                // biases cannot be represented — fall back when that happens.
                std::vector<float> bcat;
                const bool all_b = w.q_b.size() == (size_t)w.q_out &&
                                   w.k_b.size() == (size_t)w.kv_out &&
                                   w.v_b.size() == (size_t)w.kv_out;
                const bool none_b = w.q_b.empty() && w.k_b.empty() && w.v_b.empty();
                const bool bias_ok = all_b || none_b;
                if (bias_ok) {
                    if (all_b) {
                        bcat.reserve(w.q_out + 2 * w.kv_out);
                        bcat.insert(bcat.end(), w.q_b.begin(), w.q_b.end());
                        bcat.insert(bcat.end(), w.k_b.begin(), w.k_b.end());
                        bcat.insert(bcat.end(), w.v_b.begin(), w.v_b.end());
                    }
                    out.assign((size_t)(w.q_out + 2 * w.kv_out), 0.0f);
                    gpu_->matmul(dummy.data(), 1, qkv.data(), w.q_out + 2 * w.kv_out, H,
                                 all_b ? bcat.data() : nullptr, out.data(),
                                 (T + "qkv_proj").c_str());
                }
                if (!bias_ok) {
                    push("q_proj", L + "self_attn.q_proj", w.q_out, H, w.q_b);
                    push("k_proj", L + "self_attn.k_proj", w.kv_out, H, w.k_b);
                    push("v_proj", L + "self_attn.v_proj", w.kv_out, H, w.v_b);
                }
                std::vector<float>().swap(qkv);
            } else {
                push("q_proj", L + "self_attn.q_proj", w.q_out, H, w.q_b);
                push("k_proj", L + "self_attn.k_proj", w.kv_out, H, w.k_b);
                push("v_proj", L + "self_attn.v_proj", w.kv_out, H, w.v_b);
            }
            std::vector<float>().swap(qw);
            std::vector<float>().swap(kw);
            std::vector<float>().swap(vw);
        }
        push("o_proj",    L + "self_attn.o_proj", H, w.q_out, {});

        // GATE + UP FUSED into one [2·inter, H] buffer.
        //
        // run_layer asks for the tag "gate_up_proj" and gets both projections
        // from one kernel, with A uploaded once instead of twice. This is the
        // same fusion the Madhava engine uses in qkt_fused_topk.
        //
        // The two source tensors are read and concatenated on the host. That
        // peeks at 2·inter·H floats (~0.4 GB for the 4B) instead of the
        // one-matrix-at-a-time discipline used above; it is the price of the
        // fusion and it stays under the same host ceiling as before, because
        // gate+up were always the two largest matrices and now they are read
        // contiguously rather than one after the other.
        {
            std::vector<float> g = st_->to_float(st_->get(L + "mlp.gate_proj.weight"));
            std::vector<float> u = st_->to_float(st_->get(L + "mlp.up_proj.weight"));
            std::vector<float> gu2;
            gu2.reserve(g.size() + u.size());
            gu2.insert(gu2.end(), g.begin(), g.end());
            gu2.insert(gu2.end(), u.begin(), u.end());
            std::vector<float>().swap(g);
            std::vector<float>().swap(u);
            out.assign((size_t)2 * inter, 0.0f);
            gpu_->matmul(dummy.data(), 1, gu2.data(), 2 * inter, H,
                         nullptr, out.data(), (T + "gate_up_proj").c_str());
            std::vector<float>().swap(gu2);
        }
        push("down_proj", L + "mlp.down_proj", H, inter, {});
    }
    } catch (const std::exception&) {
        // Host or device ran out. Partial state is unusable (some layers
        // tagged, some not) so the caller falls back to streaming.
        //
        // But what already went up is still holding VRAM, and streaming needs
        // that VRAM for its activation buffers. Without releasing here the
        // partial upload poisons the very mode meant to replace it — measured
        // on the Qwen2.5-14B: the following stream failed with
        // CL_MEM_OBJECT_ALLOCATION_FAILURE (-4) while ~13 GB of orphaned
        // weights were still resident on the device.
        gpu_->release_weights();
        return false;
    }
    return true;
}

// Byte cost of holding every layer resident in float32.
//
// BUG FIX (2026-10-24): this used 2·H² for q_proj + o_proj, i.e. it assumed
// q_out == hidden_size. Qwen3 decouples head_dim from H/heads, so q_out is
// heads·head_dim (32·128 = 4096) and not H (2560). The estimate came out
// 1.06 GB BELOW the truth on Qwen3-4B, and the measurement shows it directly:
//
//     [gpu] fit=1 ... total=12.48GB devbytes=13.54GB
//
// The engine printed 12.48 GB and deposited 13.54 GB. Same defect the loader
// had (see load_layer) — that one was fixed, this one was not.
//
// Consequence, measured: with the wrong estimate, weights_fit() (85% budget =
// 13.16 GB) accepted f32 on a 16 GB card, which is correct by luck. On a 12 GB
// card it rejected f32 AND bf16, so the engine fell back to `stream` — the mode
// its own README documents as slower than the CPU — when bf16 would have fit
// and run on the device.
double ForwardEngine::estimate_resident_bytes() const {
    const double H = (double)cfg_.hidden_size;
    const double inter = (double)cfg_.intermediate_size;
    const double kv = (double)cfg_.head_dim * (double)cfg_.num_key_value_heads;
    // q_proj is [heads·head_dim, H] and o_proj is [H, heads·head_dim]. They are
    // NOT [H, H] when head_dim is decoupled (Qwen3). load_layer already uses
    // this shape; the estimator has to agree with it or the two disagree about
    // how big the model is.
    const double q_out = (double)cfg_.num_attention_heads * (double)cfg_.head_dim;
    const double per_layer = H * q_out            // q_proj
                           + H * q_out            // o_proj
                           + 2.0 * H * kv         // k_proj + v_proj
                           + 3.0 * H * inter      // gate + up + down
                           + 2.0 * H              // in_norm + post_norm
                           + 2.0 * kv             // k/v biases
                           + 1.0 * H;             // q bias
    return per_layer * (double)cfg_.num_hidden_layers * 4.0;   // float32
}

// RAM the process can still allocate, from /proc/meminfo MemAvailable.
double ForwardEngine::available_ram_bytes() {
    std::ifstream f("/proc/meminfo");
    std::string key;
    while (f >> key) {
        if (key == "MemAvailable:") {
            double kb = 0;
            f >> kb;
            return kb * 1024.0;
        }
        std::string rest;
        std::getline(f, rest);
    }
    return -1.0;   // unknown: caller treats as "do not go resident"
}

size_t ForwardEngine::device_weight_bytes() const {
    return gpu_ ? gpu_->device_weight_bytes() : 0;
}

void ForwardEngine::attention_raw(const float* Q, const float* K, const float* V,
                                  float* O, int seq, int ctx, int nh, int nkv,
                                  int hd, int cache_pos) {
    if (!gpu_) throw std::runtime_error("attention_raw: GPU not enabled");
    gpu_->attention(Q, K, V, O, seq, ctx, nh, nkv, hd, cache_pos);
}

bool ForwardEngine::gpu_attention() const {
    return gpu_ != nullptr && gpu_->has_attention() && device_weights_;
}

bool ForwardEngine::device_mode_viable() const {
    if (gpu_ == nullptr) return false;
    const size_t avail = gpu_->device_memory_total();
    if (avail == 0) return true;            // driver não informa; tenta
    const double bytes_resident = estimate_resident_bytes();
    // O modo device guarda bf16 quando f32 não cabe. Se nem a metade do f32
    // couber na placa, o upload vai falhar no meio — recusar é mais honesto
    // que falhar 20 minutos depois.
    return bytes_resident * 0.5 <= (double)avail * 0.85;
}

std::string ForwardEngine::weight_mode() const {
    return weight_mode_.empty() ? "cpu" : weight_mode_;
}

bool ForwardEngine::gpu_enabled() const { return gpu_ != nullptr; }
std::string ForwardEngine::gpu_device() const {
    return gpu_ ? gpu_->device_name() : std::string();
}
std::string ForwardEngine::gpu_reason() const { return ForwardGpu::reason(); }

void ForwardEngine::reset_cache() {
    const int kv_out = cfg_.head_dim * cfg_.num_key_value_heads;
    k_caches_.assign(cfg_.num_hidden_layers, std::vector<float>());
    v_caches_.assign(cfg_.num_hidden_layers, std::vector<float>());
    ctx_len_ = 0;
}

std::vector<float> ForwardEngine::embed_ids(const std::vector<int>& ids) const {
    const int H = cfg_.hidden_size;
    const SafeTensor& t = st_->get(kEmbedName);
    const int rows = (int)t.shape[0];
    std::vector<float> out((size_t)ids.size() * H);
    for (size_t i = 0; i < ids.size(); ++i) {
        const int id = ids[i];
        if (id < 0 || id >= rows)
            throw std::invalid_argument("ForwardEngine::embed_ids: token id " +
                                        std::to_string(id) + " outside [0, " +
                                        std::to_string(rows) + ")");
        std::vector<float> row = st_->read_rows(t, id, 1);
        std::memcpy(out.data() + i * H, row.data(), (size_t)H * sizeof(float));
    }
    return out;
}

std::vector<float> ForwardEngine::prefill(const std::vector<int>& ids) {
    if (ids.empty())
        throw std::invalid_argument("ForwardEngine::prefill: empty prompt");
    reset_cache();

    const int H = cfg_.hidden_size;
    const int n = (int)ids.size();
    const int kv_out = cfg_.head_dim * cfg_.num_key_value_heads;

    // Persistent caches sized for the whole prompt. forward_next() grows them
    // from here as decode continues.
    k_caches_.assign(cfg_.num_hidden_layers, std::vector<float>((size_t)n * kv_out));
    v_caches_.assign(cfg_.num_hidden_layers, std::vector<float>((size_t)n * kv_out));

    // Reserve the device-side KV cache for this sequence length. Sizing it here
    // (rather than lazily inside the first attention) keeps the allocation out
    // of any timed decode region and makes an out-of-memory failure happen at
    // prefill, where the caller can still react.
    //
    // Reserve with headroom: a caller that prefills n tokens and then decodes
    // m more would otherwise need a reallocation mid-generation. The margin is
    // capped by max_position_embeddings so a 32k context does not reserve 64k.
    if (gpu_ != nullptr) {
        const int max_pos = cfg_.max_position_embeddings > 0
                            ? cfg_.max_position_embeddings : n;
        const int want = std::min(max_pos, n + std::max(256, n / 2));
        gpu_->prepare_kv(want, cfg_.num_key_value_heads, cfg_.head_dim,
                         cfg_.num_hidden_layers);
    }

    std::vector<float> h = embed_ids(ids);
    for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
        run_layer(layer, h, n, k_caches_[layer], v_caches_[layer], 0);
    }
    ctx_len_ = n;

    std::vector<float> hnorm(H);
    rms_norm(hnorm.data(), h.data() + (size_t)(n - 1) * H, final_norm_.data(),
             1, H, cfg_.rms_norm_eps);
    std::vector<float> logits(cfg_.vocab_size);
    { ScopedTimer _t(g_phase, g_phase.lmh, g_phase.on);
      lm_head_logits(hnorm.data(), logits.data(), 1, H); }
    ++g_phase.n;
    return logits;
}

std::vector<float> ForwardEngine::step(int token_id) {
    if (ctx_len_ <= 0)
        throw std::logic_error("ForwardEngine::step: call prefill() first");
    if (ctx_len_ >= cfg_.max_position_embeddings && cfg_.max_position_embeddings > 0)
        throw std::out_of_range(
            "ForwardEngine::step: context is full (" +
            std::to_string(cfg_.max_position_embeddings) + " positions)");
    std::vector<float> h = embed_ids({token_id});
    return forward_next(h);
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
    lm_head_logits(hnorm.data(), logits.data(), 1, H);
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
        h.assign(cfg_.hidden_size, 0.0f);
        lm_head_row(tid, h.data());
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
