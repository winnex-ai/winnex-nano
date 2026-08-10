/**
 * forward.hpp — native forward pass for transformer LLMs (dense f32 + OpenCL).
 *
 * Processes hidden states through the model layers, driven by a config that
 * is READ from the model's config.json (arch-detected, model-agnostic):
 *
 *   per layer i:
 *     h = RMSNorm(h, input_layernorm)                (rms_norm)
 *     q = DENSE(h, q_proj); k = DENSE(h, k_proj); v = DENSE(h, v_proj)
 *     q,k = RoPE(q,k, position)
 *     attn = GQA_attention(q, k, v)                  (QKᵀ + top-k, Madhava-style)
 *     h = h + DENSE(attn, o_proj)
 *     h = h + MLP(RMSNorm(h), gate/up/down)          (SiLU gating)
 *   logits = DENSE(RMSNorm(h), lm_head)              (tied to embed_tokens if tied)
 *
 * The input hidden states come from the X-factor embedding of the text
 * (XFactor.project(expand_spectral(histogram))) — no BPE, no vocabulary.
 * The attention uses the Madhava selective-top-K (no softmax over all): the
 * query attends to the top-k keys by bound, exactly like the engine prunes.
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
 */
#ifndef WINNEX_NANO_FORWARD_HPP
#define WINNEX_NANO_FORWARD_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "winnex_nano/safetensors.hpp"

namespace winnex_nano {

// Model config (read from config.json — arch-detected).
struct ModelConfig {
    std::string arch = "qwen2";      // qwen2, qwen3, etc.
    int vocab_size = 0;
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int max_position_embeddings = 0;
    float rope_theta = 10000.0f;
    float rms_norm_eps = 1e-6f;
    bool tie_word_embeddings = true;
    int group_size = 128;            // GPTQ (legacy; the dense path ignores it)
    bool gptq = false;               // true when .qweight/.qzeros present
};

// Loads a ModelConfig from a config.json (minimal parser).
ModelConfig load_config(const std::string& config_path);

/**
 * ForwardEngine — runs the native forward pass.
 *
 * Owns the dequantized weight tensors it needs (lazily loaded from the
 * Safetensors). The input is a sequence of hidden states (from the X-factor
 * embedding); the output is logits over the vocab for the last position.
 *
 * For autoregressive generation the engine keeps a persistent KV cache
 * between calls (reset with reset_cache / cleared implicitly when generating
 * from a fresh prompt). The generation loop:
 *   1. prompt text → X-factor → hidden (the first "token")
 *   2. forward(hidden, 1) → logits → token_id (argmax)
 *   3. next hidden = embed_tokens[token_id]  (tied lm_head path)
 *   4. forward_next(hidden) → logits → next token_id  (uses the cache)
 *   5. repeat until EOS / max_new_tokens.
 */
class ForwardEngine {
public:
    ForwardEngine(const ModelConfig& cfg, const Safetensors& st);

    // Runs the forward pass on a batch of hidden states [seq, hidden_size]
    // with a FRESH (local) KV cache. Returns logits [vocab_size] for the LAST
    // position (or [seq, vocab_size] if all_positions). This does not touch
    // the persistent cache — use it for single-shot prefill.
    std::vector<float> forward(const std::vector<float>& hidden, int seq_len,
                               bool all_positions = false);

    // Runs a SINGLE hidden state through all layers, attending to the
    // PERSISTENT KV cache (appends this token's K/V to the cache). Returns
    // logits [vocab_size] for this token. Use for autoregressive decode.
    std::vector<float> forward_next(const std::vector<float>& hidden);

    // Resets the persistent KV cache (start a new generation).
    void reset_cache();

    // Autoregressive generation: prefill with the given prompt hidden state,
    // then decode token-by-token using the persistent cache.
    //   h_prompt:  [hidden_size] the X-factor embedding of the prompt.
    //   max_new_tokens: max tokens to generate.
    //   eos_id:   stop token id (-1 = no EOS, runs max_new_tokens).
    // Returns the generated token ids (does NOT include the prompt).
    std::vector<int> generate(const std::vector<float>& h_prompt,
                              int max_new_tokens, int eos_id = -1);

    // Greedy argmax of logits -> token id.
    static int argmax(const std::vector<float>& logits);

    // Number of parameters (approx, from loaded tensors).
    size_t param_count() const { return param_count_; }

private:
    ModelConfig cfg_;
    const Safetensors* st_;
    size_t param_count_ = 0;

    // Precomputed model-wide tensors (loaded once in the constructor).
    std::vector<float> final_norm_;      // model.norm.weight [hidden]
    std::vector<float> embed_tokens_;    // [vocab, hidden] (tied lm_head)
    bool has_separate_lm_head_ = false;  // true when lm_head.weight exists

    // Persistent KV cache: per-layer [ctx_len, kv_out].
    std::vector<std::vector<float>> k_caches_, v_caches_;
    int ctx_len_ = 0;

    // Lazy-loaded weight views (kept for the duration of a forward).
    struct LayerWeights;
    void load_layer(int layer, LayerWeights& w);

    // Core ops.
    void run_layer(int layer, std::vector<float>& h, int seq_len,
                   std::vector<float>& k_cache, std::vector<float>& v_cache,
                   int cache_pos);
};

} // namespace winnex_nano

#endif // WINNEX_NANO_FORWARD_HPP
