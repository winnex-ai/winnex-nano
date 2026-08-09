/**
 * forward.hpp — native forward pass for transformer LLMs (GPTQ int4).
 *
 * Processes hidden states through the model layers, driven by a config that
 * is READ from the model's config.json (arch-detected, model-agnostic):
 *
 *   per layer i:
 *     h = RMSNorm(h, input_layernorm)                (rms_norm)
 *     q = GPTQ(h, q_proj); k = GPTQ(h, k_proj); v = GPTQ(h, v_proj)
 *     q,k = RoPE(q,k, position)
 *     attn = GQA_attention(q, k, v)                  (QKᵀ + top-k, Madhava-style)
 *     h = h + GPTQ(attn, o_proj)
 *     h = h + MLP(RMSNorm(h), gate/up/down)          (SiLU gating)
 *   logits = GPTQ(RMSNorm(h), lm_head)               (tied to embed_tokens if tied)
 *
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
    int group_size = 128;            // GPTQ
};

// Loads a ModelConfig from a config.json (minimal parser).
ModelConfig load_config(const std::string& config_path);

/**
 * ForwardEngine — runs the native forward pass.
 *
 * Owns the dequantized weight tensors it needs (lazily loaded from the
 * Safetensors). The input is a sequence of hidden states (from the X-factor
 * embedding); the output is logits over the vocab for the last position.
 */
class ForwardEngine {
public:
    ForwardEngine(const ModelConfig& cfg, const Safetensors& st);

    // Runs the forward pass on a batch of hidden states [seq, hidden_size].
    // Returns logits [vocab_size] for the LAST position (for autoregressive
    // generation). If `all_positions` is true, returns [seq, vocab_size].
    std::vector<float> forward(const std::vector<float>& hidden, int seq_len,
                               bool all_positions = false);

    // Greedy argmax of logits -> token id.
    static int argmax(const std::vector<float>& logits);

    // Number of parameters (approx, from loaded tensors).
    size_t param_count() const { return param_count_; }

private:
    ModelConfig cfg_;
    const Safetensors* st_;
    size_t param_count_ = 0;

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
