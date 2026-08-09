/**
 * winnex_nano.hpp — native Winnex LLM inference engine (C++20, OpenCL, no CUDA).
 *
 * Reuses the winnex-madhava kernels (QK^T matmul, softmax-free attention via
 * top-k + Cauchy-Schwarz bounds, quantization, AVX2/OpenMP/OpenCL) through
 * direct source compilation — NO code duplication. This package adds the
 * LLM-only layers: BPE tokenizer, safetensors loader, tensor ops (RMSNorm,
 * RoPE, SiLU), GQA attention with KV cache, MLP, autoregressive generation,
 * and the multimodel weight balancer.
 *
 * The goal is a self-contained, pip-independent inference engine for the
 * Winnex stack — no SGLang, no CUDA, running on OpenCL (vendor-neutral GPU)
 * or CPU, following the Madhava principles (deterministic, provable).
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
 */
#ifndef WINNEX_NANO_HPP
#define WINNEX_NANO_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace winnex_nano {

// ---------------------------------------------------------------------------
// BPE tokenizer (reads a HuggingFace tokenizer.json, Qwen BPE)
// ---------------------------------------------------------------------------
class Tokenizer {
public:
    explicit Tokenizer(const std::string& tokenizer_json_path);

    std::vector<int> encode(const std::string& text, bool add_bos = true) const;
    std::string decode(const std::vector<int>& ids) const;

    int vocab_size() const { return (int)vocab_.size(); }
    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }

private:
    std::map<std::string, int> vocab_;
    std::vector<std::string> id_to_token_;
    struct Merge { std::string a, b; int rank; };
    std::vector<Merge> merges_;
    int bos_id_ = 0;
    int eos_id_ = 0;
    int unk_id_ = -1;
};

// ---------------------------------------------------------------------------
// Safetensors model loader (GPTQ int4 + f16/bf16 tensors)
// ---------------------------------------------------------------------------
struct Tensor {
    std::string name;
    int dtype = 0;                 // 0=f32, 1=f16, 2=bf16, 3=i32-packed
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;     // raw bytes (dequant handled by ops)
};

class ModelLoader {
public:
    explicit ModelLoader(const std::string& safetensors_path);

    bool has(const std::string& name) const { return tensors_.count(name) > 0; }
    const Tensor& get(const std::string& name) const;
    const std::map<std::string, Tensor>& tensors() const { return tensors_; }
    size_t count() const { return tensors_.size(); }

private:
    std::map<std::string, Tensor> tensors_;
};

// ---------------------------------------------------------------------------
// LLM config (Qwen2 / Qwen3-style)
// ---------------------------------------------------------------------------
struct LlmConfig {
    std::string arch = "qwen2";    // "qwen2" or "qwen3"
    int vocab_size = 151936;
    int hidden_size = 2048;
    int intermediate_size = 11008;
    int num_hidden_layers = 36;
    int num_attention_heads = 16;
    int num_key_value_heads = 2;
    int head_dim = 128;
    int max_position_embeddings = 32768;
    float rope_theta = 10000.0f;
    float rms_norm_eps = 1e-6f;
    bool tie_word_embeddings = false;
    int group_size = 128;          // GPTQ
    bool gptq = true;              // true when qweight/qzeros present
};

// Load config from a model config.json.
LlmConfig load_config(const std::string& config_path);

// ---------------------------------------------------------------------------
// Tensor ops (native, OpenMP/AVX2; OpenCL where reused from Madhava)
// ---------------------------------------------------------------------------
void rms_norm(float* out, const float* x, const float* weight, int n, int d,
              float eps);
void rope_apply(float* x, int seq_pos, int head_dim, int n_heads,
                float rope_theta);
void silu_gate_mul(float* out, const float* gate, const float* up, int n);

// GPTQ int4 dequant + matmul: y[out] = sum_in x[in] * W_gptq[out][in]
void gptq_matmul(float* y, const float* x, const uint8_t* qweight,
                 const uint8_t* qzeros, const float* scales, const int* g_idx,
                 int in_dim, int out_dim, int group_size);

// ---------------------------------------------------------------------------
// Weight balancer — the PsiQRH-derived multimodel blending:
//     W' = Σᵢ αᵢ · R(qᵢ) · Wᵢ      with Σᵢ αᵢ = 1
// α = operator-controlled blend weights; R(q) is an optional quaternion
// rotation applied per tensor (a "control knob" for phase, avoiding the
// destructive interference of plain linear interpolation).
// ---------------------------------------------------------------------------
struct BlendWeight {
    double alpha = 0.0;      // contribution of this model (Σ α = 1)
    double theta = 0.0;      // quaternion rotation angle (radians), 0 = no rotation
};

class WeightBalancer {
public:
    // Blends N weight maps (same architecture) into one. Each map is
    // name -> Tensor. Only matching tensor names are blended.
    std::map<std::string, Tensor> blend(
        const std::vector<std::map<std::string, Tensor>>& models,
        const std::vector<BlendWeight>& weights) const;
};

// ---------------------------------------------------------------------------
// LLM inference engine (forward + autoregressive generation)
// ---------------------------------------------------------------------------
class Llm {
public:
    Llm(const LlmConfig& cfg, const ModelLoader& loader);

    // Prefill: embed + run all layers on the input ids. Returns logits.
    std::vector<float> forward(const std::vector<int>& input_ids);

    // Greedy decode of the next token given the current logits.
    int sample_greedy(const std::vector<float>& logits) const;

    // Generate a full sequence (prefill + autoregressive decode with KV cache).
    std::vector<int> generate(const std::vector<int>& prompt_ids, int max_new_tokens,
                              float temperature = 0.0f);

private:
    LlmConfig cfg_;
    // Weights are held by the ModelLoader (shared_ptr to keep them alive).
    std::shared_ptr<const ModelLoader> loader_;
};

} // namespace winnex_nano

#endif // WINNEX_NANO_HPP
