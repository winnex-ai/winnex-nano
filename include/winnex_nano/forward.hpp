/**
 * forward.hpp — native forward pass for transformer LLMs (dense f32 + OpenCL).
 *
 * Processes hidden states through the model layers, driven by a config that
 * is READ from the model's config.json (arch-detected):
 *
 *   per layer i:
 *     h = RMSNorm(h, input_layernorm)                (rms_norm)
 *     q = DENSE(h, q_proj); k = DENSE(h, k_proj); v = DENSE(h, v_proj)
 *     q,k = RoPE(q,k, position)
 *     attn = GQA_attention(q, k, v)                  (softmax causal EXATA)
 *     h = h + DENSE(attn, o_proj)
 *     h = h + MLP(RMSNorm(h), gate/up/down)          (SiLU gating)
 *   logits = DENSE(RMSNorm(h), lm_head)              (tied to embed_tokens if tied)
 *
 * ── O QUE ESTA CLASSE FAZ, E O QUE ELA NÃO FAZ ─────────────────────────────
 *
 * A versão anterior deste cabeçalho descrevia um motor que o código não é.
 * Corrigido aqui, porque a documentação de topo é o que um usuário lê primeiro:
 *
 *   ATENÇÃO. É softmax causal EXATA — a função que os pesos pré-treinados
 *   codificam. NÃO é top-k "Madhava-style" por padrão. Existe um caminho
 *   seletivo opcional (MADHAVA_TOPK_ATTN=1) com garantia de massa descartada
 *   controlada, mas ele: (a) está DESLIGADO por padrão; (b) só existe no ramo
 *   CPU da atenção — com a atenção na GPU ele não tem efeito; (c) medido, não
 *   acelera, porque todos os scores são computados de qualquer forma. Trocar a
 *   atenção por top-k sem renormalizar já foi tentado neste código e o custo
 *   está registrado em run_layer: perplexidade 2.0e7 contra 37.4 e correlação
 *   ≈ 0 com a referência.
 *
 *   ENTRADA. O caminho de geração (prefill/step) recebe **token ids do
 *   tokenizer do checkpoint** e lê as linhas do embed_tokens por mmap. O
 *   X-factor embedding NÃO participa desse caminho. Ele existe em
 *   compute_xfactor() e era a entrada de generate(), que é legado.
 *
 *   ARQUITETURAS. Densa apenas: qwen2, qwen3, llama, mistral. MoE
 *   (qwen2_moe/qwen3_moe/mixtral), DeepSeek (MLA) e Gemma (GeGLU +
 *   softcapping) são RECUSADAS em load_config — ver o gate lá.
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
 */
#ifndef WINNEX_NANO_FORWARD_HPP
#define WINNEX_NANO_FORWARD_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <utility>   // std::pair — usado em profiling_by_tag(); não herdar de <vector>
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
    int max_position_embeddings = 0;  // required — no default, see load_config
    float rope_theta = 0.0f;          // required — 0 means "absent in config"
    float rms_norm_eps = -1.0f;       // required — negative means "absent"
    bool tie_word_embeddings = true;
    int group_size = 128;            // GPTQ (legacy; the dense path ignores it)
    bool gptq = false;               // true when .qweight/.qzeros present
    // True when config.json declares a sliding_window. This engine has no
    // windowed attention, so load_config refuses it rather than attending to
    // history the model was trained to ignore.
    bool has_sliding_window = false;  // sliding_window present in config
    int sliding_window = 0;           // its value; inert when >= max_position

    // ── rope_scaling ──────────────────────────────────────────────────────
    // AUSENTE ATÉ AQUI, E ISSO ERA UM DEFEITO SILENCIOSO. O motor lia só
    // `rope_theta` e ignorava `rope_scaling`. Para Qwen2.5 (sem scaling) isso
    // não custa nada. Para Llama-3.2 — que declara
    //
    //   "rope_scaling": {"rope_type":"llama3","factor":32.0,
    //                    "low_freq_factor":1.0,"high_freq_factor":4.0,
    //                    "original_max_position_embeddings":8192}
    //
    // — as invariantes de frequência BAIXA (wavelen > 8192, 15 das 32
    // dimensões) são divididas por 32, e as de frequência alta ficam intactas.
    // Ignorar o scaling fazia o motor rodar um modelo DIFERENTE, com erro de
    // fase que cresce com a posição: 0,13 rad em pos=128, 1,0 rad em 1024,
    // 8,1 rad em 8192. Medido contra o transformers no Llama-3.2-1B: cos dos
    // logits cai para 0,9979 em 1024 tokens, contra 0,99985 no Qwen sem scaling.
    //
    // rope_type vazio = sem scaling (o caso Qwen). Um rope_type PRESENTE e não
    // suportado é recusado em load_config, pelo mesmo contrato do gate de
    // arquitetura: recusar em vez de rodar errado.
    std::string rope_type;                 // "" | "llama3"
    float rope_factor = 1.0f;
    float rope_low_freq_factor = 1.0f;
    float rope_high_freq_factor = 4.0f;
    int rope_original_max_pos = 0;         // 0 -> usa max_position_embeddings
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
    ~ForwardEngine();

    // Runs the forward pass on a batch of hidden states [seq, hidden_size]
    // with a FRESH (local) KV cache. Returns logits [vocab_size] for the LAST
    // position (or [seq, vocab_size] if all_positions). This does not touch
    // the persistent cache — use it for single-shot prefill.
    std::vector<float> forward(const std::vector<float>& hidden, int seq_len,
                               bool all_positions = false);

    // ── Token-id interface ─────────────────────────────────────────────────
    //
    // A caller holding token ids (a tokenizer, an HTTP request, a benchmark)
    // should never have to reach into the safetensors file to build hidden
    // states. These three calls are the whole contract:

    //   reset_cache()                    start a conversation
    //   prefill(token_ids) -> logits     process the prompt, fill the KV cache
    //   step(next_id)      -> logits     one token, using the cache
    //
    // embed_ids reads ONLY the rows it needs, through the offset table, so a
    // 152k-row table costs H floats per token instead of 2.18 GB.

    // Embedding row for each id: [n_ids, hidden_size], row-major.
    std::vector<float> embed_ids(const std::vector<int>& ids) const;

    // Prefill: embeds the ids, runs them through every layer with the
    // persistent KV cache, and returns logits [vocab_size] for the LAST id.
    // The cache is reset first, so this is the start of a sequence.
    std::vector<float> prefill(const std::vector<int>& ids);

    // Decode one token: embeds `token_id`, appends its K/V to the persistent
    // cache, returns logits [vocab_size]. Use after prefill().
    std::vector<float> step(int token_id);

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

    // Shape accessors — a server needs these to validate a request against the
    // model it actually loaded, rather than against a hardcoded guess.
    int hidden_size() const { return cfg_.hidden_size; }
    int vocab_size() const { return cfg_.vocab_size; }
    int max_context() const { return cfg_.max_position_embeddings; }
    const ModelConfig& config() const { return cfg_; }

    // ── GPU projections (OpenCL) ──────────────────────────────────────────
    // When enabled, the four attention projections of every layer (Q, K, V, O)
    // run on the device instead of `dense_matmul`. RoPE, the causal mask,
    // softmax, residuals and the KV cache stay on the CPU — see the note at
    // the top of forward_gpu.cpp for why the split is drawn there.
    //
    // `opencl_lib` is the loader to dlopen; empty resolves via
    // $WINNEX_OPENCL_LIB then the vendor ICD names. There is NO silent CPU
    // fallback: if the device cannot be brought up this throws, so a benchmark
    // can never mistake CPU work for GPU work.
    void enable_gpu(const std::string& opencl_lib = "");
    bool gpu_enabled() const;
    // "cpu" | "resident" | "stream" — which weight strategy is active.
    // Residency needs the whole model in float32 RAM; when it does not fit,
    // enable_gpu() streams one layer at a time instead of failing.
    std::string weight_mode() const;

    /**
     * Metade do modelo precisa caber no device para o modo `device` valer a
     * pena. `stream` reenvia TUDO a cada forward, então um modelo grande em
     * stream não é lento — é impossível: as ativações e a camada corrente
     * competem com o que já subiu.
     *
     * Medido no Qwen2.5-14B (pesos f32 = 49,2 GB, placa 15,48 GB): o modo
     * `stream` encheu a VRAM até 15.618 MB e falhou com
     * CL_MEM_OBJECT_ALLOCATION_FAILURE (-4) no prefill. Um modelo que não cabe
     * no device precisa rodar na CPU, não no device com streaming.
     */
    bool device_mode_viable() const;

    /** Por que o device foi recusado (vazio quando não houve recusa). */
    std::string gpu_refusal() const { return gpu_refusal_; }

    /** O kernel de atenção compilou e está sendo usado. */
    bool gpu_attention() const;

    /** Chamada direta ao kernel de atenção (teste de paridade). */
    void attention_raw(const float* Q, const float* K, const float* V, float* O,
                       int seq, int ctx, int nh, int nkv, int hd, int cache_pos);

    /** Bytes of projection weights resident on the device (0 when unused). */
    size_t device_weight_bytes() const;
    // Byte cost of holding every layer resident (float32), and the RAM the
    // process can still allocate. Public so a caller can decide before calling
    // enable_gpu().
    double estimate_resident_bytes() const;
    static double available_ram_bytes();
    std::string gpu_reason() const;
    std::string gpu_device() const;

private:
    ModelConfig cfg_;
    const Safetensors* st_;
    size_t param_count_ = 0;
    class ForwardGpu* gpu_ = nullptr;    // owned; null when not enabled

    // Precomputed model-wide tensors (loaded once in the constructor).
    std::vector<float> final_norm_;      // model.norm.weight [hidden]
    // NOT the full [vocab, hidden] table. For the 7B that is 2.18 GB in f32 and
    // the forward only needs one row at a time, so lm_head is read through
    // st_->read_rows() instead. Empty here on purpose — see lm_head_row().
    std::vector<float> embed_tokens_;
    int embed_tokens_shape_h_ = 0;

    // Invariante de frequência de RoPE para a dimensão par `j` (j em [0, hd/2)).
    //
    // Sem scaling: theta^(-2j/hd), a fórmula clássica.
    //
    // Com rope_type == "llama3": a mesma base, depois
    //   - wavelen > original_max/low_freq  -> divide por `factor`
    //   - wavelen < original_max/high_freq -> fica
    //   - entre os dois                    -> interpolação suave
    // Verificado contra `m.model.rotary_emb.inv_freq` do transformers no
    // Llama-3.2-1B: erro relativo máximo 2,1e-7 em todas as 32 dimensões.
    //
    // Fica como membro (e não como lambda no laço) porque o laço chama isto
    // seq_len·hd/2 vezes e o valor depende só de j — sem estado por posição.
    float RoPEInvFreq(int j) const;

    // One row of lm_head (the tied embedding). Returns H floats. Uses the
    // offset table and converts only the requested row, so decoding a token
    // touches 14 KB instead of 2.18 GB.
    const SafeTensor& lm_head_tensor() const;
    void lm_head_row(int token_id, float* out) const;
    // logits[v] = dot(hnorm, lm_head[v]) for every v, streamed in blocks so the
    // table is never resident. `hnorm` is H floats.
    void lm_head_logits(const float* hnorm, float* logits, int n_positions, int stride) const;
    bool has_separate_lm_head_ = false;  // true when lm_head.weight exists
    // Vira true quando o upload da tabela do lm_head para o device falha.
    // `mutable` porque lm_head_logits() é const e o caminho GPU é um cache:
    // a primeira falha move o motor para a CPU de forma permanente, em vez de
    // tentar de novo a cada token (o upload que falhou custa mais que a CPU).
    mutable bool lm_head_on_cpu_ = false;
    // Tabela do lm_head em host, carregada uma vez quando o device a recebe.
    // Fica vazia quando o motor fica na CPU (o caminho por blocos le do mmap
    // e nunca materializa a tabela inteira). mutable: preenchida no primeiro
    // lm_head_logits() const.
    mutable std::vector<float> lm_head_host_;

    // Persistent KV cache: per-layer [ctx_len, kv_out].
    std::vector<std::vector<float>> k_caches_, v_caches_;
    int ctx_len_ = 0;

    // Lazy-loaded weight views (kept for the duration of a forward).
    struct LayerWeights;

    // Resident layer weights, used only on the GPU path.
    //
    // run_layer() builds a fresh LayerWeights per call, which gives every
    // weight tensor a NEW host address on every forward. The device-side weight
    // cache is keyed by host pointer, so with per-call weights it never hits and
    // all 168 weight uploads happen per forward — measured, that made the GPU
    // path 0.80x the CPU speed. Loading the layers once and keeping them alive
    // gives stable addresses, so each weight uploads exactly once.
    std::vector<std::unique_ptr<LayerWeights>> resident_;
    bool layers_resident_ = false;
    std::string weight_mode_;   // "cpu" | "resident" | "stream" | "device"
    bool device_weights_ = false;
    std::string gpu_refusal_;

public:
    // Device-side timing, forwarded to the OpenCL backend. No-ops on the CPU
    // path, where there are no kernels to time.
    void profiling_enable(bool on);
    double profiling_kernel_ms() const;
    double profiling_wall_ms() const;
    double profiling_attn_ms() const;
    int profiling_attn_calls() const;
    int profiling_kernels() const;
    std::vector<std::pair<std::string, std::pair<int, double>>> profiling_by_tag() const;
private:

    void load_layer(int layer, LayerWeights& w);

    // Device-resident mode: every projection is uploaded once and the host
    // copy is released. After warm_device_weights(), run_layer loads only the
    // norms and biases (a few MB) per layer instead of the 14 GB of matrices.
    bool warm_device_weights();
    void load_layer_shallow(int layer, LayerWeights& w);

    // Decides — once, at construction — whether the CPU path holds every layer
    // resident. Called from the constructor; enable_gpu() re-runs it only to
    // report, never to decide.
    //
    // This used to live inside enable_gpu(), which meant a caller that never
    // enabled a GPU never evaluated it: resident_ stayed empty and run_layer()
    // reloaded all seven matrices of every layer from the safetensors on every
    // layer of every token. Measured on the published 0.1.34 wheel, CPU only,
    // Qwen2.5-0.5B: 118.9 s for a 32-token prefill (0.27 tok/s), against
    // 339 tok/s of prefill with the GPU. The residency condition itself was
    // TRUE (1.43 GB estimated against a 16.8 GB budget) — it was simply never
    // reached, because it sat behind `if (gpu_ == nullptr)` inside a function
    // the CPU-only caller has no reason to call.
    void decide_cpu_residency();

    // Core ops.
    void run_layer(int layer, std::vector<float>& h, int seq_len,
                   std::vector<float>& k_cache, std::vector<float>& v_cache,
                   int cache_pos);
};

} // namespace winnex_nano

#endif // WINNEX_NANO_FORWARD_HPP
