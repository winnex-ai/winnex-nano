// test_forward.cpp — validates the native forward pass on a real model.
//
// Usage: test_forward <model_dir> <prompt>
//   model_dir must contain config.json + model.safetensors.
//
// The forward is driven by the config (arch-detected). Input: a hidden state
// sequence from the X-factor embedding of the prompt. Output: logits + next
// token id (argmax).
#include "winnex_nano/forward.hpp"
#include "winnex_nano/safetensors.hpp"
#include "winnex_nano/x_factor.hpp"
#include "winnex_nano/spectral_tokenizer.hpp"

#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace winnex_nano;

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s <model_dir> [prompt]\n", argv[0]); return 2; }
    std::string dir = argv[1];
    std::string prompt = argc > 2 ? argv[2] : "import numpy";

    try {
        ModelConfig cfg = load_config(dir + "/config.json");
        printf("config: %s | hidden=%d layers=%d heads=%d kv=%d inter=%d\n",
               cfg.arch.c_str(), cfg.hidden_size, cfg.num_hidden_layers,
               cfg.num_attention_heads, cfg.num_key_value_heads, cfg.intermediate_size);

        Safetensors st(dir + "/model.safetensors");
        printf("safetensors: %zu tensors\n", st.count());

        // X factor from embed_tokens (structural, sampled).
        auto emb = st.to_float(st.get("model.embed_tokens.weight"));
        int V = cfg.vocab_size, D = cfg.hidden_size;
        // Sample 10000 rows (like the validated reference).
        std::vector<float> Esample;
        std::mt19937 rng(42);
        for (int i = 0; i < 10000; ++i) {
            int row = (int)(rng() % V);
            for (int j = 0; j < D; ++j) Esample.push_back(emb[(size_t)row * D + j]);
        }
        XFactor xf(Esample.data(), 10000, D, 0.95);
        printf("X factor: rank=%d\n", xf.rank());

        // Embed the prompt: character histogram -> expand -> project.
        SpectralTokenizer tok(64);
        auto hist = tok.encode_histogram(prompt, 256);
        std::vector<float> expd(D, 0.0f);
        for (int j = 0; j < 256 && j < D; ++j) expd[j] = hist[j];
        float nrm = 0; for (auto v : expd) nrm += v*v;
        nrm = std::sqrt(nrm) + 1e-9f;
        for (auto& v : expd) v /= nrm;
        std::vector<float> hidden(D);
        xf.project(expd.data(), hidden.data());

        // Forward: single token (seq_len=1).
        ForwardEngine fe(cfg, st);
        std::vector<float> logits = fe.forward(hidden, 1);
        int next = ForwardEngine::argmax(logits);
        printf("forward OK | logits[0..3]=%.4f %.4f %.4f %.4f | next_token_id=%d\n",
               logits[0], logits[1], logits[2], logits[3], next);
        printf("params=%zu\n", fe.param_count());
        return 0;
    } catch (const std::exception& e) {
        printf("ERROR: %s\n", e.what());
        return 1;
    }
}
