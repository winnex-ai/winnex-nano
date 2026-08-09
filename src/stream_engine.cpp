// stream_engine.cpp — Winnex-Nano streaming engine (chunk via fast build).
#include "winnex_nano/stream_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>

#include "winnex_madhava/winnex_madhava.hpp"

namespace winnex_nano {

namespace {
// Converts a spectral-encoded chunk (sequence of quaternions) into a flat
// float vector (the "embedding" the Madhava engine indexes). We use the
// w/x complex plane magnitude per mode as the coordinate — the deterministic
// spectral signature of the text.
std::vector<float> chunk_to_vector(const std::vector<Quat>& states, int embed_dim) {
    std::vector<float> vec(embed_dim, 0.0f);
    if (states.empty()) return vec;
    const size_t n_chars = states.size() / static_cast<size_t>(embed_dim);
    // Average the per-mode magnitude over the characters of the chunk.
    for (size_t c = 0; c < n_chars; ++c) {
        for (int j = 0; j < embed_dim; ++j) {
            const Quat& q = states[c * static_cast<size_t>(embed_dim) + j];
            float mag = std::sqrt(q.w * q.w + q.x * q.x);
            vec[j] += mag;
        }
    }
    const float inv = n_chars > 0 ? 1.0f / n_chars : 1.0f;
    for (auto& v : vec) v *= inv;
    // Normalize (Madhava cosine expects normalized embeddings).
    float norm = 0.0f;
    for (auto v : vec) norm += v * v;
    norm = std::sqrt(norm) + 1e-9f;
    for (auto& v : vec) v /= norm;
    return vec;
}
} // namespace

StreamEngine::StreamEngine()
    : opts_(), tokenizer_(opts_.embed_dim) {}

StreamEngine::StreamEngine(const Options& opts)
    : opts_(opts), tokenizer_(opts.embed_dim) {}

std::string StreamEngine::stream(
    const std::string& prompt,
    const std::function<std::string(const std::string&, const std::vector<std::string>&)>& next_chunk_fn,
    const ChunkSink& sink) {

    // Working memory: the accumulated context text and its spectral vectors.
    std::string context = prompt;
    std::vector<std::vector<float>> corpus_vectors;   // one per context chunk
    std::vector<std::string> corpus_texts;            // aligned text

    // Seed the corpus with the prompt (its spectral vector).
    {
        auto states = tokenizer_.encode(prompt);
        corpus_vectors.push_back(chunk_to_vector(states, opts_.embed_dim));
        corpus_texts.push_back(prompt);
    }

    std::string full_output;

    int steps = 0;
    const int max_steps = 64; // safety cap on the autoregressive loop
    while (steps < max_steps) {
        // 1. Build the current corpus with the Madhava fast build.
        //    (n x dim) float32 normalized — Madhava quantizes internally.
        auto build_start = std::chrono::high_resolution_clock::now();
        std::vector<float> flat;
        flat.reserve(corpus_vectors.size() * opts_.embed_dim);
        for (const auto& v : corpus_vectors) flat.insert(flat.end(), v.begin(), v.end());
        const int n = (int)corpus_vectors.size();
        const int dim = opts_.embed_dim;

        std::unique_ptr<winnex_madhava::MadhavaL2> engine;
        if (n > 0) {
            // Build a bound engine over the accumulated context (cosine).
            winnex_madhava::Config cfg;
            cfg.dim = dim;
            cfg.metric = winnex_madhava::Metric::Cosine;
            cfg.k = opts_.top_k;
            cfg.stage1_dim = 32;
            cfg.stage2_dim = 64;
            cfg.k1_fraction = 0.5;
            cfg.k2_fraction = 0.2;
            cfg.k2_max = 256;
            cfg.modulation = true;
            cfg.postfilter = true;
            cfg.normalize_input = true;
            cfg.early_exit = true;
            engine = std::make_unique<winnex_madhava::MadhavaL2>(cfg);
            // uint8 corpus: quantize [-1,1] normalized floats to [0,255].
            std::vector<uint8_t> u8(flat.size());
            for (size_t i = 0; i < flat.size(); ++i) {
                float v = (flat[i] + 1.0f) * 127.5f;
                u8[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
            }
            engine->build(u8.data(), n);
        }
        auto build_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - build_start).count();

        // 2. Search top-K guided by the accumulated context (query = last chunk).
        std::vector<std::string> top_docs;
        double search_ms = 0.0;
        if (engine) {
            auto search_start = std::chrono::high_resolution_clock::now();
            const std::vector<float>& q = corpus_vectors.back();
            // Query is already normalized; pass float32 to the engine.
            winnex_madhava::SearchResult res = engine->search(q.data());
            search_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - search_start).count();
            for (int idx : res.indices) {
                if (idx >= 0 && idx < (int)corpus_texts.size()) {
                    top_docs.push_back(corpus_texts[idx]);
                }
            }
        }

        // 3. Ask the generation policy for the next chunk.
        std::string chunk = next_chunk_fn(context, top_docs);
        if (chunk.empty()) break; // EOS

        // 4. Emit the chunk (SSE) and add it to the working memory.
        StreamChunk sc;
        sc.text = chunk;
        sc.done = false;
        sc.build_ms = build_ms;
        sc.search_ms = search_ms;
        sc.doc_id = (int)corpus_texts.size() - 1;
        if (sink) sink(sc);

        full_output += chunk;
        context += chunk;

        // Index the new chunk (incremental working memory).
        auto states = tokenizer_.encode(chunk);
        corpus_vectors.push_back(chunk_to_vector(states, opts_.embed_dim));
        corpus_texts.push_back(chunk);

        ++steps;
    }

    // Emit the final done chunk.
    if (sink) {
        StreamChunk sc;
        sc.text = "";
        sc.done = true;
        sink(sc);
    }
    return full_output;
}

} // namespace winnex_nano
