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
// Converts a chunk of TEXT into the vector the Madhava engine indexes.
//
// BUG FIX (2026-10-21). The previous version took the spectral-encoded
// quaternions and averaged their per-mode MAGNITUDE sqrt(w² + x²). That
// tokenizer encodes each byte as a PURE PHASE with amplitude 1.0
// (w = cos φ, x = sin φ), so w² + x² = 1 identically: the magnitude is 1.0
// for every character and every mode, the average is 1.0, and after
// normalisation the result is the CONSTANT vector 1/sqrt(embed_dim) for ANY
// text. Measured: cos("quantum physics", "banana bread recipe") = 1.000000,
// and cos("abc","cba") = 1.000000. The StreamEngine was indexing N identical
// vectors, so retrieval returned an arbitrary member of a tie.
//
// The information lives in the PHASE and in the ORDER of the characters.
// The spectral tokenizer already ships the primitive that preserves both:
// `encode_hrr_unit` binds each symbol to a per-mode frequency and each
// position to a position-dependent phase, then sums coherently (T. Plate,
// Holographic Reduced Representations, IEEE Trans. Neural Networks 1995).
// Measured on 64-d vectors, distinct texts land at cos 0.005–0.43 instead of
// 1.000, and "abc" vs "cba" at 0.38 — order is preserved.
//
// Using the existing, documented primitive is the fix: the engine's cosine
// metric requires different texts to be different vectors, and this is
// deterministic (fixed frequencies, no PRNG), order-sensitive, and normalised.
std::vector<float> text_to_vector(const std::string& text,
                                  const SpectralTokenizer& tok, int embed_dim) {
    std::vector<float> v = tok.encode_hrr_unit(text);
    if ((int)v.size() == embed_dim) return v;
    // Different tokenizer dim: resize deterministically (truncate / zero-pad)
    // and re-normalise so the cosine contract still holds.
    std::vector<float> out(static_cast<size_t>(std::max(embed_dim, 0)), 0.0f);
    const int n = std::min((int)v.size(), embed_dim);
    for (int i = 0; i < n; ++i) out[(size_t)i] = v[(size_t)i];
    double ss = 0.0;
    for (float x : out) ss += (double)x * x;
    const double nrm = std::sqrt(ss);
    if (nrm > 1e-12) for (float& x : out) x = (float)(x / nrm);
    return out;
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
        corpus_vectors.push_back(text_to_vector(prompt, tokenizer_, opts_.embed_dim));
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
            // BUG FIX (P0, mirrors winnex-madhava 1.9.3): early_exit must be
            // OFF. Forcing it True degraded recall to 0.73 in dim >= 384
            // because the modulated bound does not order like the exact score
            // when the bound is loose (curse of dimensionality). The safe
            // default is early_exit off (exact post-filter), which
            // guarantees recall = 1.0.
            cfg.early_exit = false;
            engine = std::make_unique<winnex_madhava::MadhavaL2>(cfg);
            // BUG FIX (G2b): pass the float32 normalized corpus DIRECTLY via
            // build_float32 — the Madhava engine's float32 manifold path
            // preserves cosine exactly. The previous uint8 quantization
            // (v+1)*127.5 destroyed the [-1,1] embedding domain and degraded
            // recall even at d=64. No integer offset DC.
            engine->build_float32(flat.data(), n);
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
        corpus_vectors.push_back(text_to_vector(chunk, tokenizer_, opts_.embed_dim));
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
