// test_stream.cpp — validates the Winnex-Nano streaming engine: the chunk
// generation loop using the Madhava fast build as incremental working memory.
#include "winnex_nano/stream_engine.hpp"
#include "winnex_nano/spectral_tokenizer.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using winnex_nano::StreamEngine;

int main() {
    int failures = 0;

    StreamEngine engine(StreamEngine::Options{64, 3, 0});
    std::vector<std::string> emitted;
    double total_build_ms = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    std::string output = engine.stream(
        "O Madhava e um motor de busca vetorial deterministica.",
        // Generation policy: emit 3 fixed chunks, then EOS.
        [](const std::string& ctx, const std::vector<std::string>& top) {
            (void)ctx; (void)top;
            static int step = 0;
            const char* chunks[] = {
                " Primeira parte da resposta.",
                " Segunda parte baseada no contexto.",
                " Terceira parte concluindo.",
            };
            if (step < 3) return std::string(chunks[step++]);
            return std::string(); // EOS
        },
        [&](const winnex_nano::StreamChunk& c) {
            if (c.done) return;
            emitted.push_back(c.text);
            total_build_ms += c.build_ms;
        });
    double dt_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // 1. All 3 chunks were emitted.
    bool ok1 = (emitted.size() == 3);
    printf("3 chunks emitted: %s (%zu)\n", ok1 ? "OK" : "FAIL", emitted.size());
    if (!ok1) ++failures;

    // 2. The output concatenates the chunks.
    bool ok2 = (output.find("Primeira parte") != std::string::npos &&
                output.find("Terceira parte") != std::string::npos);
    printf("output concatenates chunks: %s\n", ok2 ? "OK" : "FAIL");
    if (!ok2) ++failures;

    // 3. The engine ran fast (per-chunk budget: build+encode+search << 100ms).
    //    The Madhava fast build is the differentiator.
    bool ok3 = (dt_ms < 500.0);
    printf("total stream < 500ms: %s (%.1f ms, build avg %.2f ms/chunk)\n",
           ok3 ? "OK" : "FAIL", dt_ms, total_build_ms / (emitted.empty() ? 1 : emitted.size()));
    if (!ok3) ++failures;

    // 4. Chunks are printed in order.
    bool ok4 = (emitted[0].find("Primeira") != std::string::npos);
    printf("chunk order correct: %s\n", ok4 ? "OK" : "FAIL");
    if (!ok4) ++failures;

    // 5. ANTI-CHEAT: different texts must produce different vectors.
    //
    //    This test did not exist, and its absence is why the engine shipped
    //    with a `chunk_to_vector` that returned a CONSTANT vector for any
    //    input: it averaged the magnitude sqrt(w²+x²) of a phase-only
    //    encoding, which is identically 1.0. Retrieval was therefore a tie
    //    among N identical vectors, while every test above still passed —
    //    they check shapes, ordering and timing, never that the representation
    //    represents anything.
    //
    //    Falsifies if: two distinct texts give cosine similarity ~1.
    {
        winnex_nano::SpectralTokenizer tok(64);
        const char* t[] = {"quantum physics", "banana bread recipe",
                           "O Madhava e deterministico.", "abc", "cba"};
        std::vector<std::vector<float>> v;
        for (auto* x : t) v.push_back(tok.encode_hrr_unit(x));
        double worst = 0.0;
        for (size_t i = 0; i < v.size(); ++i)
            for (size_t j = i + 1; j < v.size(); ++j) {
                double d = 0.0;
                for (size_t k = 0; k < 64; ++k) d += v[i][k] * v[j][k];
                if (std::fabs(d) > std::fabs(worst)) worst = d;
            }
        bool ok5 = (std::fabs(worst) < 0.90);
        printf("distinct texts give distinct vectors (max |cos| = %.4f < 0.90): %s\n",
               std::fabs(worst), ok5 ? "OK" : "FAIL");
        if (!ok5) ++failures;

        // Order must matter: "abc" and "cba" are not the same chunk.
        double d_order = 0.0;
        for (size_t k = 0; k < 64; ++k) d_order += v[3][k] * v[4][k];
        bool ok6 = (std::fabs(d_order) < 0.90);
        printf("order matters ('abc' vs 'cba' cos = %.4f < 0.90): %s\n",
               std::fabs(d_order), ok6 ? "OK" : "FAIL");
        if (!ok6) ++failures;
    }

    if (failures == 0) {
        printf("\nALL STREAM ENGINE TESTS PASSED\n");
        return 0;
    }
    printf("\n%d FAILURE(S)\n", failures);
    return 1;
}
