// test_stream.cpp — validates the Winnex-Nano streaming engine: the chunk
// generation loop using the Madhava fast build as incremental working memory.
#include "winnex_nano/stream_engine.hpp"

#include <chrono>
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

    if (failures == 0) {
        printf("\nALL STREAM ENGINE TESTS PASSED\n");
        return 0;
    }
    printf("\n%d FAILURE(S)\n", failures);
    return 1;
}
