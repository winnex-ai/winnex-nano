/**
 * stream_engine.hpp — Winnex-Nano streaming engine.
 *
 * Generates a response by CHUNKS, using the winnex-madhava fast build as an
 * incremental working memory:
 *
 *   1. Encode the current text chunk to a spectral vector (deterministic,
 *      ~0.5ms — no HTTP, no external model).
 *   2. Fast-build the accumulated context into a Madhava bound engine
 *      (~6ms for 5 chunks — the Madhava differentiator).
 *   3. Search top-K guided by the accumulated context (~0.05ms).
 *   4. Emit the chunk via SSE (OpenAI-compatible stream).
 *   5. Repeat until the stop condition.
 *
 * The Madhava engine becomes an autoregressive per-chunk working memory —
 * deterministic, instant, and reusing the winnex-madhava kernels (no code
 * duplication).
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
 */
#ifndef WINNEX_NANO_STREAM_ENGINE_HPP
#define WINNEX_NANO_STREAM_ENGINE_HPP

#include <functional>
#include <string>
#include <vector>

#include "winnex_nano/spectral_tokenizer.hpp"

namespace winnex_nano {

// A generated chunk emitted by the stream engine.
struct StreamChunk {
    std::string text;      // the chunk content
    bool done = false;     // true on the final chunk (EOS)
    int doc_id = -1;       // the Madhava doc index that produced this chunk
    double build_ms = 0;   // latency of the Madhava build for this step
    double search_ms = 0;  // latency of the top-K search
};

// Callback receiving each streamed chunk (the SSE writer hooks here).
using ChunkSink = std::function<void(const StreamChunk&)>;

/**
 * StreamEngine — generates a response chunk-by-chunk.
 *
 * The engine is MODEL-AGNOSTIC: it works on the spectral representation and
 * the Madhava bound engine, so it does not depend on any specific LLM
 * architecture (Qwen/BERT/DeepSeek/GPT). The actual "what chunk to emit next"
 * is provided by the caller via `next_chunk_fn` (the generation policy) —
 * this decouples the streaming machinery from the generation source.
 */
class StreamEngine {
public:
    struct Options {
        int embed_dim = 64;        // spectral modes per char
        int top_k = 3;             // Madhava search k
        double build_interval = 0; // rebuild every N chunks (0 = every chunk)
    };

    StreamEngine();                       // uses default Options
    explicit StreamEngine(const Options& opts);

    // Runs the stream: generates chunks and calls `sink` for each.
    //   prompt: the user message (seed context).
    //   next_chunk_fn: given the current accumulated context text and the
    //     Madhava search results, returns the next text chunk (or "" for EOS).
    //   sink: receives each emitted chunk.
    // Returns the full generated text.
    std::string stream(
        const std::string& prompt,
        const std::function<std::string(const std::string& context,
                                        const std::vector<std::string>& top_docs)>& next_chunk_fn,
        const ChunkSink& sink);

private:
    Options opts_;
    SpectralTokenizer tokenizer_;
};

} // namespace winnex_nano

#endif // WINNEX_NANO_STREAM_ENGINE_HPP
