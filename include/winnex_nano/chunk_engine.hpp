/**
 * chunk_engine.hpp — ChunkEngine: geração por chunks com memória de trabalho
 * Madhava.
 *
 * ── O QUE ESTA CLASSE É, E O QUE ELA NÃO É ────────────────────────────────
 *
 * O nome anterior era `StreamEngine`, e ele descrevia a coisa errada. "Stream"
 * em inferência de LLM quer dizer uma coisa específica: carregar os pesos em
 * fluxo, uma camada por vez, para rodar um modelo maior que a memória. **Esta
 * classe não faz isso e nunca fez.** Quem procurasse aqui o caminho para rodar
 * um 70B perderia tempo — foi exatamente o que aconteceu numa análise interna.
 *
 * O que ela faz:
 *
 *   1. Codifica o chunk de texto corrente como vetor espectral (~0,5 ms).
 *   2. Faz o fast build do contexto acumulado num engine Madhava (~6 ms).
 *   3. Busca top-K guiado pelo contexto (~0,05 ms).
 *   4. Emite o chunk pelo sink (o escritor SSE se conecta aqui).
 *   5. Repete até a condição de parada.
 *
 * Ela é uma MÁQUINA DE CHUNKS com memória de trabalho vetorial. Quem decide
 * qual é o próximo chunk é uma função que o chamador fornece (`next_chunk_fn`)
 * — a classe não carrega modelo de linguagem nenhum e não gera texto por si.
 *
 * Para inferência de verdade use `ForwardEngine` (`prefill` / `step`), que é o
 * motor nativo e está em forward.hpp. Para o modo de pesos que faz streaming
 * real por camada, veja `ForwardEngine::weight_mode() == "stream"` — e note no
 * README que esse modo é MAIS LENTO que a CPU quando o modelo não cabe no
 * device.
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
 */
#ifndef WINNEX_NANO_CHUNK_ENGINE_HPP
#define WINNEX_NANO_CHUNK_ENGINE_HPP

#include <functional>
#include <string>
#include <vector>

#include "winnex_nano/spectral_tokenizer.hpp"

namespace winnex_nano {

// Um chunk emitido pelo ChunkEngine.
struct Chunk {
    std::string text;      // the chunk content
    bool done = false;     // true on the final chunk (EOS)
    int doc_id = -1;       // the Madhava doc index that produced this chunk
    double build_ms = 0;   // latency of the Madhava build for this step
    double search_ms = 0;  // latency of the top-K search
};

// Callback que recebe cada chunk (o escritor SSE se conecta aqui).
using ChunkSink = std::function<void(const Chunk&)>;

/**
 * ChunkEngine — generates a response chunk-by-chunk.
 *
 * The engine is MODEL-AGNOSTIC: it works on the spectral representation and
 * the Madhava bound engine, so it does not depend on any specific LLM
 * architecture (Qwen/BERT/DeepSeek/GPT). The actual "what chunk to emit next"
 * is provided by the caller via `next_chunk_fn` (the generation policy) —
 * this decouples the streaming machinery from the generation source.
 */
class ChunkEngine {
public:
    struct Options {
        int embed_dim = 64;        // spectral modes per char
        int top_k = 3;             // Madhava search k
        double build_interval = 0; // rebuild every N chunks (0 = every chunk)
    };

    ChunkEngine();                       // uses default Options
    explicit ChunkEngine(const Options& opts);

    // Executa a geração: produz chunks e chama `sink` para cada um.
    //   prompt: the user message (seed context).
    //   next_chunk_fn: given the current accumulated context text and the
    //     Madhava search results, returns the next text chunk (or "" for EOS).
    //   sink: receives each emitted chunk.
    // Returns the full generated text.
    std::string generate_chunks(
        const std::string& prompt,
        const std::function<std::string(const std::string& context,
                                        const std::vector<std::string>& top_docs)>& next_chunk_fn,
        const ChunkSink& sink);

private:
    Options opts_;
    SpectralTokenizer tokenizer_;
};

// ── ALIAS DE COMPATIBILIDADE ──────────────────────────────────────────────
//
// Quem instalou antes da 0.1.33 pode ter `StreamEngine` no código. O alias
// mantém isso funcionando, mas o nome correto é `ChunkEngine` e ele é o único
// documentado. Depreciado: deve sumir na 0.2.
// O método também: `stream()` virou `generate_chunks()` — o alias do método
// fica só no binding Python, onde dois nomes podem apontar para o mesmo wrapper.
using StreamEngine = ChunkEngine;
using StreamOptions = ChunkEngine::Options;
using StreamChunk = Chunk;

} // namespace winnex_nano

#endif // WINNEX_NANO_CHUNK_ENGINE_HPP
