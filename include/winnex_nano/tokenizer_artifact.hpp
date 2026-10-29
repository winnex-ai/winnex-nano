/**
 * tokenizer_artifact.hpp — persistência do EngineTokenizer, POR MODELO.
 *
 * O PROBLEMA
 * ----------
 * O EngineTokenizer constrói um índice Madhava sobre o `embed_tokens` do
 * modelo. O build custa ~15 s no Qwen2.5-0.5B (151.936 x 896) e cresce com
 * vocab x dim. Pagar isso a cada processo é inviável.
 *
 * A solução: o índice vira um ARTEFATO em disco, um por modelo. Medido nesta
 * máquina: carregar custa 0,01 s contra 0,80 s de build (80x) num corpus de
 * 20k x 256; no Qwen 0.5B são ~15 s contra ~0,05 s.
 *
 * MULTIMODELO, SEM HARDCODING
 * ---------------------------
 * Não existe "o" arquivo de tokenizer — existe um POR MODELO, e o chamador diz
 * qual. Nada aqui presume "model.embed_tokens.weight" nem uma família:
 *
 *   - o nome do tensor vem do chamador, com uma lista de fallbacks conhecidos
 *     e ERRO se nada casar (nunca um default silencioso);
 *   - vocab e dim vêm do checkpoint, não de constante;
 *   - o cabeçalho guarda a identidade do modelo e `verify_matches` RECUSA o
 *     pareamento errado.
 *
 * O guard do pareamento é o ponto: carregar o índice do modelo A e buscar com
 * os embeddings do modelo B devolve ids válidos e errados, sem nenhum sinal.
 * É o erro mais caro do multimodelo, e o único que o formato do arquivo não
 * pega sozinho.
 *
 * O QUE É SERIALIZADO
 * -------------------
 * O índice Madhava (projeções int8 + scales + residuals + config) e um
 * cabeçalho de identidade. O `embed_tokens` NÃO é duplicado: ele já está no
 * checkpoint, e duplicá-lo custaria centenas de MB por artefato. Consequência:
 * carregar exige o checkpoint presente — daí a verificação de identidade.
 *
 * BSL 1.1 | pay@winnex.ai
 */
#ifndef WINNEX_NANO_TOKENIZER_ARTIFACT_HPP
#define WINNEX_NANO_TOKENIZER_ARTIFACT_HPP

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace winnex_nano {

/** Identidade de um modelo, como o artefato a registra. */
struct ModelIdentity {
    std::string model_name;   // livre, do chamador: "qwen2.5-0.5b"
    std::string tensor_name;  // "model.embed_tokens.weight"
    std::string engine;       // "MADHAVA_INDEX_V1"
    int vocab = 0;
    int dim = 0;
    std::string fingerprint;  // sha256 amostrado do embed_tokens

    bool compatible_with(const ModelIdentity& o) const;
    /** Descrição legível da divergência; "" se compatíveis. */
    std::string mismatch(const ModelIdentity& o) const;
};

/** Nomes de tensor de embedding conhecidos, em ordem de tentativa.
 *  Não é a única fonte válida: modelos fora dessas famílias exigem o nome
 *  explícito, e o resolver lança se nenhum casar. */
std::vector<std::string> known_embed_tensor_names();

/**
 * Resolve o nome do tensor de embedding.
 *
 * Ordem: `explicit_name` (se não-vazio) → lista conhecida.
 * LANÇA std::invalid_argument se nada casar. Nunca um default silencioso: um
 * tokenizer que escolhe o tensor errado produz ids válidos e sem sentido.
 */
std::string resolve_embed_tensor_name(
    const std::string& explicit_name,
    const std::function<bool(const std::string&)>& has);

/**
 * Serializa um índice Madhava + a identidade do modelo num arquivo.
 *
 * O índice cabe num arquivo; o `embed_tokens` fica no checkpoint. A classe
 * não constrói a engine — quem constrói é o EngineTokenizer — ela só lê e
 * escreve o índice e confere a identidade.
 */
class TokenizerArtifact {
public:
    TokenizerArtifact() = default;
    ~TokenizerArtifact() = default;
    TokenizerArtifact(const TokenizerArtifact&) = delete;
    TokenizerArtifact& operator=(const TokenizerArtifact&) = delete;

    /** Nome de arquivo canônico para um modelo, dentro de `dir`.
     *  Sanitiza o nome: barras, espaços e ".." não escapam do diretório. */
    static std::string default_path(const std::string& dir,
                                    const std::string& model_name);

    /** Cabeçalho de um artefato, sem tocar no índice. Lança se ilegível. */
    static ModelIdentity peek(const std::string& path);

    /** Salva um índice já construído, com a identidade dada. */
    bool save(const std::string& path, const ModelIdentity& id,
              const std::string& madhava_index_path) const;

    /** Carrega o índice para um arquivo temporário e devolve o caminho dele.
     *  O chamador passa esse caminho a MadhavaL2::load_index(). */
    std::string extract_index_to(const std::string& artifact_path,
                                const std::string& tmp_path) const;

    /** sha256 amostrado do embed_tokens: primeiras e últimas linhas. Não é o
     *  hash do arquivo inteiro (545 MB no Qwen) — é o suficiente para
     *  distinguir dois modelos, que é o que o guard precisa. */
    static std::string fingerprint(const float* embed_tokens, int vocab, int dim);
};

/**
 * Confere que o artefato pertence a ESTE embed_tokens.
 *
 * Lança std::invalid_argument com a divergência se não pertencer. É o guard
 * que impede o erro mais caro do multimodelo: carregar o índice do modelo A e
 * buscar com os embeddings do modelo B devolve ids válidos e errados, sem
 * nenhum sinal. O formato do arquivo é válido nos dois casos; só a identidade
 * distingue.
 */
void verify_artifact_matches(const ModelIdentity& artifact_id,
                             const float* embed_tokens, int vocab, int dim,
                             const std::string& tensor_name);

} // namespace winnex_nano

#endif // WINNEX_NANO_TOKENIZER_ARTIFACT_HPP
