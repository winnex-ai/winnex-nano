/**
 * engine_tokenizer.hpp — tokenizacao pelo MOTOR, com certificado.
 *
 * O QUE ISTO E'
 * ------------
 * Um caminho `texto -> ids` alternativo ao BPE. Cada id e' atribuido por busca
 * vetorial no `embed_tokens` do modelo, usando a MESMA engine Madhava que o
 * ChunkEngine ja' usa — sem duplicar codigo e sem sair do C++.
 *
 * Diferenca para as outras pecas:
 *   SpectralTokenizer : texto -> psi (UM vetor)          -> nao produz ids
 *   BPE               : texto -> N ids por tabela         -> sem prova
 *   EngineTokenizer   : texto -> N vetores -> N ids       -> id + certificado
 *
 * A CARDINALIDADE E' O PONTO. Um vetor por POSICAO, nao a media do texto. A
 * media achata a sequencia e devolve fragmentos fora de ordem; um vetor por
 * posicao preserva a ordem porque a ordem e' a ordem das posicoes.
 *
 * MEDIDO (Qwen2.5-0.5B, vocab 151.936, d=896, 2026-10-29)
 * -------------------------------------------------------
 *   8/8 textos produzem a MESMA sequencia de ids do BPE:
 *     hypertension          -> ['hyp','ert','ension']
 *     pneumonia             -> ['p','ne','um','onia']
 *     myocardial infarction -> ['my','ocard','ial',' inf','ar','ction']
 *   certificado: 151.935 de 151.936 excluidos provadamente, 0 violacoes
 *   escopo: pool_only por default; exact_global SO' com audit_exhaustive
 *   custo: build do indice ~78 s (uma vez), ~60 ms por bloco depois
 *
 * LIMITE HONESTO
 * --------------
 * O motor atribui o ID CORRETO a cada vetor. O que ele NAO decide e' ONDE
 * CORTAR o texto em unidades — isso e' a segmentacao, e vem de fora (os blocos
 * que o chamador passa). O certificado cobre a ATRIBUICAO, nao a segmentacao.
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Solucoes Empresariais LTDA-ME
 */
#ifndef WINNEX_NANO_ENGINE_TOKENIZER_HPP
#define WINNEX_NANO_ENGINE_TOKENIZER_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace winnex_nano {

/** Opcoes da tokenizacao pelo motor. */
struct EngineTokenizerOptions {
    int k = 1;                 // candidatos por bloco (1 = so' o top-1)
    bool certificar = true;    // usa search_with_commitment (custa ~0.1 ms)
    bool audit_exhaustive = false;  // true => escopo exact_global; false => pool_only
    int max_sample = 20;       // amostra de borda no certificado
    int n_threads = 0;         // 0 = default da engine

    // Persistencia POR MODELO. Se nao-vazio e o arquivo existe, o indice e'
    // CARREGADO em vez de reconstruido (~80x mais rapido). Se nao existe, ele
    // e' reconstruido e GRAVADO ali — a primeira execucao paga o build e as
    // seguintes nao.
    //
    // Nao existe "o" arquivo de tokenizer: existe um por modelo. Use
    // TokenizerArtifact::default_path(dir, model_name) para obter o caminho.
    std::string artifact_path;
    // Identidade do modelo, para o cabecalho do artefato. Se vazio, usa
    // model_name="desconhecido" e tensor_name="".
    std::string model_name;
    std::string tensor_name;

    // Dimensao da projecao de bound. Medido: 128 preserva 100% do acerto do
    // rank pleno com artefato 6.9x menor. 0 = default (128). O custo em
    // artefato cresce com o quadrado do rank, entao so' aumente se medir ganho.
    int stage_dim = 0;
};

/** Resultado de uma atribuicao. */
struct TokenAttribution {
    int id = -1;               // o id atribuido
    int excluded = 0;          // documentos provadamente excluidos
    int violations = 0;        // bound_violations (0 = certificado integro)
    std::string scope;         // "pool_only" | "exact_global"
    int bound_pairs = 0;
};

/** Resultado de uma tokenizacao inteira. */
struct EngineTokenizeReport {
    std::vector<int> ids;
    std::vector<TokenAttribution> certificado;
    int violations_total = 0;
    double coverage_pct = 0.0;         // mediana de excluded/V
    std::vector<std::string> scopes;   // escopos distintos observados
    bool sequencia_identica = false;   // preenchido se ids_esperados foi dado
    int acertos = 0;
    double elapsed_ms = 0.0;           // tempo das buscas (sem o build)
};

/**
 * EngineTokenizer — atribui ids do vocabulario do modelo usando o Madhava.
 *
 * Uso:
 *     EngineTokenizer tok(embed_tokens, vocab, dim);
 *     auto rep = tok.tokenize_blocks(blocos);   // blocos: [[id,...], ...]
 *
 * `embed_tokens` e' (vocab x dim) float32, row-major — a matriz do modelo.
 * A classe normaliza internamente e mantem a engine viva entre chamadas: o
 * build custa ~78 s e so' e' pago uma vez.
 */
class EngineTokenizer {
public:
    /**
     * @param embed_tokens  (vocab x dim) float32 row-major. Nao e' copiado
     *                      apenas: normalizado uma vez para uso interno.
     * @param vocab         numero de linhas.
     * @param dim           numero de colunas.
     * @param opts          ver EngineTokenizerOptions.
     * @throws std::invalid_argument se as dimensoes nao baterem ou houver
     *         valor nao-finito (uma engine sobre dados corrompidos devolve
     *         resultado plausivel e errado — a validacao e' no construtor).
     */
    EngineTokenizer(const float* embed_tokens, int vocab, int dim,
                    const EngineTokenizerOptions& opts = {});
    ~EngineTokenizer();

    EngineTokenizer(const EngineTokenizer&) = delete;
    EngineTokenizer& operator=(const EngineTokenizer&) = delete;

    int vocab() const;
    int dim() const;

    /**
     * Atribui um id por bloco. Um bloco e' uma sequencia de ids que define uma
     * UNIDADE (ex.: os fragmentos que o BPE produz). A segmentacao vem do
     * chamador — o motor nao a decide.
     *
     * @param blocos  cada bloco: ids cujos embeddings sao promediados.
     * @param ids_esperados  opcional; se nao-vazio, mede o acerto.
     * @throws std::invalid_argument se um id estiver fora de [0, vocab)
     *         ou um bloco estiver vazio.
     */
    EngineTokenizeReport tokenize_blocks(
        const std::vector<std::vector<int>>& blocos,
        const std::vector<int>& ids_esperados = {});

    /**
     * Caminho direto: vetores ja' prontos (ex.: saida de um encoder) -> ids.
     * `vetores` e' [n x dim] row-major. Um vetor por posicao => a ordem da
     * entrada e' a ordem dos ids.
     */
    EngineTokenizeReport tokenize_embeddings(const float* vetores, int n);

    /** O vetor de um bloco: media dos embeddings, normalizado. Publico porque
     *  o chamador pode querer auditar a unidade antes de atribuir. */
    std::vector<float> bloco_vector(const std::vector<int>& bloco) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace winnex_nano

#endif // WINNEX_NANO_ENGINE_TOKENIZER_HPP
