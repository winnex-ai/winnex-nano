// embed_bridge.hpp — a ponte entre uma codificação sem vocabulário e o espaço
// de embeddings de um modelo pré-treinado.
//
// ══════════════════════════════════════════════════════════════════════════
// O GARGALO QUE ISTO RESOLVE, E O QUE NÃO RESOLVE
// ══════════════════════════════════════════════════════════════════════════
//
// Medido em Qwen2.5-0.5B: alimentar o ForwardEngine com a codificação espectral
// (histograma ou HRR) produz top-1 do próximo token errado em 4 de 4 prompts,
// contra 2 de 4 do BPE. Não é bug de implementação — é diferença de contrato.
//
//   · O tokenizador produz o embedding de uma STRING, deterministicamente.
//   · O modelo espera o embedding de um TOKEN: uma linha de embed_tokens, que
//     vive numa variedade aprendida dos dados (os 16 primeiros componentes
//     explicam só 8,96% da energia das 151.936 linhas no 0.5B).
//
// Nenhuma projeção ORTOGONAL resolve: P = XXᵀ preserva o que recebe, e o que
// recebe já está fora da variedade. Ancorar à norma média também não resolve —
// medido, muda a falha ('\n', 'e', 'i') e não o resultado: a direção está
// errada, não a escala.
//
// O que resolve é APRENDER o mapa  f(espectral) → espaço de embeddings. Esta
// classe faz exatamente isso, por mínimos quadrados regularizados, e não
// substitui nada: é um objeto novo que o chamador pode usar ou ignorar.
//
// O que isto NÃO é: não é "substituir o BPE com geometria". É treino. A matriz
// W e o offset b são ajustados por regressão sobre pares (codificação, alvo), e
// a qualidade depende inteiramente de quais alvos foram fornecidos. A API pede
// os alvos explicitamente justamente para que isso não fique implícito.
//
// BSL 1.1 | pay@winnex.ai
#ifndef WINNEX_NANO_EMBED_BRIDGE_HPP
#define WINNEX_NANO_EMBED_BRIDGE_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace winnex_nano {

/**
 * EmbedBridge — regressão linear (com opção de kernel RBF) de uma codificação
 * de entrada para o espaço de embeddings de um modelo.
 *
 *   y ≈ W·x + b          (kernel="linear", padrão)
 *   y ≈ b + Σ_i α_i·k(x, c_i)   (kernel="rbf", aproximação por centros)
 *
 * `linear` resolve quando o mapa é aproximadamente linear; `rbf` quando não é.
 * Em ambos os casos o custo de aplicação é O(d_in · d_out) por token, ou
 * O(n_centers · d_in) + O(n_centers · d_out) no RBF.
 *
 * O QUE ESTA CLASSE É
 * -------------------
 * Uma ferramenta de ALINHAMENTO DE ESPAÇO VETORIAL para fins de
 * COMPATIBILIDADE: leva um vetor do espaço A para o espaço B quando os dois já
 * descrevem o mesmo conteúdo de maneiras compatíveis. O caso de uso é
 * integração — fazer um vetor calculado pelo motor conversar com um índice
 * construído por outro modelo, sem reindexar o corpus.
 *
 * O QUE ESTA CLASSE NÃO É
 * -----------------------
 * Não é — e não substitui — um codificador semântico treinado. Ela não aprende
 * significado: ajusta uma regressão supervisionada entre duas representações
 * que já existem. Se a origem não carrega o sinal, a ponte não o cria; ela
 * apenas move a fronteira de onde o sinal já estava.
 *
 * Isso é medido, não suposto. Com o tokenizer espectral como origem
 * (encode_hrr_complex, χⱼ(b)=e^{i2πjb/256}, linear, posto 128 — preserva a
 * estrutura ADITIVA dos bytes, não o significado), o cos de holdout é ~0.12
 * contra ~0.00 do acaso: acima do acaso, e muito abaixo de "alinhar os
 * espaços". Um R² de treino baixo sobre essa origem é o resultado ESPERADO,
 * não um defeito a corrigir com mais treino ou menos regularização.
 *
 * Portanto: o caminho de recuperação usa o ENCODER TREINADO, não a ponte. Uma
 * regressão linear sobre um codificador sem semântica não inventa a informação
 * que não está lá.
 */
class EmbedBridge {
public:
    enum class Kernel { Linear, Rbf };

    EmbedBridge() = default;

    /**
     * Ajusta a ponte.
     *
     *   X      — [n, d_in]   as codificações (linhas)
     *   Y      — [n, d_out]  os alvos: embeddings reais, ou o centroide do
     *                        token que o modelo esperaria naquela posição
     *   lambda — regularização ridge. 0 = mínimos quadrados puros (instável se
     *            n <= d_in); valores típicos 1e-3 a 1e1 dependendo da escala.
     *
     *            O default é 3.0, NÃO 1e-2 (2026-10-29). Medido com o caso de
     *            uso real — semântica MiniLM (384d) → hidden do Qwen 0.5B
     *            (896d), 147 textos clínicos, split 80/20:
     *
     *              lambda   R² treino   cos TESTE (texto não visto)
     *              0.00       1.0000       0.1166   (decora)
     *              0.01       0.9990       0.6209   (default antigo)
     *              0.50       0.9457       0.7237
     *              3.00       0.8392       0.7495   <- default novo
     *              10.0       0.7713       0.7409
     *
     *            A ponte é usada com n << d_in (poucos textos, muitas
     *            dimensões). Nesse regime o ridge SEM regularização
     *            suficiente decora o treino e não generaliza — o R² de treino
     *            de 0,999 do default antigo é o sintoma, não a qualidade.
     *            Quem tem muitos dados pode baixar o lambda explicitamente.
     *   kernel — Linear ou Rbf
     *   n_centers — só para Rbf: número de centros (k-means leve sobre X)
     *
     * LANÇA std::invalid_argument se n == 0, se as dimensões não baterem, ou se
     * algum valor for não-finito. Uma ponte ajustada sobre dados corrompidos
     * produz saída plausível e errada, então a validação é no ajuste.
     */
    void fit(const float* X, const float* Y, int n, int d_in, int d_out,
             double lambda = 3.0,
             Kernel kernel = Kernel::Linear,
             int n_centers = 0);

    /**
     * Ajusta a partir de pares texto→embedding, usando um codificador que o
     * chamador fornece. Existe para que o caminho comum (texto → ponte) não
     * exija que o usuário materialize a matriz X na mão.
     *
     * `encode` devolve d_in floats para o texto; `target` devolve d_out. Ambos
     * são funções do chamador porque esta classe não sabe nada sobre
     * tokenizadores nem sobre arquivos de modelo.
     */
    void fit_from_texts(
        const std::vector<std::string>& texts,
        const std::vector<float>& targets,      // [n, d_out], row-major
        int d_in, int d_out,
        void (*encode)(const std::string&, float*, void*), void* encode_ctx,
        double lambda = 3.0,
        Kernel kernel = Kernel::Linear,
        int n_centers = 0);

    /** Aplica a ponte a um vetor de entrada. `out` recebe out_dim() floats. */
    void apply(const float* x, float* out) const;

    /** Aplica a uma matriz [n, d_in]. `out` recebe [n, out_dim()]. */
    void apply_batch(const float* X, float* out, int n) const;

    /** Conveniência: codifica com `encode` e aplica. */
    void encode_and_apply(const std::string& text,
                          void (*encode)(const std::string&, float*, void*),
                          void* encode_ctx, float* out) const;

    bool fitted() const { return fitted_; }
    int in_dim() const { return d_in_; }
    int out_dim() const { return d_out_; }
    int n_samples() const { return n_fit_; }
    double train_rmse() const { return train_rmse_; }
    /** 1 - RMSE/σ(Y): quanto da variância do alvo a ponte explica. */
    double train_r2() const { return train_r2_; }
    Kernel kernel() const { return kernel_; }
    int n_centers() const { return (int)centers_.size() / (d_in_ > 0 ? d_in_ : 1); }

    /** Serializa/restaura para não reajustar a cada processo. */
    void save(const std::string& path) const;
    static EmbedBridge load(const std::string& path);

    /** Texto para diagnóstico em log. */
    std::string describe() const;

private:
    bool fitted_ = false;
    Kernel kernel_ = Kernel::Linear;
    int d_in_ = 0, d_out_ = 0, n_fit_ = 0;
    double lambda_ = 3.0;
    double train_rmse_ = 0.0, train_r2_ = 0.0;

    // Linear: W [d_in, d_out] row-major, b [d_out].
    std::vector<float> W_, b_;
    // Rbf: centers [k, d_in], alpha [k, d_out], gamma.
    std::vector<float> centers_, alpha_;
    double gamma_ = 1.0;
};

} // namespace winnex_nano

#endif // WINNEX_NANO_EMBED_BRIDGE_HPP
