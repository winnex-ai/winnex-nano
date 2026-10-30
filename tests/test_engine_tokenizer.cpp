// test_engine_tokenizer.cpp — valida a tokenização pelo motor, em C++, sem Python.
//
// O que estes testes fixam (medido em 2026-10-29, Qwen2.5-0.5B):
//   * bloco de 1 id -> o motor devolve o MESMO id (o embedding é ele próprio)
//   * sequência de N blocos preserva a ORDEM (a ordem é a das posições)
//   * certificado: 0 violações e cobertura alta com rank pleno
//   * escopo é pool_only por default e exact_global só com audit_exhaustive
//   * guard-rails: id fora do vocabulário, bloco vazio, dimensão errada,
//     embedding com NaN, linha de norma zero
//
// Não depende de checkpoint: usa um embed_tokens sintético com estrutura de
// clusters, que é o regime em que os testes de contrato fazem sentido. A
// validação contra o BPE real fica no teste Python (tests/test_tokenize_engine.py),
// porque exige o tokenizer do modelo.
#include "winnex_nano/engine_tokenizer.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace winnex_nano;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    printf("%-58s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++failures;
}

// embed_tokens sintético: K clusters em D dimensões. Cada token pertence a um
// cluster e recebe ruído pequeno — a mesma estrutura que embeddings reais têm
// (energia concentrada), e o regime em que a atribuição por vizinhança faz
// sentido.
std::vector<float> make_embed(int V, int D, int K, unsigned seed = 7) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> centers((size_t)K * D);
    for (auto& c : centers) c = nd(rng);

    // RUÍDO RELATIVO À NORMA DO CENTRO, não absoluto.
    //
    // Com ruído absoluto de 0.05 sobre centros de norma ~sqrt(64)=8, dois ids
    // do mesmo cluster ficavam a cos 0.998 — indistinguíveis. O motor NÃO PODE
    // acertar o id nesse regime porque há vários candidatos com o mesmo vetor
    // (medido nesta máquina: vizinho mais próximo do id 0 tinha cos 0.9984).
    // O teste exigia o impossível, e a culpa era do gerador.
    //
    // Com 12% da norma do centro, ids do mesmo cluster ficam a cos ~0.97 e
    // ids de clusters diferentes a cos ~0: ainda agrupados, com vencedor
    // único. É o regime dos embeddings reais (Qwen: norma 0.29-0.81, vizinho
    // mais próximo ~0.40).
    std::vector<float> E((size_t)V * D);
    for (int i = 0; i < V; ++i) {
        const int k = i % K;
        const float* c = centers.data() + (size_t)k * D;
        float cn = 0.0f;
        for (int j = 0; j < D; ++j) cn += c[j] * c[j];
        cn = std::sqrt(cn);
        const float sigma = 0.12f * cn / std::sqrt((float)D);
        for (int j = 0; j < D; ++j)
            E[(size_t)i * D + j] = c[j] + sigma * nd(rng);
    }
    return E;
}

} // namespace

int main() {
    const int V = 512, D = 64, K = 32;
    std::vector<float> E = make_embed(V, D, K);

    // ── 1. contrato básico ────────────────────────────────────────────────
    {
        EngineTokenizer tok(E.data(), V, D);
        check(tok.vocab() == V, "vocab() devolve o vocabulario");
        check(tok.dim() == D, "dim() devolve a dimensao");

        EngineTokenizeReport r = tok.tokenize_blocks({{0}});
        check(r.ids.size() == 1, "tokenize_blocks devolve 1 id para 1 bloco");
        check(r.ids.size() == 1 && r.ids[0] >= 0 && r.ids[0] < V,
              "o id devolvido esta em [0, vocab)");
        check(r.certificado.size() == 1, "certificado tem 1 entrada");
    }

    // ── 2. atribuição exata para token único ──────────────────────────────
    // Um bloco de 1 id tem como vetor o próprio embedding do id (normalizado).
    // A busca exata sobre a mesma matriz tem de devolver o mesmo id.
    {
        EngineTokenizer tok(E.data(), V, D);
        int ok = 0, n = 0;
        for (int i = 0; i < V; i += 37) {
            EngineTokenizeReport r = tok.tokenize_blocks({{i}});
            ++n;
            ok += (r.ids.size() == 1 && r.ids[0] == i) ? 1 : 0;
        }
        char msg[96];
        snprintf(msg, sizeof msg,
                 "bloco de 1 id -> o MESMO id (%d/%d)", ok, n);
        check(ok == n, msg);
    }

    // ── 3. a ordem vem das posições ───────────────────────────────────────
    // N blocos -> N ids, na ordem em que os blocos foram passados. O teste usa
    // ids-espelho e exige a sequência idêntica, o que falha se qualquer coisa
    // reordenar.
    {
        EngineTokenizer tok(E.data(), V, D);
        std::vector<std::vector<int>> blocos;
        std::vector<int> esperado;
        for (int i = 0; i < 24; ++i) {
            const int id = (i * 23) % V;
            blocos.push_back({id});
            esperado.push_back(id);
        }
        EngineTokenizeReport r = tok.tokenize_blocks(blocos, esperado);
        check(r.sequencia_identica, "sequencia preserva a ordem dos blocos");
        check(r.acertos == (int)esperado.size(), "todos os ids batem");
    }

    // ── 4. bloco multi-id: a média preserva o cluster ─────────────────────
    // Dois ids do MESMO cluster -> o vetor médio fica perto do cluster; a
    // atribuição deve cair num id daquele cluster. É a propriedade que faz a
    // segmentação funcionar.
    {
        EngineTokenizer tok(E.data(), V, D);
        int ok = 0, n = 0;
        for (int k = 0; k < K; ++k) {
            const int a = k, b = k + K;      // mesmo cluster (i % K)
            EngineTokenizeReport r = tok.tokenize_blocks({{a, b}});
            if (r.ids.size() != 1) continue;
            ++n;
            ok += (r.ids[0] % K == k) ? 1 : 0;
        }
        char msg[96];
        snprintf(msg, sizeof msg,
                 "media de 2 ids do mesmo cluster -> id do cluster (%d/%d)", ok, n);
        check(ok == n, msg);
    }

    // ── 5. certificado ────────────────────────────────────────────────────
    {
        EngineTokenizer tok(E.data(), V, D);
        EngineTokenizeReport r = tok.tokenize_blocks({{3}, {11}, {29}});
        check(r.violations_total == 0, "certificado sem violacoes");
        check(r.coverage_pct > 50.0,
              "cobertura de exclusao acima de 50% (rank pleno)");
        bool escopo_ok = r.scopes.size() == 1 &&
                         (r.scopes[0] == "pool_only" || r.scopes[0] == "exact_global");
        check(escopo_ok, "escopo declarado e' um dos dois validos");
    }

    // ── 6. escopo: rank pleno NAO da' exact_global ────────────────────────
    // Medido no Qwen real: pool_only sem audit_exhaustive, exact_global com.
    // Este teste impede que a doc volte a prometer global a partir do rank.
    {
        EngineTokenizerOptions o;
        o.audit_exhaustive = false;
        EngineTokenizer a(E.data(), V, D, o);
        EngineTokenizeReport ra = a.tokenize_blocks({{5}});
        const bool a_pool = ra.scopes.size() == 1 && ra.scopes[0] == "pool_only";
        check(a_pool, "audit_exhaustive=false -> escopo pool_only");

        o.audit_exhaustive = true;
        EngineTokenizer b(E.data(), V, D, o);
        EngineTokenizeReport rb = b.tokenize_blocks({{5}});
        const bool b_glob = rb.scopes.size() == 1 && rb.scopes[0] == "exact_global";
        check(b_glob, "audit_exhaustive=true  -> escopo exact_global");
        check(rb.violations_total == 0, "exact_global continua com 0 violacoes");
    }

    // ── 7. tokenize_embeddings: sem bloco, vetor direto ───────────────────
    {
        EngineTokenizer tok(E.data(), V, D);
        // vetor do embedding do id 42, normalizado pela própria classe
        std::vector<float> v = tok.bloco_vector({42});
        EngineTokenizeReport r = tok.tokenize_embeddings(v.data(), 1);
        check(r.ids.size() == 1 && r.ids[0] == 42,
              "tokenize_embeddings(embed[42]) -> 42");

        // ordem preservada em lote
        std::vector<float> lote;
        std::vector<int> esperado;
        for (int i = 0; i < 8; ++i) {
            const int id = (i * 71) % V;
            std::vector<float> vv = tok.bloco_vector({id});
            lote.insert(lote.end(), vv.begin(), vv.end());
            esperado.push_back(id);
        }
        EngineTokenizeReport rl = tok.tokenize_embeddings(lote.data(),
                                                         (int)esperado.size());
        check(rl.ids == esperado, "lote de vetores preserva a ordem");
    }

    // ── 8. sem certificado ainda atribui ──────────────────────────────────
    {
        EngineTokenizerOptions o;
        o.certificar = false;
        EngineTokenizer tok(E.data(), V, D, o);
        EngineTokenizeReport r = tok.tokenize_blocks({{9}, {17}});
        check(r.ids.size() == 2, "certificar=false ainda devolve ids");
        check(r.certificado.size() == 2 && r.certificado[0].scope.size() > 0,
              "certificar=false ainda preenche o escopo");
    }

    // ── 9. guard-rails ────────────────────────────────────────────────────
    {
        EngineTokenizer tok(E.data(), V, D);
        bool ok_id = false;
        try { tok.tokenize_blocks({{V + 5}}); }
        catch (const std::invalid_argument&) { ok_id = true; }
        check(ok_id, "id fora do vocabulario -> invalid_argument");

        bool ok_neg = false;
        try { tok.tokenize_blocks({{-1}}); }
        catch (const std::invalid_argument&) { ok_neg = true; }
        check(ok_neg, "id negativo -> invalid_argument");

        bool ok_vazio = false;
        try { tok.tokenize_blocks({{}}); }
        catch (const std::invalid_argument&) { ok_vazio = true; }
        check(ok_vazio, "bloco vazio -> invalid_argument");

        bool ok_nulo = false;
        try { EngineTokenizer bad(nullptr, V, D); }
        catch (const std::invalid_argument&) { ok_nulo = true; }
        check(ok_nulo, "embed_tokens nulo -> invalid_argument");

        bool ok_dims = false;
        try { EngineTokenizer bad(E.data(), 0, D); }
        catch (const std::invalid_argument&) { ok_dims = true; }
        check(ok_dims, "vocab <= 0 -> invalid_argument");

        // linha de norma zero não tem direção para atribuir
        std::vector<float> Ez = E;
        for (int j = 0; j < D; ++j) Ez[(size_t)3 * D + j] = 0.0f;
        bool ok_zero = false;
        try { EngineTokenizer bad(Ez.data(), V, D); }
        catch (const std::invalid_argument&) { ok_zero = true; }
        check(ok_zero, "linha de norma zero -> invalid_argument");

        // NaN corrompe a engine em silêncio se não for barrado no construtor
        std::vector<float> En = E;
        En[(size_t)5 * D + 2] = std::nanf("");
        bool ok_nan = false;
        try { EngineTokenizer bad(En.data(), V, D); }
        catch (const std::invalid_argument&) { ok_nan = true; }
        check(ok_nan, "embedding com NaN -> invalid_argument");
    }

    // ── 10. determinismo ──────────────────────────────────────────────────
    // Duas instâncias sobre os mesmos dados e a mesma consulta têm de dar o
    // mesmo id. Sem isso o certificado não vale nada (não se pode auditar um
    // resultado que muda entre execuções).
    {
        EngineTokenizer a(E.data(), V, D);
        EngineTokenizer b(E.data(), V, D);
        EngineTokenizeReport ra = a.tokenize_blocks({{13, 45}, {77}, {101, 133}});
        EngineTokenizeReport rb = b.tokenize_blocks({{13, 45}, {77}, {101, 133}});
        check(ra.ids == rb.ids, "duas instancias -> mesma sequencia de ids");
    }

    // ── 11. artefato: a ordem de carga NAO pode mudar o resultado ─────────
    //
    // REGRESSAO MEDIDA (2026-10-30, Qwen3-4B, V=151.936, d=2560).
    //
    // O caminho quente chamava build_float32 ANTES de load_index, refazendo o
    // trabalho que o artefato traz pronto (build 7.4 s, load 0.04 s). Inverter
    // a ordem resolve — mas a primeira tentativa deixou corpus_f32_ = nullptr,
    // e entao exact_score() caiu no ramo uint8, que e' um espaco COM PERDA:
    // 16 de 40 consultas trocaram as posicoes 2..k do top-5 (so' o top-1
    // coincidia, porque a query e' o proprio vetor).
    //
    // Corrigir a ORDEM e corrigir a EXATIDAO sao duas exigencias: um ganho que
    // muda os ids nao e' ganho, e' outro software. Este teste cobra as duas de
    // uma vez — se alguem "otimizar" o attach de novo largando o corpus float32,
    // cai aqui.
    {
        const char* art = "test_engine_tokenizer.tmp.wnnxtok";
        std::remove(art);

        EngineTokenizerOptions o;
        o.artifact_path = art;
        o.model_name = "sintetico";
        o.tensor_name = "embed_tokens";

        // BLOCOS-MEDIA, NAO BLOCOS DE 1 ID.
        //
        // Medido: com blocos de 1 id este teste passava MESMO com o defeito
        // presente — a query e' o proprio embedding, o cos e' ~1 e o top-1
        // coincide nos dois espacos. O defeito vive nas posicoes 2..k, que so'
        // se disputam quando a query e' AMBIGUA. Um bloco de 2 ids de clusters
        // DIFERENTES produz exatamente isso: a media cai na fronteira entre
        // clusters, onde a perda da quantizacao uint8 decide o vencedor.
        //
        // Era o caso do corpus real: 16 de 40 consultas divergiam, e nenhuma
        // delas era um id isolado.
        std::vector<std::vector<int>> blocos;
        for (int k = 0; k < K - 1; ++k)
            blocos.push_back({k, (k + 1) % K + K});   // um id de cada cluster

        // a fria constroi E grava o artefato
        std::vector<int> ids_frio;
        {
            EngineTokenizer frio(E.data(), V, D, o);
            ids_frio = frio.tokenize_blocks(blocos).ids;
        }
        bool criou = std::ifstream(art).good();

        // a quente CARREGA o artefato: mesmos ids, e mais rapida
        std::vector<int> ids_quente;
        {
            EngineTokenizer quente(E.data(), V, D, o);
            ids_quente = quente.tokenize_blocks(blocos).ids;
        }

        check(criou, "artefato e' gravado na primeira construcao");
        check(ids_frio == ids_quente,
              "artefato (load_index) reproduz os MESMOS ids do build");
        std::remove(art);
    }

    if (failures == 0) {
        printf("\nALL ENGINE-TOKENIZER TESTS PASSED\n");
        return 0;
    }
    printf("\n%d FAILURE(S)\n", failures);
    return 1;
}
