// engine_tokenizer.cpp — atribuicao de ids pelo motor Madhava, com certificado.
#include "winnex_nano/engine_tokenizer.hpp"

#include "winnex_madhava/winnex_madhava.hpp"
#include "winnex_nano/tokenizer_artifact.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_set>

namespace winnex_nano {

namespace {

// Mediana de um vetor de double (copia: n e' pequeno).
double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

} // namespace

struct EngineTokenizer::Impl {
    int vocab = 0;
    int dim = 0;
    EngineTokenizerOptions opts;
    std::vector<float> et;   // embed_tokens normalizado, row-major
    std::unique_ptr<winnex_madhava::MadhavaL2> engine;
    std::vector<float> qbuf; // buffer de consulta reutilizado
    bool loaded_from_artifact = false;

    const float* row(int id) const { return et.data() + (size_t)id * dim; }

    // Carrega o indice do artefato se ele existir E pertencer a este modelo.
    // Se pertencer a outro, verify_artifact_matches LANCA — carregar o indice
    // do modelo A sobre os embeddings do B devolveria ids validos e errados.
    bool try_load_artifact() {
        if (opts.artifact_path.empty()) return false;
        if (!std::ifstream(opts.artifact_path).good()) return false;

        ModelIdentity id = TokenizerArtifact::peek(opts.artifact_path);
        verify_artifact_matches(id, et.data(), vocab, dim, opts.tensor_name);

        const std::string tmp = opts.artifact_path + ".idx.tmp";
        TokenizerArtifact art;
        art.extract_index_to(opts.artifact_path, tmp);

        winnex_madhava::Config cfg;
        cfg.dim = dim;
        cfg.metric = winnex_madhava::Metric::Cosine;
        cfg.k = opts.k;
        // QUANT = NONE, nao Int8.
        //
        // O corpus e' float32 normalizado. O build_engine do Python forca
        // QuantMode::NONE para corpus float32, com o motivo medido no proprio
        // comentario dele: "Converting float32 to uint8 also destroys the
        // embedding manifold (Bug B: e(v)->1.0). So float32 corpora force
        // quant=none".
        //
        // e(v) -> 1.0 significa bound frouxo: UB nunca desce abaixo do limiar
        // e NENHUM documento e' provado descartavel. Sintoma medido: com
        // Int8 e rank 128, total_excluded_count = 0 (cobertura 0%); com None,
        // 151.935 (100%). O bug so' aparece em rank < dim — com rank pleno a
        // projecao e' o espaco inteiro e a quantizacao nao muda o resultado,
        // que foi por que passou despercebido na validacao inicial.
        cfg.quant = winnex_madhava::QuantMode::None;
        cfg.normalize_input = true;
        engine = std::make_unique<winnex_madhava::MadhavaL2>(cfg);
        // O corpus bruto NAO esta no artefato (ele esta no checkpoint): o
        // chamador re-anexa antes de buscar. E' o contrato do load_index.
        engine->build_float32(et.data(), vocab);
        const bool ok = engine->load_index(tmp);
        std::remove(tmp.c_str());
        if (!ok) {
            engine.reset();
            throw std::runtime_error(
                "EngineTokenizer: artefato ilegivel em " + opts.artifact_path);
        }
        loaded_from_artifact = true;
        return true;
    }

    void ensure_engine() {
        if (engine) return;
        if (try_load_artifact()) return;
        // Mesma construcao do ChunkEngine: a engine e' do Madhava, nao uma copia.
        winnex_madhava::Config cfg;
        cfg.dim = dim;
        cfg.metric = winnex_madhava::Metric::Cosine;
        cfg.k = opts.k;
        // RANK 128, medido — nao rank pleno.
        //
        // A hipotese "precisa de rank pleno para 100% de cobertura" era verdade
        // no corpus ISOTROPICO sintetico e FALSA no vocabulario real, que vive
        // num manifold. Medido com Qwen2.5-0.5B (151.936 x 896):
        //
        //   rank   acerto@1   artefato   build    p50
        //    896     100%      1098 MB    78.1 s  42.8 ms
        //    256     100%       316 MB      -     32.0 ms
        //    128     100%       160 MB    11.8 s  33.7 ms
        //     64     100%        81 MB      -     33.3 ms
        //
        // O acerto e' identico porque reduzir o rank nao custa informacao aqui:
        // os unicos "erros" medidos eram tokens com embedding IDENTICO (padding
        // nao treinado; cos = 1.000000 e norma 0.3010 entre eles), onde nao ha
        // resposta unica a atribuir. 128 e' onde o artefato para de crescer sem
        // ganho: de 128 para 64 o tamanho cai 2x, contra 6.9x de 896 para 128.
        //
        // NAO da' exact_global por si so' — isso exige audit_exhaustive.
        // CLAMP em dim. stage > dim NAO tem significado: em R^dim existem no
        // maximo dim vetores ortonormais, e o bound de Cauchy-Schwarz exige
        // linhas ortonormais. Com rank maior o MGS produz linhas dependentes, o
        // residuo e(v) fica subestimado e o bound SUPERESTIMADO ~2x — medido:
        // UB/exato = 2.008 e bound_violations > 0 num corpus de d=64 com
        // stage=128. Os ids continuam corretos (o bound inflado so' reduz a
        // poda); o que quebra e' o CERTIFICADO.
        //
        // Este teste estava vermelho por isso: test_engine_tokenizer usa D=64 e
        // o default era 128. O teste estava certo; o codigo e' que passava uma
        // configuracao invalida.
        const int requested = opts.stage_dim > 0 ? opts.stage_dim : 128;
        const int stage = std::min(requested, dim);
        cfg.stage1_dim = stage;
        cfg.stage2_dim = stage;
        // QUANT = NONE, nao Int8.
        //
        // O corpus e' float32 normalizado. O build_engine do Python forca
        // QuantMode::NONE para corpus float32, com o motivo medido no proprio
        // comentario dele: "Converting float32 to uint8 also destroys the
        // embedding manifold (Bug B: e(v)->1.0). So float32 corpora force
        // quant=none".
        //
        // e(v) -> 1.0 significa bound frouxo: UB nunca desce abaixo do limiar
        // e NENHUM documento e' provado descartavel. Sintoma medido: com
        // Int8 e rank 128, total_excluded_count = 0 (cobertura 0%); com None,
        // 151.935 (100%). O bug so' aparece em rank < dim — com rank pleno a
        // projecao e' o espaco inteiro e a quantizacao nao muda o resultado,
        // que foi por que passou despercebido na validacao inicial.
        cfg.quant = winnex_madhava::QuantMode::None;
        cfg.normalize_input = true;
        cfg.modulation = true;
        cfg.postfilter = true;
        // early_exit OFF, pelo mesmo motivo registrado no chunk_engine: com o
        // bound frouxo em d>=384 o early exit derruba recall para ~0.73.
        cfg.early_exit = false;
        cfg.audit_exhaustive = opts.audit_exhaustive;
        if (opts.n_threads > 0) cfg.n_threads = opts.n_threads;
        engine = std::make_unique<winnex_madhava::MadhavaL2>(cfg);
        engine->build_float32(et.data(), (int)(et.size() / dim));
        // Primeira execucao: grava o artefato para as proximas serem ~80x
        // mais rapidas. Falha de escrita NAO invalida o tokenizer — so' avisa.
        if (!opts.artifact_path.empty()) {
            try {
                const std::string tmp_idx = opts.artifact_path + ".madhava.tmp";
                if (engine->save_index(tmp_idx)) {
                    ModelIdentity id;
                    id.engine = "MADHAVA_INDEX_V1";
                    id.model_name = opts.model_name.empty() ? "desconhecido"
                                                            : opts.model_name;
                    id.tensor_name = opts.tensor_name;
                    id.vocab = vocab;
                    id.dim = dim;
                    id.fingerprint = TokenizerArtifact::fingerprint(et.data(), vocab, dim);
                    TokenizerArtifact art;
                    art.save(opts.artifact_path, id, tmp_idx);
                    std::remove(tmp_idx.c_str());
                }
            } catch (const std::exception&) {
                // Sem artefato o tokenizer continua correto, apenas mais lento.
            }
        }
    }

    // Atribui um vetor normalizado. Devolve o id e preenche a atribuicao.
    int attribute(const float* q, TokenAttribution* out) {
        ensure_engine();
        if (opts.certificar) {
            auto res = engine->search_with_commitment(q, opts.k, opts.max_sample);
            if (res.indices.empty())
                throw std::runtime_error("attribute: engine devolveu zero indices");
            if (out) {
                out->id = (int)res.indices[0];
                out->excluded = (int)res.total_excluded_count;
                out->violations = (int)res.bound_violations;
                out->scope = res.recall_guarantee;
                out->bound_pairs = (int)res.bound_pairs;
            }
            return (int)res.indices[0];
        }
        auto res = engine->search(q);
        if (res.indices.empty())
            throw std::runtime_error("attribute: engine devolveu zero indices");
        if (out) { out->id = (int)res.indices[0]; out->scope = res.recall_guarantee; }
        return (int)res.indices[0];
    }
};

EngineTokenizer::EngineTokenizer(const float* embed_tokens, int vocab, int dim,
                                 const EngineTokenizerOptions& opts)
    : impl_(new Impl) {
    if (!embed_tokens)
        throw std::invalid_argument("EngineTokenizer: embed_tokens nulo");
    if (vocab <= 0 || dim <= 0)
        throw std::invalid_argument(
            "EngineTokenizer: vocab=" + std::to_string(vocab) +
            " dim=" + std::to_string(dim) + " — ambos devem ser > 0");
    impl_->vocab = vocab;
    impl_->dim = dim;
    impl_->opts = opts;
    impl_->qbuf.assign(dim, 0.0f);

    // Normaliza UMA vez e valida finitude na mesma passada. Uma engine sobre
    // vetores corrompidos devolve resultado plausivel e errado; a validacao
    // tem de acontecer aqui, nao na primeira consulta.
    impl_->et.resize((size_t)vocab * dim);
    for (int i = 0; i < vocab; ++i) {
        const float* src = embed_tokens + (size_t)i * dim;
        float* dst = impl_->et.data() + (size_t)i * dim;
        float acc = 0.0f;
        for (int j = 0; j < dim; ++j) {
            const float v = src[j];
            if (!std::isfinite(v))
                throw std::invalid_argument(
                    "EngineTokenizer: embed_tokens[" + std::to_string(i) + "][" +
                    std::to_string(j) + "] nao e' finito");
            acc += v * v;
        }
        const float nrm = std::sqrt(acc);
        if (!(nrm > 0.0f))
            throw std::invalid_argument(
                "EngineTokenizer: embed_tokens linha " + std::to_string(i) +
                " tem norma zero — nao ha direcao para atribuir");
        const float inv = 1.0f / nrm;
        for (int j = 0; j < dim; ++j) dst[j] = src[j] * inv;
    }
}

EngineTokenizer::~EngineTokenizer() = default;

int EngineTokenizer::vocab() const { return impl_->vocab; }
int EngineTokenizer::dim() const { return impl_->dim; }

std::vector<float> EngineTokenizer::bloco_vector(
    const std::vector<int>& bloco) const {
    if (bloco.empty())
        throw std::invalid_argument("bloco_vector: bloco vazio");
    std::vector<float> v(impl_->dim, 0.0f);
    for (int id : bloco) {
        if (id < 0 || id >= impl_->vocab)
            throw std::invalid_argument(
                "bloco_vector: id " + std::to_string(id) + " fora de [0, " +
                std::to_string(impl_->vocab) +
                ") — o vocabulario e' do modelo; ids de outro tokenizer nao "
                "valem aqui");
        const float* r = impl_->row(id);
        for (int j = 0; j < impl_->dim; ++j) v[j] += r[j];
    }
    float acc = 0.0f;
    for (float x : v) acc += x * x;
    const float nrm = std::sqrt(acc);
    if (!(nrm > 0.0f))
        throw std::invalid_argument(
            "bloco_vector: media dos embeddings e' nula — bloco sem informacao");
    const float inv = 1.0f / nrm;
    for (float& x : v) x *= inv;
    return v;
}

EngineTokenizeReport EngineTokenizer::tokenize_blocks(
    const std::vector<std::vector<int>>& blocos,
    const std::vector<int>& ids_esperados) {
    EngineTokenizeReport rep;
    rep.ids.reserve(blocos.size());
    rep.certificado.reserve(blocos.size());
    std::vector<double> cov;
    std::unordered_set<std::string> sc;

    // A primeira consulta paga o build (~78 s). Cronometrar so' as buscas
    // evita atribuir o custo de build a atribuicao.
    const bool first = !impl_->engine;
    auto t_after_build = std::chrono::steady_clock::now();
    if (first) {
        impl_->ensure_engine();
        t_after_build = std::chrono::steady_clock::now();
    }

    for (const auto& bloco : blocos) {
        std::vector<float> q = bloco_vector(bloco);
        TokenAttribution a;
        rep.ids.push_back(impl_->attribute(q.data(), &a));
        rep.certificado.push_back(a);
        if (impl_->opts.certificar) {
            cov.push_back((double)a.excluded / (double)impl_->vocab);
            rep.violations_total += a.violations;
            sc.insert(a.scope);
        }
    }
    auto t_end = std::chrono::steady_clock::now();
    rep.elapsed_ms =
        std::chrono::duration<double, std::milli>(t_end - t_after_build).count();

    if (!cov.empty()) rep.coverage_pct = 100.0 * median(cov);
    rep.scopes.assign(sc.begin(), sc.end());
    std::sort(rep.scopes.begin(), rep.scopes.end());

    if (!ids_esperados.empty()) {
        const size_t n = std::min(rep.ids.size(), ids_esperados.size());
        for (size_t i = 0; i < n; ++i)
            rep.acertos += (rep.ids[i] == ids_esperados[i]) ? 1 : 0;
        rep.sequencia_identica =
            (rep.ids.size() == ids_esperados.size()) && (rep.acertos == (int)n);
    }
    return rep;
}

EngineTokenizeReport EngineTokenizer::tokenize_embeddings(const float* vetores,
                                                          int n) {
    if (!vetores || n <= 0)
        throw std::invalid_argument("tokenize_embeddings: vetores nulo ou n <= 0");
    EngineTokenizeReport rep;
    rep.ids.reserve(n);
    rep.certificado.reserve(n);
    std::vector<double> cov;
    std::unordered_set<std::string> sc;

    const bool first = !impl_->engine;
    if (first) impl_->ensure_engine();
    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < n; ++i) {
        const float* src = vetores + (size_t)i * impl_->dim;
        float acc = 0.0f;
        for (int j = 0; j < impl_->dim; ++j) {
            if (!std::isfinite(src[j]))
                throw std::invalid_argument(
                    "tokenize_embeddings: vetor " + std::to_string(i) +
                    " tem valor nao-finito");
            acc += src[j] * src[j];
        }
        const float nrm = std::sqrt(acc);
        if (!(nrm > 0.0f))
            throw std::invalid_argument(
                "tokenize_embeddings: vetor " + std::to_string(i) + " e' nulo");
        const float inv = 1.0f / nrm;
        for (int j = 0; j < impl_->dim; ++j) impl_->qbuf[j] = src[j] * inv;

        TokenAttribution a;
        rep.ids.push_back(impl_->attribute(impl_->qbuf.data(), &a));
        rep.certificado.push_back(a);
        if (impl_->opts.certificar) {
            cov.push_back((double)a.excluded / (double)impl_->vocab);
            rep.violations_total += a.violations;
            sc.insert(a.scope);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    rep.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (!cov.empty()) rep.coverage_pct = 100.0 * median(cov);
    rep.scopes.assign(sc.begin(), sc.end());
    std::sort(rep.scopes.begin(), rep.scopes.end());
    return rep;
}

} // namespace winnex_nano
