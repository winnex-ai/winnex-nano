// test_tokenizer_artifact.cpp — persistência POR MODELO, sem hardcoding.
//
// O que este teste fixa:
//   * não existe "o" arquivo: existe um por modelo, e o nome não escapa do dir
//   * dois modelos diferentes produzem dois artefatos diferentes
//   * o guard RECUSA carregar o índice de um modelo sobre outro
//   * o nome do tensor é resolvido, não presumido — e falha alto se nada casar
//   * fingerprint distingue modelos e é estável entre chamadas
//
// Não depende de checkpoint real: usa embed_tokens sintéticos. A validação
// contra o Qwen fica no teste Python.
#include "winnex_nano/tokenizer_artifact.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <cstring>
#include <string>
#include <vector>

using namespace winnex_nano;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    printf("%-62s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++failures;
}

std::vector<float> make_embed(int V, int D, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> E((size_t)V * D);
    for (auto& x : E) x = nd(rng);
    return E;
}

// Cria um arquivo qualquer para servir de "índice Madhava" no teste de save.
std::string fake_index(const std::string& path, size_t bytes, char fill) {
    std::vector<char> b(bytes, fill);
    std::ofstream o(path, std::ios::binary | std::ios::trunc);
    o.write(b.data(), (std::streamsize)b.size());
    return path;
}

} // namespace

int main() {
    const int V = 128, D = 32;
    std::vector<float> A = make_embed(V, D, /*seed=*/11);
    std::vector<float> Bv = make_embed(V, D, /*seed=*/22);   // outro modelo
    std::vector<float> A_other_dim = make_embed(64, 16, 33); // outra forma

    // ── 1. o nome do arquivo é por modelo, e não escapa do diretório ─────
    {
        const std::string p1 = TokenizerArtifact::default_path(
            "/var/cache/wnnex", "qwen2.5-0.5b");
        const std::string p2 = TokenizerArtifact::default_path(
            "/var/cache/wnnex", "llama-3.2-1b");
        check(p1 != p2, "dois modelos -> dois caminhos distintos");
        check(p1.find("qwen2.5-0.5b") != std::string::npos,
              "o caminho carrega o nome do modelo");

        // A asserção anterior era `find("..") == npos` — FALSO POSITIVO: o
        // sanitizador transforma "../../etc/passwd" em ".._.._etc_passwd", que
        // CONTÉM ".." como substring mas é um nome plano, dentro do diretório.
        // O que importa é o nome resolvido não sair do diretório: sem '/', sem
        // ser "." ou "..", e começando pela pasta esperada.
        const std::string evil = TokenizerArtifact::default_path(
            "/var/cache/wnnex", "../../etc/passwd");
        // +1: pular a barra final do diretório. Sem isso, find() acha a própria
        // separação e a asserção falha para QUALQUER nome — foi o que
        // aconteceu na primeira correção.
        const size_t dir_len = strlen("/var/cache/wnnex");
        const bool sem_barra = evil.find('/', dir_len + 1) == std::string::npos;
        const std::string tail = evil.substr(strlen("/var/cache/wnnex/"));
        const bool nome_seguro = tail != ".." && tail != "." && !tail.empty();
        const bool dentro_do_dir =
            evil.compare(0, strlen("/var/cache/wnnex/"), "/var/cache/wnnex/") == 0;
        check(sem_barra && nome_seguro && dentro_do_dir,
              "nome com ../ vira nome plano dentro do diretorio");
        bool threw = false;
        try { TokenizerArtifact::default_path("/x", ".."); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "nome '.' ou '..' -> invalid_argument");
    }

    // ── 2. fingerprint distingue modelos, é estável ───────────────────────
    {
        const std::string fa = TokenizerArtifact::fingerprint(A.data(), V, D);
        const std::string fb = TokenizerArtifact::fingerprint(Bv.data(), V, D);
        const std::string fa2 = TokenizerArtifact::fingerprint(A.data(), V, D);
        check(fa.size() == 64, "fingerprint tem 64 hex (sha256)");
        check(fa != fb, "modelos diferentes -> fingerprints diferentes");
        check(fa == fa2, "mesmo modelo -> mesmo fingerprint (estavel)");
    }

    // ── 3. resolver o tensor: explícito, conhecido, e ERRO se nada casar ──
    {
        auto has_qwen = [](const std::string& n) {
            return n == "model.embed_tokens.weight";
        };
        const std::string r1 = resolve_embed_tensor_name("", has_qwen);
        check(r1 == "model.embed_tokens.weight",
              "sem nome explicito -> acha o tensor da familia");

        auto has_gpt2 = [](const std::string& n) {
            return n == "transformer.wte.weight";
        };
        const std::string r2 = resolve_embed_tensor_name("", has_gpt2);
        check(r2 == "transformer.wte.weight",
              "outra familia -> acha o tensor dela, nao um hardcoded");

        auto has_custom = [](const std::string& n) {
            return n == "meu_modelo.embedding";
        };
        const std::string r3 = resolve_embed_tensor_name(
            "meu_modelo.embedding", has_custom);
        check(r3 == "meu_modelo.embedding",
              "nome explicito -> respeitado (modelo fora das familias)");

        bool threw = false;
        auto has_none = [](const std::string&) { return false; };
        try { resolve_embed_tensor_name("", has_none); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "nenhum tensor conhecido -> invalid_argument (sem default)");

        bool threw_exp = false;
        try { resolve_embed_tensor_name("nao_existe", has_qwen); }
        catch (const std::invalid_argument&) { threw_exp = true; }
        check(threw_exp, "nome explicito inexistente -> invalid_argument");
    }

    // ── 4. salvar e recuperar o cabeçalho ─────────────────────────────────
    const std::string idx_path = "/tmp/_wnnxtok_fake_index.bin";
    const std::string art_a = "/tmp/_wnnxtok_a.wnnxtok";
    const std::string art_b = "/tmp/_wnnxtok_b.wnnxtok";
    fake_index(idx_path, 4096, '\x5a');

    ModelIdentity ida;
    ida.engine = "MADHAVA_INDEX_V1";
    ida.model_name = "qwen2.5-0.5b";
    ida.tensor_name = "model.embed_tokens.weight";
    ida.vocab = V;
    ida.dim = D;
    ida.fingerprint = TokenizerArtifact::fingerprint(A.data(), V, D);

    ModelIdentity idb = ida;
    idb.model_name = "llama-3.2-1b";
    idb.fingerprint = TokenizerArtifact::fingerprint(Bv.data(), V, D);

    {
        TokenizerArtifact art;
        check(art.save(art_a, ida, idx_path), "save do modelo A");
        check(art.save(art_b, idb, idx_path), "save do modelo B");

        const ModelIdentity pa = TokenizerArtifact::peek(art_a);
        check(pa.model_name == "qwen2.5-0.5b", "peek recupera o nome do modelo A");
        check(pa.vocab == V && pa.dim == D, "peek recupera vocab e dim");
        check(pa.fingerprint == ida.fingerprint, "peek recupera o fingerprint");

        const ModelIdentity pb = TokenizerArtifact::peek(art_b);
        check(pb.model_name != pa.model_name,
              "os dois artefatos sao de modelos distintos");
        check(pb.fingerprint != pa.fingerprint,
              "os dois artefatos tem fingerprints distintos");
    }

    // ── 5. o guard RECUSA o pareamento errado ─────────────────────────────
    // Este é o teste central: índice do modelo A + embeddings do modelo B.
    {
        const ModelIdentity pa = TokenizerArtifact::peek(art_a);

        bool ok_certo = true;
        try {
            verify_artifact_matches(pa, A.data(), V, D,
                                    "model.embed_tokens.weight");
        } catch (...) { ok_certo = false; }
        check(ok_certo, "artefato A + embeddings A -> aceito");

        bool recusou_modelo = false;
        try {
            verify_artifact_matches(pa, Bv.data(), V, D,
                                    "model.embed_tokens.weight");
        } catch (const std::invalid_argument& e) {
            recusou_modelo = std::string(e.what()).find("NAO e' o mesmo") !=
                             std::string::npos;
        }
        check(recusou_modelo,
              "artefato A + embeddings B -> RECUSADO (o erro caro)");

        bool recusou_dim = false;
        try {
            verify_artifact_matches(pa, A_other_dim.data(), 64, 16,
                                    "model.embed_tokens.weight");
        } catch (const std::invalid_argument&) { recusou_dim = true; }
        check(recusou_dim, "artefato A + outro vocab/dim -> RECUSADO");

        bool recusou_tensor = false;
        try {
            verify_artifact_matches(pa, A.data(), V, D, "transformer.wte.weight");
        } catch (const std::invalid_argument&) { recusou_tensor = true; }
        check(recusou_tensor, "artefato A + outro tensor -> RECUSADO");
    }

    // ── 6. extrair o índice preserva o conteúdo ───────────────────────────
    {
        TokenizerArtifact art;
        const std::string out_idx = "/tmp/_wnnxtok_extracted.bin";
        art.extract_index_to(art_a, out_idx);

        std::ifstream a(idx_path, std::ios::binary), b(out_idx, std::ios::binary);
        std::string ca((std::istreambuf_iterator<char>(a)),
                       std::istreambuf_iterator<char>());
        std::string cb((std::istreambuf_iterator<char>(b)),
                       std::istreambuf_iterator<char>());
        check(ca == cb, "indice extraido byte-a-byte igual ao salvo");
    }

    // ── 7. recusas de formato ─────────────────────────────────────────────
    {
        const std::string lixo = "/tmp/_wnnxtok_lixo.bin";
        fake_index(lixo, 100, '\x00');
        bool threw = false;
        try { TokenizerArtifact::peek(lixo); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "arquivo que nao e' artefato -> invalid_argument");

        bool threw_missing = false;
        try { TokenizerArtifact::peek("/tmp/nao_existe_wnnxtok"); }
        catch (const std::invalid_argument&) { threw_missing = true; }
        check(threw_missing, "arquivo inexistente -> invalid_argument");
    }

    // ── 8. identidade incompleta não salva ────────────────────────────────
    {
        TokenizerArtifact art;
        ModelIdentity vazia;          // sem engine/tensor/fingerprint
        bool threw = false;
        try { art.save("/tmp/_wnnxtok_x.wnnxtok", vazia, idx_path); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "identidade incompleta -> invalid_argument no save");
    }

    if (failures == 0) {
        printf("\nALL TOKENIZER-ARTIFACT TESTS PASSED\n");
        return 0;
    }
    printf("\n%d FAILURE(S)\n", failures);
    return 1;
}
