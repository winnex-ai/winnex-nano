// test_forward.cpp — regression gate for the generation path.
//
// Usage:
//   test_forward <model_dir> --record         grava a referencia
//   test_forward <model_dir>                  verifica contra a referencia
//
// POR QUE ESTE ARQUIVO FOI REESCRITO
// ----------------------------------
// A versão anterior imprimia logits e passava se o programa não travasse:
//
//     forward OK | logits[0..3]=3.4510 1.6895 3.8577 -2.9207
//
// Três coisas estavam erradas, e cada uma anulava a anterior:
//
//   1. ZERO asserções. O teste não podia falhar. Uma regressão de álgebra
//      passaria como sucesso, com o número impresso como se fosse evidência.
//
//   2. NÃO TESTAVA O CAMINHO DE GERAÇÃO. Montava um vetor com
//      SpectralTokenizer → XFactor → forward(hidden, 1). O caminho que gera
//      texto é prefill(ids) → step(id), e nenhum dos dois era tocado.
//
//   3. NÃO TINHA REFERÊNCIA. Não havia contra o que comparar, nem valor
//      gravado, nem tolerância. "Imprimiu números" não é verificação.
//
// Agora: prefill + 8 passos de decode sobre ids de token reais, comparados
// com uma referência gravada, com tolerância explícita.
//
// A referência é um binário gravado por `--record`, produzido pela CPU em f32.
// Isso permite CI sem PyTorch: a regressão que importa aqui é "a álgebra
// mudou?", e ela aparece contra a CPU do mesmo build. A comparação contra o
// transformers é outra e vive em parity_gate.py.
#include "winnex_nano/forward.hpp"
#include "winnex_nano/safetensors.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace winnex_nano;

namespace {

// Tolerância absoluta. O harness usa 1e-4 para paridade ENTRE motores; aqui a
// comparação é entre duas execuções do MESMO motor, então o esperado é erro de
// arredondamento. 1e-5 é apertado o bastante para pegar uma troca de ordem de
// operação e folgado o bastante para não falhar por reordenamento de soma.
constexpr double kTolAbs = 1e-5;

// Ids fixos e sintéticos: o teste não depende de tokenizer nem de download.
// Não precisam ser texto válido — o forward só indexa embed_tokens.
std::vector<int> test_ids(int vocab, int n) {
    std::vector<int> ids;
    ids.reserve((size_t)n);
    for (int i = 0; i < n; ++i) ids.push_back((i * 7919 + 13) % (vocab - 1) + 1);
    return ids;
}

struct Ref {
    std::vector<float> logits;
    int n_steps = 0;
    int vocab = 0;
};

bool read_ref(const std::string& path, Ref& r) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int32_t hdr[3];
    f.read((char*)hdr, sizeof(hdr));
    if (!f) return false;
    r.n_steps = hdr[0];
    r.vocab = hdr[1];
    if (hdr[2] != r.n_steps * r.vocab) return false;
    r.logits.resize((size_t)hdr[2]);
    f.read((char*)r.logits.data(), sizeof(float) * (size_t)hdr[2]);
    return (bool)f;
}

void write_ref(const std::string& path, const Ref& r) {
    std::ofstream f(path, std::ios::binary);
    int32_t hdr[3] = {r.n_steps, r.vocab, (int32_t)r.logits.size()};
    f.write((const char*)hdr, sizeof(hdr));
    f.write((const char*)r.logits.data(), sizeof(float) * r.logits.size());
    printf("referencia gravada: %s (%d passos x %d logits)\n",
           path.c_str(), r.n_steps, r.vocab);
}

} // namespace

// ── Regressão: prefill curto → prefill longo → decode longo ───────────────
//
// Este cenário não existia em nenhum teste, e é exatamente o que o benchmark do
// Kaggle faz. O KV residente no device é dimensionado no prefill como
// `prompt + max(256, prompt/2)`; um prefill de 50 tokens reserva 306 posições.
// Um prefill seguinte de 512 estourava a reserva:
//
//   ForwardGpu::attention_resident: context 512 exceeds reserved 306
//
// e, ao ligar o caminho residente por padrão, derrubou o kernel inteiro.
// Nenhum teste pegava, porque todos usavam um único prefill do mesmo tamanho.
//
// Não compara contra referência: verifica que o motor sobrevive a sequências de
// tamanhos crescentes e que os logits são finitos. Um crash aqui é falha.
static int run_growth_check(const std::string& dir) {
    try {
        ModelConfig cfg = load_config(dir + "/config.json");
        Safetensors st(dir + "/model.safetensors");
        ForwardEngine fe(cfg, st);
        // O bug vive no caminho RESIDENTE DO DEVICE. Sem habilitar a GPU este
        // teste passa mesmo com o defeito presente — verificado removendo a
        // correção: continuava verde. Habilitar aqui é o que dá sentido ao
        // teste; quando não há device, ele reporta que não verificou em vez de
        // fingir que passou.
        bool gpu = false;
        try { fe.enable_gpu(); gpu = true; }
        catch (const std::exception& e) {
            printf("NAO VERIFICADO: sem device (%s).\n", e.what());
            printf("Este teste existe para o caminho residente da GPU; na CPU ele\n"
                   "nao exercita a reserva de KV e passaria com o bug presente.\n");
            return 3;
        }
        printf("device: %s | modo de pesos: %s\n", fe.gpu_device().c_str(),
               fe.weight_mode().c_str());
        (void)gpu;
        const int V = cfg.vocab_size;

        struct Case { const char* what; int len; int steps; };
        const Case cases[] = {
            {"prefill curto (50)",              50,   0},
            {"prefill longo (512) apos curto",  512,  0},
            {"decode 300 passos apos 512",      512, 300},
        };
        for (const Case& c : cases) {
            auto ids = test_ids(V, c.len);
            std::vector<float> lg = fe.prefill(ids);
            if ((int)lg.size() != V) {
                printf("FALHOU (%s): prefill devolveu %zu logits, esperado %d\n",
                       c.what, lg.size(), V);
                return 1;
            }
            int next = 0;
            for (int i = 0; i < c.steps; ++i) {
                lg = fe.step(next);
                if ((int)lg.size() != V) {
                    printf("FALHOU (%s): step %d devolveu %zu logits\n",
                           c.what, i, lg.size());
                    return 1;
                }
                float best = lg[0];
                next = 0;
                for (int j = 1; j < V; ++j) if (lg[j] > best) { best = lg[j]; next = j; }
                if (!std::isfinite(best)) {
                    printf("FALHOU (%s): step %d produziu logit nao-finito\n", c.what, i);
                    return 1;
                }
            }
            printf("OK (%s): %d posicoes, %d passos de decode\n",
                   c.what, c.len, c.steps);
        }
        printf("CRESCIMENTO OK | nenhuma reserva de KV estourou\n");
        return 0;
    } catch (const std::exception& e) {
        printf("FALHOU: %s\n", e.what());
        return 1;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s <model_dir> [--record | --growth]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const bool record = (argc > 2 && std::string(argv[2]) == "--record");
    if (argc > 2 && std::string(argv[2]) == "--growth")
        return run_growth_check(dir);
    const std::string ref_path = dir + "/forward_reference.bin";

    try {
        ModelConfig cfg = load_config(dir + "/config.json");
        Safetensors st(dir + "/model.safetensors");
        ForwardEngine fe(cfg, st);

        const int V = cfg.vocab_size;
        const int n_prompt = 24;
        const int n_steps = 8;
        auto ids = test_ids(V, n_prompt);

        // ── caminho de geração: prefill + decode greedy ────────────────
        Ref got;
        got.vocab = V;
        got.n_steps = n_steps;

        std::vector<float> lg = fe.prefill(ids);
        if ((int)lg.size() != V)
            throw std::runtime_error("prefill devolveu " + std::to_string(lg.size()) +
                                     " logits, esperado " + std::to_string(V));
        int next = ForwardEngine::argmax(lg);
        for (int s = 0; s < n_steps; ++s) {
            lg = fe.step(next);
            if ((int)lg.size() != V)
                throw std::runtime_error("step devolveu tamanho errado");
            got.logits.insert(got.logits.end(), lg.begin(), lg.end());
            next = ForwardEngine::argmax(lg);
        }

        // ── sanidade independente de referência ────────────────────────
        int bad = 0;
        for (float v : got.logits) if (!std::isfinite(v)) ++bad;
        if (bad) {
            printf("FALHOU: %d logits nao-finitos\n", bad);
            return 1;
        }

        if (record) {
            write_ref(ref_path, got);
            printf("OK (gravacao) | %d passos, %d logits cada, todos finitos\n",
                   n_steps, V);
            return 0;
        }

        Ref ref;
        if (!read_ref(ref_path, ref)) {
            printf("SEM REFERENCIA: %s nao existe.\n", ref_path.c_str());
            printf("  Rode com --record para criar e versione o arquivo junto do\n");
            printf("  checkpoint de teste. Sem ele este teste nao verifica nada —\n");
            printf("  e um teste que nao pode falhar nao e um teste.\n");
            return 3;   // ausencia de referencia e falha, nao sucesso
        }
        if (ref.vocab != V || ref.n_steps != n_steps) {
            printf("FALHOU: referencia incompativel (vocab %d/%d, passos %d/%d)\n",
                   ref.vocab, V, ref.n_steps, n_steps);
            return 1;
        }

        double max_abs = 0.0;
        size_t worst = 0;
        for (size_t i = 0; i < ref.logits.size(); ++i) {
            const double d = std::fabs((double)got.logits[i] - (double)ref.logits[i]);
            if (d > max_abs) { max_abs = d; worst = i; }
        }
        printf("logits: max|d|=%.3e (tol=%.0e) pior=passo %zu token %zu\n",
               max_abs, kTolAbs, worst / (size_t)V, worst % (size_t)V);
        if (max_abs > kTolAbs) {
            printf("FALHOU: desvio acima da tolerancia. valor=%.6f referencia=%.6f\n",
                   got.logits[worst], ref.logits[worst]);
            return 1;
        }
        printf("OK | %d passos x %d logits dentro da tolerancia\n", n_steps, V);
        return 0;
    } catch (const std::exception& e) {
        printf("ERROR: %s\n", e.what());
        return 1;
    }
}
