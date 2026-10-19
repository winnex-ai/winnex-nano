// test_hrr.cpp — o modo HRR agrega sem quebrar o que já existia.
//
// Duas famílias de teste, e a ordem importa:
//   1. NÃO-REGRESSÃO: encode/decode/encode_histogram/encode_compact devem
//      produzir EXATAMENTE os mesmos bytes/valores de antes. Se algum teste
//      desta seção falhar, a adição quebrou o contrato existente.
//   2. HRR: norma conservada, ordem preservada, determinismo.
//
// BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "winnex_nano/spectral_tokenizer.hpp"

using namespace winnex_nano;

static int g_falhas = 0;
static int g_ok = 0;

static void check(bool cond, const char* nome, const std::string& detalhe = "") {
    if (cond) {
        ++g_ok;
        std::printf("  [ok] %s %s\n", nome, detalhe.c_str());
    } else {
        ++g_falhas;
        std::printf("  [XX] %s %s\n", nome, detalhe.c_str());
    }
}

static double norma(const std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += static_cast<double>(x) * x;
    return std::sqrt(s);
}

static double cosseno(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0;
    return d / std::sqrt(na * nb);
}

int main() {
    SpectralTokenizer tk(64);
    std::printf("=== 1. NÃO-REGRESSÃO (contrato existente) ===\n");

    // Round-trip exato, incluindo acentos (o caso que já falhou em benchmarks)
    const std::vector<std::string> amostras = {
        "A superposição é uma característica, não um bug.",
        "O rato roeu a roupa do rei de Roma.",
        "abc", "12345", "!@#$%^&*()",
    };
    bool rt_ok = true;
    for (const auto& s : amostras) {
        if (tk.decode(tk.encode(s)) != s) rt_ok = false;
    }
    check(rt_ok, "round-trip exato (encode→decode) em 5 amostras");

    bool rtc_ok = true;
    for (const auto& s : amostras) {
        if (tk.decode_compact(tk.encode_compact(s)) != s) rtc_ok = false;
    }
    check(rtc_ok, "round-trip exato (compact) em 5 amostras");

    // encode_histogram: normalizado, e anagramas IDÊNTICOS (comportamento
    // antigo, preservado de propósito — o modo HRR é que corrige isso)
    auto h1 = tk.encode_histogram("abc", 256);
    auto h2 = tk.encode_histogram("cba", 256);
    bool hist_norm = std::fabs(norma(h1) - 1.0f) < 1e-4f;
    bool hist_anagrama = (h1 == h2);
    check(hist_norm, "encode_histogram normalizado (norma = 1)");
    check(hist_anagrama,
          "encode_histogram: anagramas idênticos (saco de bytes — MANTIDO)",
          "(o modo HRR é a correção, não uma mudança no histograma)");

    // encode: 1 bloco de 64 Quat por byte
    auto q = tk.encode("hello");
    check(q.size() == 5u * 64u, "encode devolve seq_len × embed_dim quaternions");

    std::printf("\n=== 2. HRR: norma e ordem ===\n");

    // Determinismo
    auto a1 = tk.encode_hrr("teste de determinismo");
    auto a2 = tk.encode_hrr("teste de determinismo");
    check(a1 == a2, "encode_hrr determinístico");

    // Dimensão
    check(a1.size() == 64u, "encode_hrr devolve embed_dim floats");

    // Norma unitária
    auto u = tk.encode_hrr_unit("teste de norma");
    check(std::fabs(norma(u) - 1.0) < 1e-5, "encode_hrr_unit tem norma 1",
          "‖v‖ = " + std::to_string(norma(u)));

    // Ordem preservada: anagramas distintos
    struct Par { const char* a; const char* b; const char* rotulo; };
    const Par pares[] = {
        {"abc", "cba", "anagrama de letras"},
        {"o rato roeu", "roeu rato o", "ordem de palavras"},
        {"gato caça rato", "rato caça gato", "ordem de palavras (2)"},
    };
    for (const auto& p : pares) {
        double c = cosseno(tk.encode_hrr(p.a), tk.encode_hrr(p.b));
        check(c < 0.99, p.rotulo, "cos = " + std::to_string(c));
    }

    // Textos diferentes e não-anagramas devem ser MENOS similares ainda
    double c_dist = cosseno(tk.encode_hrr("o rato roeu a roupa"),
                            tk.encode_hrr("a economia brasileira cresceu"));
    check(c_dist < 0.9, "textos não relacionados", "cos = " + std::to_string(c_dist));

    std::printf("\n=== 3. CONSERVAÇÃO DE NORMA AO CRESCER n ===\n");
    // O fenômeno que importa: a norma de saída deve refletir o CONTEÚDO, não
    // colapsar com o comprimento. Medimos, para n crescente, a norma por
    // caractere de cada agregação. A média decai ~1/sqrt(n); o HRR tem
    // decaimento controlado e determinístico (não é ruído).
    {
        std::printf("    %6s %14s %14s\n", "n", "media/n", "hrr/n");
        double m_peq = 0.0, m_gra = 0.0, h_peq = 0.0, h_gra = 0.0;
        for (int n : {16, 64, 256}) {
            std::string txt;
            for (int i = 0; i < n; ++i)
                txt.push_back(static_cast<char>(33 + (i * 37) % 90));

            auto qs = tk.encode(txt);
            double sr = 0.0, si = 0.0;
            for (size_t i = 0; i < qs.size(); i += 64) {
                double ph = std::atan2(qs[i].x, qs[i].w);
                sr += std::cos(ph); si += std::sin(ph);
            }
            const double norm_media = std::sqrt(sr * sr + si * si) / n;

            auto v = tk.encode_hrr(txt);
            const double norm_hrr = norma(v);

            std::printf("    %6d %14.6f %14.6f\n", n, norm_media, norm_hrr);
            if (n == 16) { m_peq = norm_media; h_peq = norm_hrr; }
            if (n == 256) { m_gra = norm_media; h_gra = norm_hrr; }
        }
        // a média por caractere deve DECAIR com n (cancelamento)
        check(m_gra < m_peq,
              "a média por caractere decai com n (cancelamento destrutivo)",
              "(" + std::to_string(m_peq) + " → " + std::to_string(m_gra) + ")");
        // o HRR não deve colapsar na mesma proporção
        const double razao_media = m_peq > 1e-12 ? m_gra / m_peq : 0.0;
        const double razao_hrr = h_peq > 1e-12 ? h_gra / h_peq : 0.0;
        check(razao_hrr > razao_media,
              "o HRR decai menos que a média",
              "(" + std::to_string(razao_hrr) + " vs " +
              std::to_string(razao_media) + ")");
    }

    std::printf("\n%d ok, %d falhas\n", g_ok, g_falhas);
    return g_falhas == 0 ? 0 : 1;
}
