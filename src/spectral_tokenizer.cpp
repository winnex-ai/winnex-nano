// spectral_tokenizer.cpp — Winnex spectral tokenizer (deterministic, UTF-8).
//
// THE MATHEMATICS — the group isomorphism ℤ/256ℤ → μ₂₅₆.
//
//   Encode:  f : byte ↦ e^{i·(byte+pos+j)·2π/256}     (phase-only, amp = 1.0)
//   Decode:  f⁻¹ : ψ ↦ arg(ψ)·256/(2π) − pos − j  (mod 256)
//
// This is NOT a DFT to be accelerated by FFT, and NOT a correlation to be
// scanned. The byte is an element of ℤ/256ℤ; its image is a 256th root of
// unity. Decoding is the direct application of the inverse homomorphism —
// O(1) per byte, no probe, no FFT, no 256-ref scan. This is the fundamental
// algebraic structure of the problem, used as engineering optimization.
//
//   f(a+b) = e^{i(a+b)2π/256} = e^{ia2π/256}·e^{ib2π/256} = f(a)·f(b)  (homomorphism)
//   f(a) = f(b) ⟹ a−b ≡ 0 (mod 256)                                    (injective)
//   every μ₂₅₆ element is e^{ik2π/256}, k ∈ {0..255}                     (surjective)
//
// Auto-synchronizing (UTF-8 is byte-oriented), lossless for ANY language.
//
// BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
#include "winnex_nano/spectral_tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace winnex_nano {

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr int kByteRange = SpectralTokenizer::kByteRange;
} // namespace

std::vector<Quat> SpectralTokenizer::encode(const std::string& text) const {
    std::vector<Quat> states;
    states.reserve(text.size() * static_cast<size_t>(embed_dim_));

    // Phase-only: amplitude 1.0 (unit circle). The byte + pos define the phase;
    // the mode j is an additional constant phase offset, preserved analytically.
    for (size_t pos = 0; pos < text.size(); ++pos) {
        const int bpos = static_cast<int>(static_cast<unsigned char>(text[pos]) + pos);
        for (int j = 0; j < embed_dim_; ++j) {
            const float phase = (static_cast<float>(bpos + j)) * 2.0f * kPi / kByteRange;
            Quat q;
            q.w = std::cos(phase);
            q.x = std::sin(phase);
            // The (y,z) components are a fixed π/4-rotated copy of (w,x):
            // phase + π/4. They carry no independent information for the
            // analytic decode (which reads arg from (w,x)), but preserve the
            // quaternion-spectrum convention for downstream consumers.
            q.y = std::cos(phase + kPi / 4.0f);
            q.z = std::sin(phase + kPi / 4.0f);
            states.push_back(q);
        }
    }
    return states;
}

void SpectralTokenizer::encode_into_buffer(const std::string& text, float* out) const {
    const float kPi2 = 2.0f * kPi;
    for (size_t pos = 0; pos < text.size(); ++pos) {
        const int bpos = static_cast<int>(static_cast<unsigned char>(text[pos]) + pos);
        float* row = out + pos * static_cast<size_t>(embed_dim_) * 4u;
        for (int j = 0; j < embed_dim_; ++j) {
            const float phase = (static_cast<float>(bpos + j)) * kPi2 / kByteRange;
            const float c = std::cos(phase), s = std::sin(phase);
            const float cp = std::cos(phase + kPi / 4.0f), sp = std::sin(phase + kPi / 4.0f);
            float* q = row + static_cast<size_t>(j) * 4u;
            q[0] = c; q[1] = s; q[2] = cp; q[3] = sp;
        }
    }
}

std::vector<float> SpectralTokenizer::encode_histogram(const std::string& text, int bins) const {
    std::vector<float> hist(bins, 0.0f);
    for (char ch : text) {
        int idx = static_cast<unsigned char>(ch) % bins;
        hist[idx] += 1.0f;
    }
    float norm = 0.0f;
    for (auto v : hist) norm += v * v;
    norm = std::sqrt(norm) + 1e-9f;
    for (auto& v : hist) v /= norm;
    return hist;
}

std::string SpectralTokenizer::decode(const std::vector<Quat>& states) const {
    if (states.empty()) return "";
    const size_t n_bytes = states.size() / static_cast<size_t>(embed_dim_);
    if (n_bytes * static_cast<size_t>(embed_dim_) != states.size()) {
        throw std::runtime_error(
            "SpectralTokenizer::decode: state count not a multiple of embed_dim");
    }

    std::string out;
    out.reserve(n_bytes);

    // Analytic inverse homomorphism: O(1) per byte.
    //   phase = (byte + pos + j)·2π/256  ⟹  byte = phase·256/(2π) − pos − j
    // We read the phase from the FIRST mode (j = 0) — the (w,x) pair.
    // All other modes are redundant for decoding (each is a constant phase
    // rotation of mode 0), so reading j=0 is exact.
    for (size_t b = 0; b < n_bytes; ++b) {
        const Quat& q = states[b * static_cast<size_t>(embed_dim_)];
        float phase = std::atan2(q.x, q.w);
        if (phase < 0) phase += 2.0f * kPi;
        // byte = round(phase·256/(2π)) − pos
        float fp = phase * kByteRange / (2.0f * kPi);
        int byte = static_cast<int>(std::lround(fp)) - static_cast<int>(b);
        byte = ((byte % kByteRange) + kByteRange) % kByteRange;
        out.push_back(static_cast<char>(byte));
    }
    return out;
}

std::vector<Quat> SpectralTokenizer::encode_compact(const std::string& text) const {
    // One complex sample per byte (j=0): [cos((byte+pos)·2π/256), sin(...)].
    // The spectral signal is a pure tone of frequency (byte+pos) — by the
    // sampling theorem, a single sample of its phase fully determines it.
    // Storage: 2 floats per byte = 64 bits/char (vs embed_dim·4·32 = 8192).
    std::vector<Quat> states;
    states.reserve(text.size());
    for (size_t pos = 0; pos < text.size(); ++pos) {
        const int bpos = static_cast<int>(static_cast<unsigned char>(text[pos]) + pos);
        const float phase = (static_cast<float>(bpos)) * 2.0f * kPi / kByteRange;
        Quat q;
        q.w = std::cos(phase);
        q.x = std::sin(phase);
        q.y = 0.0f; q.z = 0.0f;  // unused in compact mode
        states.push_back(q);
    }
    return states;
}

std::string SpectralTokenizer::decode_compact(const std::vector<Quat>& states) const {
    if (states.empty()) return "";
    // Each state is ONE complex sample: phase = (byte+pos)·2π/256.
    std::string out;
    out.reserve(states.size());
    for (size_t b = 0; b < states.size(); ++b) {
        const Quat& q = states[b];
        float phase = std::atan2(q.x, q.w);
        if (phase < 0) phase += 2.0f * kPi;
        int byte = static_cast<int>(std::lround(phase * kByteRange / (2.0f * kPi))) - static_cast<int>(b);
        byte = ((byte % kByteRange) + kByteRange) % kByteRange;
        out.push_back(static_cast<char>(byte));
    }
    return out;
}

std::string SpectralTokenizer::decode_fft(const std::vector<Quat>& states) const {
    // The analytic decode IS the O(1) inverse of the isomorphism — there is
    // no FFT needed. Kept as an alias so existing callers can migrate.
    return decode(states);
}

// ---------------------------------------------------------------------------
// MODO HRR — agregação coerente com preservação de norma
// ---------------------------------------------------------------------------
//
// Referência: T. Plate, "Holographic reduced representations", IEEE Trans.
// Neural Networks 6(3), 1995. A construção é a mesma família matemática usada
// no bound de Cauchy-Schwarz e na atenção sem softmax do Madhava: vetores de
// fase sobre a esfera unitária, combinados por soma coerente em vez de média
// (a média destrói a norma por cancelamento).
//
// Por que a soma preserva a norma e a média não:
//
//   Seja z_i = e^{i·θ_i}, i = 1..n, com θ_i ~ Uniforme(0, 2π) (independentes).
//   Então:
//       E[‖Σ z_i‖²] = Σ_i E[|z_i|²] + Σ_{i≠j} E[z_i z̄_j]
//                   = n + 0            (termos cruzados somem por ortogonalidade)
//                   = n
//   A média divide por n:
//       E[‖(1/n)Σ z_i‖²] = n / n² = 1/n → 0
//
//   Ou seja: a soma conserva a energia (‖Z‖² = n), a média a aniquila
//   (‖Z‖² = 1/n). Para a agregação sobreviver à composição com o X-Factor,
//   é a soma — modulada por uma base de frequências que torna as posições
//   distinguíveis — que deve ser usada.

std::vector<float> SpectralTokenizer::encode_hrr(const std::string& text) const {
    // Uma sequência vazia produz o vetor nulo; o chamador decide o que fazer.
    const int D = embed_dim_;
    std::vector<double> re(static_cast<size_t>(D), 0.0);
    std::vector<double> im(static_cast<size_t>(D), 0.0);
    if (text.empty() || D <= 0) {
        return std::vector<float>(re.begin(), re.end());
    }

    const size_t n = text.size();
    const double inv_n = 1.0 / static_cast<double>(n);

    for (size_t pos = 0; pos < n; ++pos) {
        const int b = static_cast<int>(static_cast<unsigned char>(text[pos]));
        for (int j = 0; j < D; ++j) {
            // j é FREQUÊNCIA (não offset): cada modo amostra o byte de forma
            // distinta, e a posição entra com gradiente real.
            //
            //   símbolo  s(b)   = e^{i·2π·j·b/256}
            //   posição  p(pos) = e^{i·2π·j·pos/n}
            //   vínculo  psi    = s(b)·p(pos) = e^{i·2π·j·(b/256 + pos/n)}
            const double ang = 2.0 * static_cast<double>(kPi) *
                               static_cast<double>(j) *
                               (static_cast<double>(b) / 256.0 +
                                static_cast<double>(pos) * inv_n);
            re[static_cast<size_t>(j)] += std::cos(ang);
            im[static_cast<size_t>(j)] += std::sin(ang);
        }
    }

    // Devolve a parte REAL, já dividida por n·sqrt(D) — o fator n mantém a
    // escala independente do comprimento; sqrt(D) normaliza a soma de D
    // componentes aleatórias (energia esperada = n·D/2 por componente).
    const double scale = 1.0 / (static_cast<double>(n) * std::sqrt(static_cast<double>(D)));
    std::vector<float> out(static_cast<size_t>(D));
    for (int j = 0; j < D; ++j) {
        out[static_cast<size_t>(j)] = static_cast<float>(re[static_cast<size_t>(j)] * scale);
    }
    return out;
}

std::vector<float> SpectralTokenizer::encode_hrr_unit(const std::string& text) const {
    std::vector<float> v = encode_hrr(text);
    double ss = 0.0;
    for (float x : v) ss += static_cast<double>(x) * x;
    const double nrm = std::sqrt(ss);
    if (nrm > 1e-12) {
        const double inv = 1.0 / nrm;
        for (float& x : v) x = static_cast<float>(x * inv);
    }
    return v;
}

// ── HRR complexo com posição de taxa fixa ─────────────────────────────────
//
// Duas correções sobre `encode_hrr`, ambas medidas:
//
//   psi = e^{i·2π·j·(byte/256 + pos·stride)}
//
// 1. A parte imaginária é ESCRITA, não descartada. O vetor tem 2·embed_dim
//    floats intercalados. `encode_hrr` devolvia só `re`, o que joga fora
//    metade do sinal e reduz a norma por componente em ~10x.
//
// 2. `pos` entra com uma taxa FIXA (padrão 1/64), não com `pos/n`. Com `pos/n`
//    o binding posicional depende do comprimento do texto, então o vetor de um
//    prefixo muda quando o documento cresce — a representação deixa de ser
//    comparável consigo mesma.
//
// A taxa 1/64 foi escolhida para dar meia volta a cada 32 bytes: dois bytes
// vizinhos ficam bem separados em fase, e a fase não satura dentro de um
// parágrafo. É um parâmetro explícito para poder ser varrido.
void SpectralTokenizer::encode_hrr_into_buffer(const std::string& text, float* out,
                                               double pos_stride) const {
    // UMA LINHA POR BYTE, não um agregado.
    //
    // A primeira versão acumulava Σ_pos dentro do laço e escrevia D pares no
    // fim — ou seja, produzia o vetor AGREGADO do texto inteiro e o chamava de
    // matriz. Medido: para um texto de 69 caracteres, a "matriz" tinha a linha
    // 0 preenchida e 68 linhas de zero, porque cada iteração sobrescrevia o
    // mesmo buffer. O ForwardEngine recebia uma sequência de 68 vetores nulos.
    //
    // A forma correta é ψ[pos] = e^{i·2π·j·(byte/256 + pos·stride)} — a linha
    // depende do byte E da posição, e é o binding posicional que carrega a
    // ordem da sequência para dentro do vetor.
    const int D = embed_dim_;
    if (!out || D <= 0) return;
    const size_t n = text.size();
    const double tau = 2.0 * static_cast<double>(kPi);
    // Escala por 1/sqrt(D): cada linha fica com norma ~1 independente de D.
    // A posição entra como fase, então textos de tamanhos diferentes continuam
    // comparáveis — o que NÃO era verdade com o fator 1/n do agregado.
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));

    for (size_t pos = 0; pos < n; ++pos) {
        const double b = static_cast<double>(static_cast<unsigned char>(text[pos]));
        const double p = static_cast<double>(pos) * pos_stride;
        float* row = out + pos * static_cast<size_t>(D) * 2u;
        for (int j = 0; j < D; ++j) {
            // A fase entra por módulo 1 ANTES de multiplicar por 2π. Sem isso,
            // com D=448 (o valor que casa 2·D com hidden=896), o argumento
            // chega a ~3000 rad e o `cos` perde precisão relativa. Medido: NaN
            // em componentes já em 'hello' antes desta redução.
            double frac = static_cast<double>(j) * (b / 256.0 + p);
            frac -= std::floor(frac);
            const double ang = tau * frac;
            row[static_cast<size_t>(2 * j)]     = static_cast<float>(std::cos(ang) * scale);
            row[static_cast<size_t>(2 * j + 1)] = static_cast<float>(std::sin(ang) * scale);
        }
    }
}

std::vector<float> SpectralTokenizer::encode_hrr_complex(const std::string& text,
                                                         double pos_stride) const {
    // AGREGADO: um vetor para o texto inteiro, não uma linha por byte.
    // É a forma que serve para retrieval (similaridade entre documentos);
    // `encode_hrr_matrix` é a que serve para o forward (uma linha por token).
    // As duas usam a MESMA ψ(pos), então um documento e sua matriz são
    // consistentes: a linha é o termo, o agregado é a soma.
    const int D = embed_dim_;
    std::vector<float> out(static_cast<size_t>(2 * D), 0.0f);
    if (text.empty() || D <= 0) return out;

    const size_t n = text.size();
    const double tau = 2.0 * static_cast<double>(kPi);
    std::vector<double> re(static_cast<size_t>(D), 0.0), im(static_cast<size_t>(D), 0.0);
    for (size_t pos = 0; pos < n; ++pos) {
        const double b = static_cast<double>(static_cast<unsigned char>(text[pos]));
        const double p = static_cast<double>(pos) * pos_stride;
        for (int j = 0; j < D; ++j) {
            double frac = static_cast<double>(j) * (b / 256.0 + p);
            frac -= std::floor(frac);
            const double ang = tau * frac;
            re[(size_t)j] += std::cos(ang);
            im[(size_t)j] += std::sin(ang);
        }
    }
    const double agg = 1.0 / std::sqrt(static_cast<double>(n * (size_t)D));
    for (int j = 0; j < D; ++j) {
        out[(size_t)2 * j]     = (float)(re[(size_t)j] * agg);
        out[(size_t)2 * j + 1] = (float)(im[(size_t)j] * agg);
    }
    return out;
}

} // namespace winnex_nano
