#include "winnex_nano/embed_bridge.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>

namespace winnex_nano {

namespace {

// Resolve (A + λI)·Z = B para Z, com A simétrica [m, m] e B [m, n_out].
// Cholesky com fallback para Gauss-Jordan pivotado: a matriz de Gram de uma
// codificação de fase fica mal condicionada com frequência, e o fallback evita
// que o ajuste exploda em vez de simplesmente ficar menos preciso.
void solve_spd(std::vector<double>& A, std::vector<double>& B, int m, int nrhs) {
    // Cholesky: A = L·Lᵀ
    std::vector<double> L((size_t)m * m, 0.0);
    bool ok = true;
    for (int i = 0; i < m && ok; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[(size_t)i * m + j];
            for (int k = 0; k < j; ++k) s -= L[(size_t)i * m + k] * L[(size_t)j * m + k];
            if (i == j) {
                if (s <= 0.0) { ok = false; break; }
                L[(size_t)i * m + i] = std::sqrt(s);
            } else {
                L[(size_t)i * m + j] = s / L[(size_t)j * m + j];
            }
        }
    }
    if (ok) {
        // Forward + back substitution per RHS.
        std::vector<double> y((size_t)m);
        for (int r = 0; r < nrhs; ++r) {
            for (int i = 0; i < m; ++i) {
                double s = B[(size_t)i * nrhs + r];
                for (int k = 0; k < i; ++k) s -= L[(size_t)i * m + k] * y[(size_t)k];
                y[(size_t)i] = s / L[(size_t)i * m + i];
            }
            for (int i = m - 1; i >= 0; --i) {
                double s = y[(size_t)i];
                for (int k = i + 1; k < m; ++k)
                    s -= L[(size_t)k * m + i] * B[(size_t)k * nrhs + r];
                B[(size_t)i * nrhs + r] = s / L[(size_t)i * m + i];
            }
        }
        return;
    }

    // Fallback: Gauss-Jordan com pivoteamento parcial, uma RHS de cada vez.
    for (int r = 0; r < nrhs; ++r) {
        std::vector<double> M((size_t)m * (m + 1));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) M[(size_t)i * (m + 1) + j] = A[(size_t)i * m + j];
            M[(size_t)i * (m + 1) + m] = B[(size_t)i * nrhs + r];
        }
        for (int col = 0; col < m; ++col) {
            int piv = col;
            for (int i = col + 1; i < m; ++i)
                if (std::fabs(M[(size_t)i * (m + 1) + col]) >
                    std::fabs(M[(size_t)piv * (m + 1) + col])) piv = i;
            if (std::fabs(M[(size_t)piv * (m + 1) + col]) < 1e-14) continue;
            if (piv != col)
                for (int j = 0; j <= m; ++j)
                    std::swap(M[(size_t)piv * (m + 1) + j], M[(size_t)col * (m + 1) + j]);
            const double d = M[(size_t)col * (m + 1) + col];
            for (int j = col; j <= m; ++j) M[(size_t)col * (m + 1) + j] /= d;
            for (int i = 0; i < m; ++i) {
                if (i == col) continue;
                const double f = M[(size_t)i * (m + 1) + col];
                if (f == 0.0) continue;
                for (int j = col; j <= m; ++j)
                    M[(size_t)i * (m + 1) + j] -= f * M[(size_t)col * (m + 1) + j];
            }
        }
        for (int i = 0; i < m; ++i) B[(size_t)i * nrhs + r] = M[(size_t)i * (m + 1) + m];
    }
}

bool all_finite(const float* p, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(p[i])) return false;
    return true;
}

} // namespace

void EmbedBridge::fit(const float* X, const float* Y, int n, int d_in, int d_out,
                      double lambda, Kernel kernel, int n_centers) {
    if (!X || !Y) throw std::invalid_argument("EmbedBridge::fit: null pointer");
    if (n <= 0) throw std::invalid_argument("EmbedBridge::fit: n must be > 0");
    if (d_in <= 0 || d_out <= 0)
        throw std::invalid_argument("EmbedBridge::fit: dimensions must be > 0");
    if (lambda < 0.0) throw std::invalid_argument("EmbedBridge::fit: lambda < 0");
    if (!all_finite(X, (size_t)n * d_in))
        throw std::invalid_argument("EmbedBridge::fit: X contains non-finite values");
    if (!all_finite(Y, (size_t)n * d_out))
        throw std::invalid_argument("EmbedBridge::fit: Y contains non-finite values");

    fitted_ = false;
    kernel_ = kernel;
    d_in_ = d_in;
    d_out_ = d_out;
    n_fit_ = n;
    lambda_ = lambda;

    // ── Média do alvo: o offset b ──────────────────────────────────────────
    // Prefixar o offset reduz muito o condicionamento: sem ele, a matriz de
    // Gram de uma codificação de fase (que tem média não nula) fica pior.
    std::vector<double> ymean((size_t)d_out, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < d_out; ++j) ymean[(size_t)j] += Y[(size_t)i * d_out + j];
    for (int j = 0; j < d_out; ++j) ymean[(size_t)j] /= (double)n;

    // σ(Y) para o R².
    double yvar = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < d_out; ++j) {
            const double d = Y[(size_t)i * d_out + j] - ymean[(size_t)j];
            yvar += d * d;
        }
    yvar /= (double)n * d_out;

    if (kernel == Kernel::Rbf) {
        // ── Centros por k-means leve (subamostra para manter O(k·n·d) barato) ──
        int k = (n_centers > 0) ? n_centers : std::min(256, std::max(8, n / 16));
        k = std::min(k, n);
        // Inicialização determinística por amostragem espaçada — evita a
        // dependência de semente que faria dois ajustes idênticos divergirem.
        centers_.assign((size_t)k * d_in, 0.0f);
        for (int c = 0; c < k; ++c) {
            const int src = (int)((long long)c * n / k);
            std::memcpy(centers_.data() + (size_t)c * d_in,
                        X + (size_t)src * d_in, (size_t)d_in * sizeof(float));
        }
        std::vector<int> assign((size_t)n, 0);
        for (int iter = 0; iter < 8; ++iter) {
            for (int i = 0; i < n; ++i) {
                double best = 1e300; int bi = 0;
                for (int c = 0; c < k; ++c) {
                    double s = 0.0;
                    const float* xc = centers_.data() + (size_t)c * d_in;
                    const float* xi = X + (size_t)i * d_in;
                    for (int j = 0; j < d_in; ++j) {
                        const double dd = (double)xi[j] - xc[j];
                        s += dd * dd;
                    }
                    if (s < best) { best = s; bi = c; }
                }
                assign[(size_t)i] = bi;
            }
            std::vector<double> acc((size_t)k * d_in, 0.0);
            std::vector<int> cnt((size_t)k, 0);
            for (int i = 0; i < n; ++i) {
                const int c = assign[(size_t)i];
                cnt[(size_t)c]++;
                for (int j = 0; j < d_in; ++j)
                    acc[(size_t)c * d_in + j] += X[(size_t)i * d_in + j];
            }
            for (int c = 0; c < k; ++c)
                if (cnt[(size_t)c] > 0)
                    for (int j = 0; j < d_in; ++j)
                        centers_[(size_t)c * d_in + j] =
                            (float)(acc[(size_t)c * d_in + j] / cnt[(size_t)c]);
        }

        // gamma: heurística da mediana das distâncias aos centros.
        std::vector<double> dmin((size_t)n, 1e300);
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < k; ++c) {
                double s = 0.0;
                const float* xc = centers_.data() + (size_t)c * d_in;
                const float* xi = X + (size_t)i * d_in;
                for (int j = 0; j < d_in; ++j) {
                    const double dd = (double)xi[j] - xc[j];
                    s += dd * dd;
                }
                dmin[(size_t)i] = std::min(dmin[(size_t)i], s);
            }
        std::vector<double> tmp = dmin;
        std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
        const double med = std::max(tmp[tmp.size() / 2], 1e-12);
        gamma_ = 1.0 / med;

        // Features φ(x) = [1, k(x,c_1), ..., k(x,c_k)] — k+1 dimensões.
        const int m = k + 1;
        std::vector<double> A((size_t)m * m, 0.0);
        std::vector<double> B((size_t)m * d_out, 0.0);
        std::vector<double> phi((size_t)m);
        for (int i = 0; i < n; ++i) {
            phi[0] = 1.0;
            const float* xi = X + (size_t)i * d_in;
            for (int c = 0; c < k; ++c) {
                double s = 0.0;
                const float* xc = centers_.data() + (size_t)c * d_in;
                for (int j = 0; j < d_in; ++j) {
                    const double dd = (double)xi[j] - xc[j];
                    s += dd * dd;
                }
                phi[(size_t)c + 1] = std::exp(-gamma_ * s);
            }
            for (int a = 0; a < m; ++a) {
                for (int b2 = 0; b2 < m; ++b2) A[(size_t)a * m + b2] += phi[(size_t)a] * phi[(size_t)b2];
                for (int o = 0; o < d_out; ++o)
                    B[(size_t)a * d_out + o] += phi[(size_t)a] *
                        ((double)Y[(size_t)i * d_out + o] - ymean[(size_t)o]);
            }
        }
        for (int a = 0; a < m; ++a) A[(size_t)a * m + a] += lambda;
        solve_spd(A, B, m, d_out);
        alpha_.resize((size_t)m * d_out);
        for (size_t i = 0; i < alpha_.size(); ++i) alpha_[i] = (float)B[i];
    } else {
        // ── Linear: (XᵀX + λI)·W = Xᵀ(Y − ȳ) ──────────────────────────────
        std::vector<double> A((size_t)d_in * d_in, 0.0);
        std::vector<double> B((size_t)d_in * d_out, 0.0);
        for (int i = 0; i < n; ++i) {
            const float* xi = X + (size_t)i * d_in;
            const float* yi = Y + (size_t)i * d_out;
            for (int a = 0; a < d_in; ++a) {
                const double xa = xi[a];
                for (int b2 = 0; b2 < d_in; ++b2)
                    A[(size_t)a * d_in + b2] += xa * xi[b2];
                for (int o = 0; o < d_out; ++o)
                    B[(size_t)a * d_out + o] += xa * ((double)yi[o] - ymean[(size_t)o]);
            }
        }
        for (int a = 0; a < d_in; ++a) A[(size_t)a * d_in + a] += lambda;
        solve_spd(A, B, d_in, d_out);
        W_.resize((size_t)d_in * d_out);
        for (size_t i = 0; i < W_.size(); ++i) W_[i] = (float)B[i];
        centers_.clear();
        alpha_.clear();
    }

    b_.resize((size_t)d_out);
    for (int o = 0; o < d_out; ++o) b_[o] = (float)ymean[(size_t)o];

    // fitted_ precisa ser verdadeiro ANTES da medição de erro: apply() recusa
    // rodar num objeto não ajustado, e a verificação abaixo chama apply().
    // Marcar depois fazia todo fit() falhar com "not fitted" no primeiro uso.
    fitted_ = true;

    // ── Erro de treino e R² ────────────────────────────────────────────────
    std::vector<float> pred((size_t)d_out);
    double sse = 0.0;
    for (int i = 0; i < n; ++i) {
        apply(X + (size_t)i * d_in, pred.data());
        for (int o = 0; o < d_out; ++o) {
            const double d = pred[(size_t)o] - Y[(size_t)i * d_out + o];
            sse += d * d;
        }
    }
    train_rmse_ = std::sqrt(sse / ((double)n * d_out));
    train_r2_ = (yvar > 1e-30) ? 1.0 - (train_rmse_ * train_rmse_) / yvar : 0.0;
}

void EmbedBridge::fit_from_texts(
    const std::vector<std::string>& texts, const std::vector<float>& targets,
    int d_in, int d_out,
    void (*encode)(const std::string&, float*, void*), void* encode_ctx,
    double lambda, Kernel kernel, int n_centers) {
    if (!encode) throw std::invalid_argument("fit_from_texts: null encode");
    const int n = (int)texts.size();
    if (n <= 0) throw std::invalid_argument("fit_from_texts: no texts");
    if ((size_t)n * d_out != targets.size())
        throw std::invalid_argument("fit_from_texts: targets must be [n, d_out]");

    std::vector<float> X((size_t)n * d_in);
    std::vector<float> buf((size_t)d_in);
    for (int i = 0; i < n; ++i) {
        encode(texts[(size_t)i], buf.data(), encode_ctx);
        std::memcpy(X.data() + (size_t)i * d_in, buf.data(), (size_t)d_in * sizeof(float));
    }
    fit(X.data(), targets.data(), n, d_in, d_out, lambda, kernel, n_centers);
}

void EmbedBridge::apply(const float* x, float* out) const {
    if (!fitted_)
        throw std::logic_error("EmbedBridge::apply: not fitted");
    if (!x || !out) throw std::invalid_argument("EmbedBridge::apply: null pointer");

    if (kernel_ == Kernel::Rbf) {
        const int k = (int)centers_.size() / d_in_;
        const int m = k + 1;
        std::vector<double> acc((size_t)d_out_, 0.0);
        const double* alpha0 = nullptr;
        (void)alpha0;
        // Intercepto
        for (int o = 0; o < d_out_; ++o) acc[(size_t)o] += (double)alpha_[(size_t)o];
        for (int c = 0; c < k; ++c) {
            double s = 0.0;
            const float* xc = centers_.data() + (size_t)c * d_in_;
            for (int j = 0; j < d_in_; ++j) {
                const double dd = (double)x[j] - xc[j];
                s += dd * dd;
            }
            const double kv = std::exp(-gamma_ * s);
            const float* ac = alpha_.data() + (size_t)(c + 1) * d_out_;
            for (int o = 0; o < d_out_; ++o) acc[(size_t)o] += kv * ac[o];
        }
        for (int o = 0; o < d_out_; ++o)
            out[o] = (float)(acc[(size_t)o] + b_[(size_t)o]);
        (void)m;
        return;
    }

    for (int o = 0; o < d_out_; ++o) {
        double s = 0.0;
        for (int j = 0; j < d_in_; ++j) s += (double)W_[(size_t)j * d_out_ + o] * x[j];
        out[o] = (float)(s + b_[(size_t)o]);
    }
}

void EmbedBridge::apply_batch(const float* X, float* out, int n) const {
    for (int i = 0; i < n; ++i)
        apply(X + (size_t)i * d_in_, out + (size_t)i * d_out_);
}

void EmbedBridge::encode_and_apply(const std::string& text,
                                   void (*encode)(const std::string&, float*, void*),
                                   void* encode_ctx, float* out) const {
    if (!encode) throw std::invalid_argument("encode_and_apply: null encode");
    std::vector<float> buf((size_t)d_in_);
    encode(text, buf.data(), encode_ctx);
    apply(buf.data(), out);
}

// ── Persistência ──────────────────────────────────────────────────────────
// Formato binário simples, documentado aqui para poder ser lido de outro lugar:
//   magic "WNXB1" | int32 kernel, d_in, d_out, n_fit, n_centers
//   float64 lambda, gamma, train_rmse, train_r2
//   W    [d_in*d_out]   floats   (vazio no Rbf)
//   b    [d_out]        floats
//   C    [k*d_in]       floats   (vazio no linear)
//   A    [k+1 * d_out]  floats   (vazio no linear)
void EmbedBridge::save(const std::string& path) const {
    if (!fitted_) throw std::logic_error("EmbedBridge::save: not fitted");
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("EmbedBridge::save: cannot open " + path);
    const char magic[6] = {'W', 'N', 'X', 'B', '1', '\0'};
    f.write(magic, 6);
    int32_t hdr[5] = {(int32_t)kernel_, d_in_, d_out_, n_fit_,
                      (int32_t)(centers_.size() / (d_in_ > 0 ? d_in_ : 1))};
    f.write((char*)hdr, sizeof(hdr));
    double dbl[4] = {lambda_, gamma_, train_rmse_, train_r2_};
    f.write((char*)dbl, sizeof(dbl));
    auto wr = [&](const std::vector<float>& v) {
        if (!v.empty()) f.write((const char*)v.data(), (std::streamsize)(v.size() * sizeof(float)));
    };
    wr(W_); wr(b_); wr(centers_); wr(alpha_);
}

EmbedBridge EmbedBridge::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("EmbedBridge::load: cannot open " + path);
    char magic[6] = {0};
    f.read(magic, 6);
    if (std::string(magic, 5) != "WNXB1")
        throw std::runtime_error("EmbedBridge::load: bad magic in " + path);
    int32_t hdr[5] = {0};
    f.read((char*)hdr, sizeof(hdr));
    double dbl[4] = {0};
    f.read((char*)dbl, sizeof(dbl));

    EmbedBridge b;
    b.kernel_ = (Kernel)hdr[0];
    b.d_in_ = hdr[1];
    b.d_out_ = hdr[2];
    b.n_fit_ = hdr[3];
    const int k = hdr[4];
    b.lambda_ = dbl[0]; b.gamma_ = dbl[1];
    b.train_rmse_ = dbl[2]; b.train_r2_ = dbl[3];

    auto rd = [&](std::vector<float>& v, size_t n) {
        v.resize(n);
        if (n) f.read((char*)v.data(), (std::streamsize)(n * sizeof(float)));
    };
    if (b.kernel_ == Kernel::Linear) {
        rd(b.W_, (size_t)b.d_in_ * b.d_out_);
        rd(b.b_, (size_t)b.d_out_);
    } else {
        rd(b.b_, (size_t)b.d_out_);
        rd(b.centers_, (size_t)k * b.d_in_);
        rd(b.alpha_, (size_t)(k + 1) * b.d_out_);
    }
    b.fitted_ = true;
    return b;
}

std::string EmbedBridge::describe() const {
    if (!fitted_) return "EmbedBridge(unfitted)";
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "EmbedBridge(%s, %d->%d, n=%d, rmse=%.4g, r2=%.4f)",
                  kernel_ == Kernel::Linear ? "linear" : "rbf",
                  d_in_, d_out_, n_fit_, train_rmse_, train_r2_);
    return std::string(buf);
}

} // namespace winnex_nano
