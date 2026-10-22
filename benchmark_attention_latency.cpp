// benchmark_attention_latency.cpp — softmax completo vs atenção seletiva com
// massa controlada, medidos ISOLADAMENTE (só o passo de atenção).
//
// Por que existe: a hipótese era que a poda escala O(N) e portanto o ganho
// cresceria com o contexto. Medição mostra o oposto, e o motivo é estrutural:
//
//   1. Todos os scores têm de ser calculados para saber a massa. O custo
//      O(N·d·N) do QK^T é intocável por uma poda que precisa do softmax
//      completo para decidir o que podar.
//   2. A atenção do Qwen2.5 é concentrada, mas não a ponto de dispensar a
//      maioria das chaves: em ε=1e-4 o conjunto sobrevivente é ~50% do
//      contexto (medido em 540 tokens reais, camadas 0/12/23).
//
// Resultado: a seletiva é 0.78-0.83x (ou seja, MAIS LENTA) em N de 256 a 4096.
// O sort/ordenação que ela adiciona custa mais do que a acumulação de valores
// que ela economiza.
//
// Build: c++ -O3 -std=c++20 benchmark_attention_latency.cpp -o attn_bench
// Uso:   ./attn_bench [epsilon] [min_keep]
// Microbenchmark ISOLADO da atenção: softmax completo vs seletiva com massa
// controlada. Mede apenas o passo de atenção (scores + softmax + acumulação de
// valores), que é o que a poda pode afetar. As projeções densas ficam de fora.
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <cstdlib>
#include <limits>

struct Cfg { int n; int hd; int nh; };

// baseline: softmax causal completo
double attn_full(const std::vector<float>& q, const std::vector<float>& k,
                 const std::vector<float>& v, Cfg c) {
    std::vector<float> sc(c.n);
    double sink = 0.0;
    const float scale = 1.0f / std::sqrt((float)c.hd);
    for (int t = 0; t < c.n; ++t) {
        const float* qh = &q[(size_t)t * c.hd];
        int lim = t + 1;
        float m = -1e30f;
        for (int u = 0; u < lim; ++u) {
            const float* kh = &k[(size_t)u * c.hd];
            float s = 0; for (int j = 0; j < c.hd; ++j) s += qh[j] * kh[j];
            s *= scale; sc[u] = s; if (s > m) m = s;
        }
        double den = 0;
        for (int u = 0; u < lim; ++u) { double e = std::exp((double)sc[u] - m); sc[u] = (float)e; den += e; }
        float inv = (float)(1.0 / (den > 0 ? den : 1.0));
        for (int u = 0; u < lim; ++u) {
            float w = sc[u] * inv;
            if (w == 0.0f) continue;
            const float* vh = &v[(size_t)u * c.hd];
            for (int j = 0; j < c.hd; ++j) sink += w * vh[j];
        }
    }
    return sink;
}

// seletiva: massa controlada (epsilon)
double attn_sel(const std::vector<float>& q, const std::vector<float>& k,
                const std::vector<float>& v, Cfg c, double eps, int minkeep,
                long long* pruned_out = nullptr) {
    std::vector<float> sc(c.n);
    std::vector<std::pair<float,int>> ranked;
    double sink = 0.0; long long pruned = 0;
    const float scale = 1.0f / std::sqrt((float)c.hd);
    for (int t = 0; t < c.n; ++t) {
        const float* qh = &q[(size_t)t * c.hd];
        int lim = t + 1;
        float m = -1e30f;
        for (int u = 0; u < lim; ++u) {
            const float* kh = &k[(size_t)u * c.hd];
            float s = 0; for (int j = 0; j < c.hd; ++j) s += qh[j] * kh[j];
            s *= scale; sc[u] = s; if (s > m) m = s;
        }
        int keep = lim;
        if (lim > minkeep) {
            ranked.resize(lim);
            for (int u = 0; u < lim; ++u) ranked[u] = {sc[u], u};
            std::sort(ranked.begin(), ranked.end(),
                      [](auto&a, auto&b){ return a.first > b.first; });
            double z_all = 0;
            for (int u = 0; u < lim; ++u) z_all += std::exp((double)sc[u] - m);
            const double zt = (1.0 - eps) * z_all;
            double zk = 0; int kk = 0;
            for (; kk < lim; ++kk) {
                zk += std::exp((double)ranked[kk].first - m);
                if (kk + 1 >= minkeep && zk >= zt) break;
            }
            keep = std::min(lim, kk + 1);
            pruned += lim - keep;
        }
        if (keep == lim) {
            double den = 0;
            for (int u = 0; u < lim; ++u) { sc[u] = (float)std::exp((double)sc[u] - m); den += sc[u]; }
            float inv = (float)(1.0 / (den > 0 ? den : 1.0));
            for (int u = 0; u < lim; ++u) {
                float w = sc[u] * inv;
                const float* vh = &v[(size_t)u * c.hd];
                for (int j = 0; j < c.hd; ++j) sink += w * vh[j];
            }
        } else {
            double den = 0;
            for (int i = 0; i < keep; ++i) den += std::exp((double)ranked[i].first - m);
            float inv = (float)(1.0 / (den > 0 ? den : 1.0));
            for (int i = 0; i < keep; ++i) {
                float w = (float)std::exp((double)ranked[i].first - m) * inv;
                const float* vh = &v[(size_t)ranked[i].second * c.hd];
                for (int j = 0; j < c.hd; ++j) sink += w * vh[j];
            }
        }
    }
    if (pruned_out) *pruned_out = pruned;
    return sink;
}

int main(int argc, char** argv) {
    const double EPS = argc > 1 ? atof(argv[1]) : 1e-4;
    const int MINKEEP = argc > 2 ? atoi(argv[2]) : 64;
    // Atenção de UMA cabeça (nh=1) para isolar o custo da atenção.
    Cfg c; c.hd = 64; c.nh = 1;
    std::mt19937 rng(42); std::normal_distribution<float> nd(0, 1);
    printf("%8s %12s %12s %10s %12s %10s\n",
           "N", "full (ms)", "sel (ms)", "speedup", "keys kept", "pruned");
    printf("%s\n", std::string(68, '-').c_str());
    for (int n : {256, 512, 1024, 2048, 4096}) {
        c.n = n;
        std::vector<float> q((size_t)n*c.hd), k((size_t)n*c.hd), v((size_t)n*c.hd);
        for (auto& x : q) x = nd(rng);
        for (auto& x : k) x = nd(rng);
        for (auto& x : v) x = nd(rng);
        auto t0 = std::chrono::high_resolution_clock::now();
        double s1 = attn_full(q,k,v,c);
        auto t1 = std::chrono::high_resolution_clock::now();
        long long pruned = 0;
        double s2 = attn_sel(q,k,v,c,EPS,MINKEEP,&pruned);
        auto t2 = std::chrono::high_resolution_clock::now();
        double ms1 = std::chrono::duration<double,std::milli>(t1-t0).count();
        double ms2 = std::chrono::duration<double,std::milli>(t2-t1).count();
        long long total = (long long)n*(n+1)/2;
        printf("%8d %12.1f %12.1f %9.2fx %12lld %10lld\n",
               n, ms1, ms2, ms1/ms2, total-pruned, pruned);
        if (std::fabs(s1-s2) > 1e-3*std::fabs(s1)+1e-2)
            printf("         (sanity: saidas divergem s1=%.4f s2=%.4f)\n", s1, s2);
    }
}
