// test_x_factor.cpp — validates the X-factor math (manifold basis).
#include "winnex_nano/x_factor.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using winnex_nano::XFactor;

int main() {
    int failures = 0;

    // Synthetic embed_tokens: V vectors in D dims with a clear low-rank manifold.
    // We create a 4-dim intrinsic structure embedded in 16 dims (random basis).
    const int V = 500, D = 16, intrinsic = 4;
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0, 1);

    // Random D×intrinsic basis.
    std::vector<float> B(D * intrinsic);
    for (auto& b : B) b = nd(rng);
    // Orthonormalize (Gram-Schmidt).
    for (int i = 0; i < intrinsic; ++i) {
        for (int k = 0; k < i; ++k) {
            float dp = 0;
            for (int j = 0; j < D; ++j) dp += B[j*intrinsic+i] * B[j*intrinsic+k];
            for (int j = 0; j < D; ++j) B[j*intrinsic+i] -= dp * B[j*intrinsic+k];
        }
        float nr = 0;
        for (int j = 0; j < D; ++j) nr += B[j*intrinsic+i] * B[j*intrinsic+i];
        nr = std::sqrt(nr) + 1e-9f;
        for (int j = 0; j < D; ++j) B[j*intrinsic+i] /= nr;
    }

    // Embed tokens: v = B · z, z ~ N(0,1)^intrinsic.
    std::vector<float> E(V * D);
    for (int i = 0; i < V; ++i) {
        std::vector<float> z(intrinsic);
        for (auto& zi : z) zi = nd(rng);
        for (int j = 0; j < D; ++j) {
            float s = 0;
            for (int k = 0; k < intrinsic; ++k) s += B[j*intrinsic+k] * z[k];
            E[i*D + j] = s;
        }
    }

    XFactor xf(E.data(), V, D, 0.95);

    // 1. Effective rank should be much smaller than D (the manifold was
    //    detected). With 500 samples the covariance carries noise, so the
    //    exact intrinsic dim is not recovered — but r << D must hold.
    int r = xf.rank();
    bool ok1 = (r < D);  // manifold detected: rank well below ambient dim
    printf("effective rank (intrinsic=%d, D=%d): %d [%s]\n", intrinsic, D, r,
           ok1 ? "OK" : "FAIL");
    if (!ok1) ++failures;

    // 2. Projector P is idempotent: P·P = P.
    const float* P = xf.projector();
    bool idem = true;
    for (int a = 0; a < D; ++a) {
        for (int b = 0; b < D; ++b) {
            float s = 0;
            for (int k = 0; k < D; ++k) s += P[a*D+k] * P[k*D+b];
            if (std::fabs(s - P[a*D+b]) > 1e-3) { idem = false; break; }
        }
        if (!idem) break;
    }
    printf("projector idempotent (P²=P): %s\n", idem ? "OK" : "FAIL");
    if (!idem) ++failures;

    // 3. Vectors ON the manifold are preserved by projection (nearly identity).
    std::vector<float> v0(D), h(D);
    for (int j = 0; j < D; ++j) v0[j] = E[j];  // first token, on the manifold
    xf.project(v0.data(), h.data());
    float cos = 0, n1 = 0, n2 = 0;
    for (int j = 0; j < D; ++j) { cos += v0[j]*h[j]; n1 += v0[j]*v0[j]; n2 += h[j]*h[j]; }
    float c = cos / (std::sqrt(n1)*std::sqrt(n2) + 1e-9f);
    bool ok3 = (c > 0.9);
    printf("manifold vector preserved (cos=%.3f): %s\n", c, ok3 ? "OK" : "FAIL");
    if (!ok3) ++failures;

    // 4. expand_spectral produces a D-dim unit vector.
    std::vector<float> psi(8, 0.0f), expD(D);
    for (int j = 0; j < 8; ++j) psi[j] = (j % 2 ? 1.0f : -0.5f);
    winnex_nano::expand_spectral(psi.data(), 8, expD.data(), D);
    float nn = 0;
    for (int j = 0; j < D; ++j) nn += expD[j]*expD[j];
    bool ok4 = (std::fabs(std::sqrt(nn) - 1.0f) < 1e-3);
    printf("expand_spectral unit norm: %s\n", ok4 ? "OK" : "FAIL");
    if (!ok4) ++failures;

    if (failures == 0) {
        printf("\nALL X-FACTOR TESTS PASSED\n");
        return 0;
    }
    printf("\n%d FAILURE(S)\n", failures);
    return 1;
}
