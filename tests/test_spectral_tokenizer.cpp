// test_spectral_tokenizer.cpp — round-trip validation of the Winnex spectral
// tokenizer (encode(text) -> decode(encode(text)) == text) and the
// multimodel weight balancer (W' = Σαᵢ·R(qᵢ)·Wᵢ).
#include "winnex_nano/spectral_tokenizer.hpp"
#include "winnex_nano/weight_balancer.hpp"

#include <cstdio>
#include <string>
#include <vector>

using winnex_nano::SpectralTokenizer;
using winnex_nano::WeightBalancer;
using winnex_nano::WeightMap;

int main() {
    int failures = 0;

    // --- Spectral tokenizer round-trip --------------------------------------
    SpectralTokenizer tok(64);

    const std::vector<std::string> samples = {
        "Winnex AI",
        "Madhava engine",
        "Winnex spectral tokenizer",
        "A B C 123 !@#",
        "Deterministic round-trip test 456",
        // pt-BR (bytes > 126 — the historical ASCII-clamp failure).
        "café",
        "ação",
        "não",
        "português",
        "Não é possível fazer isso.",
        // Other UTF-8 scripts (multibyte).
        "こんにちは",
        "Привет мир",
        "مرحبا بالعالم",
    };

    for (const auto& s : samples) {
        auto states = tok.encode(s);
        auto out = tok.decode(states);
        bool ok = (out == s);
        printf("spectral round-trip: '%s' -> %s [%s]\n",
               s.c_str(), ok ? out.c_str() : "MISMATCH", ok ? "OK" : "FAIL");
        if (!ok) ++failures;

        // decode_fft (the analytic O(1) inverse) must agree with decode().
        auto out_fft = tok.decode_fft(states);
        bool ok_fft = (out_fft == s);
        printf("  decode_fft: '%s' [%s]\n",
               ok_fft ? out_fft.c_str() : "MISMATCH", ok_fft ? "OK" : "FAIL");
        if (!ok_fft) ++failures;
    }

    // Cross-check: different chars must encode to different spectra.
    auto a = tok.encode("A");
    auto b = tok.encode("B");
    bool distinct = false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].w != b[i].w || a[i].x != b[i].x) { distinct = true; break; }
    }
    printf("distinct spectra for A vs B: %s\n", distinct ? "OK" : "FAIL");
    if (!distinct) ++failures;

    // --- Weight balancer (multimodel fusion) --------------------------------
    WeightMap m1, m2;
    m1["layer.0.weight"] = {1,2,3,4, 5,6,7,8};
    m2["layer.0.weight"] = {10,20,30,40, 50,60,70,80};

    WeightBalancer wb;
    auto out1 = wb.blend({m1, m2}, {{1.0,0,0,0},{0.0,0,0,0}});
    bool ok1 = (out1["layer.0.weight"] == m1["layer.0.weight"]);
    printf("balancer alpha=[1,0] = model1: %s\n", ok1 ? "OK" : "FAIL");
    if (!ok1) ++failures;

    auto out3 = wb.blend({m1, m2}, {{0.5,0,0,0},{0.5,0,0,0}});
    bool ok3 = true;
    for (size_t i = 0; i < 8; ++i) {
        float expect = 0.5f * m1["layer.0.weight"][i] + 0.5f * m2["layer.0.weight"][i];
        if (out3["layer.0.weight"][i] != expect) ok3 = false;
    }
    printf("balancer alpha=[0.5,0.5] mean: %s\n", ok3 ? "OK" : "FAIL");
    if (!ok3) ++failures;

    auto out4 = wb.blend_with_rotation({m1, m2}, {{1.0,0,0,0},{0.0,0,0,0}});
    bool ok4 = (out4["layer.0.weight"] == m1["layer.0.weight"]);
    printf("balancer rotation theta=0 identity: %s\n", ok4 ? "OK" : "FAIL");
    if (!ok4) ++failures;

    if (failures == 0) {
        printf("\nALL WINNEX-NANO CORE TESTS PASSED\n");
        return 0;
    }
    printf("\n%d FAILURE(S)\n", failures);
    return 1;
}
