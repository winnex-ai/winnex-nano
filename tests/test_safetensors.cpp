// test_safetensors.cpp — validates the native safetensors reader.
#include "winnex_nano/safetensors.hpp"

#include <cstdio>
#include <string>

using winnex_nano::Safetensors;

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s <model.safetensors>\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];

    try {
        Safetensors st(path);
        printf("loaded %zu tensors\n", st.count());

        // Check key tensors.
        const char* checks[] = {
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_proj.qweight",
            "model.layers.0.mlp.gate_proj.qweight",
            "lm_head.weight",
        };
        int ok = 0;
        for (const char* name : checks) {
            if (!st.has(name)) { printf("MISSING: %s\n", name); continue; }
            const auto& t = st.get(name);
            printf("  %s: %d %s [", name, (int)t.dtype, name);
            for (size_t i = 0; i < t.shape.size(); ++i) printf("%lld%s", (long long)t.shape[i], i+1<t.shape.size()?",":"");
            printf("]\n");
            ok++;
        }
        printf("found %d/%zu key tensors\n", ok, sizeof(checks)/sizeof(checks[0]));

        // Read embed_tokens as float (F16) and check a few values.
        if (st.has("model.embed_tokens.weight")) {
            auto emb = st.to_float(st.get("model.embed_tokens.weight"));
            printf("embed_tokens[0][:4] = %.4f %.4f %.4f %.4f\n",
                   emb[0], emb[1], emb[2], emb[3]);
        }
        // Read a GPTQ qweight as int32.
        if (st.has("model.layers.0.self_attn.q_proj.qweight")) {
            auto qw = st.to_int32(st.get("model.layers.0.self_attn.q_proj.qweight"));
            printf("qweight[0] = %d (int32, packed int4)\n", qw[0]);
        }
        return 0;
    } catch (const std::exception& e) {
        printf("ERROR: %s\n", e.what());
        return 1;
    }
}
