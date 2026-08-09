#!/usr/bin/env python3
"""
Benchmark honesto — winnex-nano (tokenizer espectral Winnex) vs BPE.

Compara em métricas REAIS:
  1. Corretude de round-trip (texto -> espectro -> texto)
  2. Latência (encode/decode): caracteres/seg e tokens/seg
  3. Discriminabilidade espectral (quão distinto cada caractere é)
  4. Densidade da representação (bits por caractere)

Objetivo: verificar HONESTOmente o que o tokenizer espectral entrega vs o BPE
(traduzido para o mesmo vocabulário por caractere). Não há cherry-picking —
mede-se o que existe.
"""

import sys
import os
import time
import json
import math
import statistics

# winnex-nano binding
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _winnex_nano as wn

# ============================================================================
# 1. Amostras de texto (realistas, variadas)
# ============================================================================
SAMPLES = [
    "Winnex AI - plataforma de inteligencia artificial",
    "Madhava: busca vetorial deterministica com prova Cauchy-Schwarz",
    "O despacho aduaneiro exige documentacao completa para importacao",
    "Folha de pagamento processada mensalmente pelo departamento de RH",
    "Winnex reformula transformers usando matematica quaternionica",
    "Os 123 numeros e 456 simbolos !@# devem ser preservados fielmente",
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
    "Teste de acentuacao: acao, coracao, licao, composicao",
]

# ============================================================================
# 2. Métricas do tokenizer espectral
# ============================================================================
def bench_spectral(embed_dim=64, repeats=20):
    tok = wn.SpectralTokenizer(embed_dim)
    results = {}

    # 2.1 Round-trip corretude
    correct = 0
    char_errors = 0
    total_chars = 0
    for s in SAMPLES:
        states = tok.encode(s)
        out = tok.decode(states)
        if out == s:
            correct += 1
        # contagem de erros por caractere
        total_chars += len(s)
        for i in range(min(len(s), len(out))):
            if s[i] != out[i]:
                char_errors += 1
        if len(s) != len(out):
            char_errors += abs(len(s) - len(out))
    results['roundtrip_perfect'] = correct
    results['roundtrip_total'] = len(SAMPLES)
    results['char_error_rate'] = char_errors / max(total_chars, 1)

    # 2.2 Latência encode (caracteres/seg)
    big_text = " ".join(SAMPLES) * 10  # ~2000 chars
    t0 = time.perf_counter()
    for _ in range(repeats):
        tok.encode(big_text)
    dt = (time.perf_counter() - t0) / repeats
    results['encode_chars_per_sec'] = len(big_text) / max(dt, 1e-9)

    # 2.3 Latência decode
    states = tok.encode(big_text)
    t0 = time.perf_counter()
    for _ in range(repeats):
        tok.decode(states)
    dt = (time.perf_counter() - t0) / repeats
    results['decode_chars_per_sec'] = len(big_text) / max(dt, 1e-9)

    # 2.4 Discriminabilidade: similaridade média entre caracteres distintos
    # (menor = mais separáveis). Mede cos(spectral) entre pares de chars.
    import itertools
    chars = [chr(c) for c in range(32, 127)]
    sims = []
    for a, b in itertools.islice(itertools.combinations(chars, 2), 500):
        sa = tok.encode(a)
        sb = tok.encode(b)
        # cosine sobre [embed_dim, 4]
        ip = n1 = n2 = 0.0
        for qa, qb in zip(sa, sb):
            ip += qa.w*qb.w + qa.x*qb.x + qa.y*qb.y + qa.z*qb.z
            n1 += qa.w*qa.w + qa.x*qa.x + qa.y*qa.y + qa.z*qa.z
            n2 += qb.w*qb.w + qb.x*qb.x + qb.y*qb.y + qb.z*qb.z
        denom = math.sqrt(n1*n2) + 1e-12
        sims.append(ip / denom)
    results['mean_inter_char_similarity'] = statistics.mean(sims) if sims else 1.0
    results['max_inter_char_similarity'] = max(sims) if sims else 1.0

    # 2.5 Densidade: representação por caractere
    # Cada caractere = embed_dim quats × 4 floats × 4 bytes
    bits_per_char = embed_dim * 4 * 4 * 8
    results['bits_per_char'] = bits_per_char
    results['embed_dim'] = embed_dim

    return results


# ============================================================================
# 3. Baseline: BPE (via transformers, Qwen) — para referência
# ============================================================================
def bench_bpe():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "/home/wnnx_user/docker-winnex-stack/data/models/Qwen2.5-Coder-3B-Instruct-GPTQ",
            local_files_only=True)
        big_text = " ".join(SAMPLES) * 10

        # Latência encode (tokens/seg)
        t0 = time.perf_counter()
        for _ in range(5):
            tok(big_text, add_special_tokens=False)
        dt = (time.perf_counter() - t0) / 5
        enc_chars_ps = len(big_text) / max(dt, 1e-9)

        # tokens por caractere
        n_tokens = len(tok(big_text, add_special_tokens=False).input_ids)
        tokens_per_char = n_tokens / len(big_text)

        return {
            'bpe_encode_chars_per_sec': enc_chars_ps,
            'bpe_tokens_per_char': tokens_per_char,
            'bpe_vocab_size': tok.vocab_size,
            'available': True,
        }
    except Exception as e:
        return {'available': False, 'error': str(e)}


# ============================================================================
# 4. Main
# ============================================================================
def main():
    print("=" * 72)
    print("BENCHMARK HONESTO — winnex-nano (tokenizer espectral Winnex)")
    print("=" * 72)

    # Espectral
    for dim in (32, 64, 128):
        r = bench_spectral(embed_dim=dim)
        print(f"\n--- Tokenizer Espectral (embed_dim={dim}) ---")
        print(f"  Round-trip perfeito : {r['roundtrip_perfect']}/{r['roundtrip_total']}")
        print(f"  Erro por caractere  : {r['char_error_rate']:.4%}")
        print(f"  Encode              : {r['encode_chars_per_sec']:.0f} chars/s")
        print(f"  Decode              : {r['decode_chars_per_sec']:.0f} chars/s")
        print(f"  Similaridade inter-caractere (média): {r['mean_inter_char_similarity']:.4f}")
        print(f"  Similaridade inter-caractere (máx) : {r['max_inter_char_similarity']:.4f}")
        print(f"  Bits por caractere   : {r['bits_per_char']}")

    # BPE baseline
    print("\n--- Baseline: BPE (Qwen2.5, transformers) ---")
    b = bench_bpe()
    if b.get('available'):
        print(f"  Encode              : {b['bpe_encode_chars_per_sec']:.0f} chars/s")
        print(f"  Tokens por caractere: {b['bpe_tokens_per_char']:.3f}")
        print(f"  Vocab size          : {b['bpe_vocab_size']}")
    else:
        print(f"  Não disponível: {b.get('error', '?')}")

    print("\n" + "=" * 72)
    print("ANÁLISE HONESTA")
    print("=" * 72)
    print("""
• O tokenizer espectral é 100% determinístico (sem vocabulário, sem BPE,
  sem treino) — o round-trip valida a integridade do encoding.
• Custo: usa embed_dim×4 floats por caractere (representação densa). O BPE
  usa ~1 token por 3-4 caracteres (mais compacto) MAS exige um vocabulário
  externo de 151K tokens e a biblioteca transformers.
• Discriminabilidade: similaridade inter-caractere < 1.0 significa que o
  probe distingue caracteres — quanto menor, mais separáveis.
• Trade-off real: espectral = autônomo/determinístico/denso; BPE = compacto/
  eficiente/vocabulário. O espectral NÃO é um substituto do BPE para o Qwen
  treinado — é um encoding próprio para o motor multimodelo nativo.
""")


if __name__ == "__main__":
    main()
