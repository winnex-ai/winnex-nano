#!/usr/bin/env python3
"""
benchmark_spectral.py — benchmark da camada Python do winnex-nano.

Segue o princípio: benchmarks na camada SUPERIOR (Python), onde o cliente
real consome a API. Mede:

  [1] CORREÇÃO   — round-trip em múltiplos idiomas e tamanhos (CER)
  [2] PERFORMANCE— throughput (chars/s) e latência para vários tamanhos
  [3] QUALIDADE  — compressão (bits/char) vs baselines (BPE 4.6, char 8.0)
  [4] ESCALABILIDADE — decode O(1) até 10M bytes

Os resultados são salvos em JSON para comparação com baselines futuros.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import winnex_nano as wn


# ---------------------------------------------------------------------------
# [1] CORREÇÃO — round-trip multiidioma
# ---------------------------------------------------------------------------
def benchmark_roundtrip(tok, corpora):
    """Round-trip em múltiplos idiomas e tamanhos."""
    results = []
    for name, text in corpora.items():
        states = tok.encode(text)
        decoded = tok.decode(states)
        match = decoded == text
        n = max(len(text), 1)
        char_error_rate = sum(a != b for a, b in zip(text, decoded)) / n
        results.append({
            "corpus": name,
            "chars": len(text),
            "match": match,
            "cer": round(char_error_rate, 6),
        })
    return results


CORPORA = {
    "ascii_short": "Hello, World!",
    "ascii_long": "The quick brown fox jumps over the lazy dog. " * 50,
    "pt_br": "São Paulo é a maior cidade do Brasil. Ação, coração, educação.",
    "pt_br_long": "A educação é a arma mais poderosa para mudar o mundo. " * 50,
    "japanese": "東京は日本の首都です。",
    "russian": "Москва — столица России.",
    "arabic": "القاهرة هي عاصمة مصر.",
    "emoji": "Hello 🌍🎉🚀",
    "code": "def hello():\n    print('world')\n",
    "binary": bytes(range(256)).decode("latin-1"),
}


# ---------------------------------------------------------------------------
# [2] PERFORMANCE — throughput e latência
# ---------------------------------------------------------------------------
def benchmark_throughput(tok, text_sizes=(100, 1_000, 10_000, 100_000)):
    """Throughput (chars/s) e latência para diferentes tamanhos."""
    results = []
    base_text = "São Paulo é a maior cidade do Brasil. " * 1000
    for size in text_sizes:
        text = base_text[:size]
        if not text:
            continue

        # Encode
        t0 = time.perf_counter()
        states = tok.encode(text)
        t1 = time.perf_counter()
        encode_s = t1 - t0

        # Decode (analítico O(1))
        t0 = time.perf_counter()
        decoded = tok.decode(states)
        t1 = time.perf_counter()
        decode_s = t1 - t0

        results.append({
            "chars": size,
            "encode_ms": round(encode_s * 1000, 3),
            "decode_ms": round(decode_s * 1000, 3),
            "encode_chars_per_sec": round(size / encode_s, 1) if encode_s > 0 else None,
            "decode_chars_per_sec": round(size / decode_s, 1) if decode_s > 0 else None,
            "roundtrip_ok": decoded == text,
        })
    return results


# ---------------------------------------------------------------------------
# [3] QUALIDADE — compressão (bits/char)
# ---------------------------------------------------------------------------
def benchmark_compression(tok, text):
    """Quantos bits por caractere o tokenizador usa?"""
    states = tok.encode(text)
    n_chars = max(len(text), 1)
    n_quats = len(states)
    # Cada estado quaternion = 4 floats32 = 4×32 bits
    spectral_bits = (n_quats * 4 * 32) / n_chars
    return {
        "spectral_bits_per_char": round(spectral_bits, 1),
        "bpe_bits_per_char_ref": 4.6,
        "char_bits_per_char_ref": 8.0,
        "bytes_per_char_quats": round(n_quats * 4 / n_chars, 2),
    }


# ---------------------------------------------------------------------------
# [4] ESCALABILIDADE — decode O(1) até 10M
# ---------------------------------------------------------------------------
def benchmark_scaling(tok, sizes=(250_000, 1_000_000)):
    """Escalabilidade do encode batch e decode analítico.

    O encode usa encode_batch (numpy direto, sem objetos Quat Python) e o
    decode é o analítico O(1)/char. 1M chars → array (1M, 64, 4) float32
    (1GB) — o tamanho da memória é o limite prático na camada Python.
    """
    results = []
    base = ("São Paulo é a maior cidade do Brasil. Ação e coração. " * 2000)
    for size in sizes:
        data = (base * (size // len(base) + 1))[:size].encode()

        t0 = time.perf_counter()
        arr = tok.encode_batch(data)
        t1 = time.perf_counter()
        encode_s = t1 - t0

        t0 = time.perf_counter()
        decoded = tok.decode_array(arr)
        t1 = time.perf_counter()
        decode_s = t1 - t0

        results.append({
            "chars": size,
            "encode_ms": round(encode_s * 1000, 2),
            "decode_ms": round(decode_s * 1000, 3),
            "encode_chars_per_sec": round(size / encode_s, 1),
            "decode_chars_per_sec": round(size / decode_s, 1),
            "ns_per_char_decode": round(decode_s * 1e9 / size, 1),
            "roundtrip_ok": decoded == data.decode(),
        })
    return results


def main() -> None:
    tok = wn.SpectralTokenizer(embed_dim=64)
    report = {
        "package_version": getattr(wn, "__version__", "?"),
        "embed_dim": tok.embed_dim,
    }

    print("=" * 64)
    print("  WINNEX-NANO — BENCHMARK ESPECTRAL (camada Python)")
    print(f"  versão: {report['package_version']} | embed_dim={report['embed_dim']}")
    print("=" * 64)

    # [1] CORREÇÃO
    print("\n[1] CORREÇÃO (round-trip multiidioma)")
    rt = benchmark_roundtrip(tok, CORPORA)
    n_ok = sum(1 for r in rt if r["match"])
    for r in rt:
        st = "✅" if r["match"] else "❌"
        print(f"  {st} {r['corpus']:<14} {r['chars']:>6} chars  CER={r['cer']:.5f}")
    print(f"  → {n_ok}/{len(rt)} corpora ok")
    report["roundtrip"] = {"ok": n_ok, "total": len(rt), "details": rt}

    # [2] PERFORMANCE
    print("\n[2] PERFORMANCE (throughput e latência)")
    tp = benchmark_throughput(tok)
    print("  {:<8} {:<10} {:<10} {:<14} {:<14}".format(
        "chars", "encode_ms", "decode_ms", "enc_chars/s", "dec_chars/s"))
    for r in tp:
        print("  {:<8} {:<10} {:<10} {:<14} {:<14}".format(
            r["chars"], r["encode_ms"], r["decode_ms"],
            r["encode_chars_per_sec"], r["decode_chars_per_sec"]))
    report["throughput"] = tp

    # [3] QUALIDADE (compressão)
    print("\n[3] QUALIDADE (bits/char)")
    comp_text = "São Paulo é a maior cidade do Brasil. " * 200
    comp = benchmark_compression(tok, comp_text)
    for k, v in comp.items():
        print(f"  {k:<24} {v}")
    report["compression"] = comp

    # [4] ESCALABILIDADE
    print("\n[4] ESCALABILIDADE (encode vs decode)")
    sc = benchmark_scaling(tok)
    print("  {:<10} {:<12} {:<12} {:<14} {:<14} {}".format(
        "chars", "encode_ms", "decode_ms", "enc_chars/s", "dec_chars/s", "rt"))
    for r in sc:
        st = "✅" if r["roundtrip_ok"] else "❌"
        print("  {:<10} {:<12} {:<12} {:<14,} {:<14,} {}".format(
            r["chars"], r["encode_ms"], r["decode_ms"],
            r["encode_chars_per_sec"], r["decode_chars_per_sec"], st))
    report["scaling"] = sc

    # Salvar JSON
    out = Path(__file__).with_suffix(".results.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nResultados salvos em {out}")

    ok = all(r["match"] for r in rt) and all(r["roundtrip_ok"] for r in sc)
    print("\n" + ("=== TODOS OS BENCHMARKS PASSARAM ===" if ok else "=== HÁ FALHAS ==="))


if __name__ == "__main__":
    main()
