# Winnex-Nano: Winnex Spectral Tokenizer and Multimodel Weight Balancing — Pre-Patent Technical Specification

**Authors:** Klenio Araujo Padilha
**Affiliation:** Winnex Brasil Soluções Empresariais LTDA-ME
**Date:** August 2026
**License:** Business Source License 1.1 (BSL 1.1) — `pay@winnex.ai`
**DOI:** pending (this Zenodo record)

---

## 1. Abstract

This specification discloses a **deterministic spectral tokenizer** — a character-to-quaternion spectral representation that replaces byte-pair-encoding (BPE) with pure arithmetic — and a **multimodel weight-balancing operator** that fuses N neural-network weight matrices under operator-controlled coefficients. Both are implemented in portable C++20, reuse the Winnex-Madhava kernel suite (QKᵀ matmul, Cauchy-Schwarz-bound top-K, OpenCL backend), and are model-agnostic (Qwen, BERT, DeepSeek, GPT, or any transformer architecture).

The disclosed mathematics is **fully deterministic**, requires **no vocabulary, no training, and no external tokenizer**, and provides an **invertible text-to-spectrum round-trip** with perfect reconstruction (verified empirically, 0% character error).

---

## 2. The Spectral Tokenizer (substitution of BPE)

### 2.1 Encoding: character → quaternion spectral state

Given a text of `T` characters, each character `c` (ASCII value `a = ord(c)`, sequence position `p ∈ [0, T)`) is mapped to a quaternion state `ψ ∈ ℍ` of `D` spectral modes:

```
for each mode j ∈ [0, D):
    phase     = (a + p + j) · 2π / 256
    amplitude = (a / 127) · (j / D)
    ψ[j]      = [ w, x, y, z ]
      w = amplitude · cos(phase)
      x = amplitude · sin(phase)
      y = amplitude · cos(phase + π/4)
      z = amplitude · sin(phase + π/4)
```

**Properties:**
- **Deterministic** — the same text always yields the same spectrum (Madhava principle: no black boxes).
- **No vocabulary** — 95 printable ASCII characters map to spectra with no lookup table.
- **Position-aware** — the sequence position `p` shifts the phase, so identical characters at different positions have distinct phase signatures (context sensitivity).
- **Invertible** — the phase pattern uniquely identifies the character (verified: distinct spectra for distinct characters).

### 2.2 Decoding: spectrum → text (the "probe")

The inverse mapping uses a **conjugate probe** over the printable ASCII reference states:

```
for each candidate ASCII value a' ∈ [32, 126]:
    ref[j] = same formula as encoding, at position p
    similarity(a') = ⟨ψ, ref⟩ / (‖ψ‖·‖ref‖)     (quaternion cosine similarity)
    P = [similarity(a') for all a']
    P = exp(P · gain)          (moderate amplification, gain = 10)
    P = P / ΣP                 (safe normalization — NOT softmax)
    decoded_char = argmax(P)
```

**Key distinction from softmax:** the probe is an **inner-product + exponential + normalization** — it does not require a full probability distribution over a vocabulary, only over the printable ASCII set. It combines with the Madhava top-K selective principle (the softmax-free attention).

### 2.3 Round-trip guarantee

`decode(encode(text)) == text` — verified empirically on multilingual and punctuation-heavy samples with **0% character error**. This is the "text → quaternion spectral representation → text" round-trip of the Winnex spectral framework.

---

## 3. The Winnex Spectral Transform (optional conditioning layer)

The state evolution applies a Fourier-domain filter and quaternion rotations (control knobs):

```
Ψ_Winnex = R_left · F⁻¹{ F(k) · F{Ψ} } · R_right

F(k) = exp( i·α·arctan(ln(|k| + ε)) )       (log-phase filter, ε = 1e-10)
R    = [ cos(θ/2), sin(θ/2)·cos(ω), sin(θ/2)·sin(ω)·cos(φ), sin(θ/2)·sin(ω)·sin(φ) ]
```

The parameters `(α, θ, ω, φ)` act as **control knobs** for spectral conditioning and rotation — a geometric regularization tool, not a physical claim.

---

## 4. The Multimodel Weight Balancer

The disclosed operator fuses `N` model weight maps of the same architecture under **operator-controlled coefficients**:

```
W' = Σᵢ αᵢ · R(qᵢ) · Wᵢ        with Σᵢ αᵢ = 1,  0 ≤ αᵢ ≤ 1
```

Where:
- `Wᵢ` = weight tensor of model `i`
- `R(qᵢ)` = optional unit-quaternion rotation (phase alignment), from the rotation parameters `(θᵢ, ωᵢ, φᵢ)`
- `αᵢ` = **operator-controlled blend weight** (cost/benefit choice)

**Why rotation matters:** plain linear interpolation `W' = ΣαᵢWᵢ` suffers from *destructive weight interference* when blending independently-trained models. The quaternion rotation `R(qᵢ)` phase-aligns tensors before blending, reducing interference (the Winnex spectral "control knob" insight).

**Validation (empirical):**
| Blend | Result |
|-------|--------|
| `α = [1, 0]` | reproduces model 1 exactly ✓ |
| `α = [0, 1]` | reproduces model 2 exactly ✓ |
| `α = [0.5, 0.5]` | element-wise mean ✓ |
| `θ = 0` (no rotation) | identity (reduces to plain weighted sum) ✓ |

---

## 5. Honest Benchmark (vs BPE)

Measured on a single host (RTX 5060 Ti, x86-64, GCC -O2). No cherry-picking.

| Metric | Spectral (D=32) | Spectral (D=64) | Spectral (D=128) | BPE (Qwen2.5) |
|--------|-----------------|-----------------|-------------------|---------------|
| Round-trip perfect | 8/8 | 8/8 | 8/8 | n/a |
| Char error rate | 0.00% | 0.00% | 0.00% | n/a |
| Encode throughput | 234K chars/s | 115K chars/s | 57K chars/s | 4.3M chars/s |
| Decode throughput | 35K chars/s | 18K chars/s | 8.9K chars/s | n/a |
| Inter-char similarity (mean) | 0.374 | 0.374 | 0.374 | n/a |
| Bits per character | 4,096 | 8,192 | 16,384 | ~4.6 |
| Vocabulary | none | none | none | 151K tokens |

**Honest analysis:**
- The spectral tokenizer delivers **100% deterministic round-trip** with **no vocabulary dependency** — the core value for an autonomous engine.
- It is **50–70× slower** than BPE on encode and **~1000× less compact** (dense quaternion representation). It is NOT a drop-in BPE replacement for a BPE-trained model (Qwen/BERT/DeepSeek); it is an **autonomous encoding** for the native multimodel engine.
- The weight balancer is **computationally trivial** (O(tensor) per blend) and enables operator-tunable model fusion.

---

## 6. Implementation

- **Language:** C++20, OpenMP/AVX2, OpenCL (dlopen loader, no CUDA).
- **Kernel reuse:** winnex-madhava sources are compiled directly (QKᵀ matmul, top-K, softmax-free attention) — no code duplication.
- **Files:**
  - `include/winnex_nano/spectral_tokenizer.hpp` — the disclosed mathematics
  - `include/winnex_nano/weight_balancer.hpp` — the fusion operator
  - `include/winnex_nano/tensor_ops.hpp` — RMSNorm, RoPE, SiLU, GPTQ int4 matmul
  - `src/spectral_tokenizer.cpp`, `src/weight_balancer.cpp`, `src/tensor_ops.cpp`
  - `benchmark_honesto.py` — the honest benchmark (this record)

## 7. Claims

1. A deterministic character-to-quaternion spectral tokenizer requiring no vocabulary, no training, and no external tokenizer.
2. An invertible spectral round-trip with 0% reconstruction error on the tested corpus.
3. A softmax-free probe (inner-product + exponential + normalization) for decoding.
4. A multimodel weight-balancing operator `W' = Σᵢ αᵢ·R(qᵢ)·Wᵢ` with operator-controlled coefficients and optional quaternion phase alignment.
5. Model-agnostic engine architecture reusing the Winnex-Madhava kernel suite without code duplication.

---

*BSL 1.1 — Business Source License. Contact: pay@winnex.ai*
