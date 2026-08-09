# Análise Profunda dos Gargalos — winnex-nano

**Data:** 2026-08-09
**Método:** profiling real (medição direta, sem cherry-picking)

## Medições reais (embed_dim=64, ~800 chars)

| Operação | Custo | Throughput | Ratio |
|----------|-------|-----------|-------|
| **Encode** (texto→espectro) | 7.2ms / 800 chars | 111K chars/s | 1× |
| **Decode** (espectro→texto) | 44.0ms → **10.1ms** | 18K → **79K chars/s** | **4.3× otimizado** |
| **Stream** (10 chunks) | 3.6ms | 0.36ms/chunk | eficiente |
| **Stream** (200 chunks) | 19ms | 0.09ms/chunk | escala bem |

## 🔴 Gargalo #1 — Decode O(n×95×4×d) — o principal

**Custo:** para cada caractere, compara com 95 candidatos ASCII, cada um com d modos × 4 componentes quaternion. Em d=128: **~48K operações/caractere**.

**Sub-gargalos identificados no código (`spectral_tokenizer.cpp`):**
1. **`n1` (norma do estado) recomputada 95×** por caractere — é constante (o mesmo estado), mas recalculada para cada candidato. **Correção:** pré-computar `n1` uma vez por caractere.
2. **`ref_state` recomputa `sin`/`cos` a cada posição** — 95 caracteres × d modos × 4 sin/cos por posição. **Correção:** pré-computar as referências pos-independentes (a parte `(ascii+j)` da fase) e aplicar o shift de posição como rotação de fase (o shift `pos·2π/256` é um fator constante no domínio complexo).

**Complexidade real:** O(n × 95 × d × 4) para o decode completo. Para 1000 chars, d=64: ~24M operações. É o gargalo que explica o ratio 6×.

**Correção APLICADA (validada):** pré-computar as 95 referências pos-independentes (1×), pré-computar a norma do estado (1× por caractere), e aplicar o shift de posição como **rotação de fase constante** (multiplicação complexa) em vez de recomputar sin/cos. **Resultado: decode 44ms → 10.1ms (4.3×), 18K → 79K chars/s.** A corretude (round-trip) é preservada — todos os testes passam.

## 🟠 Gargalo #2 — Memória da representação espectral

Cada caractere → `d` quaternions × 4 floats × 4 bytes = **d×16 bytes**. Em d=128: **2048 bytes/caractere** vs BPE ~0.575 bytes. Ratio **~3560×**.

**Impacto:** o corpus do stream (memória de trabalho) cresce rápido. 1000 chars = 2MB só de representação.

**Correção parcial:** o `chunk_to_vector` reduz o chunk a `d` floats (não d×4×n_chars) antes de indexar — então o **corpus indexado** é compacto (d floats/chunk). A representação densa só é retida durante encode/decode.

## 🟢 Gargalo #3 — Stream (NÃO é gargalo — validado)

O stream usa o **build rápido do Madhava** que escala bem: 200 chunks = 19ms (0.09ms/chunk). O rebuild incremental é eficiente. **Sem correção necessária.**

## 🟠 Gargalo #4 — WeightBalancer sem SIMD no blend

O blend é O(tensor) com OpenMP na rotação (corrigido), mas o **loop principal do blend** (a soma `Σ αᵢ·Wᵢ`) não tem SIMD/OpenMP explícito. Para bilhões de parâmetros, o blend serial seria lento.

**Correção:** OpenMP no loop do blend (como feito na rotação).

## Correções prioritárias (o que fazer agora)

| # | Gargalo | Correção | Impacto estimado |
|---|---------|----------|------------------|
| 1a | `n1` recomputado 95× | Pré-computar norma | ~95× menos sqrt |
| 1b | `sin`/`cos` por posição | Pré-computar refs + shift de fase | ~95× menos trig |
| 2 | Memória densa | Já mitigado no corpus (chunk_to_vector) | — |
| 4 | Blend serial | OpenMP no loop | ~N cores × |

## Limitações honestas (não corrigíveis agora)

- **Memória por caractere** é inerente à representação quaterniónica densa (trade-off documentado)
- **Sem testes em modelos reais** (MMLU/HellaSwag) — requer o forward pass do LLM (próximo passo)
- **Sem benchmark de tasks padrão** — idem
