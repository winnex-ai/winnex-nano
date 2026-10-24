# winnex-nano

**Native Winnex inference engine — deterministic spectral tokenizer, multimodel weight balancing, and chunk generation via the Madhava fast build.**

Model-agnostic (Qwen / BERT / DeepSeek / GPT), C++20/OpenCL, no CUDA. Reuses the
[winnex-madhava](https://pypi.org/project/winnex-madhava/) engine kernels (QKᵀ matmul,
selective-top-K with Cauchy-Schwarz bounds) — no code duplication.

## Components

| Module | Purpose |
|--------|---------|
| `SpectralTokenizer` | Character → quaternion spectrum (deterministic, no BPE, no vocabulary) |
| `WeightBalancer` | Multimodel fusion: `W' = Σᵢ αᵢ·R(qᵢ)·Wᵢ` |
| `ChunkEngine` | Geração por chunks com memória de trabalho Madhava — **não** faz stream de pesos |
| `server_sse` | OpenAI-compatible `/v1/chat/completions` (stream + non-stream) |

## Installation

```bash
pip install winnex-nano
```

Requires `winnex-madhava>=1.8.1` (the engine), `numpy`. Optional: `winnex-nano[server]`
for the SSE server (`fastapi`, `uvicorn`, `httpx`).

## Quick start

```python
import winnex_nano as wn

# 1. Deterministic spectral tokenizer (no BPE, no vocabulary)
tok = wn.SpectralTokenizer(embed_dim=64)
states = tok.encode("Winnex AI")       # list of quaternions
text = tok.decode(states)              # round-trip: "Winnex AI"

# 2. Multimodel weight balancing: W' = Σ αᵢ·R(qᵢ)·Wᵢ
bal = wn.WeightBalancer()
model_a = {"layer.0.weight": [1, 2, 3, 4]}
model_b = {"layer.0.weight": [10, 20, 30, 40]}
a = wn.BlendWeight(); a.alpha = 0.5
b = wn.BlendWeight(); b.alpha = 0.5
fused = bal.blend([model_a, model_b], [a, b])   # [5.5, 11, 16.5, 22]

# 3. ChunkEngine — geração por chunks com memória de trabalho Madhava
#
# ATENÇÃO AO ESCOPO: isto NÃO faz streaming de pesos. A decisão do próximo
# chunk é uma função que você fornece; a classe não carrega modelo nenhum.
# Para inferência de verdade, use ForwardEngine (prefill/step), abaixo.
engine = wn.ChunkEngine()
chunks = []
engine.generate_chunks(
    "O Madhava e deterministico.",
    lambda ctx, top: " chunk" if len(chunks) < 3 else "",
    lambda c: chunks.append(c.text) if not c.done else None,
)
```

> **Sobre o nome.** Esta classe se chamava `StreamEngine` até a 0.1.33. O nome
> estava errado: em inferência, "stream" quer dizer carregar pesos em fluxo para
> rodar um modelo maior que a memória, e esta classe nunca fez isso. O alias
> `StreamEngine` e o método `.stream()` continuam funcionando, mas estão
> depreciados e somem na 0.2. O modo de pesos que faz streaming real por camada
> é `ForwardEngine::weight_mode() == "stream"` — e ele é **mais lento que a CPU**
> quando o modelo não cabe no device (medido: 90,8 s contra 67,0 s no 7B).

## Server (OpenAI-compatible, real inference)

```bash
pip install "winnex-nano[server]"
python -m winnex_nano.server_sse --model /path/to/checkpoint --port 30002
```

```bash
curl -N http://localhost:30002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"The capital of France is"}],
       "max_tokens":32,"stream":true}'
```

It runs the native engine end to end: tokenizer → `prefill` → sample → `step` →
detokenize, on the KV cache, with greedy or temperature/top-k sampling, stop
sequences, a real chat template when the tokenizer has one, and SSE in OpenAI's
chunk format. `/health` reports the device and the weight mode that were
actually selected.

```python
eng.prefill([...ids])   # -> logits for the last prompt token; fills the KV cache
eng.step(next_id)       # -> logits for the next token; appends to the cache
eng.embed_ids([...ids]) # -> [n, hidden] rows, without touching the whole table
```

`prefill`+`step` is bit-identical to a full `forward` at every position (max
difference 0.0 over 85 tokens) and matches `transformers` to 1.2e-4.

**Throughput, honestly.** Qwen3-4B on an RTX 5060 Ti in `device` mode: prefill
40.6 tok/s, decode 2.6 tok/s. Decode is the slow one because attention, RoPE,
RMSNorm and the residual stream stay on the CPU by design — at `seq_len=1` the
per-projection device round-trip dominates. Moving the whole layer to OpenCL is
the next real gain and is not done.

> **Changed in 0.1.28.** Previous versions of this server answered every
> request with 12-character slices of the prompt. It was commented as a
> placeholder, but at the HTTP layer it was indistinguishable from a real
> completion. If you are running an older version, it is not inferring.

## Honest benchmark (vs BPE)

Run the full benchmark on Kaggle — installs `winnex-nano` from PyPI and
reports round-trip, throughput and O(1) decode scaling on real multilingual
text:

[![Kaggle](https://img.shields.io/badge/Kaggle-Spectral%20Tokenizer%20benchmark-20BEFF?logo=kaggle)](https://www.kaggle.com/code/kleniopadilha/winnex-nano-benchmark-spectral)

| Metric | winnex-nano spectral | BPE (Qwen2.5) |
|--------|---------------------|---------------|
| Round-trip perfect | 8/8 | n/a |
| Char error rate | 0.00% | n/a |
| Encode throughput | 234K chars/s | 4.3M chars/s |
| Stream (3 chunks) | 1.7 ms | ~2.1 s (HTTP) |
| Vocabulary | none | 151K tokens |

The spectral tokenizer is 100% deterministic with no vocabulary dependency, but
50–70× slower and ~1000× less compact than BPE. It is an autonomous encoding for
the native multimodel engine, NOT a drop-in BPE replacement for a BPE-trained model.

## Native forward pass — memory and speed

`ForwardEngine` runs a Qwen2/Qwen2.5/Qwen3/Llama checkpoint without torch, without
transformers and without CUDA. Weights come straight out of the `.safetensors`
file; the heavy matrix work goes to OpenCL.

```python
import struct, numpy as np, winnex_nano as wn

# Map the payload instead of copying it. The 7B payload is 14.2 GB and the
# constructor copies it; view() borrows it and lets the kernel page it in.
with open(P, "rb") as f:
    hlen = struct.unpack("<Q", f.read(8))[0]
    header = f.read(hlen).decode()
st = wn.Safetensors.view(
    np.memmap(P, dtype=np.uint8, mode="r", offset=8 + hlen), header)

eng = wn.ForwardEngine(wn.load_config(cfg_path), st)
eng.enable_gpu()            # decides the weight strategy, see below
print(eng.weight_mode)      # "device" | "resident" | "stream" | "cpu"
```

`enable_gpu()` picks one of three strategies, in this order:

| Mode | What lives where | When |
|------|------------------|------|
| `device` | every projection on the GPU (bf16 when f32 does not fit); host keeps only norms and biases | the model fits in VRAM |
| `stream` | one layer at a time through a scratch buffer | half the model fits the device but the whole does not |
| `cpu` | nothing on the device | the model does not fit even at bf16 |

> **`resident` is not reachable through `enable_gpu()`.** It is computed only
> when no device was created — and `enable_gpu()` creates the device first, and
> throws if it cannot. The mode remains in `weight_mode()`'s vocabulary and in
> the code that selects it, but there is no public call that reaches it without
> a GPU. Treat the table above as the actual contract; `resident` is dead code
> reachable only by editing the source.

Measured on an RTX 5060 Ti (16 GB) and a 31 GB host, Qwen2.5-7B, 131 tokens:

| Path | PPL | Time | RSS | Anonymous heap |
|------|-----|------|-----|----------------|
| CPU | 15.55172 | 69.8 s | 14.2 GB | 0.69 GB |
| GPU `device` | 15.55171 | 7.5 s | 14.4 GB | 0.74 GB |

The perplexity difference is 7e-6 relative — the checkpoint is already bf16, so
the GPU path's bf16 weights are the same numbers the CPU reads after widening
them to f32. Nothing is lost that was ever in the file.

RSS stays high in both rows because the payload is *mapped*: those pages are the
file, evictable under pressure, not heap. The number that decides whether a model
fits a machine is the anonymous column.

## Attention: exact softmax, and the Madhava mode

The default is **exact causal softmax** — the function the pretrained weights
encode. It is not negotiable: replacing it with a top-k filter that does not
renormalise was tried in this codebase, and the cost is recorded in `run_layer`:
perplexity 2.0e7 against 37.4, correlation ≈ 0 with the reference.

`MADHAVA_TOPK_ATTN=1` enables the **selective mode**: keys are pruned while the
mass they account for stays within `MADHAVA_EPSILON_MASS` (default 1e-4) of the
full softmax mass. The dropped set is renormalised, so the weights still form a
distribution. The guarantee is exact, not a score bound:
`‖attn_full − attn_pruned‖₁ ≤ 2ε`.

Three facts measured on 2026-10-23, Qwen2.5-0.5B, 80 LAMBADA items:

| Arm | LAMBADA | prefill tok/s |
|---|---:|---:|
| CPU, softmax | 35.0% | 50.83 |
| CPU, `MADHAVA_TOPK_ATTN=1`, ε=1e-4 | 35.0% | 50.54 |
| CPU, `MADHAVA_TOPK_ATTN=1`, ε=0.05 | 36.2% | 50.53 |

- **Quality is preserved** — all three inside the confidence interval
  (the guarantee holds).
- **It does not accelerate.** Computing every score is O(n·d), the same work as
  full softmax; what is saved is the value accumulation, which only dominates
  when the context is long. At 97 tokens the difference is noise.
- **It is CPU-only.** The selective block lives inside the CPU branch of
  `run_layer`. With attention on the GPU the variable has **no effect** —
  verified: identical LAMBADA and prefill. And the CPU attention arm is 7.4×
  slower than the GPU one (50.8 vs 376 tok/s), so reaching the filter costs far
  more than it can save at short context.

Honest use: an **auditable discard guarantee**, not a speed lever. Tested at
512 → 8192 tokens, 25 combinations of epsilon (1e-4 to 0.2) x context length,
on the CPU arm where the filter actually exists:

```
softmax exact:  512 54.3 tok/s | 1024 50.8 | 2048 43.4 | 4096 32.8 | 8192 21.8
selective:      time ratio min 0.997, max 1.015, mean 1.003
                logit divergence: max 0.00e+00
```

The time does not fall at any length, for any epsilon, and the logits are
bit-identical — at these sizes the filter discards less than float noise. The
code comment claimed the saving lives in the value accumulation "which dominates
only when n is large"; **that hypothesis is falsified up to 8k tokens**. The
score computation is the bottleneck and the filter does not avoid it. Porting it
to the device is therefore not justified by these numbers.

## Supported architectures

Dense only: **qwen2, qwen3, llama, mistral**. `load_config` refuses everything
else by name, because a model whose maths differs loads cleanly and produces a
plausible-looking perplexity that is wrong:

| Refused | Why |
|---|---|
| `qwen2_moe`, `qwen3_moe`, `mixtral` | need a top-k router over N experts; not implemented |
| `deepseek_v2`, `deepseek_v3` | need MLA (latent attention); not implemented |
| `gemma`, `gemma2` | GeGLU, logit softcapping, sliding window |

This list is a contract, not an aspiration. It previously named four MoE
architectures the engine had no code for — the failure surfaced as "tensor not
found" rather than "architecture unsupported", with the list telling the user
the model was supported.

### Why it is built this way

- **`view` instead of a copy.** The constructor reads the whole payload into a
  `std::vector`. For the 7B that is 14.2 GB before a single weight is used, and
  it is what made the Kaggle run die with "Kernel died". `view()` follows the
  MadhavaL2 contract: the caller owns the pointer, the engine never copies.
- **Weights keyed by tensor name, not by host pointer.** In `stream` mode
  `run_layer` builds a fresh `LayerWeights` per call, so every matrix gets a new
  address and the device cache never hits — the GPU then re-uploads 14.2 GB per
  forward and measures *slower* than the CPU (90.8 s against 67.0 s). Naming the
  weights (`L3.gate_proj`) lets the device keep them for the life of the engine
  regardless of what the host allocator does.
- **bf16 weights when f32 does not fit.** The 7B in f32 is 26.1 GB against a
  16 GB card. The dtype is decided once for the whole model from the declared
  total; deciding per matrix mixes formats on the device and corrupts every
  launch after the first switch.
- **Tiled GEMM.** The original kernel was one work-item per output element with
  no blocking — arithmetic intensity 1 FLOP/byte against a machine balance of
  ~26, i.e. the kernel waited on its own weight reads. 16x16 local-memory tiles
  cut global traffic per MAC by ~8x.
- **lm_head streamed.** The embedding table is the largest single tensor
  (152064x3584). Both of its uses — one row per decoded token, and the logits
  projection — go through the offset table in 512-row blocks, so it costs 7.3 MB
  instead of 2.18 GB.

### Splitting a model into shards

`ForwardEngine` takes one `Safetensors`. Checkpoints that arrive as
`model-00001-of-0000N.safetensors` must be merged into a single file first —
concatenating the payloads byte-for-byte, without decoding, preserves bf16 and
avoids the 30 GB a float32 merge would need.

## License

Business Source License 1.1 (BSL 1.1) — `pay@winnex.ai`.

Pre-patent: https://zenodo.org/records/21861809
