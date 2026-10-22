# winnex-nano

**Native Winnex inference engine — deterministic spectral tokenizer, multimodel weight balancing, and chunk streaming via the Madhava fast build.**

Model-agnostic (Qwen / BERT / DeepSeek / GPT), C++20/OpenCL, no CUDA. Reuses the
[winnex-madhava](https://pypi.org/project/winnex-madhava/) engine kernels (QKᵀ matmul,
selective-top-K with Cauchy-Schwarz bounds) — no code duplication.

## Components

| Module | Purpose |
|--------|---------|
| `SpectralTokenizer` | Character → quaternion spectrum (deterministic, no BPE, no vocabulary) |
| `WeightBalancer` | Multimodel fusion: `W' = Σᵢ αᵢ·R(qᵢ)·Wᵢ` |
| `StreamEngine` | Chunk generation via the Madhava fast build (working memory) |
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

# 3. Streaming via the Madhava fast build
engine = wn.StreamEngine()
chunks = []
engine.stream(
    "O Madhava e deterministico.",
    lambda ctx, top: " chunk" if len(chunks) < 3 else "",
    lambda c: chunks.append(c.text) if not c.done else None,
)
```

## SSE server (OpenAI-compatible)

```bash
pip install "winnex-nano[server]"
python -m winnex_nano.server_sse 30002
```

```bash
curl -N http://localhost:30002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"O Madhava e deterministico"}],"stream":true}'
```

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
| `resident` | every projection in host RAM, stable pointers | fits in 60% of available RAM, no usable GPU |
| `stream` | one layer at a time through a scratch buffer | otherwise |

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
