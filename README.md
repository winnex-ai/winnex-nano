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

## License

Business Source License 1.1 (BSL 1.1) — `pay@winnex.ai`.

Pre-patent: https://zenodo.org/records/21861809
