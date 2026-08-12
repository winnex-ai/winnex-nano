"""
winnex-nano — native Winnex inference engine.

Deterministic spectral tokenizer, X-factor manifold embedding, native dense
forward pass (no GPTQ, no CUDA, no SGLang, no transformers), multimodel weight
balancing, and chunk streaming via the Madhava fast build. Model-agnostic
(Qwen/BERT/DeepSeek/GPT), C++/OpenCL/AVX2, no CUDA. Reuses the winnex-madhava
bound-engine kernels.

Components:
    - SpectralTokenizer: character → quaternion spectrum (deterministic, no BPE)
    - XFactor:           manifold basis of a model's embedding space (D×r) —
                         exposed via compute_xfactor (projector P, rank, var)
    - Safetensors:       native .safetensors loader (no torch/transformers)
    - ForwardEngine:     native dense f32 forward pass (RMSNorm, QKV, RoPE,
                         GQA attention, SiLU MLP, lm_head)
    - sample_embeddings: read N rows of a tensor without materializing it
    - WeightBalancer:    W' = Σᵢ αᵢ·R(qᵢ)·Wᵢ  (multimodel fusion)
    - StreamEngine:      chunk generation via Madhava fast build (SSE)

The inference path (X-Factor + forward, no BPE):
    text → SpectralTokenizer.encode_histogram → expand_spectral(D)
        → XFactor.project (P·ψ̃) → ForwardEngine.forward → logits → argmax

Requires: winnex-madhava (engine), numpy.
"""

from ._winnex_nano import (
    Quat,
    SpectralTokenizer,
    WeightBalancer,
    BlendWeight,
    StreamEngine,
    StreamOptions,
    StreamChunk,
    Safetensors,
    ModelConfig,
    ForwardEngine,
    XFactor,
    compute_xfactor,
    sample_embeddings,
    expand_spectral,
    dense_matmul,
    rms_norm,
    load_config,
)

# Mathematical model fusion / conversion (replaces the refuted W·P projection).
from . import model_fusion
from .model_fusion import (
    linear_merge,
    slerp_merge,
    ties_merge,
    dare_merge,
    lora_adapter,
    quantize,
)

__version__ = "0.1.8"

__all__ = [
    "Quat",
    "SpectralTokenizer",
    "WeightBalancer",
    "BlendWeight",
    "StreamEngine",
    "StreamOptions",
    "StreamChunk",
    "Safetensors",
    "ModelConfig",
    "ForwardEngine",
    "XFactor",
    "compute_xfactor",
    "sample_embeddings",
    "expand_spectral",
    "dense_matmul",
    "rms_norm",
    "load_config",
    "linear_merge",
    "slerp_merge",
    "ties_merge",
    "dare_merge",
    "lora_adapter",
    "quantize",
    "model_fusion",
    "__version__",
]
