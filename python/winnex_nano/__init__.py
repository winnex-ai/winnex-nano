"""
winnex-nano — native Winnex inference engine.

Deterministic spectral tokenizer, multimodel weight balancing, and chunk
streaming via the Madhava fast build. Model-agnostic (Qwen/BERT/DeepSeek/GPT),
C++/OpenCL, no CUDA. Depends on winnex-madhava for the bound-engine kernels.

Components:
    - SpectralTokenizer: character → quaternion spectrum (deterministic, no BPE)
    - WeightBalancer:    W' = Σᵢ αᵢ·R(qᵢ)·Wᵢ  (multimodel fusion)
    - StreamEngine:      chunk generation via Madhava fast build (SSE)

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
)

__version__ = "0.1.0"

__all__ = [
    "Quat",
    "SpectralTokenizer",
    "WeightBalancer",
    "BlendWeight",
    "StreamEngine",
    "StreamOptions",
    "StreamChunk",
    "__version__",
]
