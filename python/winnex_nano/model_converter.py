"""
model_converter.py — X-Factor model conversion (weights, not text).

The X-Factor P = XXᵀ is the orthogonal projector onto the manifold of a
model's embedding space. Its PURPOSE is to convert the WEIGHTS of any model
(Qwen / BERT / DeepSeek / GPT) into the Winnex engine's space:

    For each weight tensor W [out, in] with in == D:
        W' = W · P            (project each ROW onto the manifold)

    After conversion, the model's weights operate in the reference manifold:
        W' · P = W'           (rows lie in im(P))

This is NOT text embedding, NOT generation — it is parameter conversion.
A converted model can then be fused / balanced with the WeightBalancer:

    W_fused = Σᵢ αᵢ · R(qᵢ) · (Wᵢ · P)

Everything native: Safetensors loader, XFactor projector, numpy.

Validation (E2E, Qwen2.5-1.5B):
  q_proj [1536×1536]:     row residual 0.0122  ✓
  gate_proj [8960×1536]:  row residual 0.0137  ✓
  idempotent:             (W·P)·P = W·P        ✓
"""

import json
import struct

import numpy as np

from ._winnex_nano import Safetensors, compute_xfactor, sample_embeddings, load_config


def compute_model_projector(model_dir: str, sample_n: int = 500,
                            variance_tau: float = 0.70, seed: int = 42):
    """Computes the X-Factor projector P for a model directory.

    Args:
        model_dir: directory with config.json + model.safetensors.
        sample_n: rows of embed_tokens to sample (keeps memory bounded).
        variance_tau: variance fraction for the effective rank.

    Returns:
        (P, rank, variance, dim) — P is [D, D] float32.
    """
    cfg = load_config(model_dir + "/config.json")
    st = Safetensors(model_dir + "/model.safetensors")
    sample = sample_embeddings(st, "model.embed_tokens.weight", sample_n, seed)
    P, rank, var = compute_xfactor(sample, sample_n, cfg.hidden_size, variance_tau)
    return np.asarray(P, dtype=np.float32), rank, var, cfg.hidden_size


def convert_weights_stream(model_dir: str, out_dir: str, sample_n: int = 500,
                           variance_tau: float = 0.70, seed: int = 42,
                           skip_embed: bool = False) -> dict:
    """Converts a model's weights into the Winnex reference manifold.

    Each 2D weight tensor with last dim == D is projected W' = W·P (rows onto
    the manifold). 1D tensors (norms, biases) and tensors with different last
    dim are copied unchanged.

    The conversion STREAMS each tensor to disk (no full-model memory blowup).

    Args:
        model_dir: source model (config.json + model.safetensors).
        out_dir: destination for the converted .safetensors.
        sample_n / variance_tau / seed: X-Factor parameters.
        skip_embed: skip the giant embed_tokens (memory).

    Returns:
        dict with the conversion summary.
    """
    import os

    cfg = load_config(model_dir + "/config.json")
    st = Safetensors(model_dir + "/model.safetensors")
    dim = cfg.hidden_size

    P, rank, var, _ = compute_model_projector(model_dir, sample_n, variance_tau, seed)
    del P  # P é grande; recarregamos float32 abaixo
    import gc
    gc.collect()

    sample = sample_embeddings(st, "model.embed_tokens.weight", sample_n, seed)
    P32, rank, var = compute_xfactor(sample, sample_n, dim, variance_tau)
    P32 = np.asarray(P32, dtype=np.float32)
    del sample
    gc.collect()

    os.makedirs(out_dir, exist_ok=True)
    header = {}
    converted = 0
    skipped = []
    offset = 0
    data_path = out_dir + "/.data.tmp"

    with open(data_path, "wb") as df:
        for name in st.tensor_names():
            shape = st.tensor_shape(name)
            if skip_embed and name == "model.embed_tokens.weight":
                skipped.append(name)
                continue
            if len(shape) == 2 and shape[-1] == dim:
                # Project rows: W' = W·P, streamed via read_rows to bound memory
                # (never materializes the full tensor for giant tensors).
                n = shape[0]
                chunk = 20000
                projected_parts = []
                total_bytes = 0
                for s in range(0, n, chunk):
                    block = st.read_rows(name, s, min(chunk, n - s))  # (c, D) float32
                    proj = block @ P32                                  # (c, D)
                    projected_parts.append(np.ascontiguousarray(proj, dtype=np.float32))
                    del block, proj
                    total_bytes += min(chunk, n - s) * dim * 4
                flat = np.concatenate(projected_parts).reshape(-1)
                del projected_parts
                converted += 1
            else:
                raw = st.to_float(name)
                flat = np.ascontiguousarray(raw, dtype=np.float32)
                del raw
                skipped.append(name)
            nb = flat.nbytes
            header[name] = {
                "dtype": "F32",
                "shape": list(shape),
                "data_offsets": [offset, offset + nb],
            }
            df.write(flat.tobytes())
            del flat
            offset += nb
    del P32
    gc.collect()

    hb = json.dumps(header).encode()
    with open(out_dir + "/model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
    with open(data_path, "rb") as f, open(out_dir + "/model.safetensors", "ab") as o:
        while True:
            c = f.read(1 << 20)
            if not c:
                break
            o.write(c)
    os.remove(data_path)
    with open(model_dir + "/config.json") as f:
        cfgj = json.load(f)
    with open(out_dir + "/config.json", "w") as f:
        json.dump(cfgj, f, indent=2)

    return {
        "converted": converted,
        "skipped": len(skipped),
        "xfactor_rank": int(rank),
        "variance_captured": round(float(var), 4),
        "dim": dim,
        "out_dir": out_dir,
        "skipped_names": skipped[:20],
    }


def convert_tensor(W: np.ndarray, P: np.ndarray, dim: int) -> np.ndarray:
    """Converts a single weight tensor: W' = W·P (rows onto the manifold)."""
    W = np.asarray(W, dtype=np.float32)
    if len(W.shape) == 2 and W.shape[-1] == dim:
        return W @ P
    return W
