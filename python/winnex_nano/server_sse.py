#!/usr/bin/env python3
"""
Winnex-Nano server — OpenAI-compatible /v1/chat/completions with real inference.

This runs the native ForwardEngine: tokenizer -> prefill -> step -> detokenize,
with a KV cache, on the weights of a real checkpoint. It is not a mock and not
a stream of the prompt back to the caller.

WHAT CHANGED (2026-10-22)
------------------------
The previous version answered every request with 12-character slices of the
prompt. It was honest in its own comment ("a placeholder generation policy —
the point is the STREAM MACHINERY, not the language model itself"), but the
endpoint claimed to be an OpenAI-compatible chat completion, and a caller had
no way to tell the difference from the HTTP response:

    request : "What is the capital of France? Please answer in one word."
    response: "user: What is the capital of France? ...user: What is the..."

The engine could already generate (`ForwardEngine.generate`). What was missing
was the wiring, and one primitive: a way to go from TOKEN IDS to hidden states.
`prefill()` / `step()` / `embed_ids()` were added to the engine for that, so the
server no longer reaches into the safetensors file to build inputs.

Usage
-----
    python -m winnex_nano.server_sse --model /path/to/checkpoint [--port 30002]

    curl -N http://localhost:30002/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"messages":[{"role":"user","content":"The capital of France is"}],
           "max_tokens":32,"stream":true}'

Endpoints
---------
    GET  /health                  model, mode, device, context length
    GET  /v1/models               OpenAI-compatible model list
    POST /v1/chat/completions     stream + non-stream
"""
import argparse
import json
import logging
import os
import sys
import time
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

from . import _winnex_nano as wn

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("winnex-nano")

app = FastAPI(title="Winnex-Nano", version="0.2.0")

# Filled by build_engine() / the __main__ entrypoint.
_state = {
    "engine": None,
    "tokenizer": None,
    "model_dir": None,
    "model_name": "winnex-nano",
    "device": os.environ.get("WINNEX_OPENCL_LIB", ""),
    # Holds the Safetensors alive; ForwardEngine keeps a non-owning pointer.
    "safetensors": None,
}


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------
def build_engine(model_dir: str, use_gpu: bool = True, opencl_lib: str = ""):
    """Loads a checkpoint into the native engine.

    Gives every layer's weights to the device when they fit; otherwise the
    engine picks resident/stream itself and says which in /health.
    """
    from transformers import AutoTokenizer

    cfg = wn.load_config(os.path.join(model_dir, "config.json"))

    # A checkpoint is either ONE file (model.safetensors) or a directory of
    # shards. open_dir() mmaps each shard and never copies, so it is the right
    # path for both — but it needs the directory, not the file. Passing the
    # file to open_dir() scans its parent and looks for .safetensors entries,
    # which yields an empty view and surfaces later as
    # "tensor not found: model.embed_tokens.weight" at the first generate.
    if not hasattr(wn.Safetensors, "open_dir"):
        raise RuntimeError(
            "winnex-nano is too old for this server (no Safetensors.open_dir); "
            "upgrade to >= 0.1.27")
    st = wn.Safetensors.open_dir(model_dir)

    tok = AutoTokenizer.from_pretrained(model_dir)
    eng = wn.ForwardEngine(cfg, st)

    device = "cpu"
    if use_gpu:
        try:
            eng.enable_gpu(opencl_lib or os.environ.get("WINNEX_OPENCL_LIB", ""))
            device = eng.gpu_device or "gpu"
        except Exception as e:
            # No silent fallback: this is the one place where degrading to CPU
            # is the right call, because a server that refuses to start is
            # worse than a slow one — and /health reports which it got.
            logger.warning("GPU unavailable, serving on CPU: %s", e)
            device = "cpu"

    # `st` MUST be kept alive: ForwardEngine holds a non-owning pointer to it
    # (the MadhavaL2 contract — the caller owns the data). Letting it fall out
    # of scope here freed the Safetensors while the engine still pointed at it,
    # and the first generate then failed with
    # "Safetensors: tensor not found: model.embed_tokens.weight" — after the
    # model had reported a successful load.
    _state.update(engine=eng, tokenizer=tok, safetensors=st, model_dir=model_dir,
                  model_name=os.path.basename(model_dir.rstrip("/")) or "winnex-nano",
                  device=device)
    logger.info("loaded %s | %d layers | mode=%s | device=%s",
                _state["model_name"], cfg.num_hidden_layers,
                eng.weight_mode, device)
    return eng, tok


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "winnex-nano"
    messages: List[ChatMessage] = []
    max_tokens: int = 256
    temperature: float = 0.0
    top_k: int = 0
    stream: bool = False
    stop: Optional[List[str]] = None


def _render_prompt(messages: List[dict], tok) -> str:
    """Applies the model's own chat template when it has one.

    Falling back to "role: content" lines would give the model a format it was
    never trained on and quietly degrade every answer, so the template is the
    default and the manual join is the exception.
    """
    if tok is not None and getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
        except Exception as e:
            logger.warning("chat template failed (%s); using plain join", e)
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}"
                     for m in messages) + "\nassistant:"


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _sample(logits, temperature: float, top_k: int):
    """Token id from logits. temperature <= 0 is greedy."""
    import numpy as np
    lg = np.asarray(logits, dtype=np.float64)
    if temperature <= 0.0:
        return int(lg.argmax())
    lg = lg / max(temperature, 1e-5)
    lg -= lg.max()
    p = np.exp(lg)
    if top_k and 0 < top_k < p.size:
        # Keep the k largest, zero the rest. Cheaper than a full sort and it
        # matches what OpenAI's top_k does.
        keep = np.argpartition(-p, top_k - 1)[:top_k]
        mask = np.zeros_like(p, dtype=bool)
        mask[keep] = True
        p = np.where(mask, p, 0.0)
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        return int(np.asarray(logits).argmax())
    p /= total
    return int(np.random.choice(p.size, p=p))


def _generate(prompt: str, max_tokens: int, temperature: float, top_k: int,
              stop: Optional[List[str]]):
    """Yields decoded text pieces. Runs prefill then step-by-step decode."""
    eng, tok = _state["engine"], _state["tokenizer"]
    if eng is None:
        raise RuntimeError("engine not loaded — start with --model")

    ids = tok(prompt).input_ids
    if not ids:
        return
    # Leave room for the completion inside the model's window.
    room = eng.max_context - len(ids)
    if room <= 0:
        raise ValueError(
            f"prompt is {len(ids)} tokens but the model context is "
            f"{eng.max_context}")
    max_tokens = min(max_tokens, room)

    logits = eng.prefill(ids)
    eos = getattr(tok, "eos_token_id", None)
    emitted = []

    for _ in range(max_tokens):
        nxt = _sample(logits, temperature, top_k)
        if eos is not None and nxt == eos:
            return
        piece = tok.decode([nxt])
        if not piece:
            # Byte-level BPE can emit a token that decodes to nothing until the
            # next one completes the sequence. Keep decoding rather than stop.
            logits = eng.step(nxt)
            continue
        emitted.append(piece)
        text_so_far = "".join(emitted)

        cut = _stop_hit(text_so_far, stop)
        if cut is not None:
            # Emit only the part before the stop sequence.
            head = text_so_far[:cut]
            tail = head[len("".join(emitted[:-1])):]
            if tail:
                yield tail
            return

        yield piece
        logits = eng.step(nxt)


def _stop_hit(text: str, stop: Optional[List[str]]):
    if not stop:
        return None
    best = None
    for s in stop:
        if s and s in text:
            i = text.index(s)
            if best is None or i < best:
                best = i
    return best


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    eng = _state["engine"]
    if eng is None:
        return JSONResponse({"status": "no-model"}, status_code=503)
    return {
        "status": "ok",
        "model": _state["model_name"],
        "device": _state["device"],
        "weight_mode": eng.weight_mode,
        "hidden_size": eng.hidden_size,
        "vocab_size": eng.vocab_size,
        "max_context": eng.max_context,
        "device_weight_bytes": eng.device_weight_bytes,
    }


@app.get("/v1/models")
def models():
    return {"object": "list",
            "data": [{"id": _state["model_name"], "object": "model",
                      "owned_by": "winnex"}]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    eng, tok = _state["engine"], _state["tokenizer"]
    if eng is None:
        return JSONResponse(
            {"error": {"message": "no model loaded; start the server with --model",
                       "type": "server_error"}}, status_code=503)
    if not req.messages:
        return JSONResponse(
            {"error": {"message": "empty messages", "type": "invalid_request_error"}},
            status_code=400)

    prompt = _render_prompt([m.dict() for m in req.messages], tok)
    model = _state["model_name"]
    cid = f"chatcmpl-{int(time.time() * 1000)}"

    if req.stream:
        async def event_stream():
            yield _sse({"id": cid, "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": model,
                        "choices": [{"index": 0, "delta": {"role": "assistant"},
                                     "finish_reason": None}]})
            try:
                for piece in _generate(prompt, req.max_tokens, req.temperature,
                                       req.top_k, req.stop):
                    yield _sse({"id": cid, "object": "chat.completion.chunk",
                                "created": int(time.time()), "model": model,
                                "choices": [{"index": 0,
                                             "delta": {"content": piece},
                                             "finish_reason": None}]})
            except Exception as e:
                logger.exception("generation failed")
                yield _sse({"id": cid, "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [{"index": 0, "delta": {},
                                         "finish_reason": "error"}],
                            "error": str(e)})
            yield _sse({"id": cid, "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": model,
                        "choices": [{"index": 0, "delta": {},
                                     "finish_reason": "stop"}]})
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    t0 = time.perf_counter()
    try:
        text = "".join(_generate(prompt, req.max_tokens, req.temperature,
                                 req.top_k, req.stop))
    except Exception as e:
        logger.exception("generation failed")
        return JSONResponse(
            {"error": {"message": str(e), "type": "invalid_request_error"}},
            status_code=400)

    dt = time.perf_counter() - t0
    prompt_tokens = len(tok(prompt).input_ids)
    completion_tokens = len(tok(text).input_ids) if text else 0
    return {
        "id": cid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens,
                  "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
        "winnex": {"seconds": round(dt, 3),
                   "tokens_per_second": round(completion_tokens / dt, 2) if dt else None,
                   "weight_mode": _state["engine"].weight_mode},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Winnex-Nano inference server")
    ap.add_argument("--model", required=True,
                    help="checkpoint directory (config.json + weights + tokenizer)")
    ap.add_argument("--port", type=int, default=30002)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--cpu", action="store_true", help="do not enable OpenCL")
    ap.add_argument("--opencl-lib", default="", help="explicit ICD loader path")
    args = ap.parse_args(argv)

    build_engine(args.model, use_gpu=not args.cpu, opencl_lib=args.opencl_lib)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
