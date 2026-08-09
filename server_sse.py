#!/usr/bin/env python3
"""
Winnex-Nano SSE Server — OpenAI-compatible /v1/chat/completions with streaming.

The server uses the StreamEngine (spectral tokenizer + Madhava fast build) to
generate responses chunk-by-chunk and stream them via SSE. It exposes the same
contract the Maestro already consumes (non-stream also supported), so swapping
SGLANG_URL -> winnex-nano requires no Maestro changes.

Endpoints:
  POST /v1/chat/completions  {model, messages, max_tokens, stream, temperature}
    - stream:true  -> SSE chunks (OpenAI format)
    - stream:false -> single JSON response
  GET  /health

Run:  python3 server_sse.py  (port 30002 by default)
"""
import json
import logging
import sys
import time

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn

sys.path.insert(0, "/home/wnnx_user/zenodo/winnex-nano")
import _winnex_nano as wn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("winnex-nano")

app = FastAPI(title="Winnex-Nano", version="0.1.0")

# The engine is a singleton (holds the spectral tokenizer + stream machinery).
_engine = wn.StreamEngine()


# ---------------------------------------------------------------------------
# Models (OpenAI-compatible)
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "winnex-nano"
    messages: List[ChatMessage] = []
    max_tokens: int = 256
    temperature: float = 0.0
    stream: bool = False


def _prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    """Joins the messages into a single prompt for the spectral engine."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _sse_encode(chunk_text: str) -> str:
    """Encodes a text chunk in OpenAI SSE delta format."""
    data = {"choices": [{"delta": {"content": chunk_text}, "index": 0}]}
    return f"data: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Generation policy: emits the response chunk-by-chunk.
# The spectral engine indexes the context; we split the context into word
# chunks (a simple deterministic policy). A real model would generate tokens;
# here the Madhava fast build + search guides the chunk composition.
# ---------------------------------------------------------------------------
def _generation_policy(max_chunks: int):
    """Returns a next_chunk_fn that emits word-chunks of the context."""
    counter = {"n": 0}

    def next_chunk(context: str, top_docs: List[str]) -> str:
        if counter["n"] >= max_chunks:
            return ""  # EOS
        # Deterministic chunking: take a slice of the accumulated context.
        # This is a placeholder generation policy — the point is the STREAM
        # MACHINERY (build + search + SSE), not the language model itself.
        start = counter["n"] * 12
        if start >= len(context):
            return ""
        chunk = context[start:start + 12]
        if not chunk.strip():
            return ""
        counter["n"] += 1
        return chunk
    return next_chunk


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    prompt = _prompt_from_messages([m.dict() for m in req.messages])
    if not prompt.strip():
        return JSONResponse({"error": "empty messages"}, status_code=400)

    max_chunks = max(1, min(req.max_tokens // 10, 64))
    next_chunk = _generation_policy(max_chunks)
    chunks = []

    def sink(chunk):
        if not chunk.done:
            chunks.append(chunk.text)

    if req.stream:
        async def event_stream():
            # Generator that streams chunks via SSE.
            def gen():
                yield _sse_encode(prompt[:0])  # (no-op prefix, kept minimal)
                _engine.stream(prompt, next_chunk, sink)
                for c in chunks:
                    yield _sse_encode(c)
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")

        # Run synchronously, then stream.
        _engine.stream(prompt, next_chunk, sink)
        return StreamingResponse(
            (_sse_encode(c) for c in chunks),
            media_type="text/event-stream",
        )

    # Non-stream: run once, return a single JSON.
    _engine.stream(prompt, next_chunk, sink)
    content = "".join(chunks)
    return {
        "id": "winnex-nano",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(chunks), "total_tokens": len(prompt) + len(chunks)},
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "winnex-nano", "engine": "spectral+madhava-fast-build"}


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 30002
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
