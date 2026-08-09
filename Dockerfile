# Winnex-Nano — native inference server (spectral tokenizer + Madhava fast build).
# Exposes OpenAI-compatible /v1/chat/completions (stream + non-stream) on 30002.
FROM python:3.10-slim

WORKDIR /app

# Copy the winnex-nano package (Python + prebuilt C++ binding).
COPY server_sse.py /app/
COPY _winnex_nano.cpython-310-x86_64-linux-gnu.so /app/
COPY benchmark_honesto.py /app/

# Runtime deps only (the binding is prebuilt; no compiler needed at runtime).
RUN pip install --no-cache-dir numpy fastapi uvicorn httpx pydantic

EXPOSE 30002

CMD ["python3", "server_sse.py", "30002"]
