# Status da Stack Winnex — 2026-08-09

## Camada de IA (Maestro ERP)

| Componente | Estado | Detalhe |
|-----------|--------|---------|
| SGLang (inferência) | ✅ operacional | Recriado com GPU RTX 5060 Ti; responde `2+2=4` |
| winnex-embedding (Qwen3 1024d) | ✅ | Serviço de embedding (CPU) |
| UnifiedAIService (failover) | ✅ | Prioridade via WSafe |
| WSafe (providers) | ✅ | `['winnex_local','zai','deepseek','openai']` com type+priority |
| ai_models config | ✅ | inference + embedding JSON-driven |
| Madhava RAG | ✅ | quantização, métrica, persistência, auto-indexação |
| RAG injector (async) | ✅ | fallbacks visíveis |

## Motor nativo (winnex-nano)

| Componente | Estado | Métrica |
|-----------|--------|---------|
| Tokenizer espectral | ✅ | round-trip 0% erro, 490µs/chunk |
| WeightBalancer `W'=ΣαᵢR(qᵢ)Wᵢ` | ✅ | α=[1,0]=model1, α=[0.5,0.5]=mean |
| Tensor ops (RMSNorm/RoPE/GPTQ) | ✅ | — |
| **StreamEngine** (chunk via build rápido) | ✅ | **1.7ms/3 chunks, 0.37ms build** |
| **Servidor SSE** `/v1/chat/completions` | ✅ | stream + non-stream, OpenAI-compat |
| Pre-patente Zenodo | ✅ | DOI 10.5281/zenodo.21861809 |

## Comparação honesta (benchmark)

| Métrica | winnex-nano | SGLang/HTTP |
|---------|-------------|-------------|
| Encode de chunk | 490µs | ~700ms (embedding HTTP) |
| Build rápido Madhava | 0.37ms/chunk | n/a |
| Stream 3 chunks | 1.7ms | ~2.1s |
| Dependência externa | nenhuma | SGLang + transformers |

## Próximos passos

1. **Substituir o SGLang**: trocar `SGLANG_URL` → winnex-nano (servidor SSE) no Maestro
2. **Integração**: `ai_models_config` → winnex_local aponta para winnex-nano
3. **e2e failover**: simular queda do winnex_local → verificar fallback para zai/deepseek
4. **Commitar** o progresso nos repositórios
