#!/usr/bin/env python3
"""Gate de paridade numérica: CPU(f32) vs GPU, ANTES de qualquer métrica.

Um benchmark de qualidade medido num braço numérico que não reproduz o outro
mede o braço, não o motor. Este script é o portão: se `max|Δ|` ou o erro
relativo passarem do limiar, as métricas daquele modelo não devem ser citadas.

Mede, para cada modelo:
  - logits do mesmo prompt, CPU f32 vs GPU (bf16 quando o motor usa bf16 nas
    projeções de device), em várias posições;
  - `max|Δ|` absoluto e relativo (norma da diferença / norma da referência);
  - concordância de argmax (o que decide a geração);
  - e o mesmo contra o `transformers` em f32, como referência externa.

Uso: python3 parity_gate.py [model_dir ...]
"""
import os, sys, json
import numpy as np
import winnex_nano as W
from transformers import AutoTokenizer
from datasets import load_dataset

LIBS = ["/home/wnnx_user/ocl-local/lib/libOpenCL.so.1"]


def find_lib():
    for c in LIBS + ["/usr/lib/x86_64-linux-gnu/libOpenCL.so.1"]:
        if os.path.exists(c):
            return c
    return ""


def run(M):
    print(f"\n{'='*78}\n{M}\n{'='*78}")
    cfg = W.load_config(M + "/config.json")
    st = W.Safetensors.open_dir(M)
    tok = AutoTokenizer.from_pretrained(M)

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(r["text"].strip() for r in ds if r["text"].strip())
    ids_all = tok(text, add_special_tokens=False).input_ids

    cpu = W.ForwardEngine(cfg, st)
    gpu = W.ForwardEngine(cfg, st)
    gpu_ok = True
    try:
        gpu.enable_gpu(find_lib())
    except Exception as e:
        gpu_ok = False
        print(f"  GPU indisponivel: {type(e).__name__}: {e}")

    out = {"modelo": os.path.basename(M.rstrip('/')), "cpu_mode": cpu.weight_mode,
           "gpu_mode": gpu.weight_mode if gpu_ok else None}
    if not gpu_ok:
        return out

    print(f"  cpu modo={cpu.weight_mode}  gpu modo={gpu.weight_mode}  "
          f"atenção na GPU={gpu.attention_on_gpu}")
    print(f"  {'n':>5}{'max|Δ|':>14}{'rel':>13}{'cos':>11}{'argmax=':>9}")
    rows = []
    for n in (64, 128, 256, 512):
        seg = ids_all[:n]
        H = np.ascontiguousarray(cpu.embed_ids(seg), dtype=np.float32)
        lc = np.asarray(cpu.forward(H, n, False), dtype=np.float64)
        lg = np.asarray(gpu.forward(H, n, False), dtype=np.float64)
        d = np.abs(lc - lg).max()
        rel = np.linalg.norm(lc - lg) / np.linalg.norm(lc)
        cos = float(lc @ lg / (np.linalg.norm(lc) * np.linalg.norm(lg)))
        same = int(lc.argmax()) == int(lg.argmax())
        print(f"  {n:>5}{d:>14.3e}{rel:>13.3e}{cos:>11.6f}{str(same):>9}")
        rows.append(dict(n=n, max_abs=float(d), rel=float(rel), cos=cos, argmax_same=same))
    out["paridade_cpu_gpu"] = rows

    # referência externa: transformers f32 em CPU
    try:
        import torch
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()
        seg = ids_all[:256]
        H = np.ascontiguousarray(cpu.embed_ids(seg), dtype=np.float32)
        lc = np.asarray(cpu.forward(H, 256, False), dtype=np.float64)
        with torch.no_grad():
            lh = m(input_ids=torch.tensor([seg])).logits[0, -1].float().numpy().astype(np.float64)
        # o transformers recebe os mesmos tokens, não os mesmos hidden states:
        # o forward(H) do winnex parte do embed que ele próprio calculou.
        Hemb = np.asarray(m.get_input_embeddings()(torch.tensor(seg)).float().numpy(),
                          dtype=np.float64)
        lc2 = np.asarray(cpu.forward(np.ascontiguousarray(Hemb, dtype=np.float32), 256, False),
                         dtype=np.float64)
        rel = float(np.linalg.norm(lc2 - lh) / np.linalg.norm(lh))
        cos = float(lc2 @ lh / (np.linalg.norm(lc2) * np.linalg.norm(lh)))
        print(f"  winnex(cpu) vs transformers(f32): rel={rel:.3e}  cos={cos:.8f}"
              f"  argmax={int(lc2.argmax())==int(lh.argmax())}")
        out["vs_transformers"] = dict(rel=rel, cos=cos)
    except Exception as e:
        out["vs_transformers"] = {"erro": f"{type(e).__name__}: {e}"}
    return out


if __name__ == "__main__":
    models = sys.argv[1:] or [
        "/home/wnnx_user/models/Qwen2.5-0.5B-Instruct",
        "/home/wnnx_user/models/Llama-3.2-1B-Instruct",
    ]
    res = [run(m) for m in models]
    json.dump(res, open("/tmp/lowrank_study/parity_gate.json", "w"), indent=1)
    print("\n# -> /tmp/lowrank_study/parity_gate.json")
