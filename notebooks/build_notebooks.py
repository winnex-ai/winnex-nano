#!/usr/bin/env python3
"""Gera os dois notebooks de benchmark: um por motor.

Dois notebooks, mesmo harness (`bench_core.py`), mesma ordem de células. A
única diferença entre eles é o braço que carregam. Isso é deliberado: se os
dois motores fossem medidos por códigos diferentes, a diferença entre os
números poderia ser a diferença entre os códigos.

Uso: python3 build_notebooks.py
Escreve: bench_winnex_nano.ipynb e bench_transformers.ipynb
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": t.splitlines(keepends=True)}


# ── cabeçalho comum ─────────────────────────────────────────────────────────

def header(arm, other):
    return md(f"""# Benchmark de inferência — `{arm}`

Braço **{arm}** do benchmark pareado. O outro braço é `{other}`, e roda no
notebook irmão com **o mesmo harness** (`bench_core.py`) e a **mesma ordem de
células**. Rode os dois no mesmo tipo de máquina e compare as saídas JSON.

## O que este notebook mede, e em que dtype

| bloco | dtype | por quê |
|---|---|---|
| **Paridade de logits** | **f32** | é o gate. Em f32 os dois motores têm que dar o mesmo argmax em 100% dos tokens gerados; se não derem, nada abaixo é interpretável |
| Perplexidade | **f32** | medida de qualidade — comparar motores, não implementações |
| MMLU / LAMBADA | **f32** | idem; zero-shot, IC de Wilson |
| Prefill / Decode | **bf16** | medida de desempenho, no dtype em que o modelo seria servido |
| Memória | o do bloco | anônimo, RSS e VRAM de pico |

**O dtype está declarado em cada resultado**, no campo `dtype`. Um número sem
ele não é comparável com nada.

## O que este notebook NÃO mede

- **Não é MMLU de 5-shot.** É zero-shot, então os valores absolutos não são
  comparáveis com a literatura.
- **Não mede batch > 1.** Decode é batch 1, que é o caso de um chat.
- **Prefill e decode são regimes diferentes** e têm métricas diferentes:
  prefill é throughput (tokens/s, TTFT), decode é latência (ms/token). Somar
  os dois num único "tokens/s" esconde o que importa.

## Como ler os números de tempo

10 iterações, as 2 primeiras descartadas como warmup (JIT do OpenCL, alocação
preguiçosa, cache frio). Reporta **média, mediana e P99**. O P99 é o que expõe
gargalo de sincronização ou de coleta de lixo: uma mediana boa com P99 ruim é
um serviço que derruba 1 em 100 requisições.
""")


def instalar(arm):
    if arm == "winnex-nano":
        return md("""## 1. Instalação

O pacote vem do PyPI. `winnex-madhava` é dependência declarada e é instalado
junto. O motor precisa de um ICD OpenCL; no Kaggle (T4) ele vem com o driver
da NVIDIA.

Se a instalação falhar, a célula abaixo imprime a causa — em particular se o
wheel da versão não existir para este Python/plataforma, o que é diferente de
"a instalação deu erro".
""") , code("""import subprocess, sys, platform
print("python", platform.python_version(), "|", platform.platform())
VERSAO = "0.1.35"
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    f"winnex-nano=={VERSAO}"],
                   capture_output=True, text=True)
print("rc =", r.returncode)
if r.returncode != 0:
    print("--- stdout ---"); print(r.stdout[-2500:])
    print("--- stderr ---"); print(r.stderr[-2500:])
    raise SystemExit("instalacao falhou — veja acima qual wheel faltou")
import winnex_nano as wn
print("winnex-nano", wn.__version__)
print("SIMD (AVX2+FMA):", wn.compiled_with_simd())
""")
    return md("""## 1. Instalação

`transformers` e `torch` já vêm na imagem do Kaggle. A célula confirma as
versões — importa saber qual delas produziu o número, porque o desempenho do
`transformers` muda entre versões.
"""), code("""import torch, transformers, platform
print("python", platform.python_version(), "|", platform.platform())
print("torch", torch.__version__, "| cuda disponivel:", torch.cuda.is_available())
print("transformers", transformers.__version__)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU:", p.name, "| VRAM", round(p.total_memory/2**30, 2), "GB")
    CAPS = ((p.major, p.minor) >= (8, 0))
else:
    CAPS = False
print("bf16 suportado nativamente:", CAPS)
"""), code("""!pip -q install -U datasets tokenizers 2>&1 | tail -2
import datasets; print("datasets", datasets.__version__)
""")


def harness(arm):
    return md("""## 2. O harness

`bench_core.py` é o mesmo arquivo nos dois notebooks. Copiado para cá na
célula seguinte para que o notebook seja autossuficiente no Kaggle.
"""), code("""import os
CORE = r'''""" + open(os.path.join(HERE, "bench_core.py")).read().replace("'''", '\\x27\\x27\\x27') + """'''
with open("bench_core.py", "w") as f:
    f.write(CORE)
import bench_core
print("harness carregado;", len(CORE), "bytes")
"""), md("""## 3. Memória: linha de base (ponto 1 de 3)

Medida **antes** de carregar o modelo. Heap anônimo é a métrica honesta para o
winnex, que lê o checkpoint por mmap: as páginas de arquivo aparecem no RSS sem
serem memória alocada pelo processo.
"""), code("""import gc
gc.collect()
rss0, anon0 = bench_core.smaps()
vram0 = bench_core.vram()
BASELINE = {"rss_gb": rss0/2**30, "anon_gb": anon0/2**30, "vram_gb": vram0/2**30}
print(f"baseline  RSS={BASELINE['rss_gb']:.3f} GB  "
      f"anonimo={BASELINE['anon_gb']:.3f} GB  VRAM={BASELINE['vram_gb']:.3f} GB")
""")


def download(arm):
    return md("""## 4. Modelo

Qwen2.5-0.5B-Instruct: pequeno o bastante para caber na T4 em f32 (1,0 GB de
pesos), o que permite rodar o gate de paridade e a PPL **em f32** — condição
para comparar motores sem a confusão do bf16.

Trocar o `MODELO` para um Llama-3.2 exige que o winnex-nano honre
`rope_scaling` (corrigido na 0.1.35); numa versão anterior ele rodaria errado
sem avisar.
"""), code("""import os
MODELO_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODELO = "/kaggle/working/modelo"
if not os.path.isdir(MODELO):
    from huggingface_hub import snapshot_download
    snapshot_download(MODELO_ID, local_dir=MODELO,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])
print(sorted(os.listdir(MODELO)))
""")


def carga(arm):
    cls = "WinnexArm" if arm == "winnex-nano" else "HFArm"
    return md(f"""## 5. Carga do modelo (ponto 2 de 3 de memória)

`{cls}`, dispositivo `gpu`, dtype `f32` — o gate e a qualidade precisam de f32.
Se a GPU não estiver disponível o braço cai para CPU e **declara isso** em
`arm.device`.
"""), code(f"""import time, importlib
importlib.reload(bench_core)

t0 = time.perf_counter()
arm = bench_core.{cls}(MODELO, device="gpu", dtype="f32").load_tokenizer()
t_load = time.perf_counter() - t0

rss1, anon1 = bench_core.smaps()
CARGA = {{"load_s": round(t_load, 2),
         "anon_delta_gb": round((anon1-anon0)/2**30, 3),
         "vram_delta_gb": round((bench_core.vram()-vram0)/2**30, 3)}}
print(f"carregou em {{t_load:.2f}} s | modo={{arm.device}}")
print(f"anonimo +{{CARGA['anon_delta_gb']:.3f}} GB | VRAM +{{CARGA['vram_delta_gb']:.3f}} GB")
print("vocab", arm.V, "| contexto maximo", arm.max_ctx)
""")


def paridade(arm, other):
    return md(f"""## 6. GATE: paridade de logits (f32)

Prompt de 50 tokens, 50 tokens gerados, argmax de cada um comparado com a
referência publicada pelo braço `{other}`.

**Se este gate falhar, nada abaixo vale.** É por isso que ele vem antes da
perplexidade e das tarefas: medir qualidade num motor que não reproduz o outro
mede o motor, não o modelo.

A referência é gravada em `referencia_paridade.json` pelo notebook do
`{other}`; rode os dois e copie o arquivo entre eles (ou compare os dois JSON
no fim).
"""), code(f"""PROMPT_TEXTO = ("The history of the Roman Empire is a long and complicated "
                "story that begins with the founding of the city")
prompt_ids = arm.encode(PROMPT_TEXTO, False)[:50]
print("prompt:", len(prompt_ids), "tokens")

t0 = time.perf_counter()
gerado = bench_core.task_parity(arm, prompt_ids, n_gen=50)
t_par = time.perf_counter() - t0
print("50 tokens gerados em", round(t_par, 2), "s")
print("tokens:", gerado[:20], "...")

import json, os
REF = "referencia_paridade.json"
if os.path.exists(REF):
    ref = json.load(open(REF))
    igual = sum(a == b for a, b in zip(gerado, ref["tokens"]))
    print(f"\\nGATE: {{igual}}/{{len(ref['tokens'])}} argmax identicos "
          f"({{100*igual/len(ref['tokens']):.1f}}%)")
    assert igual == len(ref["tokens"]), "GATE FALHOU — nao siga para as metricas"
    print("GATE PASSOU")
else:
    json.dump({{"arm": "{arm}", "dtype": "f32", "prompt_ids": prompt_ids,
               "tokens": gerado}}, open(REF, "w"), indent=1)
    print("referencia gravada em", REF, "— rode o outro notebook e volte")
""")


def ppl(arm):
    return md("""## 7. Perplexidade de corpus (f32)

Wikitext-2 test, janelas de 512 com passo 256. `PPL = exp(Σnll/Σtok)`.

O IC é bootstrap sobre **tokens**, não sobre janelas: as janelas se sobrepõem
em 50% e compartilham metade dos tokens, então tratá-las como amostras
independentes produz um intervalo que não se sustenta.
"""), code("""t0 = time.perf_counter()
r_ppl = bench_core.task_ppl(arm, seq=512, stride=256, n_windows=40)
r_ppl["segundos"] = round(time.perf_counter()-t0, 1)
print(f"PPL = {r_ppl['ppl']:.4f}  IC95 = [{r_ppl['ppl_ci95'][0]:.4f}, {r_ppl['ppl_ci95'][1]:.4f}]")
print(f"n = {r_ppl['n_tokens']} tokens, {r_ppl['n_janelas']} janelas, "
      f"seq={r_ppl['seq']} stride={r_ppl['stride']}, {r_ppl['segundos']}s")
print("dtype:", r_ppl["dtype"])
""")


def tarefas(arm):
    return md("""## 8. Acurácia zero-shot (f32)

MMLU e LAMBADA por **log-verossimilhança**, não por geração: pontua cada
alternativa pela soma (ou média, na variante `ll-norm`) dos log-probs da
continuação e escolhe a maior. É o protocolo do MMLU original.

IC de Wilson. Com n=150 o intervalo de uma acurácia perto de 35% tem ~±8
pontos — diferenças menores que isso não são detectáveis e não devem ser
reportadas como vantagem.

`n` é ajustável; cada item de MMLU custa 4 forwards completos.
"""), code("""N_ITENS = 150
t0 = time.perf_counter()
r_mmlu_raw  = bench_core.task_mmlu(arm, n=N_ITENS, normalize=False)
r_mmlu_norm = bench_core.task_mmlu(arm, n=N_ITENS, normalize=True)
r_lambada   = bench_core.task_lambada(arm, n=N_ITENS)
print("tempo total:", round(time.perf_counter()-t0, 1), "s")
for r in (r_mmlu_raw, r_mmlu_norm, r_lambada):
    lo, hi = r["acc_ci95"]
    print(f"  {r['task']:<16} {r['acc']*100:5.1f}%  n={r['n']:<4} "
          f"IC95=[{lo*100:.1f}, {hi*100:.1f}]  ({r['dtype']})")
print("\\nNota: zero-shot. O MMLU da literatura e 5-shot — nao compare o valor absoluto.")
""")


def desempenho(arm):
    return md("""## 9. Prefill — throughput (bf16)

Recarrega o modelo em **bf16**, que é o dtype em que ele seria servido. O
dtype é declarado no resultado.

512 / 1024 / 2048 tokens. Métrica: **TTFT** (tempo até o primeiro token) e
**tokens/s**. 10 iterações, 2 descartadas.

Prefill e decode são regimes de memória e computação diferentes. Misturá-los
num único "tokens/s" esconde o gargalo: aqui há paralelismo sobre a sequência;
no decode há um token por vez.
"""), code("""%%time
del arm
gc.collect()
arm16 = bench_core.%s(MODELO, device="gpu", dtype="bf16").load_tokenizer()
print("modo:", arm16.device, "| dtype:", arm16.dtype)

r_prefill = bench_core.task_prefill(arm16, lengths=(512, 1024, 2048), iters=10, warmup=2)
for r in r_prefill:
    if r.get("status"):
        print(f"  {r['n_tokens']:>5} tokens: {r['status']} — {r.get('erro','')[:90]}")
    else:
        print(f"  {r['n_tokens']:>5} tokens: TTFT {r['ttft_ms']:8.1f} ms  "
              f"{r['tok_per_s']:8.1f} tok/s  "
              f"(mediana {r['median_ms']:.1f}, P99 {r['p99_ms']:.1f})")
""" % ("WinnexArm" if arm == "winnex-nano" else "HFArm"))


def decode(arm):
    return md("""## 10. Decode — latência (bf16)

Prompt de 50 tokens, 128 tokens gerados, **batch = 1**. Métrica:
**ITL** (inter-token latency, ms/token) e tokens/s.

O P99 importa mais que a média: é o que determina se um timeout de 200 ms
derruba uma requisição em 100. Uma mediana boa com P99 ruim é um serviço
instável, mesmo com throughput médio bom.
"""), code("""r_decode = bench_core.task_decode(arm16, prompt_len=50, n_gen=128, iters=10, warmup=2)
if r_decode.get("status"):
    print(r_decode["status"], "-", r_decode.get("erro","")[:120])
else:
    print(f"ITL mediana = {r_decode['itl_ms']:.2f} ms/token  "
          f"({r_decode['tok_per_s']:.2f} tok/s)")
    print(f"  media  {r_decode['mean_ms']:.2f} ms")
    print(f"  P99    {r_decode['p99_ms']:.2f} ms   <- expos gargalo de sync/GC")
    print(f"  min/max {r_decode['min_ms']:.2f} / {r_decode['max_ms']:.2f} ms")
    print(f"  n = {r_decode['n']} execucoes x {r_decode['n_gen']} tokens, dtype {r_decode['dtype']}")
""")


def memoria(arm):
    return md("""## 11. Memória — pico durante a geração (ponto 3 de 3)

- **Heap anônimo**: a RAM realmente alocada pelo processo. Ignora páginas de
  arquivo do mmap, que o SO pode descartar. É a métrica honesta para o winnex.
- **RSS**: memória residente total; mostra o impacto do mmap no sistema.
- **VRAM pico**: amostrada em thread durante a geração, incluindo KV cache.
"""), code("""with bench_core.MemTracker() as mt:
    pb = list(range(1000, 1000 + min(512, arm16.max_ctx - 40)))
    arm16.reset(); lg = arm16.prefill(pb)
    cur = int(lg.argmax())
    for _ in range(64):
        lg = arm16.decode_step(cur); cur = int(lg.argmax())
    arm16.sync()

rss2, anon2 = bench_core.smaps()
PICO = {"anon_gb": anon2/2**30,
        "rss_gb": rss2/2**30,
        "vram_peak_gb": (mt.peak - BASELINE["vram_gb"]*2**30)/2**30}
print(f"heap anonimo    {PICO['anon_gb']:7.3f} GB   (baseline {BASELINE['anon_gb']:.3f})")
print(f"RSS             {PICO['rss_gb']:7.3f} GB")
print(f"VRAM pico       {PICO['vram_peak_gb']:7.3f} GB")
print(f"  -> anonimo e a metrica honesta: o mmap do checkpoint aparece no RSS")
print(f"     sem ser memoria alocada pelo processo.")
""")


def limpeza(arm):
    return md("""## 12. Limpeza e vazamento

Libera o motor, força coleta e mede o heap anônimo de novo. O delta depois da
limpeza é o vazamento. **Deve ser ~0**: memória que sobrevive ao `del` é
memória que um servidor não recupera entre requisições.
"""), code("""del arm16
gc.collect()
if torch_ok := ("torch" in sys.modules):
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
time.sleep(1)
rss3, anon3 = bench_core.smaps()
VAZAMENTO = {"anon_gb": (anon3-anon0)/2**30,
             "vram_gb": (bench_core.vram()-vram0)/2**30}
print(f"apos limpeza: anonimo {anon3/2**30:.3f} GB  (baseline {anon0/2**30:.3f})")
print(f"vazamento anonimo: {VAZAMENTO['anon_gb']:+.3f} GB")
print(f"vazamento VRAM   : {VAZAMENTO['vram_gb']:+.3f} GB")
""")


def exporta(arm):
    tag = arm.replace("-", "_")
    return md("""## 13. Exportar

Um JSON por braço, com **todos** os números e o dtype de cada um. É esse
arquivo que a comparação lê — não a saída impressa, que não carrega n nem IC.
"""), code(f"""import json, platform
saida = {{
    "arm": "{arm}",
    "maquina": platform.platform(),
    "python": platform.python_version(),
    "modelo": MODELO_ID,
    "data": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "memoria": {{"baseline": BASELINE, "carga": CARGA, "pico": PICO, "vazamento": VAZAMENTO}},
    "gate_paridade": {{"dtype": "f32", "n_tokens": len(gerado),
                       "primeiros": gerado[:10]}},
    "resultados": [r_ppl, r_mmlu_raw, r_mmlu_norm, r_lambada,
                   *r_prefill, r_decode],
}}
ARQ = "resultado_{tag}.json"
json.dump(saida, open(ARQ, "w"), indent=2, default=float)
print("gravado:", ARQ)
print(json.dumps({{k: v for k, v in saida.items() if k != "resultados"}}, indent=2, default=float)[:900])
""")


def slug(arm):
    return arm.replace("-", "_")


def montar(arm, other):
    cells = [header(arm, other)]
    cells += list(instalar(arm))
    cells += list(harness(arm))
    cells += list(download(arm))
    cells += list(carga(arm))
    cells += list(paridade(arm, other))
    cells += list(ppl(arm))
    cells += list(tarefas(arm))
    cells += list(desempenho(arm))
    cells += list(decode(arm))
    cells += list(memoria(arm))
    cells += list(limpeza(arm))
    cells += list(exporta(arm))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    for arm, other in (("winnex-nano", "transformers"),
                       ("transformers", "winnex-nano")):
        nb = montar(arm, other)
        p = os.path.join(HERE, f"bench_{slug(arm)}.ipynb")
        with open(p, "w") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"{p}  ({len(nb['cells'])} celulas)")
