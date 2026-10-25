#!/usr/bin/env python3
"""Gera os dois notebooks de benchmark para o Kaggle: um por motor.

Mesmo harness, mesma ordem de células, mesma amostra. A única diferença entre os
dois arquivos é a classe que carregam. Se os dois motores fossem medidos por
códigos diferentes, a diferença entre os números poderia ser a diferença entre os
códigos.

Ambos instalam do PyPI. Nada é compilado: o que se mede é o artefato publicado,
que é o que um usuário instala.

Uso: python3 build_notebooks.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
VERSAO = "0.1.35"

# Os dois braços têm que ver os MESMOS itens. O seed é fixo e a amostra é
# canônica (ordenada por conteúdo antes de embaralhar), então o hash dos itens
# é comparável entre execuções e entre máquinas.
SEED = 0
N_MMLU = 200
N_LAMBADA = 200


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": t.splitlines(keepends=True)}


# ══ 0. cabeçalho ═══════════════════════════════════════════════════════════

def header(arm, other):
    cls = "WinnexArm" if arm == "winnex-nano" else "HFArm"
    return md(f"""# Benchmark de inferência — `{arm}`

Braço **{arm}** de um par. O outro braço é `{other}`, e roda no notebook irmão
com **o mesmo harness** e a **mesma ordem de células**. A única diferença entre
os dois arquivos é a classe que carregam: `{cls}`.

## O que este notebook mede

| bloco | dtype | o que responde | métrica |
|---|---|---|---|
| ambiente | — | a GPU está mesmo sendo usada? | `assert device != "cpu"` |
| tokenizer | — | os dois braços tokenizam igual? | hash dos ids |
| **paridade** | **f32** | os dois calculam a mesma função? | argmax de 50 tokens (gate) |
| perplexidade | f32 | reproduzem o mesmo modelo? | `exp(Σnll/Σtok)` + IC bootstrap |
| MMLU / LAMBADA | f32 | acertam o mesmo? | acurácia + IC Wilson |
| prefill | bf16 | quanto custa processar o prompt? | TTFT, tok/s, **decomposição** |
| decode | bf16 | quanto custa cada token? | ITL em 3 contextos |
| memória | — | quanto RAM/VRAM | anônimo, RSS, VRAM pico |

## O que este notebook NÃO afirma

- **Não é MMLU 5-shot.** É zero-shot: o valor absoluto não é comparável com a
  literatura. A comparação que ele sustenta é winnex vs transformers, no mesmo
  protocolo.
- **Batch = 1.** É o caso de um chat; não é serving com throughput.
- **VRAM vem do `nvidia-smi`, que é do dispositivo**, não do processo. Se outra
  coisa estiver usando a GPU, o número inclui. Está anotado no relatório.
- **Uma placa, um modelo, uma execução.** O IC cobre a variância amostral das
  tarefas, não a variância entre hardware.

## Reprodutibilidade

- Seed fixa e amostra **canônica**: os itens são ordenados por conteúdo antes de
  embaralhar, então duas execuções escolhem os mesmos itens mesmo que o dataset
  tenha sido reordenado. O hash dos itens vai no JSON e os dois braços têm que
  produzir o mesmo.
- 10 iterações, as 2 primeiras descartadas como warmup.
- Reporta média, mediana, P99, IQR e **ruído relativo** `(p99−mediana)/mediana`.
  Uma diferença menor que o ruído não é uma diferença.
""")


# ══ 1. ambiente ════════════════════════════════════════════════════════════

def ambiente(arm):
    return md("""## 1. Ambiente

Três verificações antes de instalar qualquer coisa, porque elas decidem se o
benchmark é interpretável: GPU presente, OpenCL carregável (só o braço do
winnex usa), e disco.
"""), code("""import subprocess, sys, os, shutil, platform
def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout
print("python", platform.python_version(), "|", platform.platform())
print(sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv"))
print(sh("free -g | head -2"))
print(sh("df -h /kaggle/working | tail -1"))
"""), md("""### 1.1 OpenCL

O motor do winnex usa **OpenCL, não CUDA**. Sem um ICD carregável ele lança na
inicialização — que é o comportamento correto, e o oposto de cair para CPU e
reportar um número de GPU que não existe.

O Kaggle não garante o loader; a célula instala se faltar e imprime o que achou.
"""), code("""if shutil.which("apt-get") and os.geteuid() == 0:
    sh("apt-get install -y -q ocl-icd-libopencl1 ocl-icd-opencl-dev 2>&1 | tail -2")
print("libOpenCL:", sh("ls /usr/lib/x86_64-linux-gnu/libOpenCL.so* 2>&1"))
print("vendors  :", sh("ls /etc/OpenCL/vendors/ 2>&1"))
""")


# ══ 2. instalação ══════════════════════════════════════════════════════════

def instalar(arm):
    if arm == "winnex-nano":
        return md(f"""## 2. Instalação — do PyPI

`winnex-nano=={VERSAO}`, instalado do PyPI. **Nada é compilado aqui.** Se o
wheel não existir para este Python, o `pip` falha e a célula mostra a mensagem —
o que é diferente de "o código tem um bug", e a diferença importa para saber o
que consertar.
"""), code(f"""import subprocess, sys, importlib.metadata as pm
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "winnex-nano=={VERSAO}", "transformers>=4.40",
                    "datasets", "tokenizers", "huggingface_hub"],
                   capture_output=True, text=True)
print("pip rc =", r.returncode)
if r.returncode != 0:
    print("--- stdout ---"); print(r.stdout[-3000:])
    print("--- stderr ---"); print(r.stderr[-3000:])
    raise SystemExit("instalacao falhou — veja qual wheel faltou para este Python")
for p in ("winnex-nano", "winnex-madhava", "transformers", "torch", "datasets"):
    try: print(f"  {{p:<18}} {{pm.version(p)}}")
    except Exception: print(f"  {{p:<18}} AUSENTE")
import winnex_nano as wn
print("\\nwinnex_nano", wn.__version__, "| SIMD AVX2:", wn.compiled_with_simd())
""")
    return md("""## 2. Instalação

`transformers` já vem na imagem do Kaggle; a célula confirma as versões — o
desempenho muda entre versões, então o número precisa carregar a versão que o
produziu.
"""), code("""import subprocess, sys, importlib.metadata as pm
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                    "transformers>=4.40", "datasets", "tokenizers"],
                   capture_output=True, text=True)
print("pip rc =", r.returncode)
if r.returncode != 0:
    print(r.stderr[-2000:])
for p in ("transformers", "torch", "datasets", "tokenizers"):
    try: print(f"  {p:<14} {pm.version(p)}")
    except Exception: print(f"  {p:<14} AUSENTE")
import torch
print("\\ncuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    pr = torch.cuda.get_device_properties(0)
    print("GPU:", pr.name, "|", round(pr.total_memory/2**30, 2), "GB",
          "| bf16 nativo:", (pr.major, pr.minor) >= (8, 0))
""")


# ══ 3. harness ═════════════════════════════════════════════════════════════

def harness(arm):
    core = open(os.path.join(HERE, "bench_core.py")).read()
    return md("""## 3. O harness

`bench_core.py`, o mesmo arquivo nos dois notebooks. Escrito aqui para o
notebook ser autossuficiente no Kaggle.
"""), code("CORE = r'''" + core.replace("'''", "\\x27\\x27\\x27") + "'''\n"
              "open('bench_core.py','w').write(CORE)\n"
              "import bench_core\n"
              "print('harness carregado:', len(CORE), 'bytes')\n")


# ══ 4. modelo ══════════════════════════════════════════════════════════════

def download(arm):
    return md("""## 4. Modelo

Qwen2.5-0.5B-Instruct: 1,0 GB em f32, cabe na T4 **em f32** — condição para o
gate de paridade e a PPL compararem motores sem a confusão do arredondamento
bf16.

Trocar por um Llama-3.2 exige `winnex-nano >= 0.1.35`, que passou a honrar
`rope_scaling`; antes disso o motor rodava um modelo diferente sem avisar.
"""), code("""import os
from huggingface_hub import snapshot_download
MODELO_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODELO = "/kaggle/working/modelo"
if not os.path.isdir(MODELO):
    snapshot_download(MODELO_ID, local_dir=MODELO,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])
arqs = [f for f in os.listdir(MODELO) if os.path.isfile(os.path.join(MODELO, f))]
print(sorted(arqs))
print("tamanho:", round(sum(os.path.getsize(os.path.join(MODELO,f)) for f in arqs)/2**30, 2), "GB")
""")


# ══ 5. baseline de memória ═════════════════════════════════════════════════

def memoria_base(arm):
    return md("""## 5. Memória: linha de base (ponto 1 de 3)

Antes de carregar o modelo. O heap anônimo é a métrica honesta para o winnex:
ele lê o checkpoint por mmap, e páginas de arquivo aparecem no RSS sem serem
memória alocada pelo processo.
"""), code("""import gc, bench_core
gc.collect()
rss0, anon0 = bench_core.smaps()
vram0 = bench_core.vram()
BASE = {"rss_gb": rss0/2**30, "anon_gb": anon0/2**30, "vram_gb": vram0/2**30}
print(f"baseline  RSS={BASE['rss_gb']:.3f} GB  anonimo={BASE['anon_gb']:.3f} GB  "
      f"VRAM={BASE['vram_gb']:.3f} GB")
print("(VRAM e do dispositivo, nao do processo)")
""")


# ══ 6. carga ═══════════════════════════════════════════════════════════════

def carga(arm):
    cls = "WinnexArm" if arm == "winnex-nano" else "HFArm"
    return md(f"""## 6. Carga do modelo (ponto 2 de 3)

`{cls}` com `device="gpu"`, `dtype="f32"`.

O `assert` não é decorativo: se o braço cair para CPU, o bloco de desempenho
mediria CPU e o relatório estaria errado sem parecer errado.
"""), code(f"""import time, bench_core
t0 = time.perf_counter()
arm = bench_core.{cls}(MODELO, device="gpu", dtype="f32").load_tokenizer()
t_load = time.perf_counter() - t0
rss1, anon1 = bench_core.smaps()
CARGA = {{"load_s": round(t_load, 2),
         "anon_delta_gb": round((anon1-anon0)/2**30, 3),
         "vram_delta_gb": round((bench_core.vram()-vram0)/2**30, 3)}}
print(f"carga: {{t_load:.2f}} s | modo: {{arm.device}} | dtype: {{arm.dtype}}")
print(f"  anonimo +{{CARGA['anon_delta_gb']:.3f}} GB | VRAM +{{CARGA['vram_delta_gb']:.3f}} GB")
print(f"  vocab={{arm.V}}  contexto_max={{arm.max_ctx}}")
assert arm.device != "cpu", "braco caiu para CPU — desempenho nao seria interpretavel"

# Prova de que os dois bracos tokenizam igual. Sem isto, uma diferenca de
# acuracia poderia ser diferenca de tokenizacao: ja aconteceu neste projeto
# (o tokenizer do HF inserindo o token de inicio, LAMBADA 0/300).
TOKF = bench_core.tok_fingerprint(arm)
print("tokenizer:", TOKF)
""")


# ══ 7. gate ════════════════════════════════════════════════════════════════

def paridade(arm, other):
    return md(f"""## 7. GATE: paridade de logits (f32)

Prompt de 50 tokens, 50 tokens gerados, argmax de cada um.

**Se este gate falhar, nada abaixo vale** — é por isso que ele vem antes da
qualidade: medir a acurácia de um motor que não reproduz o outro mede o motor,
não o modelo.

O gate tem **três estados**, e o terceiro é o que costuma ser esquecido:

- `passou` — 50/50 idênticos
- `falhou` — mostra o índice da primeira divergência
- `nao_verificado` — a referência do braço `{other}` não está aqui, ou o hash
  do prompt não bate

"não verificado" **não** é "passou". O JSON registra qual dos três ocorreu.

Para cruzar: rode o outro notebook e adicione o `referencia_paridade.json` dele
como *Add Data → Your Work → Output*, ou copie o arquivo para
`/kaggle/working/`.
"""), code("""import time, json, os, bench_core
PROMPT = ("The history of the Roman Empire is a long and complicated story "
          "that begins with the founding of the city")
prompt_ids = arm.encode(PROMPT, False)[:50]
print("prompt:", len(prompt_ids), "tokens")

t0 = time.perf_counter()
PAR = bench_core.task_parity(arm, prompt_ids, n_gen=50)
print(f"50 tokens em {time.perf_counter()-t0:.1f} s ->", PAR["tokens"][:12])
print("hash do prompt:", PAR["prompt_hash"])

CAND = ["/kaggle/working/referencia_paridade.json", "referencia_paridade.json"]
REF = None
for c in CAND:
    if os.path.exists(c):
        d = json.load(open(c))
        if d.get("arm") != arm.name:
            REF = d
            break
GATE = bench_core.compare_parity(PAR, REF)
print("\\nGATE:", json.dumps(GATE, indent=2))
if GATE["status"] == "falhou":
    raise AssertionError(f"GATE FALHOU: diverge no token {GATE['primeira_divergencia']} "
                         f"({GATE['identicos']}/{GATE['n']} identicos)")
json.dump(PAR, open("referencia_paridade.json", "w"), indent=1)
print("referencia gravada")
""")


# ══ 8. perplexidade ════════════════════════════════════════════════════════

def ppl(arm):
    return md("""## 8. Perplexidade de corpus (f32)

Wikitext-2 test, janelas de 512 com passo 256. `PPL = exp(Σnll/Σtok)`.

O IC é bootstrap sobre **tokens**, não sobre janelas: as janelas se sobrepõem em
50% e compartilham metade dos tokens, então tratá-las como amostras
independentes produz um intervalo que não se sustenta. A estatística é a razão
de somas, não a média de razões.
"""), code("""import bench_core
r_ppl = bench_core.task_ppl(arm, seq=512, stride=256, n_windows=40)
lo, hi = r_ppl["ppl_ci95"]
print(f"PPL = {r_ppl['ppl']:.6f}   IC95 = [{lo:.4f}, {hi:.4f}]")
print(f"n = {r_ppl['n_tokens']} tokens | {r_ppl['n_janelas']} janelas "
      f"seq={r_ppl['seq']} stride={r_ppl['stride']} | {r_ppl['segundos']} s")
print("dtype:", r_ppl["dtype"])
""")


# ══ 9. acurácia ════════════════════════════════════════════════════════════

def tarefas(arm):
    return md(f"""## 9. Acurácia zero-shot (f32)

MMLU e LAMBADA por **log-verossimilhança**: pontua cada alternativa pelos
log-probs da continuação e escolhe a maior. É o protocolo do MMLU original.

A geração greedy fica de fora: ela aceita "a primeira letra A–D que aparecer",
é sensível a formatação, e não é comparável com nada publicado.

Duas leituras do MMLU, na mesma passagem:
- `ll-raw` — soma crua dos log-probs. **É o protocolo clássico.**
- `ll-norm` — a mesma soma dividida pelo número de tokens da alternativa.
  Variante que favorece alternativas curtas; **não é o número publicado.**
  Reportar as duas evita escolher uma depois de ver o resultado.

A amostra é **canônica** (seed {SEED}, ordenada por conteúdo antes de
embaralhar), então o hash dos itens é idêntico nos dois notebooks. Se não for,
a comparação é entre tarefas diferentes, e o JSON mostra isso.

n={N_MMLU} dá IC de cerca de ±6 pontos numa acurácia perto de 35%. Diferenças
menores não são detectáveis.
"""), code(f"""import bench_core
r_mmlu_raw  = bench_core.task_mmlu(arm, n={N_MMLU}, seed={SEED}, normalize=False)
r_mmlu_norm = bench_core.task_mmlu(arm, n={N_MMLU}, seed={SEED}, normalize=True)
r_lambada   = bench_core.task_lambada(arm, n={N_LAMBADA}, seed={SEED})
for r in (r_mmlu_raw, r_mmlu_norm, r_lambada):
    lo, hi = r["acc_ci95"]
    print(f"  {{r['task']:<16}} {{r['acc']*100:5.1f}}%  n={{r['n']:<4}} "
          f"IC95=[{{lo*100:.1f}}, {{hi*100:.1f}}]  status={{r['status']}}")
    print(f"      itens_hash={{r.get('itens_hash')}}")
print("\\nZero-shot: nao comparar o valor absoluto com MMLU 5-shot da literatura.")
""")


# ══ 10. prefill ════════════════════════════════════════════════════════════

def desempenho(arm):
    cls = "WinnexArm" if arm == "winnex-nano" else "HFArm"
    return md("""## 10. Prefill — throughput e decomposição (bf16)

Recarrega em **bf16**, o dtype em que o modelo seria servido. O dtype vai em
cada resultado.

Mede TTFT e tokens/s em 512 / 1024 / 2048 tokens, e **decompõe** o custo:

```
TTFT(n) = custo_fixo + custo_por_token · n
```

A decomposição é o que isola o gargalo. Um intercepto alto é custo que não
depende do comprimento (setup, primeira transferência); uma inclinação alta é
custo por token. Os dois pedem correções diferentes, e a média de tok/s esconde
qual é qual. O `r2` do ajuste vai no resultado — se for baixo, o modelo linear
não descreve a curva e a decomposição não vale.
"""), code(f"""import gc, bench_core
del arm
gc.collect()
try:
    import torch; torch.cuda.empty_cache()
except Exception: pass

arm16 = bench_core.{cls}(MODELO, device="gpu", dtype="bf16").load_tokenizer()
print("modo:", arm16.device, "| dtype:", arm16.dtype)

r_prefill = bench_core.task_prefill(arm16, lengths=(512, 1024, 2048), iters=10, warmup=2)
for r in r_prefill:
    st = r.get("status")
    if st != "ok":
        print(f"  {{r['task']:<22}} {{r.get('n_tokens','')}} -> {{st}}: {{r.get('erro','')[:90]}}")
    elif r["task"] == "prefill/decomposicao":
        print(f"  decomposicao: fixo={{r['custo_fixo_ms']:.1f}} ms  "
              f"marginal={{r['custo_por_token_ms']:.4f}} ms/tok  r2={{r['r2']}}")
    else:
        print(f"  {{r['n_tokens']:>5}} tok: TTFT {{r['ttft_ms']:9.1f}} ms  "
              f"{{r['tok_per_s']:9.1f}} tok/s  P99 {{r['p99_ms']:.1f}}  ruido {{r['ruido_rel']}}")
""")


# ══ 11. decode ═════════════════════════════════════════════════════════════

def decode(arm):
    return md("""## 11. Decode — latência em três contextos (bf16)

Prompt de 50 tokens, 128 tokens gerados, **batch = 1**. Métrica: **ITL**
(inter-token latency, ms/token) e tokens/s.

Medido em **ctx = 128, 512 e 1024**, e é isso que isola o gargalo: se o ITL
cresce com o contexto, o custo está na atenção/KV cache; se é plano, está no
custo fixo por token (as projeções). Um único número de ITL não distingue os
dois casos, e os dois pedem correções diferentes.

O P99 importa mais que a média: é o que decide se um timeout derruba uma
requisição em 100.
"""), code("""import bench_core
r_decode = bench_core.task_decode(arm16, prompt_len=50, n_gen=128, iters=10,
                                  warmup=2, ctx_list=(128, 512, 1024))
for r in r_decode:
    if r.get("status") != "ok":
        print(f"  ctx={r.get('ctx')} -> {r['status']}: {r.get('erro','')[:100]}")
        continue
    print(f"  ctx={r['ctx']:>5}: ITL {r['itl_ms']:7.2f} ms/tok  "
          f"({r['tok_per_s']:7.2f} tok/s)  media {r['mean_ms']:.2f}  "
          f"P99 {r['p99_ms']:.2f}  ruido {r['ruido_rel']}")
print("\\nSe o ITL cresce com ctx, o gargalo e atencao/KV; se e plano, e o custo")
print("fixo por token das projecoes.")
""")


# ══ 12. memória pico ═══════════════════════════════════════════════════════

def memoria_pico(arm):
    return md("""## 12. Memória: pico durante a geração (ponto 3 de 3)

- **Heap anônimo** — RAM realmente alocada pelo processo. Ignora páginas de
  arquivo do mmap, que o SO pode descartar. É a métrica honesta para o winnex.
- **RSS** — residente total; mostra o impacto do mmap no sistema.
- **VRAM pico** — amostrada em thread durante a geração, incluindo KV cache.
  **Vem do `nvidia-smi`, que é do dispositivo**: inclui qualquer outro processo
  na GPU. O relatório diz isso em vez de apresentá-lo como consumo do motor.
"""), code("""import bench_core
with bench_core.MemTracker() as mt:
    ids = list(range(1000, 1000 + min(512, arm16.max_ctx - 40)))
    arm16.reset()
    lg = arm16.prefill(ids)
    cur = int(lg.argmax())
    for _ in range(64):
        lg = arm16.decode_step(cur)
        cur = int(lg.argmax())
    arm16.sync()
rss2, anon2 = bench_core.smaps()
PICO = {"anon_gb": anon2/2**30, "rss_gb": rss2/2**30,
        "vram_peak_gb": max(0.0, (mt.peak - BASE["vram_gb"]*2**30))/2**30}
print(f"heap anonimo  {PICO['anon_gb']:7.3f} GB  (baseline {BASE['anon_gb']:.3f})")
print(f"RSS           {PICO['rss_gb']:7.3f} GB")
print(f"VRAM pico     {PICO['vram_peak_gb']:7.3f} GB  (do dispositivo)")
print("anonimo e a metrica honesta: o mmap do checkpoint entra no RSS sem ser")
print("memoria alocada pelo processo.")
""")


# ══ 13. limpeza ════════════════════════════════════════════════════════════

def limpeza(arm):
    return md("""## 13. Limpeza e vazamento

Libera o motor, força coleta e mede de novo. O delta depois da limpeza é o
vazamento. **Deve ser ~0**: memória que sobrevive ao `del` é memória que um
servidor não recupera entre requisições.
"""), code("""import gc, time, bench_core
del arm16
gc.collect()
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
except Exception: pass
time.sleep(1)
rss3, anon3 = bench_core.smaps()
VAZ = {"anon_gb": (anon3-anon0)/2**30, "vram_gb": (bench_core.vram()-vram0)/2**30}
print(f"apos limpeza: anonimo {anon3/2**30:.3f} GB (baseline {anon0/2**30:.3f})")
print(f"vazamento anonimo: {VAZ['anon_gb']:+.3f} GB")
print(f"vazamento VRAM   : {VAZ['vram_gb']:+.3f} GB")
""")


# ══ 14. exportar ═══════════════════════════════════════════════════════════

def exportar(arm):
    tag = arm.replace("-", "_")
    return md("""## 14. Exportar

Um JSON por braço, com **todos** os números, o dtype de cada um, o `status`, e
as falhas — que são resultado, não algo a omitir.

No Kaggle ele fica em `/kaggle/working/` e aparece em *Output*.
"""), code(f"""import json, time, platform
RES = [r_ppl, r_mmlu_raw, r_mmlu_norm, r_lambada, *r_prefill, *r_decode]
saida = {{
    "arm": arm.name,
    "pacote": "winnex-nano {VERSAO}" if arm.name == "winnex-nano" else "transformers",
    "modelo": MODELO_ID,
    "maquina": platform.platform(),
    "python": platform.python_version(),
    "data": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "tokenizer": TOKF,
    "gate": GATE,
    "memoria": {{"baseline": BASE, "carga": CARGA, "pico": PICO, "vazamento": VAZ}},
    "resultados": RES,
}}
# Contagem explicita de falhas: o relatorio tem que mostrar o que nao funcionou.
falhas = [r for r in RES if r.get("status") not in (None, "ok")]
saida["resumo_falhas"] = {{"n": len(falhas),
    "itens": [{{"task": r["task"], "status": r.get("status"),
                "erro": r.get("erro", "")[:200]}} for r in falhas]}}
ARQ = "resultado_{tag}.json"
json.dump(saida, open(ARQ, "w"), indent=2, default=float)
print("gravado:", ARQ)
print("falhas:", len(falhas))
"""), md("""### 14.1 Tabela resumo

Uma linha por métrica, com valor, n, IC, dtype e status. É o que vai para o
relatório — a saída crua não carrega n nem IC.
"""), code("""import bench_core
print(bench_core.resumo(RES))
print("\\nGATE de paridade:", GATE["status"])
if GATE["status"] != "passou":
    print("  ->", GATE.get("motivo") or f"diverge em {GATE.get('primeira_divergencia')}")
""")


def montar(arm, other):
    cells = [header(arm, other)]
    cells += list(ambiente(arm))
    cells += list(instalar(arm))
    cells += list(harness(arm))
    cells += list(download(arm))
    cells += list(memoria_base(arm))
    cells += list(carga(arm))
    cells += list(paridade(arm, other))
    cells += list(ppl(arm))
    cells += list(tarefas(arm))
    cells += list(desempenho(arm))
    cells += list(decode(arm))
    cells += list(memoria_pico(arm))
    cells += list(limpeza(arm))
    cells += list(exportar(arm))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def slug(arm):
    return arm.replace("-", "_")


if __name__ == "__main__":
    for arm, other in (("winnex-nano", "transformers"),
                       ("transformers", "winnex-nano")):
        nb = montar(arm, other)
        p = os.path.join(HERE, f"bench_{slug(arm)}.ipynb")
        with open(p, "w") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"{p}  ({len(nb['cells'])} celulas)")
