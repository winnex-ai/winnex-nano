#!/usr/bin/env python3
"""bench_core.py — harness de medição compartilhado pelos dois notebooks.

UM harness para os dois braços. Se o `transformers` e o `winnex-nano` fossem
medidos por códigos diferentes, a diferença entre os números poderia ser a
diferença entre os códigos.

PRINCÍPIOS QUE ESTE ARQUIVO IMPÕE
---------------------------------
1. TODA métrica carrega (valor, n, IC) e o dtype. Número sem os três é saída
   inválida.
2. Os dois braços veem os MESMOS itens. Não "o mesmo n": os mesmos itens,
   identificados por um hash do conteúdo. Amostrar por índice posicional sobre
   uma lista embaralhada não é reprodutível entre execuções em máquinas
   diferentes, e foi o defeito da versão anterior.
3. Falha é resultado. Toda tarefa devolve `status` em {"ok","oom","erro"} e o
   relatório lista as falhas em vez de omiti-las.
4. Gargalo é isolado, não agregado. Prefill é decomposto em custo fixo e custo
   marginal por token; decode é medido em três comprimentos de contexto. É
   isso que mostra ONDE o motor perde, e é simétrico entre os braços.

MEDIÇÕES
--------
  paridade      f32, argmax de 50 tokens gerados. Gate.
  ppl           f32, wikitext-2, exp(Σnll/Σtok), IC bootstrap sobre tokens.
  mmlu/lambada  f32, log-verossimilhança, IC de Wilson. Zero-shot.
  prefill       bf16, TTFT e tokens/s em 512/1024/2048 + decomposição.
  decode        bf16, ITL em ctx 128/512/1024, batch 1.
  memoria       heap anônimo, RSS, VRAM (pico amostrado em thread).
"""
import gc
import hashlib
import math
import os
import statistics
import sys
import time

import numpy as np


# ══ estatística ═════════════════════════════════════════════════════════════

def wilson(k, n, z=1.96):
    """IC de Wilson para proporção. Correto para n pequeno e p perto de 0/1."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def boot_ratio_ci(num, den, n_boot=4000, alpha=0.05, seed=0, exponentiate=True):
    """IC percentil de Σnum/Σden, reamostrando as unidades independentes.

    A estatística da perplexidade é uma RAZÃO de somas. Reamostrar
    `exp(nll_janela/n_janela)` e tirar a média — o que o harness anterior fazia
    — bootstrapa a média de razões, que é outro estimador, sobre janelas
    sobrepostas que compartilham metade dos tokens.
    """
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    if num.size == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(n_boot, num.size))
    r = num[idx].sum(1) / den[idx].sum(1)
    if exponentiate:
        r = np.exp(r)
    return (float(np.quantile(r, alpha / 2)), float(np.quantile(r, 1 - alpha / 2)))


def pct(v, q):
    v = sorted(v)
    if not v:
        return None
    k = (len(v) - 1) * (q / 100.0)
    f = int(math.floor(k))
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def describe(times_ms):
    """Média, mediana, P99, IQR, e o ruído relativo (p99-mediana)/mediana.

    O ruído relativo é o que impede reportar como diferença entre motores algo
    que está dentro da dispersão da própria medição.
    """
    if not times_ms:
        return {}
    med = statistics.median(times_ms)
    p99 = pct(times_ms, 99)
    return {"n": len(times_ms),
            "mean_ms": round(statistics.fmean(times_ms), 3),
            "median_ms": round(med, 3),
            "p99_ms": round(p99, 3),
            "iqr_ms": round(pct(times_ms, 75) - pct(times_ms, 25), 3),
            "min_ms": round(min(times_ms), 3),
            "max_ms": round(max(times_ms), 3),
            "ruido_rel": round((p99 - med) / med, 4) if med else None}


def fit_linear(xs, ys):
    """Ajuste y = a + b·x por mínimos quadrados. Devolve (a, b, r2).

    Usado para decompor o prefill: `a` é o custo fixo (que não depende do
    comprimento) e `b` é o custo marginal por token. É a diferença entre "o
    motor é lento" e "o motor paga 300 ms antes de fazer qualquer coisa".
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 2:
        return (None, None, None)
    b, a = np.polyfit(xs, ys, 1)
    pred = a + b * xs
    ss_res = float(((ys - pred) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return (float(a), float(b), float(r2))


# ══ reprodutibilidade ══════════════════════════════════════════════════════

def fingerprint(texts):
    """SHA-256 do conteúdo, não da posição.

    Identifica o CONJUNTO de itens que um braço viu. Os dois notebooks têm que
    produzir o mesmo hash; se não produzirem, a comparação de acurácia entre
    eles é entre tarefas diferentes, não entre motores.
    """
    h = hashlib.sha256()
    for t in sorted(texts):
        h.update(t.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def canonical_sample(rows, n, seed, key):
    """Amostra determinística e independente da ordem do dataset.

    Ordena por `key` antes de embaralhar, para que duas execuções sobre
    revisões diferentes do dataset (mesmo conteúdo, ordem diferente) escolham
    os mesmos itens. A amostra anterior usava índice posicional sobre a ordem
    do dataset — reprodutível na mesma revisão, e só nela.
    """
    import random
    rows = sorted(rows, key=key)
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    return [rows[i] for i in idx[:n]]


def tok_fingerprint(arm):
    """Hash dos ids de um texto fixo. Prova que os dois braços tokenizam igual.

    Sem esta verificação, uma diferença de acurácia poderia ser uma diferença
    de tokenização, não de modelo. Já aconteceu neste projeto: o tokenizer do
    HF inseria o token de início e o LAMBADA deu 0/300.
    """
    texto = ("The quick brown fox jumps over the lazy dog. "
             "O rato roeu a roupa do rei de Roma. 0123456789")
    ids = arm.encode(texto, False)
    return {"n_tokens": len(ids), "hash": fingerprint([",".join(map(str, ids))]),
            "ids_head": ids[:8]}


# ══ memória ════════════════════════════════════════════════════════════════

def smaps():
    """(RSS, Anonymous) do processo, em bytes.

    Anônimo é a métrica honesta para o winnex: ele lê o checkpoint por mmap, e
    as páginas de arquivo aparecem no RSS sem serem memória alocada. Contar RSS
    pune o mmap duas vezes.
    """
    rss = anon = 0
    try:
        with open("/proc/self/smaps") as f:
            for line in f:
                if line.startswith("Rss:"):
                    rss += int(line.split()[1]) * 1024
                elif line.startswith("Anonymous:"):
                    anon += int(line.split()[1]) * 1024
    except OSError:
        pass
    return rss, anon


def vram():
    """VRAM usada, em bytes. ATENÇÃO: é do DISPOSITIVO, não do processo.

    Se outra coisa estiver usando a GPU, este número inclui. O notebook diz
    isso no relatório em vez de apresentá-lo como consumo do motor.
    """
    try:
        import subprocess
        o = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        lines = [l for l in o.stdout.strip().splitlines() if l.strip()]
        return sum(int(l) for l in lines) * 2**20
    except Exception:
        return 0


class MemTracker:
    """Amostra o pico de VRAM em thread, porque `nvidia-smi` é caro."""

    def __init__(self):
        self.peak = 0
        self.base = 0
        self._run = False
        self._t = None

    def __enter__(self):
        import threading
        self.base = vram()
        self.peak = self.base
        self._run = True

        def loop():
            while self._run:
                v = vram()
                if v > self.peak:
                    self.peak = v
                time.sleep(0.05)

        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._run = False
        if self._t:
            self._t.join(timeout=2)


# ══ os dois braços ═════════════════════════════════════════════════════════

class WinnexArm:
    name = "winnex-nano"

    def __init__(self, model_dir, device="gpu", dtype="f32"):
        import winnex_nano as wn
        self.wn = wn
        self.model_dir = model_dir
        self.dtype = dtype
        if dtype == "bf16":
            os.environ["WINNEX_GPU_DTYPE"] = "bf16"
        else:
            os.environ.pop("WINNEX_GPU_DTYPE", None)
        self.cfg = wn.load_config(os.path.join(model_dir, "config.json"))
        self.st = wn.Safetensors.open_dir(model_dir)
        self.eng = wn.ForwardEngine(self.cfg, self.st)
        self.device = "cpu"
        if device == "gpu":
            # enable_gpu() LANÇA quando não há dispositivo. Deixar propagar é o
            # comportamento correto: um benchmark que se diz de GPU não pode
            # continuar em CPU em silêncio.
            self.eng.enable_gpu(os.environ.get("WINNEX_OPENCL_LIB", ""))
            self.device = self.eng.weight_mode
        self.V = self.cfg.vocab_size
        self.max_ctx = int(self.eng.max_context)
        self._tk = None

    def load_tokenizer(self):
        from tokenizers import Tokenizer
        self._tk = Tokenizer.from_file(os.path.join(self.model_dir, "tokenizer.json"))
        return self

    def encode(self, text, add_special_tokens=False):
        return self._tk.encode(text, add_special_tokens=add_special_tokens).ids

    def logits_all(self, ids):
        H = np.ascontiguousarray(self.eng.embed_ids(list(ids)), dtype=np.float32)
        out = self.eng.forward(H, len(ids), True)
        return np.asarray(out, dtype=np.float64).reshape(len(ids), self.V)

    def reset(self):
        self.eng.reset_cache()

    def prefill(self, ids):
        return np.asarray(self.eng.prefill(list(ids)), dtype=np.float64)

    def decode_step(self, token_id):
        return np.asarray(self.eng.step(int(token_id)), dtype=np.float64)

    def sync(self):
        # `step`/`prefill` terminam com leitura bloqueante + clFinish, então o
        # relógio do host já inclui o trabalho de device. Não há o que esperar.
        pass

    def device_time_ms(self):
        """Tempo de KERNEL relatado pelo OpenCL desde o último enable, se houver.

        Serve para separar "o motor é lento" de "o dispositivo é lento e o host
        espera". Devolve None quando o braço não expõe a instrumentação.
        """
        try:
            return float(self.eng.profiling_attn_ms)
        except Exception:
            return None


class HFArm:
    name = "transformers"

    def __init__(self, model_dir, device="gpu", dtype="f32"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.model_dir = model_dir
        self.dtype = dtype
        self.dev = "cuda" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
        if device == "gpu" and self.dev != "cuda":
            raise RuntimeError("device='gpu' pedido, mas torch.cuda não está disponível")
        td = torch.float32 if dtype == "f32" else torch.bfloat16
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=td).to(self.dev).eval()
        self.device = self.dev
        self.V = self.m.config.vocab_size
        self.max_ctx = int(getattr(self.m.config, "max_position_embeddings", 4096))
        self._past = None

    def load_tokenizer(self):
        return self

    def encode(self, text, add_special_tokens=False):
        return self.tok(text, add_special_tokens=add_special_tokens).input_ids

    def logits_all(self, ids):
        t = self.torch
        with t.no_grad():
            o = self.m(t.tensor([list(ids)], device=self.dev))
        return o.logits[0].float().cpu().numpy().astype(np.float64)

    def reset(self):
        self._past = None

    def prefill(self, ids):
        t = self.torch
        with t.no_grad():
            o = self.m(t.tensor([list(ids)], device=self.dev), use_cache=True)
        self._past = o.past_key_values
        return o.logits[0, -1].float().cpu().numpy().astype(np.float64)

    def decode_step(self, token_id):
        t = self.torch
        with t.no_grad():
            o = self.m(t.tensor([[int(token_id)]], device=self.dev),
                       past_key_values=self._past, use_cache=True)
        self._past = o.past_key_values
        return o.logits[0, -1].float().cpu().numpy().astype(np.float64)

    def sync(self):
        if self.dev == "cuda":
            self.torch.cuda.synchronize()

    def device_time_ms(self):
        return None


# ══ tarefas ════════════════════════════════════════════════════════════════

def _logrow(lg):
    m = lg.max(1, keepdims=True)
    return (lg - m) - np.log(np.exp(lg - m).sum(1, keepdims=True))


def task_parity(arm, prompt_ids, n_gen=50):
    """Gate: argmax de cada token gerado. Devolve tokens e o hash do prompt."""
    arm.reset()
    lg = arm.prefill(prompt_ids)
    got, cur = [], int(np.argmax(lg))
    for _ in range(n_gen):
        got.append(cur)
        lg = arm.decode_step(cur)
        cur = int(np.argmax(lg))
    return {"tokens": got, "prompt_hash": fingerprint([",".join(map(str, prompt_ids))]),
            "prompt_len": len(prompt_ids), "dtype": arm.dtype}


def compare_parity(a, b):
    """Resultado do gate em três estados — nunca um silêncio.

    Devolve status em {"passou","falhou","nao_verificado"} e, quando falha, o
    primeiro índice onde divergiu. Um gate que "não se aplica" é indistinguível
    de um gate que passou, se não for dito.
    """
    if a is None or b is None:
        return {"status": "nao_verificado",
                "motivo": "referencia do outro braco ausente"}
    if a["prompt_hash"] != b["prompt_hash"]:
        return {"status": "nao_verificado",
                "motivo": "prompts diferentes (hash nao bate) — nao comparavel"}
    ta, tb = a["tokens"], b["tokens"]
    n = min(len(ta), len(tb))
    igual = sum(x == y for x, y in zip(ta, tb))
    div = next((i for i in range(n) if ta[i] != tb[i]), None)
    return {"status": "passou" if igual == n and len(ta) == len(tb) else "falhou",
            "n": n, "identicos": igual, "fracao": igual / n if n else 0.0,
            "primeira_divergencia": div}


def task_ppl(arm, seq=512, stride=256, n_windows=40):
    """exp(Σnll/Σtok) em janelas deslizantes, com IC bootstrap sobre tokens."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(r["text"].strip() for r in ds if r["text"].strip())
    ids = arm.encode(text, False)
    wins, i = [], 0
    while len(wins) < n_windows and i + seq <= len(ids):
        wins.append(ids[i:i + seq])
        i += stride
    nlls, counts = [], []
    t0 = time.perf_counter()
    for w in wins:
        ls = _logrow(arm.logits_all(w)[:-1])
        v = -ls[np.arange(len(w) - 1), np.asarray(w[1:])]
        nlls.append(v)
        counts.append(np.ones_like(v))
    dt = time.perf_counter() - t0
    nll = np.concatenate(nlls)
    cnt = np.concatenate(counts)
    lo, hi = boot_ratio_ci(nll, cnt)
    return {"task": "ppl/wikitext-2", "status": "ok", "dtype": arm.dtype,
            "ppl": float(math.exp(nll.sum() / cnt.sum())),
            "ppl_ci95": [lo, hi], "n_tokens": int(cnt.sum()),
            "n_janelas": len(wins), "seq": seq, "stride": stride,
            "segundos": round(dt, 1),
            "nota": "IC bootstrap sobre tokens; janelas tem passo 256 e se sobrepoem em 50%"}


def _mcq(arm, items, name):
    """As DUAS leituras do MMLU numa passagem só.

    `ll-raw` (soma crua, protocolo clássico) e `ll-norm` (média por token,
    variante) saem dos MESMOS scores por alternativa. Rodar a tarefa duas
    vezes para obter as duas lê a mesma coisa duas vezes: no braço do winnex
    isso custa 800 forwards e ~11 min no Kaggle, para produzir um segundo
    número que é uma divisão do primeiro.

    Devolve uma lista com os dois resultados, mais a contagem de itens que
    falharam — que é resultado, não algo a descartar.
    """
    hit_raw = hit_norm = n = erros = 0
    for it in items:
        q = it["question"].strip()
        raw, norm = [], []
        for ch in it["choices"]:
            prompt = arm.encode(q + "\nAnswer:", False)
            cont = arm.encode(" " + ch.strip(), False)
            ids = prompt + cont
            if len(ids) < 2 or len(ids) > arm.max_ctx - 2:
                raw.append(-1e30)
                norm.append(-1e30)
                continue
            ls = _logrow(arm.logits_all(ids)[:-1])
            s = sum(float(ls[len(prompt) - 1 + j, cont[j]]) for j in range(len(cont)))
            raw.append(s)
            norm.append(s / max(1, len(cont)))
        if not raw or len(raw) != len(it["choices"]):
            erros += 1
            continue
        hit_raw += int(int(np.argmax(raw)) == it["answer"])
        hit_norm += int(int(np.argmax(norm)) == it["answer"])
        n += 1
    lo, hi = wilson(hit_raw, n)
    lo2, hi2 = wilson(hit_norm, n)
    base = {"status": "ok" if n else "erro", "dtype": arm.dtype,
            "n": n, "itens_com_erro": erros}
    return [
        dict(base, task=f"{name}/ll-raw", acc=hit_raw / n if n else None,
             acc_ci95=[lo, hi], acertos=hit_raw,
             nota="log-verossimilhanca SOMADA, sem normalizar (protocolo MMLU "
                  "classico); zero-shot; IC de Wilson"),
        dict(base, task=f"{name}/ll-norm", acc=hit_norm / n if n else None,
             acc_ci95=[lo2, hi2], acertos=hit_norm,
             nota="mesmos scores divididos pelo numero de tokens da alternativa "
                  "(variante, NAO e o numero publicado); IC de Wilson"),
    ]


def load_rows(name, split="test", **kw):
    """Carrega um dataset tentando os identificadores VÁLIDOS antes do legado.

    `load_dataset("lambada", "plain_text")` funciona no `datasets` antigo e
    FALHA no novo, que exige `namespace/nome`: o erro foi

        HfUriError: Invalid HF URI 'hf://datasets/lambada@.../...'
        Repository id must be 'namespace/name', got 'lambada'

    Isso derrubou o notebook inteiro no Kaggle — LAMBADA não é opcional, e a
    exceção subiu até o fim da célula. Aqui a lista de candidatos é tentada em
    ordem e, se todas falharem, a tarefa devolve `status="erro"` com a causa em
    vez de matar o notebook: a PPL e as outras métricas continuam válidas.
    """
    from datasets import load_dataset
    tentativas = (
        ["EleutherAI/lambada_openai"] if name == "lambada" else []
    ) + (
        ["cais/mmlu"] if name == "mmlu" else []
    ) + (
        ["Salesforce/wikitext"] if name == "wikitext" else []
    )
    erros = []
    for cand in tentativas:
        for cfg in ((kw.get("config"), None) if kw.get("config") else (None,)):
            try:
                return load_dataset(cand, cfg, split=split) if cfg else \
                       load_dataset(cand, split=split)
            except Exception as e:
                erros.append(f"{cand}/{cfg}: {type(e).__name__}: {str(e)[:110]}")
    raise RuntimeError(" | ".join(erros))


def task_mmlu(arm, n=200, seed=0, rows=None):
    """MMLU zero-shot, amostra estável por conteúdo. Devolve lista de 2 resultados."""
    if rows is None:
        try:
            ds = load_rows("mmlu", config="all")
            rows = [{"question": ds[i]["question"], "choices": list(ds[i]["choices"]),
                     "answer": int(ds[i]["answer"])} for i in range(len(ds))]
            src = "cais/mmlu all/test"
        except Exception as e:
            msg = str(e)[:200]
            return [{"task": f"mmlu/{t}", "status": "erro", "dtype": arm.dtype,
                     "erro": msg, "n": 0} for t in ("ll-raw", "ll-norm")]
    else:
        src = "fornecido pelo chamador"
    items = canonical_sample(rows, n, seed, key=lambda r: r["question"])
    out = _mcq(arm, items, "mmlu")
    for r in out:
        r["itens_hash"] = fingerprint([i["question"] for i in items])
        r["fonte"] = f"{src}, amostra n={len(items)} seed={seed}"
    return out


def task_lambada(arm, n=200, seed=0, rows=None):
    """LAMBADA: prever a última palavra. Falha de dataset vira status, não exceção."""
    if rows is None:
        try:
            ds = load_rows("lambada", config="plain_text")
            rows = [{"text": ds[i]["text"]} for i in range(len(ds))]
        except Exception as e:
            return {"task": "lambada", "status": "erro", "dtype": arm.dtype,
                    "acc": None, "acc_ci95": [None, None], "acertos": 0, "n": 0,
                    "erro": str(e)[:200],
                    "nota": "dataset indisponivel — a tarefa NAO rodou, o que e "
                            "diferente de acuracia zero"}
    items = canonical_sample(rows, n, seed, key=lambda r: r["text"])
    hits, n_, skip = 0, 0, 0
    for it in items:
        t = it["text"].strip()
        if " " not in t:
            skip += 1
            continue
        prefix, target = t.rsplit(" ", 1)
        ip, itg = arm.encode(prefix, False), arm.encode(" " + target, False)
        if not itg:
            skip += 1
            continue
        ids = ip + itg
        if len(ids) > arm.max_ctx - 2:
            ids = ids[-(arm.max_ctx - 2):]
            if len(ids) <= len(itg):
                skip += 1
                continue
        lg = arm.logits_all(ids)
        hits += int(int(lg[len(ip) - 1].argmax()) == itg[0])
        n_ += 1
    lo, hi = wilson(hits, n_)
    return {"task": "lambada", "status": "ok" if n_ else "erro", "dtype": arm.dtype,
            "acc": hits / n_ if n_ else None, "acc_ci95": [lo, hi],
            "acertos": hits, "n": n_, "pulados": skip,
            "itens_hash": fingerprint([i["text"] for i in items]),
            "fonte": f"lambada plain_text, amostra n={len(items)} seed={seed}",
            "nota": "argmax exato do primeiro token do alvo; IC de Wilson"}


def _is_oom(msg):
    m = msg.lower()
    return any(k in m for k in ("out of memory", "oom", "cl_mem",
                                "allocate", "memoryerror", "cuda error"))


def task_prefill(arm, lengths=(512, 1024, 2048), iters=10, warmup=2):
    """TTFT e throughput por comprimento, mais a DECOMPOSIÇÃO do custo.

    A decomposição é o que isola o gargalo: ajusta TTFT(n) = fixo + marginal·n.
    Um intercepto alto significa custo que não depende do comprimento (setup,
    transferência inicial); uma inclinação alta significa custo por token. Os
    dois pedem correções diferentes, e a média de tok/s esconde qual é qual.
    """
    out, xs, ys = [], [], []
    for L in lengths:
        ids = list(range(1000, 1000 + min(L, arm.max_ctx - 8)))
        try:
            for _ in range(warmup):
                arm.reset()
                arm.prefill(ids)
                arm.sync()
            times = []
            for _ in range(iters):
                arm.reset()
                t0 = time.perf_counter()
                arm.prefill(ids)
                arm.sync()
                times.append((time.perf_counter() - t0) * 1e3)
            d = describe(times)
            d.update(task="prefill", status="ok", dtype=arm.dtype,
                     n_tokens=len(ids), iters=iters, warmup=warmup,
                     ttft_ms=d["median_ms"],
                     tok_per_s=round(len(ids) / (d["median_ms"] / 1e3), 1))
            out.append(d)
            xs.append(len(ids))
            ys.append(d["median_ms"])
        except (RuntimeError, MemoryError, Exception) as e:
            msg = str(e)[:200]
            out.append({"task": "prefill", "status": "oom" if _is_oom(msg) else "erro",
                        "dtype": arm.dtype, "n_tokens": L, "erro": msg})
            break
    if len(xs) >= 2:
        a, b, r2 = fit_linear(xs, ys)
        decomp = {"custo_fixo_ms": round(a, 2), "custo_por_token_ms": round(b, 4),
                  "r2": round(r2, 5) if r2 == r2 else None,
                  "nota": "ajuste TTFT(n) = fixo + marginal*n; r2 baixo = o modelo linear nao descreve"}
        out.append({"task": "prefill/decomposicao", "status": "ok",
                    "dtype": arm.dtype, **decomp})
    return out


def task_decode(arm, prompt_len=50, n_gen=128, iters=10, warmup=2,
                ctx_list=(128, 512, 1024)):
    """ITL em três comprimentos de contexto, batch 1.

    Medir em três contextos isola o gargalo: se o ITL cresce com o contexto, o
    custo está na atenção/KV; se é plano, está no custo fixo por token (as
    projeções). Um único número de ITL não distingue os dois.
    """
    out = []
    for ctx in ctx_list:
        ctx = min(ctx, arm.max_ctx - n_gen - 8)
        if ctx <= prompt_len:
            continue
        ids = list(range(1000, 1000 + prompt_len))
        try:
            for _ in range(warmup):
                arm.reset()
                lg = arm.prefill(ids)
                cur = int(np.argmax(lg))
                for _ in range(16):
                    cur = int(np.argmax(arm.decode_step(cur)))
                arm.sync()
            runs = []
            for _ in range(iters):
                arm.reset()
                # Preenche até o contexto alvo e decodifica os n_gen seguintes,
                # para que o ITL reflita o KV cache do tamanho declarado.
                lg = arm.prefill(list(range(1000, 1000 + ctx)))
                arm.sync()
                cur = int(np.argmax(lg))
                t0 = time.perf_counter()
                for _ in range(n_gen):
                    lg = arm.decode_step(cur)
                    cur = int(np.argmax(lg))
                arm.sync()
                runs.append((time.perf_counter() - t0) * 1e3 / n_gen)
            d = describe(runs)
            d.update(task="decode", status="ok", dtype=arm.dtype, ctx=ctx,
                     n_gen=n_gen, iters=iters, warmup=warmup, batch=1,
                     itl_ms=d["median_ms"],
                     tok_per_s=round(1000.0 / d["median_ms"], 2) if d["median_ms"] else None)
            out.append(d)
        except (RuntimeError, MemoryError, Exception) as e:
            msg = str(e)[:200]
            out.append({"task": "decode", "status": "oom" if _is_oom(msg) else "erro",
                        "dtype": arm.dtype, "ctx": ctx, "erro": msg})
            break
    return out


def resumo(resultados):
    """Tabela texto com (valor, n, IC) para cada resultado que os tenha."""
    linhas = [f"{'tarefa':<20}{'valor':>11}{'n':>8}  {'IC95':<22}{'dtype':>7} {'status'}"]
    for r in resultados:
        st = r.get("status", "?")
        dt = r.get("dtype", "")
        if st != "ok":
            linhas.append(f"{r['task']:<20}{'—':>11}{'—':>8}  {'—':<22}{dt:>7} {st}")
            continue
        if r.get("acc") is not None:
            lo, hi = r["acc_ci95"]
            linhas.append(f"{r['task']:<20}{r['acc']*100:>10.2f}%{r['n']:>8}  "
                          f"[{lo*100:.1f}, {hi*100:.1f}]{'':<6}{dt:>7} {st}")
        elif r.get("ppl") is not None:
            lo, hi = r["ppl_ci95"]
            linhas.append(f"{r['task']:<20}{r['ppl']:>11.4f}{r['n_tokens']:>8}  "
                          f"[{lo:.2f}, {hi:.2f}]{'':<6}{dt:>7} {st}")
        elif r.get("itl_ms") is not None:
            linhas.append(f"{r['task']} ctx={r['ctx']:<4}{r['itl_ms']:>10.2f}"
                          f"{r['n']:>8}  P99={r['p99_ms']:<14.1f}{dt:>7} {st}")
        elif r.get("ttft_ms") is not None:
            linhas.append(f"{r['task']} n={r['n_tokens']:<5}{r['ttft_ms']:>9.1f}"
                          f"{r['n']:>8}  P99={r['p99_ms']:<14.1f}{dt:>7} {st}")
        elif r.get("custo_fixo_ms") is not None:
            linhas.append(f"{r['task']:<20}{r['custo_fixo_ms']:>11.1f}{'':>8}  "
                          f"marginal {r['custo_por_token_ms']:.4f} ms/tok{'':<1}{dt:>7} {st}")
        else:
            linhas.append(f"{r['task']:<20}{'—':>11}{'—':>8}  {'—':<22}{dt:>7} {st}")
    return "\n".join(linhas)
