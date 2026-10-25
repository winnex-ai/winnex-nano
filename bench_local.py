#!/usr/bin/env python3
"""Benchmark local — o que mede, com que n, e o que NÃO mede.

Escrito a partir da revisão do `eval_nano.py`: corrige o IC da perplexidade
(bootstrap sobre janelas sobrepostas -> sobre tokens), mede MMLU pelo protocolo
clássico (log-verossimilhança das 4 alternativas, normalizada por token) além do
antigo MMLU por geração, e recusa rodar sem passar o gate de paridade.

REGRA: todo número sai com (valor, n, IC). Um número sem esses três campos é
saída inválida, não resultado.

Protocolos:
  PPL       wikitext-2 test, janela deslizante, PPL = exp(Σnll/Σtok).
            IC por bootstrap sobre TOKENS (não sobre janelas).
  LAMBADA   prever a última palavra; argmax exato do primeiro token do alvo.
            IC de Wilson.
  MMLU-LL   log-verossimilhança média por token de cada uma das 4 opções;
            escolhe a maior. É o protocolo do MMLU original (sem few-shot).
  MMLU-GEN  geração greedy, aceita a letra se aparecer na saída. Métrica
            separada, NÃO comparável com a literatura — fica como diagnóstico.
  ARC-LL    mesmo protocolo do MMLU-LL, em ARC-Challenge.

Uso:
  python bench_local.py --model DIR --which winnex --arm gpu --out r.json
  python bench_local.py --model DIR --which hf     --arm gpu --out r.json
"""
import argparse, json, math, os, sys, random, time
import numpy as np


# ── estatística ──────────────────────────────────────────────────────────────

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def boot_mean_ci(x, n_boot=4000, alpha=0.05, seed=0):
    """IC percentil da média, reamostrando as unidades INDEPENDENTES de `x`."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    m = x[idx].mean(axis=1)
    return (float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2)))


def boot_ratio_ci(num, den, n_boot=4000, alpha=0.05, seed=0, exponentiate=False):
    """IC percentil de Σnum/Σden reamostrando as unidades em pares.

    É o caso da perplexidade: a estatística é uma RAZÃO de somas, não a média
    de razões. Reamostrar `per_win = exp(nll_win/n_win)` e tirar a média mede
    uma coisa diferente da que se reporta — foi o defeito do harness anterior.

    `exponentiate=True` devolve o IC de exp(razão), que é o IC da PERPLEXIDADE;
    sem isso o retorno é o IC da entropia cruzada em nats. O ponto e o IC têm
    que estar na mesma unidade — a primeira versão devolvia exp() do ponto e
    nats no intervalo.
    """
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    if num.size == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(n_boot, num.size))
    r = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    if exponentiate:
        r = np.exp(r)
    return (float(np.quantile(r, alpha / 2)), float(np.quantile(r, 1 - alpha / 2)))


# ── dados ────────────────────────────────────────────────────────────────────

def wikitext_windows(tok, seq_len, n_windows, stride=None):
    from datasets import load_dataset
    ds = None
    for name, cfg in (("Salesforce/wikitext", "wikitext-103-raw-v1"),
                      ("Salesforce/wikitext", "wikitext-2-raw-v1")):
        try:
            ds = load_dataset(name, cfg, split="test")
            src = f"{name}/{cfg}"
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError("wikitext indisponivel")
    text = "\n\n".join(r["text"].strip() for r in ds if r["text"].strip())
    ids = tok(text, add_special_tokens=False).input_ids
    stride = stride or seq_len
    out, i = [], 0
    while len(out) < n_windows and i + seq_len <= len(ids):
        out.append(ids[i:i + seq_len])
        i += stride
    return out, src, len(ids)


def lambada_items(n, seed=0):
    from datasets import load_dataset
    for name, cfg in (("lambada", "plain_text"), ("EleutherAI/lambada_openai", None)):
        try:
            kw = {"split": "test"}
            if cfg:
                kw["name"] = cfg
            ds = load_dataset(name, **kw)
            idx = list(range(len(ds)))
            random.Random(seed).shuffle(idx)
            return [ds[i]["text"] for i in idx[:n]], name
        except Exception:
            continue
    raise RuntimeError("lambada indisponivel")


def mmlu_items(n, seed=0):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = [{"question": ds[i]["question"], "choices": list(ds[i]["choices"]),
            "answer": int(ds[i]["answer"]), "subject": ds[i]["subject"]}
           for i in idx[:n]]
    return out, len(ds)


def arc_items(n, seed=0):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    k2i = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
    out = []
    for i in idx:
        r = ds[i]
        texts = list(r["choices"]["text"])
        try:
            ans = k2i[str(r["answerKey"]).strip().upper()]
        except KeyError:
            continue
        if len(texts) != 4 or ans >= 4:
            continue
        out.append({"question": r["question"], "choices": texts, "answer": ans})
        if len(out) >= n:
            break
    return out, len(ds)


# ── motores ──────────────────────────────────────────────────────────────────

def log_softmax_rows(lg):
    m = lg.max(axis=1, keepdims=True)
    z = lg - m
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def tok_ids(tok, text, add_special_tokens=False):
    """Tokeniza SEM especiais, de forma uniforme entre os dois motores.

    O `AutoTokenizer` do HF (backend `tokenizers`) insere automaticamente o
    token de início de sequência (`<|begin_of_text|>` no Llama, `<|eot_id|>`/
    BOS variando por checkpoint) em cada chamada. O harness concatena
    `prefixo + alvo` como DUAS chamadas, então o BOS do alvo caía no MEIO da
    sequência e o modelo era alimentado com `... is <|begin_of_text|> Paris`.

    Isso zerou o LAMBADA no braço HF: 0/300, com IC de Wilson [0,000; 0,013].
    Não era o modelo nem o motor — era o harness inserindo um token que o
    modelo nunca vê nessa posição. Medido antes e depois da correção.
    """
    return tok(text, add_special_tokens=add_special_tokens).input_ids


class _TokWrap:
    """Adaptador mínimo do tokenizers puro para a interface usada aqui.

    Existe para que o caminho do winnex não importe `transformers`/`torch`. Só
    os métodos que o harness usa: `.input_ids` (via `encode`), `.eos_token_id`,
    `.decode`. `encode` NÃO adiciona tokens especiais, igual ao
    `tok(text).input_ids` sem `add_special_tokens` que o harness usava.
    """

    def __init__(self, tk):
        self.tk = tk
        self.eos_token_id = tk.token_to_id("<|eot_id|>")
        if self.eos_token_id is None:
            self.eos_token_id = tk.token_to_id("</s>")
        if self.eos_token_id is None:
            self.eos_token_id = tk.token_to_id("<|endoftext|>")

    def encode(self, text, add_special_tokens=False):
        class _R:
            pass
        r = _R()
        r.input_ids = self.tk.encode(text, add_special_tokens=add_special_tokens).ids
        return r

    def __call__(self, text, add_special_tokens=False):
        return self.encode(text, add_special_tokens)

    def decode(self, ids, skip_special_tokens=True):
        return self.tk.decode(ids, skip_special_tokens=skip_special_tokens)


class WinnexScorer:
    def __init__(self, model_dir, arm, opencl_lib=""):
        # `transformers` NÃO é importado aqui. Ele traz `torch`, que abre o
        # próprio pool de threads no processo do winnex e contamina a medição:
        # medido, um benchmark de PPL que devia levar ~260 s passou de 15 min
        # com o processo a 265% de CPU e a GPU a 0%. O winnex só precisa do
        # `tokenizers` (Rust), que é o mesmo vocabulário do checkpoint e não
        # traz torch.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import winnex_nano as wn
        from tokenizers import Tokenizer
        self.wn = wn
        self.tok = _TokWrap(Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json")))
        self.cfg = wn.load_config(os.path.join(model_dir, "config.json"))
        self.st = wn.Safetensors.open_dir(model_dir)
        t0 = time.perf_counter()
        self.eng = wn.ForwardEngine(self.cfg, self.st)
        self.load_s = time.perf_counter() - t0
        self.mode = self.eng.weight_mode
        self.gpu_error = None
        if arm != "cpu":
            t0 = time.perf_counter()
            try:
                self.eng.enable_gpu(opencl_lib)
                self.mode = self.eng.weight_mode
            except Exception as e:
                self.gpu_error = f"{type(e).__name__}: {e}"
            self.setup_s = time.perf_counter() - t0
        self.V = self.cfg.vocab_size
        self.max_ctx = self.eng.max_context
        self.n_heads = self.cfg.num_attention_heads

    def logits(self, ids):
        H = np.ascontiguousarray(self.eng.embed_ids(ids), dtype=np.float32)
        out = self.eng.forward(H, len(ids), True)
        return np.asarray(out, dtype=np.float64).reshape(len(ids), self.V)

    def generate(self, text, max_new=12):
        ids = self.tok(text).input_ids
        if len(ids) > self.max_ctx - max_new - 2:
            ids = ids[-(self.max_ctx - max_new - 2):]
        lg = self.eng.prefill(ids)
        out = []
        for _ in range(max_new):
            nid = int(np.argmax(np.asarray(lg, dtype=np.float64)))
            out.append(nid)
            if self.tok.eos_token_id is not None and nid == self.tok.eos_token_id:
                break
            lg = self.eng.step(nid)
        return self.tok.decode(out, skip_special_tokens=True)


class HFScorer:
    def __init__(self, model_dir, arm):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        # `arm` decide o dispositivo; não há parâmetro `device` separado. A
        # versão anterior tinha `device="cpu"` como segundo argumento
        # posicional e `arm` como terceiro, então `HFScorer(dir, "gpu")`
        # interpretava "gpu" como NOME DE DISPOSITIVO, caía no default `cpu`,
        # e o braço "HF/gpu" media CPU — o mesmo erro que o harness existe para
        # evitar. Medido: 8,2 s/item de MMLU em vez de ~0,3 s.
        self.device = "cpu" if arm == "cpu" else "cuda"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("arm=gpu pedido, mas torch.cuda.is_available() é False")
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        t0 = time.perf_counter()
        self.m = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=dtype).to(self.device).eval()
        self.load_s = time.perf_counter() - t0
        self.mode = self.device
        self.gpu_error = None
        self.V = self.m.config.vocab_size
        self.max_ctx = getattr(self.m.config, "max_position_embeddings", 4096)

    def logits(self, ids):
        t = self.torch
        with t.no_grad():
            o = self.m(t.tensor([ids], device=self.device))
        return o.logits[0].float().cpu().numpy().astype(np.float64)

    def generate(self, text, max_new=12):
        t = self.torch
        ids = self.tok(text).input_ids
        if len(ids) > self.max_ctx - max_new - 2:
            ids = ids[-(self.max_ctx - max_new - 2):]
        with t.no_grad():
            o = self.m.generate(t.tensor([ids], device=self.device),
                                max_new_tokens=max_new, do_sample=False,
                                pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(o[0][len(ids):], skip_special_tokens=True)


# ── tarefas ──────────────────────────────────────────────────────────────────

def t_perplexity(sc, windows, label):
    nlls, toks, t0 = [], [], time.perf_counter()
    for ids in windows:
        if len(ids) < 8:
            continue
        lg = sc.logits(ids)
        ls = log_softmax_rows(lg[:-1])
        v = -ls[np.arange(len(ids) - 1), np.asarray(ids[1:])]
        nlls.append(v)
        toks.append(np.ones_like(v))
    if not nlls:
        return {"task": "ppl", "arm": label, "erro": "sem janelas"}
    nll = np.concatenate(nlls)
    tot = np.concatenate(toks)
    ppl = float(math.exp(nll.sum() / tot.sum()))
    lo, hi = boot_ratio_ci(nll, tot, exponentiate=True)
    dt = time.perf_counter() - t0
    return {"task": "ppl/wikitext", "arm": label, "ppl": ppl,
            "ppl_ci95": [lo, hi], "n_tokens": int(tot.sum()),
            "n_janelas": len(windows), "segundos": round(dt, 1),
            "nota": "IC bootstrap sobre tokens; estatistica exp(Sum nll/Sum n)"}


def t_lambada(sc, items, label):
    hits, n, skipped = 0, 0, 0
    t0 = time.perf_counter()
    for text in items:
        text = text.strip()
        if " " not in text:
            skipped += 1
            continue
        prefix, target = text.rsplit(" ", 1)
        ip = tok_ids(sc.tok, prefix)
        it = tok_ids(sc.tok, " " + target)
        if not it:
            skipped += 1
            continue
        ids = ip + it
        if len(ids) > sc.max_ctx - 2:
            ids = ids[-(sc.max_ctx - 2):]
            if len(ids) <= len(it):
                skipped += 1
                continue
        lg = sc.logits(ids)
        hits += int(int(lg[len(ip) - 1].argmax()) == it[0])
        n += 1
    lo, hi = wilson(hits, n)
    return {"task": "lambada", "arm": label, "acc": hits / n if n else None,
            "acc_ci95": [lo, hi], "acertos": hits, "n": n, "pulados": skipped,
            "segundos": round(time.perf_counter() - t0, 1),
            "nota": "argmax exato do primeiro token do alvo; IC de Wilson"}


def t_mcq_ll(sc, items, label, name):
    """Log-verossimilhança das alternativas — o protocolo MMLU, em DUAS leituras.

    Uma passagem só, os dois números: `ll_raw` (soma crua dos log-probs da
    continuação, que é o PROTOCOLO CLÁSSICO do MMLU no Harness) e `ll_norm`
    (média por token, variante que favorece alternativas curtas). Escolher uma
    e chamá-la de "MMLU" seria escolher o número depois de ver o resultado;
    as duas saem com n e IC, e a diferença entre elas fica visível.

    Uma passagem e não duas porque o custo domina: cada item custa 4 forwards
    completos, e refazer tudo para a outra normalização dobrava o tempo sem
    acrescentar informação.

    O prompt termina em "Answer:" e a alternativa é tokenizada COM o espaço
    inicial, que é o formato que o tokenizer do checkpoint espera. O mesmo
    texto é usado nos dois motores: a comparação é de motor, não de prompt.
    """
    hits_raw = hits_norm = n = 0
    t0 = time.perf_counter()
    for it in items:
        q = it["question"].strip()
        raw, norm = [], []
        for ch in it["choices"]:
            prompt = tok_ids(sc.tok, q + "\nAnswer:")
            ch_ids = tok_ids(sc.tok, " " + ch.strip())
            ids = prompt + ch_ids
            if len(ids) < 2:
                raw.append(-1e30)
                norm.append(-1e30)
                continue
            lg = sc.logits(ids)
            ls = log_softmax_rows(lg[:-1])
            start = len(prompt) - 1
            s = sum(float(ls[start + j, ch_ids[j]]) for j in range(len(ch_ids)))
            raw.append(s)
            norm.append(s / max(1, len(ch_ids)))
        if not raw:
            continue
        hits_raw += int(int(np.argmax(raw)) == it["answer"])
        hits_norm += int(int(np.argmax(norm)) == it["answer"])
        n += 1
    lo, hi = wilson(hits_raw, n)
    lo2, hi2 = wilson(hits_norm, n)
    dt = round(time.perf_counter() - t0, 1)
    base = {"arm": label, "n": n, "segundos": dt}
    return [
        dict(base, task=f"{name}/ll-raw", acc=hits_raw / n if n else None,
             acc_ci95=[lo, hi], acertos=hits_raw,
             nota="log-verossimilhanca SOMADA da continuacao -- protocolo MMLU "
                  "classico, sem normalizar por token; sem few-shot; IC de Wilson"),
        dict(base, task=f"{name}/ll-norm", acc=hits_norm / n if n else None,
             acc_ci95=[lo2, hi2], acertos=hits_norm,
             nota="mesmos scores divididos pelo numero de tokens da continuacao "
                  "-- variante, NAO e o numero publicado do MMLU; IC de Wilson"),
    ]


def t_mcq_gen(sc, items, label, name, max_new=8):
    letters = "ABCD"
    hits, n = 0, 0
    t0 = time.perf_counter()
    for it in items:
        q = it["question"].strip()
        opts = "\n".join(f"{letters[i]}. {c.strip()}" for i, c in enumerate(it["choices"]))
        out = sc.generate(f"Question: {q}\n{opts}\nAnswer with the letter only.\nAnswer:",
                          max_new_tokens=max_new)
        pred = next((letters.index(ch) for ch in out if ch in letters), None)
        hits += int(pred == it["answer"])
        n += 1
    lo, hi = wilson(hits, n)
    return {"task": f"{name}/gen", "arm": label, "acc": hits / n if n else None,
            "acc_ci95": [lo, hi], "acertos": hits, "n": n,
            "segundos": round(time.perf_counter() - t0, 1),
            "nota": "geracao greedy; DIAGNOSTICO, nao comparavel com a literatura"}


# ── driver ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--which", choices=["winnex", "hf"], required=True)
    ap.add_argument("--arm", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--tasks", nargs="+",
                    default=["ppl", "lambada", "mmlu_ll", "mmlu_gen", "arc_ll"])
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--ppl-windows", type=int, default=40)
    ap.add_argument("--ppl-stride", type=int, default=0,
                    help="0 = passo = seq_len (janelas disjuntas)")
    ap.add_argument("--lambada-n", type=int, default=300)
    ap.add_argument("--mmlu-n", type=int, default=300)
    ap.add_argument("--arc-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opencl", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sc = (WinnexScorer(a.model, a.arm, a.opencl) if a.which == "winnex"
          else HFScorer(a.model, a.arm))
    label = f"{a.which}/{a.arm}"
    res = {"motor": a.which, "arm": a.arm,
           "modelo": os.path.basename(a.model.rstrip("/")),
           "modo": sc.mode, "gpu_error": getattr(sc, "gpu_error", None),
           "vocab": sc.V, "max_ctx": sc.max_ctx, "seed": a.seed,
           "resultados": []}

    def add(r):
        res["resultados"].append(r)
        print(json.dumps(r, indent=1), flush=True)

    if "ppl" in a.tasks:
        wins, src, ntok = wikitext_windows(sc.tok, a.seq_len, a.ppl_windows,
                                           a.ppl_stride or None)
        res["ppl_fonte"] = src
        add(t_perplexity(sc, wins, label))
    if "lambada" in a.tasks:
        items, src = lambada_items(a.lambada_n, a.seed)
        res["lambada_fonte"] = src
        add(t_lambada(sc, items, label))
    if "mmlu_ll" in a.tasks or "mmlu_gen" in a.tasks:
        items, total = mmlu_items(a.mmlu_n, a.seed)
        res["mmlu_fonte"] = f"cais/mmlu all/test ({total} itens), amostra n={len(items)}"
        if "mmlu_ll" in a.tasks:
            for r in t_mcq_ll(sc, items, label, "mmlu"):
                add(r)
        if "mmlu_gen" in a.tasks:
            add(t_mcq_gen(sc, items, label, "mmlu"))
    if "arc_ll" in a.tasks:
        items, total = arc_items(a.arc_n, a.seed)
        res["arc_fonte"] = f"ai2_arc ARC-Challenge/test ({total}), n={len(items)}"
        for r in t_mcq_ll(sc, items, label, "arc"):
            add(r)

    print("RESULT " + json.dumps({k: v for k, v in res.items() if k != "resultados"}),
          flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # GUARDA DE CONTAMINAÇÃO. Este host tem um `.pth` da NVIDIA
    # (`_cuda_bindings_redirector.pth`) que importa `torch` no STARTUP de
    # qualquer processo Python, inclusive no braço do winnex. Medido: com
    # torch carregado, o PPL do Llama levava >15 min em vez de ~260 s, com o
    # processo a 265% de CPU e a GPU a 0% -- o pool de threads do torch
    # competia com o do winnex. O braço do winnex só precisa de `numpy`,
    # `tokenizers` e do módulo `_winnex_nano`, nenhum dos quais precisa de
    # torch. `python -S` desliga os .pth; o launcher passa os caminhos à mão.
    if os.environ.get("WN_BENCH_NO_SITE") == "1" and "torch" in sys.modules:
        sys.exit("bench_local: torch foi carregado neste processo (braço winnex). "
                 "Rode com `python3 -S` — ver o cabeçalho de bench_local.py.")
    main()
