#!/usr/bin/env python3
"""bench_core.py - harness compartilhado pelos dois notebooks de benchmark.

Um unico harness para os dois bracos. Se o transformers e o winnex-nano fossem
medidos por codigos diferentes, a diferenca entre os numeros poderia ser a
diferenca entre os codigos.

Mede:
  PARIDADE (gate f32)  prompt de 50 tokens, argmax de cada token gerado.
  PPL (f32)            exp(Sum nll/Sum tok), wikitext-2, janela 512 stride 256.
  ACURACIA (f32)       MMLU e LAMBADA por log-verossimilhanca, IC de Wilson.
  PREFILL (bf16)       512/1024/2048 tokens: TTFT e tokens/s.
  DECODE (bf16)        prompt 50, 128 tokens, batch 1: ITL ms/token.
  MEMORIA              3 pontos: apos import, apos carga, pico na geracao.
"""
import gc
import math
import os
import statistics
import sys
import time

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def boot_ratio_ci(num, den, n_boot=4000, alpha=0.05, seed=0, exponentiate=True):
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


def pct(values, q):
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * (q / 100.0)
    f = int(math.floor(k))
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def describe(times_ms):
    if not times_ms:
        return {}
    return {"n": len(times_ms),
            "mean_ms": round(statistics.fmean(times_ms), 3),
            "median_ms": round(statistics.median(times_ms), 3),
            "p99_ms": round(pct(times_ms, 99), 3),
            "min_ms": round(min(times_ms), 3),
            "max_ms": round(max(times_ms), 3)}


def smaps():
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
            lib = os.environ.get("WINNEX_OPENCL_LIB", "")
            self.eng.enable_gpu(lib)
            self.device = self.eng.weight_mode
        self.V = self.cfg.vocab_size
        self.max_ctx = int(self.eng.max_context)

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
        pass


class HFArm:
    name = "transformers"

    def __init__(self, model_dir, device="gpu", dtype="f32"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.model_dir = model_dir
        self.dtype = dtype
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        td = torch.float32 if dtype == "f32" else torch.bfloat16
        self.dev = "cuda" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
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


def task_parity(arm, prompt_ids, n_gen=50):
    arm.reset()
    lg = arm.prefill(prompt_ids)
    got = []
    cur = int(np.argmax(lg))
    for _ in range(n_gen):
        got.append(cur)
        lg = arm.decode_step(cur)
        cur = int(np.argmax(lg))
    return got


def task_ppl(arm, seq=512, stride=256, n_windows=40):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(r["text"].strip() for r in ds if r["text"].strip())
    ids = arm.encode(text, False)
    wins, i = [], 0
    while len(wins) < n_windows and i + seq <= len(ids):
        wins.append(ids[i:i + seq])
        i += stride
    nlls, counts = [], []
    for w in wins:
        lg = arm.logits_all(w)
        m = lg[:-1].max(1, keepdims=True)
        ls = (lg[:-1] - m) - np.log(np.exp(lg[:-1] - m).sum(1, keepdims=True))
        v = -ls[np.arange(len(w) - 1), np.asarray(w[1:])]
        nlls.append(v)
        counts.append(np.ones_like(v))
    nll = np.concatenate(nlls)
    cnt = np.concatenate(counts)
    lo, hi = boot_ratio_ci(nll, cnt)
    return {"task": "ppl/wikitext-2", "dtype": arm.dtype,
            "ppl": float(math.exp(nll.sum() / cnt.sum())),
            "ppl_ci95": [lo, hi], "n_tokens": int(cnt.sum()),
            "n_janelas": len(wins), "seq": seq, "stride": stride}


def _mcq_ll(arm, items, name, normalize=True):
    hits, n = 0, 0
    for it in items:
        q = it["question"].strip()
        sc = []
        for ch in it["choices"]:
            prompt = arm.encode(q + "\nAnswer:", False)
            cont = arm.encode(" " + ch.strip(), False)
            ids = prompt + cont
            if len(ids) < 2 or len(ids) > arm.max_ctx - 2:
                sc.append(-1e30)
                continue
            lg = arm.logits_all(ids)
            m = lg[:-1].max(1, keepdims=True)
            ls = (lg[:-1] - m) - np.log(np.exp(lg[:-1] - m).sum(1, keepdims=True))
            s = sum(float(ls[len(prompt) - 1 + j, cont[j]]) for j in range(len(cont)))
            sc.append(s / max(1, len(cont)) if normalize else s)
        if not sc:
            continue
        hits += int(int(np.argmax(sc)) == it["answer"])
        n += 1
    lo, hi = wilson(hits, n)
    return {"task": "{}/{}".format(name, "ll-norm" if normalize else "ll-raw"),
            "dtype": arm.dtype, "acc": hits / n if n else None,
            "acc_ci95": [lo, hi], "acertos": hits, "n": n}


def task_mmlu(arm, n=150, seed=0, normalize=True):
    from datasets import load_dataset
    import random
    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    items = [{"question": ds[i]["question"], "choices": list(ds[i]["choices"]),
              "answer": int(ds[i]["answer"])} for i in idx[:n]]
    return _mcq_ll(arm, items, "mmlu", normalize)


def task_lambada(arm, n=150, seed=0):
    from datasets import load_dataset
    import random
    ds = load_dataset("lambada", "plain_text", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    texts = [ds[i]["text"] for i in idx[:n]]
    hits, nn, skipped = 0, 0, 0
    for t in texts:
        t = t.strip()
        if " " not in t:
            skipped += 1
            continue
        prefix, target = t.rsplit(" ", 1)
        ip = arm.encode(prefix, False)
        it = arm.encode(" " + target, False)
        if not it:
            skipped += 1
            continue
        ids = ip + it
        if len(ids) > arm.max_ctx - 2:
            ids = ids[-(arm.max_ctx - 2):]
            if len(ids) <= len(it):
                skipped += 1
                continue
        lg = arm.logits_all(ids)
        hits += int(int(lg[len(ip) - 1].argmax()) == it[0])
        nn += 1
    lo, hi = wilson(hits, nn)
    return {"task": "lambada", "dtype": arm.dtype,
            "acc": hits / nn if nn else None, "acc_ci95": [lo, hi],
            "acertos": hits, "n": nn, "pulados": skipped}


def _oom(msg):
    return any(k in msg.lower() for k in
               ("out of memory", "oom", "cl_mem", "allocate", "memoryerror"))


def task_prefill(arm, lengths=(512, 1024, 2048), iters=10, warmup=2):
    out = []
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
            d.update(task="prefill", dtype=arm.dtype, n_tokens=len(ids),
                     ttft_ms=d["median_ms"],
                     tok_per_s=round(len(ids) / (d["median_ms"] / 1e3), 1))
            out.append(d)
        except (RuntimeError, MemoryError) as e:
            m = str(e)[:140]
            out.append({"task": "prefill", "dtype": arm.dtype, "n_tokens": L,
                        "status": "OOM" if _oom(m) else "ERRO", "erro": m})
            break
    return out


def task_decode(arm, prompt_len=50, n_gen=128, iters=10, warmup=2):
    ids = list(range(1000, 1000 + min(prompt_len, arm.max_ctx - n_gen - 8)))
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
            lg = arm.prefill(ids)
            arm.sync()
            cur = int(np.argmax(lg))
            t0 = time.perf_counter()
            for _ in range(n_gen):
                lg = arm.decode_step(cur)
                cur = int(np.argmax(lg))
            arm.sync()
            runs.append((time.perf_counter() - t0) * 1e3 / n_gen)
        d = describe(runs)
        d.update(task="decode", dtype=arm.dtype, prompt_len=len(ids), n_gen=n_gen,
                 batch=1, itl_ms=d["median_ms"],
                 tok_per_s=round(1000.0 / d["median_ms"], 2) if d["median_ms"] else None)
        return d
    except (RuntimeError, MemoryError) as e:
        m = str(e)[:140]
        return {"task": "decode", "dtype": arm.dtype,
                "status": "OOM" if _oom(m) else "ERRO", "erro": m}
