#!/usr/bin/env python3
"""
benchmark_selective_attention.py — selective attention with a CONTROLLED lost
mass, measured against the full-softmax baseline.

The rule (see EpsilonMass() in src/forward.cpp): keep the most probable keys
until they account for (1 - ε) of the softmax mass. The dropped mass is then
< ε by construction, and the attention output moves by at most 2ε in L1 — a
rigorous bound, not a score cut.

An important finding about the obvious approach: bounding the SCORE with
Cauchy-Schwarz (s_i = q·k_i ≤ ‖q‖‖k_i‖) is sound but useless in practice.
Measured on Qwen2.5-0.5B layer 0: real scores sit at ~80 while the CS bound is
~1974 — a gap of ~1895. The stopping test never fires and every key is kept.
Since the true mass is available for free once the scores are computed, the
decision is made on it directly and the guarantee becomes exact.

Usage:
    python benchmark_selective_attention.py                    # baseline
    MADHAVA_TOPK_ATTN=1 MADHAVA_EPSILON_MASS=1e-4 python ...   # selective
"""
import sys, os, numpy as np, torch
import os as _o
sys.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)) + '/python')
from transformers import AutoTokenizer, AutoModelForCausalLM
M=os.environ.get('WINNEX_MODEL','/home/wnnx_user/models/Qwen2.5-0.5B-Instruct')
tok=AutoTokenizer.from_pretrained(M); m=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float32).eval()
E=m.get_input_embeddings().weight.detach().numpy().astype(np.float32)
import winnex_nano as wn
_st=wn.Safetensors(M+'/model.safetensors')
eng=wn.ForwardEngine(wn.load_config(M+'/config.json'), _st)
TEXTS=["The capital of France is Paris and the weather is nice today.",
 "O rato roeu a roupa do rei de Roma.",
 "Machine learning models require substantial computational resources.",
 "A inferência geométrica preserva a estrutura da linguagem.",
 "def f(x): return x**2 + 1",
 "Quanto é dois mais dois? A resposta é quatro, evidentemente.",
 "The quick brown fox jumps over the lazy dog.",
 "A geometria localiza; ela não infere."]
LONG=("The quick brown fox jumps over the lazy dog. "*20)[:700]
def ppl(t):
    ids=tok(t).input_ids
    H=np.ascontiguousarray(E[ids[:-1]],dtype=np.float32)
    lg=eng.forward(H,len(ids)-1,True).reshape(len(ids)-1,151936).astype(np.float64)
    nll=0.0
    for p,tg in enumerate(ids[1:]):
        z=lg[p]-lg[p].max(); nll-=(z-np.log(np.exp(z).sum()))[tg]
    return float(np.exp(nll/(len(ids)-1)))
short=np.mean([ppl(t) for t in TEXTS]); lp=ppl(LONG)
print(f"{short:.4f} {lp:.6f} {len(tok(LONG).input_ids)-1}")
