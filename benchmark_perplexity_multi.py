#!/usr/bin/env python3
"""
benchmark_perplexity_multi.py — expanded perplexity benchmark across models.

20 texts, 7 languages (EN/PT/FR/DE/ZH/EL/code), 270 tokens, two architectures:

    Qwen2.5-0.5B-Instruct   24 layers, d=896,  530M params, tied lm_head
    Qwen3-4B-Instruct-2507  36 layers, d=2560, 4.1B params, decoupled head_dim,
                            per-head Q/K RMSNorm

Cross-model correctness caught three defects that a single model never
exposes:

  * head_dim was always derived as hidden/heads. Qwen3 decouples them
    (2560/32 = 80, but the model uses 128), so every projection reshaped
    wrong. Also q_out had to become heads*head_dim instead of hidden_size,
    or q was under-allocated and the o_proj read past it (malloc SIGABRT).
  * Qwen3 applies an RMSNorm over head_dim on each head of Q and K before
    RoPE. Without it correlation dropped from 1.000000 (1 token) to ~0.47
    (n>=2) — a single token cannot reveal it because attention is trivial.

Usage:
    python benchmark_perplexity_multi.py <model_dir> <safetensors> <label>
"""
import sys, os, numpy as np, torch
import os as _o
sys.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)) + '/python')
from transformers import AutoTokenizer, AutoModelForCausalLM
import winnex_nano as wn
TEXTS=["The capital of France is Paris and the weather is nice today.",
 "O rato roeu a roupa do rei de Roma.",
 "Machine learning models require substantial computational resources.",
 "A inferência geométrica preserva a estrutura da linguagem.",
 "def f(x): return x**2 + 1",
 "Quanto é dois mais dois? A resposta é quatro, evidentemente.",
 "The quick brown fox jumps over the lazy dog.",
 "A geometria localiza; ela não infere.",
 "Les mathématiques sont le langage de la nature.",
 "Die Quantenmechanik beschreibt die Natur im Kleinsten.",
 "机器学习需要大量的计算资源。",
 "Οι μαθηματικοί αποδεικνύουν θεωρήματα.",
 "A derivada mede a taxa de variação instantânea.",
 "Water boils at one hundred degrees Celsius at sea level.",
 "A prova por contradição assume o oposto do que se quer provar.",
 "In the beginning was the word, and the word was with meaning.",
 "Le chat est sur la table et le chien est sous la chaise.",
 "Der Beweis durch Widerspruch setzt das Gegenteil voraus.",
 "A soma dos ângulos internos de um triângulo é 180 graus.",
 "Every finite field has prime power order."]
MP,SF,NAME = sys.argv[1], sys.argv[2], sys.argv[3]
tok=AutoTokenizer.from_pretrained(MP)
st=wn.Safetensors(SF)
cfg=wn.load_config(MP+'/config.json')
eng=wn.ForwardEngine(cfg,st)
E=np.asarray(st.to_float('model.embed_tokens.weight'),dtype=np.float32).reshape(-1,cfg.hidden_size)
V=E.shape[0]
def ppl(t):
    ids=tok(t).input_ids
    H=np.ascontiguousarray(E[ids[:-1]],dtype=np.float32)
    lg=eng.forward(H,len(ids)-1,True).reshape(len(ids)-1,V).astype(np.float64)
    nll=0.0
    for p,tg in enumerate(ids[1:]):
        z=lg[p]-lg[p].max(); nll-=(z-np.log(np.exp(z).sum()))[tg]
    return float(np.exp(nll/(len(ids)-1)))
print(f"{NAME}: {cfg.num_hidden_layers} camadas, d={cfg.hidden_size}, params={eng.param_count:,}")
v=[ppl(t) for t in TEXTS]
print(f"  textos: {len(TEXTS)} | tokens: {sum(len(tok(t).input_ids)-1 for t in TEXTS)}")
print(f"  PPL media: {np.mean(v):.4f} | min {min(v):.2f} | max {max(v):.2f}")
import json; json.dump({'name':NAME,'ppl':v,'mean':float(np.mean(v))},open(f'/tmp/{NAME}.json','w'))
