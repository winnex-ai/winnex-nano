#!/usr/bin/env python3
"""
benchmark_perplexity.py — perplexity of the NATIVE 24-layer forward.

This is the measurement the stack was missing: it runs the winnex-nano
ForwardEngine end to end (all layers, RMSNorm, RoPE, GQA, SwiGLU, lm_head)
over real weights and compares against transformers on the SAME checkpoint.

It exists because three defects made the native forward unusable and none of
them showed up in the stack's unit tests (which check shapes, not quality):

  1. RoPE used the interleaved-pair convention with a -2j/hd exponent, while
     Qwen2 pairs element j with j+hd/2 ("rotate_half") at -2j/hd for j in
     [0, hd/2).
  2. The binding copied shape[0] elements, so the documented (seq, hidden)
     input collapsed to seq floats and every 2D call raised.
  3. Q/K/V attention BIASES (present in all 24 Qwen2.5 layers) were never
     loaded or applied.

Additionally, the attention kept only the top-16 keys with weight
score/Σscore and NO softmax and NO causal mask — not the function the
pretrained weights encode. Softmax + causal masking are now the default.

Usage:
    python benchmark_perplexity.py [model_dir]
"""
import sys, time, numpy as np, torch
import os

import winnex_nano as WN
from transformers import AutoModelForCausalLM, AutoTokenizer
M=os.environ.get('WINNEX_MODEL','/home/wnnx_user/models/Qwen2.5-0.5B-Instruct')
tok=AutoTokenizer.from_pretrained(M)
m=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float32).eval()
st=WN.Safetensors(M+'/model.safetensors'); cfg=WN.load_config(M+'/config.json')
t=time.time(); eng=WN.ForwardEngine(cfg,st); build=time.time()-t
E=m.get_input_embeddings().weight.detach().numpy().astype(np.float32)
TEXTS=["The capital of France is Paris and the weather is nice today.",
 "O rato roeu a roupa do rei de Roma.",
 "Machine learning models require substantial computational resources.",
 "A inferência geométrica preserva a estrutura da linguagem.",
 "def f(x): return x**2 + 1",
 "Quanto é dois mais dois? A resposta é quatro, evidentemente.",
 "The quick brown fox jumps over the lazy dog.",
 "A geometria localiza; ela não infere."]
def ev(text):
    ids=tok(text).input_ids
    H=np.ascontiguousarray(E[ids[:-1]],dtype=np.float32)
    t0=time.time(); lg=eng.forward(H,len(ids)-1,True); dt=time.time()-t0
    lg=lg.reshape(len(ids)-1,151936).astype(np.float64)
    nll=0.0; hit=0
    for p,tg in enumerate(ids[1:]):
        z=lg[p]-lg[p].max(); nll-=(z-np.log(np.exp(z).sum()))[tg]
        hit+= int(lg[p].argmax()==tg)
    with torch.no_grad(): r=m(torch.tensor([ids])).logits[0,:-1].float().numpy().astype(np.float64)
    nll_r=0.0
    for p,tg in enumerate(ids[1:]):
        z=r[p]-r[p].max(); nll_r-=(z-np.log(np.exp(z).sum()))[tg]
    n=len(ids)-1
    return np.exp(nll/n), np.exp(nll_r/n), hit/n, dt, n
print("="*78)
print("BENCHMARK DE PERPLEXIDADE — winnex-nano ForwardEngine (24 camadas nativas)")
print("="*78)
print(f"build: {build:.2f}s | params: {eng.param_count:,} | sem torch no caminho nativo")
print()
print(f"{'texto':<44}{'n':>3}{'PPL nat':>10}{'PPL ref':>10}{'top1':>7}{'ms':>7}")
print("-"*78)
A=[];B=[];C=[];D=[]
for t in TEXTS:
    a,b,h,dt,n=ev(t); A.append(a);B.append(b);C.append(h);D.append(dt)
    print(f"{t[:42]:<44}{n:>3}{a:>10.2f}{b:>10.2f}{h:>7.0%}{dt*1000:>7.0f}")
print("-"*78)
print(f"{'MEDIA':<44}{'':>3}{np.mean(A):>10.2f}{np.mean(B):>10.2f}{np.mean(C):>7.0%}{np.mean(D)*1000:>7.0f}")
print()
print(f"  razao PPL nativo/referencia : {np.mean(A)/np.mean(B):.6f}   (1.000000 = identico)")
print(f"  erro relativo medio         : {abs(np.mean(A)-np.mean(B))/np.mean(B):.2e}")
