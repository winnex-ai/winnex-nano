"""Tokenizacao pelo MOTOR, com certificado — wrapper sobre a implementacao C++.

A IMPLEMENTACAO ESTA EM C++
--------------------------
`src/engine_tokenizer.cpp` faz a atribuicao usando a MESMA engine Madhava que o
ChunkEngine ja' usa (mesmo include, mesma biblioteca estatica, nenhuma copia).
Este modulo apenas re-exporta as classes do binding e oferece dois aliases em
snake_case para quem prefere dict/getattr ao estilo do pybind.

Se o modulo compilado nao expuser as classes (build antigo), a importacao
falha alto — nao ha' caminho Python alternativo, porque um fallback Python
silencioso mediria outra coisa que nao a engine.

O QUE ELE FAZ
-------------
`texto -> N vetores -> N ids`, um vetor por POSICAO. A media do texto achataria
a sequencia; um vetor por posicao preserva a ordem porque a ordem e' a ordem das
posicoes.

MEDIDO (Qwen2.5-0.5B, vocab 151.936, d=896, 2026-10-29)
-------------------------------------------------------
  8/8 textos produzem a MESMA sequencia de ids do BPE
  151.935 de 151.936 documentos provadamente excluidos, 0 violacoes
  escopo: pool_only por default; exact_global SO' com audit_exhaustive=True
  custo: build do indice ~78 s (uma vez), ~60 ms por bloco depois

LIMITE HONESTO
--------------
O motor atribui o ID CORRETO a cada vetor. O que ele NAO decide e' ONDE CORTAR
o texto em unidades — a segmentacao vem do chamador. O certificado cobre a
ATRIBUICAO, nao a segmentacao.

BSL 1.1 | pay@winnex.ai
"""
from __future__ import annotations

from ._winnex_nano import (  # noqa: F401
    EngineTokenizer,
    EngineTokenizerOptions,
    EngineTokenizeReport,
    TokenAttribution,
    # Persistencia POR MODELO. Ficou de fora da primeira exportacao: o binding
    # C++ expunha TokenizerArtifact/ModelIdentity/verify_artifact_matches, mas
    # nem este modulo nem o __init__ os reexportavam — entao a funcionalidade
    # existia no .so e era inalcancavel em Python. Medido ao validar o 0.1.44:
    # `from winnex_nano import TokenizerArtifact` -> ImportError.
    TokenizerArtifact,
    ModelIdentity,
    verify_artifact_matches,
)

__all__ = [
    "EngineTokenizer",
    "EngineTokenizerOptions",
    "EngineTokenizeReport",
    "TokenAttribution",
    "TokenizerArtifact",
    "ModelIdentity",
    "verify_artifact_matches",
    "SegmentationReport",
]


class SegmentationReport(dict):
    """Visao dict de um `EngineTokenizeReport`, para quem prefere chaves.

    Nao substitui o objeto C++: e' uma leitura dele. Existe porque os
    benchmarks deste repo comparam com `r["ids"]` / `r.get("coverage_pct")`.
    """

    @classmethod
    def from_report(cls, rep):
        d = dict(
            ids=list(rep.ids),
            certificado=[
                dict(id=c.id, excluded=c.excluded, violations=c.violations,
                     scope=c.scope, bound_pairs=c.bound_pairs)
                for c in rep.certificado
            ],
            violations_total=rep.violations_total,
            cobertura_pct=rep.coverage_pct,
            escopos=list(rep.scopes),
            sequencia_identica=rep.sequencia_identica,
            acertos=rep.acertos,
            elapsed_ms=rep.elapsed_ms,
        )
        return cls(d)

    @property
    def ids(self):
        return self["ids"]

    @property
    def certificado(self):
        return self["certificado"]
