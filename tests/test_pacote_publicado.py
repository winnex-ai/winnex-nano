"""O WHEEL PUBLICADO tem de expor o EngineTokenizer.

O defeito que este arquivo impede
---------------------------------
A tag `v0.1.43` foi criada em `8e2a362`, um commit ANTES de `b5458d2`
("tokenização pelo motor"). O código estava no `main`, o `setuptools_scm`
calculou `0.1.44.dev1` corretamente, mas a tag apontava para o passado — e o
wheel publicado como 0.1.43 saiu **sem** `EngineTokenizer`,
`EngineTokenizerOptions`, `TokenizerArtifact` e sem o módulo
`winnex_nano.tokenize_engine`. O `.so` compilado exportava as classes; o
`__init__.py` empacotado não as importava.

Nada falhou no CI: o build passou, o `python -c "print(__version__)"` passou, o
upload passou. O produto simplesmente não tinha a funcionalidade, e só um
`hasattr` no artefato instalado revelaria isso.

Este teste roda contra o pacote INSTALADO (não contra o fonte), que é o que o
usuário recebe. Ele existe para que "publiquei" signifique "o artefato tem o
que eu disse que tem".
"""
import os
import subprocess
import sys

import pytest


def test_engine_tokenizer_esta_no_pacote_instalado():
    """`import winnex_nano` tem de expor o tokenizer do motor."""
    wn = pytest.importorskip("winnex_nano")
    faltando = [n for n in ("EngineTokenizer", "EngineTokenizerOptions",
                            "TokenizerArtifact")
                if not hasattr(wn, n)]
    assert not faltando, (
        f"o pacote instalado (winnex-nano {getattr(wn, '__version__', '?')}) "
        f"não expõe {faltando}. O wheel foi construído de um commit anterior "
        f"ao que adiciona o EngineTokenizer — provavelmente a tag aponta para "
        f"antes do commit. Verifique `git describe --tags` e recrie a tag.")


def test_modulo_tokenize_engine_importavel():
    """O submódulo precisa existir no wheel, não só o símbolo no __init__."""
    pytest.importorskip("winnex_nano")
    from winnex_nano import tokenize_engine
    for nome in ("EngineTokenizer", "EngineTokenizerOptions",
                 "EngineTokenizeReport", "SegmentationReport"):
        assert hasattr(tokenize_engine, nome), f"tokenize_engine sem {nome}"


def test_pacote_instalado_contem_os_arquivos_do_tokenizer():
    """O RECORD do wheel lista os arquivos que o código importa.

    Detectar a ausência pelo `importlib.metadata` pega o caso em que o build
    excluiu o módulo por engano (configuração de pacote), sem depender de o
    ambiente ter o `.so` certo.
    """
    pytest.importorskip("winnex_nano")
    from importlib.metadata import distribution

    dist = distribution("winnex-nano")
    arquivos = {str(f) for f in (dist.files or [])}
    if not arquivos:
        pytest.skip("distribuição sem RECORD (instalação de fonte)")

    esperados = [s for s in arquivos if "tokenize_engine" in s or
                 "engine_tokenizer" in s]
    assert esperados, (
        "o RECORD do wheel não contém nem `tokenize_engine.py` nem o header/"
        "fonte do engine tokenizer — o build excluiu a funcionalidade")


def test_versao_do_pacote_bate_com_a_tag_mais_recente():
    """A versão instalada não pode ser anterior à tag do repositório.

    Este é o teste que teria pego a divergência na origem: se o pacote diz
    `0.1.43` mas o repositório já está em `v0.1.44`, o artefato publicado
    descreve um estado que não existe mais.
    """
    wn = pytest.importorskip("winnex_nano")
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(wn.__file__)))
    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=raiz, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("repositório git indisponível")
    if out.returncode != 0 or not out.stdout.strip():
        pytest.skip("sem tag git acessível")

    tag = out.stdout.strip().lstrip("v")
    instalada = getattr(wn, "__version__", "")
    if "dev" in instalada or "+" in instalada:
        pytest.skip(f"instalação de desenvolvimento ({instalada})")

    def _tup(s):
        partes = []
        for x in s.split("."):
            d = "".join(c for c in x if c.isdigit())
            if not d:
                break
            partes.append(int(d))
        return tuple(partes)

    assert _tup(instalada) >= _tup(tag), (
        f"pacote instalado {instalada} é ANTERIOR à tag v{tag}. O artefato "
        f"foi construído de um commit antes da tag.")
