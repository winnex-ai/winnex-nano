#!/usr/bin/env python3
"""
Publish the Winnex X-Factor mathematics document as a new Zenodo record.

Usage:
    ZENODO_TOKEN=<token> python3 publish_xfactor.py
    python3 publish_xfactor.py --token <token>

Creates a fresh record (new DOI) titled "Winnex X-Factor: Deterministic Manifold
Embedding for Native LLM Inference", uploads the PDF + Markdown source, then
publishes. The prior winnex-nano pre-patent record (10.5281/zenodo.21861809)
is referenced as a related identifier and left untouched.
"""
import argparse
import os
import sys
import requests

API = "https://zenodo.org/api"
FILES = ["XFACTOR_MATHEMATICS.pdf", "XFACTOR_MATHEMATICS.md"]

DESCRIPTION = """# Winnex X-Factor

Deterministic Manifold Embedding for Native LLM Inference.

## Abstract

The X-Factor is a **deterministic, training-free** operator that embeds arbitrary
text into the latent space of a pretrained language model, enabling native inference
without fine-tuning. It is computed **structurally** from a model's embedding matrix
— never by running the model. This record gives the complete mathematics.

## Contents

- The complete mathematical derivation: the manifold hypothesis, centering,
  covariance, eigendecomposition (power iteration with deflation), effective rank,
  the projector, spectral tokenization (character histogram), dimensionality
  expansion, and the full operator.
- Complexity analysis (O(D²r·iters) vs O(D³)) and relation to the Madhava engine
  philosophy: bound-guided selection, deterministic provable operators, structural
  reads — no forward pass, no training.

## Related record

- Supplement to the winnex-nano pre-patent:
  [10.5281/zenodo.21861809](https://zenodo.org/records/21861809)

## License

BSL 1.1 | pay@winnex.ai | Author: Klenio Araujo Padilha |
Winnex Brasil Soluções Empresariais LTDA - ME
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--token", default=os.environ.get("ZENODO_TOKEN"), help="Zenodo API token")
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="directory containing the files to upload")
    args = ap.parse_args()
    if not args.token:
        sys.exit("No Zenodo token. Set ZENODO_TOKEN=<token> or pass --token <token>.")

    here = args.dir

    # Step 1: Create deposit
    payload = {"metadata": {
        "title": "Winnex X-Factor: Deterministic Manifold Embedding for Native LLM Inference",
        "description": DESCRIPTION,
        "creators": [{"name": "Padilha, Klenio Araujo", "affiliation": "Winnex AI"}],
        "access_right": "open",
        "license": "busl-1.1",
        "upload_type": "publication",
        "publication_type": "technicalnote",
        "keywords": [
            "Winnex X-Factor", "deterministic embedding", "manifold",
            "LLM inference", "principal component analysis", "power iteration",
            "character histogram", "spectral tokenization", "training-free",
            "structural read", "Madhava", "Winnex AI",
            "EU AI Act", "LGPD", "auditable AI", "deterministic AI"
        ],
        "related_identifiers": [
            {
                "identifier": "10.5281/zenodo.21861809",
                "relation": "isSupplementTo",
                "resource_type": "publication-technicalnote"
            }
        ],
        "notes": "BSL 1.1 | pay@winnex.ai | Author: Klenio Araujo Padilha | Winnex Brasil Solucoes Empresariais LTDA - ME"
    }}

    r = requests.post(
        f"{API}/deposit/depositions?access_token={args.token}",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    if r.status_code != 201:
        sys.exit(f"ERROR creating deposit: {r.status_code} {r.text[:500]}")

    dep = r.json()
    did = dep["id"]
    bucket = dep["links"]["bucket"]
    doi = dep.get("doi", "?")
    print(f"Created deposit {did}, DOI: {doi}")

    # Step 2: Upload files
    for fn in FILES:
        path = os.path.join(here, fn)
        if not os.path.exists(path):
            print(f"  SKIP {fn} (not found)")
            continue
        size = os.path.getsize(path)
        print(f"  Uploading {fn} ({size/1024:.1f} KB)...")
        with open(path, "rb") as f:
            resp = requests.put(f"{bucket}/{fn}?access_token={args.token}", data=f, timeout=120)
        if resp.status_code in (200, 201):
            print("    OK")
        else:
            print(f"    FAIL: {resp.status_code} {resp.text[:300]}")

    # Step 3: Publish
    rp = requests.post(
        f"{API}/deposit/depositions/{did}/actions/publish?access_token={args.token}",
        timeout=30,
    )
    if rp.status_code == 202:
        data = rp.json()
        print(f"\n✅ PUBLISHED: https://zenodo.org/records/{data['id']}")
        print(f"   DOI: {data.get('doi', '?')}")
        print(f"   URL: {data.get('doi_url', '?')}")
    else:
        print(f"\n❌ Publish failed: {rp.status_code} {rp.text[:500]}")


if __name__ == "__main__":
    main()
