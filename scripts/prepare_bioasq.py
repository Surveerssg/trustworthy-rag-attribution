"""BioASQ (second medical dataset): processed data, chunks, gold chunks and the precomputed retrieval plan, without ChromaDB.

python -m scripts.prepare_bioasq              # data (fetches PubMed abstracts once, cached) + chunks + gold chunk ids + plan
python -m scripts.prepare_bioasq --skip-plan  # without the plan (no sentence-transformers needed)
Needs data/raw/bioasq/BioASQ-training13b/training13b.json (registered download from participants-area.bioasq.org).
The plan uses exactly the rules of scripts/prepare_tatqa.py (dense main / oracle / removed). Unlike TAT-QA, BioASQ questions are
self-contained, so the natural (main) condition is meaningful and used.
"""
import argparse

from scripts.prepare_tatqa import build_chunks, build_plan
from src.pipeline.data import prepare_bioasq

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-plan", action="store_true")
    a = ap.parse_args()
    prepare_bioasq()
    chunks, qs = build_chunks("bioasq")
    if not a.skip_plan:
        build_plan(chunks, qs, domain="bioasq", note="")
