"""python -m scripts.prepare [domains...]  -> data/processed + ChromaDB indices."""
import json
import sys

from src.common import DOMAINS, RESULTS
from src.pipeline.data import PREPARERS
from src.pipeline.index import build_index

if __name__ == "__main__":
    doms = sys.argv[1:] or DOMAINS
    stats = []
    for d in doms:
        PREPARERS[d]()
        stats.append(build_index(d))
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / "corpus_stats.json"
    old = json.load(open(p)) if p.exists() else {}
    old.update({s["domain"]: s for s in stats})
    json.dump(old, open(p, "w"), indent=2)
