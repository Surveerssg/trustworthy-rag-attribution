"""Add the 'improved' retrieval condition to data/plans/<domain>.jsonl (existing conditions are kept unchanged).

Per-domain retriever, chosen by principle (not tuned on test questions):
  general (web-like text)  -> hybrid BM25+dense with the web-trained cross-encoder reranker
  medical                  -> dense (already 0.99 Hit@5)
  legal / financial        -> entity-scoped hybrid (restrict to the named contract / company filings)
python -m scripts.add_improved_plans [--domains cuad financebench ...]
"""
import argparse

from src.common import DATA, read_jsonl, write_jsonl
from src.pipeline.retrieval_plus import PlusRetriever

IMPROVED = {"hotpotqa": "rerank", "squad2": "rerank", "pubmedqa": "dense", "cuad": "entity", "financebench": "entity"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=list(IMPROVED))
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    for d in a.domains:
        pf = DATA / "plans" / f"{d}.jsonl"
        rows = read_jsonl(pf)
        qs = read_jsonl(DATA / "processed" / d / "questions.jsonl")
        text = {q["qid"]: q["question"] for q in qs}
        hits = PlusRetriever(d, IMPROVED[d]).retrieve_many([text[r["qid"]] for r in rows], k=a.k)
        for r, h in zip(rows, hits):
            r["improved"] = h
            r["improved_mode"] = IMPROVED[d]
        write_jsonl(pf, rows)
        print(d, IMPROVED[d], len(rows), flush=True)
