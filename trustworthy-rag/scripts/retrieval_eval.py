"""Retrieval-only evaluation on all answerable questions (no LLM). Chunk- and document-level metrics at k.

python -m scripts.retrieval_eval [--k 5 10 20]
"""
import argparse

import numpy as np
import pandas as pd

from src.common import PROC, RESULTS, read_jsonl
from src.evaluation.metrics import retrieval_metrics
from src.pipeline.index import Retriever

DOMS = ["hotpotqa", "squad2", "pubmedqa", "cuad", "financebench"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", nargs="+", type=int, default=[5, 10, 20])
    a = ap.parse_args()
    rows = []
    for d in DOMS:
        qs = [q for q in read_jsonl(PROC / d / "questions.jsonl") if q["answerable"] and q["gold_chunk_ids"]]
        hits = Retriever(d).retrieve_many([q["question"] for q in qs], k=max(a.k))
        for k in a.k:
            ch = [retrieval_metrics([c["chunk_id"] for c in h], q["gold_chunk_ids"], k) for q, h in zip(qs, hits)]
            dc = [retrieval_metrics(list(dict.fromkeys(c["doc_id"] for c in h[:k])), q["gold_doc_ids"]) for q, h in zip(qs, hits)]
            rows.append({"domain": d, "k": k, "n": len(qs),
                         **{f"chunk_{m}": np.mean([x[m] for x in ch]) for m in ["hit", "recall", "precision", "mrr", "ndcg"]},
                         "doc_hit": np.mean([x["hit"] for x in dc]), "doc_recall": np.mean([x["recall"] for x in dc]),
                         "top1_score": np.mean([h[0]["score"] for h in hits])})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "retrieval_all.csv", index=False)
    print(df.round(3).to_string(index=False))
