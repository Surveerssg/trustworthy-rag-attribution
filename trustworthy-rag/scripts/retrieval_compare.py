"""Retrieval-variant comparison (dense / bm25 / hybrid / rerank / scoped) at k=5 on all answerable questions."""
import time
import numpy as np
import pandas as pd
from src.common import PROC, RESULTS, read_jsonl
from src.evaluation.metrics import retrieval_metrics
from src.pipeline.retrieval_plus import PlusRetriever

DOMS = ["hotpotqa", "squad2", "pubmedqa", "cuad", "financebench"]
if __name__ == "__main__":
    rows = []
    for d in DOMS:
        qs = [q for q in read_jsonl(PROC / d / "questions.jsonl") if q["answerable"] and q["gold_chunk_ids"]]
        modes = ["dense", "bm25", "hybrid", "rerank"] + (["scoped"] if d in ("cuad", "financebench") else [])
        for m in modes:
            t0 = time.time()
            hits = PlusRetriever(d, m).retrieve_many([q["question"] for q in qs], k=5)
            sec = (time.time() - t0) / len(qs)
            ch = [retrieval_metrics([c["chunk_id"] for c in h], q["gold_chunk_ids"]) for q, h in zip(qs, hits)]
            rows.append({"domain": d, "mode": m, "n": len(qs), **{k: np.mean([x[k] for x in ch]) for k in ["hit", "recall", "mrr", "ndcg"]},
                         "sec_per_q": sec})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "retrieval_variants.csv", index=False)
    print(df.round(3).to_string(index=False))
