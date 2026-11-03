import numpy as np
from src.common import RUNS, read_jsonl
DOMS = ["squad2", "hotpotqa", "pubmedqa", "cuad", "financebench"]
def acc(recs, mode):
    per, cited_share = [], []
    for r in recs:
        if r["refused"] or not r["answerable"]:
            continue
        sc = r["attribution"].get("scores", {}); sents = r["attribution"].get("sentences", [])
        gold = {c["chunk_id"] for c in r["retrieved"]} & set(r["gold_chunk_ids"])
        if not gold or "sim_sent" not in sc or "nli_set" not in sc:
            continue
        F = 0.5 * np.asarray(sc["sim_sent"]) + 0.5 * np.asarray(sc["nli_set"])
        hits = []
        for i, s in enumerate(sents):
            cites = [c for c in s.get("cites", []) if 1 <= c <= len(r["retrieved"])]
            if mode == "hybrid" and cites:
                j = max((c - 1 for c in cites), key=lambda j: F[i, j])   # best of the cited passages
            else:
                j = int(F[i].argmax())
            hits.append(r["retrieved"][j]["chunk_id"] in gold)
            cited_share.append(bool(cites))
        if hits:
            per.append(np.mean(hits))
    return np.mean(per), len(per), np.mean(cited_share)
for run in ["main_llama8b", "main_qwen1.5b"]:
    print(f"\n{run} (trust-prompt answers): attribution accuracy over ALL sentences | gold retrieved")
    print(f"{'domain':13s} {'fused':>7s} {'hybrid':>7s} {'n':>5s} {'% sentences cited':>18s}")
    for d in DOMS:
        T = read_jsonl(RUNS / run / f"{d}__trust_rag.scored.jsonl")
        f, n, _ = acc(T, "fused"); h, _, cs = acc(T, "hybrid")
        print(f"{d:13s} {f:7.3f} {h:7.3f} {n:5d} {cs:18.2f}")
