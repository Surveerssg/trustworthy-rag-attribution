"""The core scoreboard: retrieval, attribution, hallucination, refusal — per domain, baseline vs improved,
plus the DOMAIN GAP (best minus worst domain). Success = weak domains rise and the gap shrinks.

python -m scripts.scoreboard --run main_llama8b [--improved-run v2_llama8b]
Rows (per domain):
  retrieval_hit5      gold passage in top 5 (baseline dense vs improved per-domain retriever, from data/plans)
  attribution_acc     top-attributed passage is gold | gold retrieved (similarity baseline vs fused v2)
  unsupported_rate    share of answer sentences no passage supports (lower is better)
  wrong_answer_rate   share of ALL questions answered wrongly (incl. answering unanswerable) — the hallucination
                      the user actually sees (plain RAG vs + implicit-refusal detection)
  refusal_f1          on datasets with unanswerable questions (SQuAD 2.0, CUAD)
"""
import argparse
import json

import numpy as np
import pandas as pd

from scripts.improve_system import fused_attr_acc
from src.common import DATA, PROC, RESULTS, RUNS, read_jsonl
from src.evaluation import metrics as M
from src.pipeline.prompts import is_implicit_refusal

DOMS = ["squad2", "hotpotqa", "pubmedqa", "cuad", "financebench"]


def retrieval(d):
    rows = read_jsonl(DATA / "plans" / f"{d}.jsonl")
    qs = {q["qid"]: q for q in read_jsonl(PROC / d / "questions.jsonl")}
    out = {}
    for cond in ["main", "improved"]:
        m = [retrieval_metrics(r, qs, cond) for r in rows]
        m = [x for x in m if x is not None]
        out[cond] = float(np.mean(m)) if m and all(x == x for x in m) else np.nan
    return out


def retrieval_metrics(r, qs, cond):
    q = qs[r["qid"]]
    if not q["answerable"] or not q.get("gold_chunk_ids") or not r.get(cond):
        return None
    return M.retrieval_metrics([c["chunk_id"] for c in r[cond]], q["gold_chunk_ids"])["hit"]


def answer_metrics(recs, implicit):
    ref = [r["refused"] or (implicit and is_implicit_refusal(r["answer"])) for r in recs]
    wrong = [(not rf) and ((not r["answerable"]) or not r["correct"]) for r, rf in zip(recs, ref)]
    uns = [1 - r["attr_metrics"]["nli_set"]["eas"] for r, rf in zip(recs, ref)
           if not rf and r["attr_metrics"].get("nli_set")]
    out = {"wrong_answer_rate": float(np.mean(wrong)), "unsupported_rate": float(np.mean(uns)) if uns else np.nan,
           "coverage": 1 - float(np.mean(ref))}
    if any(not r["answerable"] for r in recs):
        out["refusal_f1"] = M.prf(ref, [not r["answerable"] for r in recs])["f1"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--improved-run", default=None, help="Stage-2 run with the improved-retrieval arm")
    a = ap.parse_args()
    rows = []
    for d in DOMS:
        V = read_jsonl(RUNS / a.run / f"{d}__vanilla_rag.scored.jsonl")
        ans = [r for r in V if r["answerable"] and not r["refused"]]
        ret = retrieval(d)
        base, imp_imp = answer_metrics(V, False), answer_metrics(V, True)
        row = {"domain": d,
               "retrieval_hit5 (base)": ret["main"], "retrieval_hit5 (improved)": ret["improved"],
               "attribution_acc (base)": fused_attr_acc(ans, 1.0)[0], "attribution_acc (fused)": fused_attr_acc(ans, 0.5)[0],
               "unsupported_rate (base)": base["unsupported_rate"],
               "wrong_answer_rate (base)": base["wrong_answer_rate"], "wrong_answer_rate (+implicit)": imp_imp["wrong_answer_rate"],
               "coverage (+implicit)": imp_imp["coverage"],
               "refusal_f1 (base)": base.get("refusal_f1", np.nan), "refusal_f1 (+implicit)": imp_imp.get("refusal_f1", np.nan)}
        if a.improved_run:
            p = RUNS / a.improved_run / f"{d}__improved_vanilla.scored.jsonl"
            if p.exists():
                I = read_jsonl(p)
                ia = [r for r in I if r["answerable"] and not r["refused"]]
                im = answer_metrics(I, True)
                row.update({"attribution_acc (improved arm)": fused_attr_acc(ia, 0.5)[0],
                            "unsupported_rate (improved arm)": im["unsupported_rate"],
                            "wrong_answer_rate (improved arm)": im["wrong_answer_rate"],
                            "accuracy_answerable (improved arm)": float(np.mean([bool(r["correct"]) for r in I if r["answerable"]]))})
        row["accuracy_answerable (base)"] = float(np.mean([bool(r["correct"]) for r in V if r["answerable"]]))
        from scripts.cite_hybrid import acc as _hy
        T = read_jsonl(RUNS / a.run / f"{d}__trust_rag.scored.jsonl")
        row["attribution_acc (hybrid: citations+fused, trust prompt)"] = _hy(T, "hybrid")[0]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("domain").T
    df["DOMAIN GAP (best-worst)"] = df.max(1) - df.min(1)
    out = RESULTS / a.run / "scoreboard.csv"
    df.to_csv(out)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
