"""Mechanistic error taxonomy + case studies per domain.

python -m scripts.error_taxonomy --run main_qwen1.5b [--system vanilla_rag]
Every wrong outcome is assigned exactly one mechanism (evaluated in this order):
  false_answer       unanswerable question answered
  over_refusal       answerable question refused (split: gold retrieved / not retrieved)
  retrieval_miss     no gold chunk in top-k
  evidence_ignored   gold retrieved, but the answer's best-supporting chunk (nli_set) is not gold / answer unsupported
  evidence_misread   answer attributed to a gold chunk and still wrong (reasoning / numeric / label error)
Outputs a domain x mechanism table and paper/findings/case_studies/case_studies_<run>.md with worked examples (successes + failures).
"""
import argparse
import random

import numpy as np
import pandas as pd

from src.attribution.methods import strip_citations
from src.common import PAPER, RESULTS, RUNS, read_jsonl

DOMS = ["squad2", "hotpotqa", "pubmedqa", "cuad", "financebench"]


def mechanism(r):
    if not r["answerable"]:
        return "correct_refusal" if r["refused"] else "false_answer"
    gold = set(r["gold_chunk_ids"])
    hit = any(c["chunk_id"] in gold for c in r["retrieved"])
    if r["refused"]:
        return "over_refusal_gold_retrieved" if hit else "over_refusal_no_gold"
    if r["correct"]:
        return "correct"
    if not hit:
        return "retrieval_miss"
    sc = r["attribution"].get("scores", {})
    S = sc.get("nli_set") or sc.get("nli_window")
    if not S:
        return "evidence_ignored"
    S = np.asarray(S)
    best_chunk = S.max(0).argmax()
    supported = S.max() >= 0.5
    if supported and r["retrieved"][best_chunk]["chunk_id"] in gold:
        return "evidence_misread"
    return "evidence_ignored"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--system", default="vanilla_rag")
    a = ap.parse_args()
    rows, examples = [], {}
    for d in DOMS:
        p = RUNS / a.run / f"{d}__{a.system}.scored.jsonl"
        if not p.exists():
            continue
        for r in read_jsonl(p):
            m = mechanism(r)
            rows.append({"domain": d, "mechanism": m})
            examples.setdefault((d, m), []).append(r)
    df = pd.DataFrame(rows)
    tab = pd.crosstab(df.domain, df.mechanism, normalize="index").round(3)
    tab["n"] = df.groupby("domain").size()
    out = RESULTS / a.run
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / f"error_taxonomy_{a.system}.csv")
    print(tab.to_string())

    md = [f"# Case studies — run `{a.run}`, system `{a.system}`\n",
          "Automatically selected (seeded) examples per domain and mechanism; authors add the analysis paragraph.\n"]
    rng = random.Random(0)
    for d in DOMS:
        for m in ["correct", "retrieval_miss", "evidence_ignored", "evidence_misread", "over_refusal_gold_retrieved",
                  "false_answer", "correct_refusal"]:
            ex = examples.get((d, m), [])
            if not ex:
                continue
            r = rng.choice(ex)
            gold = set(r["gold_chunk_ids"])
            ev = "\n".join(f"> [{i + 1}]{' **(gold)**' if c['chunk_id'] in gold else ''} {c['text'][:300]}…"
                           for i, c in enumerate(r["retrieved"]))
            sc = r["attribution"].get("scores", {})
            S = sc.get("nli_set") or sc.get("nli_window")
            sup = "" if not S else " | per-sentence best support: " + ", ".join(f"{max(x):.2f}" for x in S)
            md.append(f"## {d} — {m}\n**Q:** {r['question']}\n\n**Gold:** {'; '.join(r['gold_answers'])[:300] or '(unanswerable)'}\n\n"
                      f"**Answer:** {strip_citations(r['answer'])[:500]}\n\n*correct={r.get('correct')}{sup}*\n\n**Retrieved:**\n{ev}\n")
    out_dir = PAPER / "findings" / "case_studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"case_studies_{a.run}_{a.system}.md").write_text("\n".join(md), encoding="utf8")


if __name__ == "__main__":
    main()
