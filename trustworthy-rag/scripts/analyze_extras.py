"""Zero-cost analyses on stored outputs.

python -m scripts.analyze_extras --run main_llama8b
E  retrieval help/harm : closed-book vs vanilla-RAG outcome transitions per domain
                         (harm = closed-book correct -> RAG wrong; McNemar test on the paired outcomes)
D  position bias       : oracle conditions shuffle the gold chunk position; accuracy and attribution accuracy
                         by gold position 1..k per domain (needs the 'extensions' stage)
"""
import argparse

import numpy as np
import pandas as pd

from src.common import RESULTS, RUNS, read_jsonl
from src.evaluation import metrics as M

DOMS = ["squad2", "hotpotqa", "pubmedqa", "cuad", "financebench"]


def load(run, system):
    out = {}
    for d in DOMS:
        p = RUNS / run / f"{d}__{system}.scored.jsonl"
        if p.exists():
            out[d] = {r["qid"]: r for r in read_jsonl(p)}
    return out


def help_harm(run):
    C, V = load(run, "closed_book"), load(run, "vanilla_rag")
    rows = []
    for d in DOMS:
        if d not in C or d not in V:
            continue
        q = [k for k in V[d] if k in C[d] and V[d][k]["answerable"]]
        cb = np.array([bool(C[d][k]["correct"]) for k in q]); rg = np.array([bool(V[d][k]["correct"]) for k in q])
        rows.append({"domain": d, "n": len(q), "closed_book_acc": cb.mean(), "rag_acc": rg.mean(),
                     "help (CB wrong->RAG right)": (~cb & rg).mean(), "harm (CB right->RAG wrong)": (cb & ~rg).mean(),
                     "both_right": (cb & rg).mean(), "both_wrong": (~cb & ~rg).mean(),
                     "mcnemar_p": M.mcnemar(rg, cb)["p"]})
    return pd.DataFrame(rows)


def position_bias(run):
    rows = []
    for system in ["oracle_vanilla", "oracle_trust"]:
        S = load(run, system)
        for d, recs in S.items():
            for r in recs.values():
                if not r["answerable"] or not r["gold_chunk_ids"]:
                    continue
                gold = set(r["gold_chunk_ids"])
                pos = [i + 1 for i, c in enumerate(r["retrieved"]) if c["chunk_id"] in gold]
                if not pos:
                    continue
                am = r["attr_metrics"].get("nli_set") or {}
                rows.append({"system": system, "domain": d, "gold_pos": min(pos), "correct": r["correct"],
                             "attr_acc": am.get("attr_acc", np.nan)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.groupby(["system", "domain", "gold_pos"]).agg(n=("correct", "size"), accuracy=("correct", "mean"),
                                                            attr_acc=("attr_acc", "mean")).reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    out = RESULTS / a.run
    out.mkdir(parents=True, exist_ok=True)
    hh = help_harm(a.run)
    hh.to_csv(out / "retrieval_help_harm.csv", index=False)
    print("=== E: does retrieval help or hurt? (answerable questions) ===\n", hh.round(3).to_string(index=False))
    pb = position_bias(a.run)
    if len(pb):
        pb.to_csv(out / "position_bias.csv", index=False)
        print("\n=== D: accuracy by position of the gold chunk (oracle context) ===\n", pb.round(3).to_string(index=False))
    else:
        print("\n(D) position bias: no oracle outputs yet (run the 'extensions' stage)")


# ---------------------------------------------------------------- C: numeric grounding
def numeric_grounding(run, system="vanilla_rag"):
    """Fraction of numbers in the answer (years excluded) found in the retrieved evidence (x1e3/x1e6 unit shifts
    allowed). Evaluated as a detector of wrong answers, alone and vs / combined with NLI support."""
    from src.attribution.methods import strip_citations
    rows = []
    for d, recs in load(run, system).items():
        for r in recs.values():
            if not r["answerable"] or r["refused"] or not r["retrieved"]:
                continue
            nums = [x for x in M.extract_numbers(strip_citations(r["answer"])) if not (1900 <= abs(x) <= 2100 and float(x).is_integer())]
            if not nums:
                continue
            ev = M.extract_numbers(" ".join(c["text"] for c in r["retrieved"]))
            ok = [any(np.isclose(abs(n) * s, abs(e), rtol=0.005, atol=0.006) for e in ev for s in (1, 1e3, 1e-3, 1e6, 1e-6)) for n in nums]
            f = r["features"]
            rows.append({"domain": d, "num_grounded": float(np.mean(ok)), "nli_cov": f.get("nli_cov_mean", 0.0),
                         "correct": r["correct"]})
    df = pd.DataFrame(rows)
    out = []
    for d, x in df.groupby("domain"):
        if len(x) < 15 or x.correct.nunique() < 2:
            continue
        comb = (x.num_grounded + x.nli_cov) / 2
        out.append({"domain": d, "n_answers_with_numbers": len(x), "frac_fully_grounded": (x.num_grounded == 1).mean(),
                    "acc_if_grounded": x.correct[x.num_grounded == 1].mean(), "acc_if_not": x.correct[x.num_grounded < 1].mean(),
                    "AUROC_numeric": M.auroc(x.num_grounded, x.correct), "AUROC_nli": M.auroc(x.nli_cov, x.correct),
                    "AUROC_combined": M.auroc(comb, x.correct)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    ng = numeric_grounding(a.run)
    ng.to_csv(out / "numeric_grounding.csv", index=False)
    print("\n=== C: numeric grounding as a hallucination / error detector ===\n", ng.round(3).to_string(index=False))
