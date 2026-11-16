"""Section 9: baseline comparison + ablation + significance.

Systems (all built from the same stored generations; no extra LLM calls):
  closed_book        LLM only
  vanilla_rag        plain RAG prompt
  full               trust prompt (cite + may refuse) + NLI hallucination filter + confidence gate
  full -gate         (no confidence estimation/refusal gate)
  full -filter       (no hallucination detection)
  full -trustprompt  vanilla prompt + filter + gate (no attribution-aware prompting / self-refusal)
Per-question reliability utility u: +1 correct answer or correct refusal, 0 refusal of answerable, -1 wrong/hallucinated answer.
Gate threshold is fitted on SQuAD2 only (cross-fitted halves for SQuAD2 itself), then transferred.
"""
import argparse

import numpy as np
import pandas as pd

from src.attribution.methods import answer_sentences, nli_probs
from src.common import RESULTS, RUNS, read_jsonl
from src.evaluation import metrics as M

DOMS = ["squad2", "hotpotqa", "pubmedqa", "cuad", "financebench"]
FILTER_T = 0.5


def load(run, system):
    out = {}
    for d in DOMS:
        p = RUNS / run / f"{d}__{system}.scored.jsonl"
        if p.exists():
            out[d] = {r["qid"]: r for r in read_jsonl(p)}
    return out


def trust_score(r):
    f = r["features"]
    if r["refused"]:
        return 0.0
    return (f["ret_top1"] + f.get("nli_cov_mean", 0) + 1 - f.get("contra_max", 1)) / 3


def filtered_answer(r):
    """Drop answer sentences whose best NLI-window entailment < FILTER_T. Returns (text, n_kept, n_total)."""
    sc = r["attribution"].get("scores", {}); S = sc.get("nli_set", sc.get("nli_window"))
    sents = r["attribution"].get("sentences", [])
    if not S:
        return r["answer"], len(sents), len(sents)
    keep = [s["text"] for s, row in zip(sents, S) if max(row) >= FILTER_T]
    return " ".join(keep), len(keep), len(sents)


def judge(text, r):
    if not r["answerable"]:
        return None
    c, _ = M.correctness(text, r["gold_answers"], r["answer_type"])
    if c is None:
        p = nli_probs([text, r["gold_answers"][0]], [r["gold_answers"][0], text])[:, 0]
        c = float(max(p) >= 0.5)
    return c


def evaluate(r, use_filter, gate_t):
    """Returns dict(answered, correct, utility, unsupported_rate)."""
    refused = r["refused"]
    text, unsup = r["answer"], None
    if not refused and r["attribution"].get("scores"):
        S = np.asarray(r["attribution"]["scores"].get("nli_set", r["attribution"]["scores"]["nli_window"]))
        unsup = float((S.max(1) < FILTER_T).mean())
    if not refused and use_filter:
        text, kept, tot = filtered_answer(r)
        if kept == 0:
            refused = True
        else:
            unsup = 0.0
    if not refused and gate_t is not None and trust_score(r) < gate_t:
        refused = True
    if refused:
        return {"answered": 0, "correct": float(not r["answerable"]), "utility": 1.0 if not r["answerable"] else 0.0,
                "unsup": None}
    if not r["answerable"]:
        return {"answered": 1, "correct": 0.0, "utility": -1.0, "unsup": unsup}
    c = judge(text, r) if use_filter else r["correct"]
    return {"answered": 1, "correct": c, "utility": 1.0 if c else -1.0, "unsup": unsup}


def fit_gate(rows):
    x = np.array([trust_score(r) for r in rows]); y = np.array([not r["answerable"] for r in rows])
    return M.threshold_sweep(x, y)[1]["t"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    V, T, C = load(a.run, "vanilla_rag"), load(a.run, "trust_rag"), load(a.run, "closed_book")
    # gate thresholds from SQuAD2 (cross-fitted halves for squad2 itself)
    sq = {"trust": list(T.get("squad2", {}).values()), "vanilla": list(V.get("squad2", {}).values())}
    gates = {k: fit_gate(v) for k, v in sq.items() if v}
    half = {k: (fit_gate(v[0::2]), fit_gate(v[1::2])) for k, v in sq.items() if v}
    print("gate thresholds (trust score) fitted on SQuAD2:", {k: round(v, 3) for k, v in gates.items()})

    systems = {
        "closed_book": (C, "none", False, None), "vanilla_rag": (V, "none", False, None),
        "full": (T, "trust", True, "gate"), "full -gate": (T, "trust", True, None),
        "full -filter": (T, "trust", False, "gate"), "full -trustprompt": (V, "vanilla", True, "gate")}
    per_q, rows = {}, []
    for d in DOMS:
        if d not in V:
            continue
        qids = [q for q in V[d] if all(q in S.get(d, {}) for S, *_ in systems.values() if S)]
        for name, (S, gkey, filt, gate) in systems.items():
            if d not in S:
                continue
            res = []
            for i, q in enumerate(qids):
                r = S[d][q]
                t = None
                if gate:
                    t = (half[gkey][1] if i % 2 == 0 else half[gkey][0]) if d == "squad2" else gates[gkey]
                    # squad2 qids iterate in file order; cross-fitting: even rows use threshold from odd half
                res.append(evaluate(r, filt, t))
            per_q[(d, name)] = res
            ans = [x for x, q in zip(res, qids) if S[d][q]["answerable"]]
            un = [x for x, q in zip(res, qids) if not S[d][q]["answerable"]]
            answered_ans = [x for x in ans if x["answered"]]
            us = [x["unsup"] for x in res if x["unsup"] is not None]
            row = {"domain": d, "system": name, "n": len(res),
                   "accuracy": np.mean([x["correct"] for x in ans]) if ans else np.nan,
                   "coverage": np.mean([x["answered"] for x in ans]) if ans else np.nan,
                   "selective_acc": np.mean([x["correct"] for x in answered_ans]) if answered_ans else np.nan,
                   "unsupported_rate": np.mean(us) if us else np.nan,
                   "reliability_u": np.mean([x["utility"] for x in res])}
            if un:
                row.update({f"refusal_{k}": v for k, v in M.prf([not x["answered"] for x in res],
                                                               [not S[d][q]["answerable"] for q in qids]).items()})
            rows.append(row)
    df = pd.DataFrame(rows)
    out = RESULTS / a.run
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "systems_ablation.csv", index=False)
    print("\n=== Systems / ablation ===\n", df.round(3).to_string(index=False))

    # significance: full vs vanilla_rag and each ablation vs full, paired on the same questions
    sig = []
    for d in DOMS:
        if (d, "full") not in per_q:
            continue
        f = np.array([x["utility"] for x in per_q[(d, "full")]])
        for other in ["vanilla_rag", "closed_book", "full -gate", "full -filter", "full -trustprompt"]:
            if (d, other) in per_q:
                o = np.array([x["utility"] for x in per_q[(d, other)]])
                b = M.paired_bootstrap(f, o)
                sig.append({"domain": d, "comparison": f"full vs {other}", **b,
                            "t_test_p": M.paired_ttest(f, o)["p"] if np.any(f != o) else 1.0})
    sig = pd.DataFrame(sig)
    sig.to_csv(out / "systems_significance.csv", index=False)
    print("\n=== Paired significance on reliability utility (full minus other) ===\n", sig.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
