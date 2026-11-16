"""Direction A: cross-domain attribution quality with retrieval metrics alongside.

python -m scripts.analyze_crossdomain --run main_qwen1.5b [--system vanilla_rag]
Outputs: experiments/results/<run>/crossdomain_*.csv, paper/figures/<run>_attribution_by_domain.png
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.common import PAPER, RESULTS, RUNS, read_jsonl

DOM_ORDER = ["hotpotqa", "squad2", "pubmedqa", "cuad", "financebench"]
DOM_LABEL = {"hotpotqa": "General (HotpotQA)", "squad2": "General (SQuAD2)", "pubmedqa": "Medical (PubMedQA)",
             "cuad": "Legal (CUAD)", "financebench": "Financial (FinanceBench)"}
METHODS = ["sim_chunk", "sim_sent", "nli_chunk", "nli_window", "nli_set", "self_cite"]


def boot_ci(x, n=2000, seed=0):
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)], float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.quantile(bs, .025), np.quantile(bs, .975)


def load(run, system):
    rows = []
    for d in DOM_ORDER:
        p = RUNS / run / f"{d}__{system}.scored.jsonl"
        if p.exists():
            rows += read_jsonl(p)
    return rows


def fmt(m, lo, hi):
    return "—" if np.isnan(m) else f"{m:.3f} [{lo:.2f},{hi:.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--system", default="vanilla_rag")
    a = ap.parse_args()
    out = RESULTS / a.run
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in load(a.run, a.system) if r["answerable"]]
    doms = [d for d in DOM_ORDER if any(r["domain"] == d for r in rows)]

    # ---------------- Table 1: retrieval + generation per domain
    t1 = []
    for d in doms:
        R = [r for r in rows if r["domain"] == d]
        ret = [r["retrieval"] for r in R if r.get("retrieval")]
        row = {"domain": d, "n": len(R)}
        for m in ["recall", "hit", "precision", "mrr", "ndcg"]:
            row[f"{m}@5"] = np.mean([x[m] for x in ret]) if ret else np.nan
        row["accuracy"] = np.mean([r["correct"] for r in R])
        row["refusal_rate"] = np.mean([r["refused"] for r in R])
        row["acc_given_hit"] = np.mean([r["correct"] for r in R if r.get("retrieval", {}).get("hit")] or [np.nan])
        row["acc_given_miss"] = np.mean([r["correct"] for r in R if r.get("retrieval") and not r["retrieval"]["hit"]] or [np.nan])
        t1.append(row)
    t1 = pd.DataFrame(t1)
    t1.to_csv(out / f"crossdomain_retrieval_{a.system}.csv", index=False)
    print(f"\n=== Retrieval & generation ({a.system}) ===\n", t1.round(3).to_string(index=False))

    # ---------------- Table 2: attribution per domain x method (conditioned on gold retrieved)
    t2, long = [], []
    for d in doms:
        R = [r for r in rows if r["domain"] == d and not r["refused"]]
        for m in METHODS:
            vals = [r["attr_metrics"].get(m) for r in R if r["attr_metrics"].get(m)]
            if not vals:
                continue
            acc = [v["attr_acc"] for v in vals if "attr_acc" in v]
            eas = [v["eas"] for v in vals]
            ap_ = [v["attr_p"] for v in vals if "attr_p" in v]
            ar_ = [v["attr_r"] for v in vals if "attr_r" in v]
            A, E = boot_ci(acc), boot_ci(eas)
            t2.append({"domain": d, "method": m, "n_answers": len(vals), "n_gold_retrieved": len(acc),
                       "AttrAcc": fmt(*A), "EAS": fmt(*E), "AttrP": np.mean(ap_) if ap_ else np.nan,
                       "AttrR": np.mean(ar_) if ar_ else np.nan})
            long.append({"domain": d, "method": m, "AttrAcc": A[0], "lo": A[1], "hi": A[2], "EAS": E[0]})
    t2 = pd.DataFrame(t2)
    t2.to_csv(out / f"crossdomain_attribution_{a.system}.csv", index=False)
    print(f"\n=== Attribution (AttrAcc = top-attributed chunk is gold | gold retrieved) ===\n", t2.round(3).to_string(index=False))

    # ---------------- Unsupported claim rate (NLI-window, sentence level)
    t3 = []
    for d in doms:
        R = [r for r in rows if r["domain"] == d and not r["refused"] and "nli_window" in r["attr_metrics"]]
        ucr = [1 - r["attr_metrics"]["nli_window"]["eas"] for r in R]
        corr = [r for r in R if r["correct"]]
        wrong = [r for r in R if not r["correct"]]
        t3.append({"domain": d, "unsupported_claim_rate": np.mean(ucr) if ucr else np.nan,
                   "UCR_correct_answers": np.mean([1 - r["attr_metrics"]["nli_window"]["eas"] for r in corr]) if corr else np.nan,
                   "UCR_wrong_answers": np.mean([1 - r["attr_metrics"]["nli_window"]["eas"] for r in wrong]) if wrong else np.nan})
    t3 = pd.DataFrame(t3)
    t3.to_csv(out / f"crossdomain_unsupported_{a.system}.csv", index=False)
    print("\n=== Unsupported claim rate (NLI-window < 0.5) ===\n", t3.round(3).to_string(index=False))

    # ---------------- LOO faithfulness agreement
    t4 = []
    for d in doms:
        R = [r for r in rows if r["domain"] == d and r.get("loo") and r["loo"]["influence"]]
        if not R:
            continue
        for m in METHODS:
            agree, rho = [], []
            for r in R:
                S = r["attribution"].get("scores", {}).get(m)
                if not S:
                    continue
                chunk_score = np.asarray(S).max(0)
                infl = np.asarray(r["loo"]["influence"])
                agree.append(float(chunk_score.argmax() == infl.argmax()))
                if np.std(chunk_score) > 0 and np.std(infl) > 0:
                    rho.append(spearmanr(chunk_score, infl).correlation)
            t4.append({"domain": d, "method": m, "n": len(agree), "top1_agree_with_LOO": np.mean(agree) if agree else np.nan,
                       "spearman_with_LOO": np.mean(rho) if rho else np.nan,
                       "frac_answers_insensitive": np.mean([max(r["loo"]["influence"]) < 0.5 for r in R])})
    if t4:
        t4 = pd.DataFrame(t4)
        t4.to_csv(out / f"crossdomain_loo_{a.system}.csv", index=False)
        print("\n=== Agreement with leave-one-out (causal) attribution ===\n", t4.round(3).to_string(index=False))

    # ---------------- Cost table (seconds per answer)
    t5 = []
    for d in doms:
        R = [r for r in rows if r["domain"] == d and r["attribution"].get("seconds")]
        row = {"domain": d}
        for m in METHODS:
            s = [r["attribution"]["seconds"].get(m) for r in R if m in r["attribution"]["seconds"]]
            row[m] = np.mean(s) if s else np.nan
        L = [r["loo"]["seconds"] for r in rows if r["domain"] == d and r.get("loo")]
        row["loo"] = np.mean(L) if L else np.nan
        row["generation"] = np.mean([r["gen_seconds"] for r in rows if r["domain"] == d and not r.get("cached")] or [np.nan])
        t5.append(row)
    t5 = pd.DataFrame(t5)
    t5.to_csv(out / f"cost_seconds_{a.system}.csv", index=False)
    print("\n=== Wall-clock seconds per answer ===\n", t5.round(3).to_string(index=False))

    # ---------------- Figure: grouped bars AttrAcc (domain x method)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = pd.DataFrame(long)
    if len(L):
        ms = [m for m in METHODS if m in set(L.method)]
        fig, ax = plt.subplots(figsize=(9, 3.8))
        w = 0.8 / len(ms)
        colors = ["#8fb3d9", "#3a78b5", "#f2b279", "#d9661f", "#6aa56a"]
        for i, m in enumerate(ms):
            sub = L[L.method == m].set_index("domain").reindex(doms)
            x = np.arange(len(doms)) + i * w - 0.4 + w / 2
            ax.bar(x, sub.AttrAcc, w, yerr=[sub.AttrAcc - sub.lo, sub.hi - sub.AttrAcc], capsize=2,
                   label=m, color=colors[i % len(colors)])
        ax.set_xticks(range(len(doms)), [DOM_LABEL[d] for d in doms], fontsize=8)
        ax.set_ylabel("Attribution accuracy\n(top chunk is gold | gold retrieved)")
        ax.set_ylim(0, 1.05); ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        (PAPER / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(PAPER / "figures" / f"{a.run}_{a.system}_attribution_by_domain.png", dpi=200)


if __name__ == "__main__":
    main()
