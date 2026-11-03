"""Direction C: refusal behaviour + cross-domain confidence calibration.

python -m scripts.analyze_refusal --run main_qwen1.5b
  1. Refusal on datasets with gold unanswerables (SQuAD2, CUAD): self-refusal vs post-hoc gate vs combined.
  2. Bridge: per-domain AUROC of each confidence signal for answer correctness.
  3. Transfer: calibrator + threshold fitted on general domain, applied to medical/legal/financial.
  4. Few-shot recalibration (ours): refit Platt scaling with n in {10,20,50} in-domain labels.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.common import PAPER, RESULTS, RUNS, read_jsonl
from src.evaluation import metrics as M

DOMS = ["squad2", "hotpotqa", "pubmedqa", "cuad", "financebench"]
SIGNALS = ["ret_top1", "ret_mean", "ret_gap", "sim_cov_mean", "nli_cov_mean", "nli_cov_min",
           "not_contra", "frac_chunks_support", "trust_score"]
LR_FEATS = ["ret_top1", "ret_mean", "ret_gap", "sim_cov_mean", "nli_cov_mean", "nli_cov_min", "not_contra",
            "frac_chunks_support"]


def feats(r):
    f = dict(r["features"])
    for k in ["nli_cov_mean", "nli_cov_min", "frac_chunks_support", "sim_cov_mean"]:
        f.setdefault(k, 0.0)  # refused answers carry no evidence support
    f["not_contra"] = 1 - f.get("contra_max", 1.0)
    f["trust_score"] = (f["ret_top1"] + f["nli_cov_mean"] + f["not_contra"]) / 3  # unsupervised composite
    return f


def frame(run, system):
    rows = []
    for d in DOMS:
        p = RUNS / run / f"{d}__{system}.scored.jsonl"
        if p.exists():
            for r in read_jsonl(p):
                rows.append({"qid": r["qid"], "domain": d, "answerable": r["answerable"], "refused": r["refused"],
                             "correct": r["correct"], **feats(r)})
    return pd.DataFrame(rows)


def fit_lr(df):
    sc = StandardScaler().fit(df[LR_FEATS])
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(df[LR_FEATS]), df["correct"].astype(int))
    return lambda x: lr.predict_proba(sc.transform(x[LR_FEATS]))[:, 1], lambda x: lr.decision_function(sc.transform(x[LR_FEATS]))


def refusal_table(V, T):
    """V: vanilla_rag frame, T: trust_rag frame. Gate threshold fitted on SQuAD2 (best F1), transferred to CUAD."""
    out, curves = [], {}
    gate_t = {}
    for d in ["squad2", "cuad"]:
        for name, df in [("vanilla", V), ("trust", T)]:
            x = df[df.domain == d]
            if x.empty or x.answerable.all():
                continue
            should = ~x.answerable.values
            if name == "trust":
                out.append({"domain": d, "mechanism": "self-refusal (trust prompt)", **M.prf(x.refused.values, should),
                            "refusal_rate": x.refused.mean(), "threshold": None})
            for sig in ["trust_score", "ret_top1", "nli_cov_mean"]:
                rows, best = M.threshold_sweep(x[sig].values, should)
                curves[f"{d}|{name}|{sig}"] = rows
                if d == "squad2":
                    gate_t[(name, sig)] = best["t"]
                t_src = gate_t.get((name, sig), best["t"])
                gate = x[sig].values < t_src
                if name == "trust":
                    gate = gate | x.refused.values
                lab = f"{'combined' if name == 'trust' else 'post-hoc gate'} [{sig}]"
                out.append({"domain": d, "mechanism": lab + (" (t from SQuAD2)" if d != "squad2" else " (best t)"),
                            **M.prf(gate, should), "refusal_rate": gate.mean(), "threshold": t_src,
                            "oracle_f1_in_domain": best["f1"]})
    return pd.DataFrame(out), curves


def bridge(V):
    """AUROC of each signal for correctness on ANSWERABLE questions, per domain."""
    rows = []
    A = V[V.answerable & ~V.refused]
    for d in DOMS:
        x = A[A.domain == d]
        if len(x) < 10:
            continue
        row = {"domain": d, "n": len(x), "accuracy": x.correct.mean()}
        for s in SIGNALS:
            row[s] = M.auroc(x[s].values, x.correct.values)
        rows.append(row)
    return pd.DataFrame(rows)


def transfer(V, source=("squad2", "hotpotqa"), target_prec=0.8, n_draws=50):
    """Fit LR calibrator on general-domain answerable questions, evaluate ECE / threshold transfer elsewhere,
    and few-shot Platt recalibration with n in-domain labels."""
    A = V[V.answerable & ~V.refused].reset_index(drop=True)
    src = A[A.domain.isin(source)]
    prob, logit = fit_lr(src)
    # threshold on source that gives >= target precision (accuracy among answered)
    ps = prob(src)
    ts = np.sort(np.unique(ps))
    t_star = next((t for t in ts if src.correct.values[ps >= t].mean() >= target_prec and (ps >= t).mean() >= 0.1), ts[len(ts) // 2])
    rng = np.random.default_rng(0)
    rows = []
    for d in DOMS:
        x = A[A.domain == d]
        if len(x) < 30:
            continue
        p, z, y = prob(x), logit(x), x.correct.values.astype(int)
        ans = p >= t_star
        row = {"domain": d, "n": len(x), "base_acc": y.mean(), "mean_conf": p.mean(),
               "ECE_transfer": M.ece(p, y), "Brier_transfer": M.brier(p, y), "AUROC": M.auroc(p, y),
               "coverage@t_src": ans.mean(), "sel_acc@t_src": y[ans].mean() if ans.any() else np.nan}
        # in-domain oracle threshold for the same precision target
        tt = np.sort(np.unique(p))
        t_in = next((t for t in tt if y[p >= t].mean() >= target_prec and (p >= t).mean() >= 0.05), None)
        row["t_src"], row["t_in_domain"] = float(t_star), (float(t_in) if t_in is not None else np.nan)
        row["coverage@t_in"] = (p >= t_in).mean() if t_in is not None else 0.0
        # few-shot Platt recalibration
        for n in [10, 20, 50]:
            if len(x) <= n + 10:
                continue
            eces = []
            for _ in range(n_draws):
                idx = rng.permutation(len(x))
                tr, te = idx[:n], idx[n:]
                if len(set(y[tr])) < 2:
                    eces.append(M.ece(np.full(len(te), y[tr].mean()), y[te]))  # degenerate: constant
                    continue
                pl = LogisticRegression(C=1.0).fit(z[tr, None], y[tr])
                eces.append(M.ece(pl.predict_proba(z[te, None])[:, 1], y[te]))
            row[f"ECE_fewshot{n}"] = float(np.mean(eces))
        rows.append(row)
    return pd.DataFrame(rows), float(t_star)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    out = RESULTS / a.run
    out.mkdir(parents=True, exist_ok=True)
    V, T = frame(a.run, "vanilla_rag"), frame(a.run, "trust_rag")

    rt, curves = refusal_table(V, T)
    rt.to_csv(out / "refusal_table.csv", index=False)
    json.dump(curves, open(out / "refusal_curves.json", "w"))
    print("\n=== Grounded refusal (positive class = should refuse) ===\n", rt.round(3).to_string(index=False))

    b = bridge(V)
    b.to_csv(out / "bridge_auroc.csv", index=False)
    print("\n=== Bridge: AUROC of confidence signal vs answer correctness (answerable, vanilla RAG) ===\n",
          b.round(3).to_string(index=False))

    tr, t_star = transfer(V)
    tr.to_csv(out / "calibration_transfer.csv", index=False)
    print(f"\n=== Calibrator trained on general domain (t_src={t_star:.3f} targets 80% selective accuracy) ===\n",
          tr.round(3).to_string(index=False))

    # figure: risk-coverage curves per domain for the transferred calibrator + trust score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = V[V.answerable & ~V.refused]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for d in DOMS:
        x = A[A.domain == d]
        if len(x) < 30:
            continue
        for j, s in enumerate(["trust_score", "nli_cov_mean"]):
            o = np.argsort(-x[s].values)
            c = x.correct.values[o]
            cov = np.arange(1, len(c) + 1) / len(c)
            ax[j].plot(cov, np.cumsum(c) / np.arange(1, len(c) + 1), label=d)
    for j, s in enumerate(["trust_score", "nli_cov_mean"]):
        ax[j].set_title(f"Selective accuracy by {s}", fontsize=9)
        ax[j].set_xlabel("coverage (fraction answered)"); ax[j].set_ylabel("accuracy on answered")
        ax[j].spines[["top", "right"]].set_visible(False)
    ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / f"{a.run}_risk_coverage.png", dpi=200)




# ---------------------------------------------------------------- B: conformal refusal under domain shift
def crc_threshold(conf, wrong, alpha):
    """Conformal risk control (Angelopoulos et al., 2024) for loss L(t) = 1[conf >= t and wrong]:
    smallest t with (n/(n+1)) * mean L(t) + 1/(n+1) <= alpha  =>  E[L_test] <= alpha (exchangeable data)."""
    conf, wrong = np.asarray(conf, float), np.asarray(wrong, bool)
    n = len(conf)
    for t in np.sort(np.unique(np.r_[conf, np.inf])):
        if (n / (n + 1)) * np.mean((conf >= t) & wrong) + 1 / (n + 1) <= alpha:
            return t
    return np.inf


def conformal_shift(V, alphas=(0.1, 0.2), source=("squad2", "hotpotqa"), ns=(20, 50), draws=200, signal="trust_score"):
    """Calibrate on general-domain questions, test per domain; then few-shot in-domain calibration."""
    A = V[V.answerable].reset_index(drop=True)
    A = A.assign(wrong=(A.correct < 1) | A.refused)          # refused counts as not-answered, see below
    A["conf"] = np.where(A.refused, -np.inf, A[signal])       # self-refusal = never answered
    src = A[A.domain.isin(source)]
    rng = np.random.default_rng(0)
    rows = []
    for alpha in alphas:
        t_src = crc_threshold(src.conf, src.wrong & ~src.refused, alpha)
        for d in DOMS:
            x = A[A.domain == d]
            if len(x) < 40:
                continue
            L = (x.conf >= t_src) & x.wrong & ~x.refused
            row = {"alpha": alpha, "domain": d, "n": len(x), "risk@t_general": L.mean(),
                   "coverage@t_general": (x.conf >= t_src).mean(), "violates": L.mean() > alpha}
            for n in ns:
                if len(x) <= n + 20:
                    continue
                risks, covs = [], []
                for _ in range(draws):
                    idx = rng.permutation(len(x)); cal, te = x.iloc[idx[:n]], x.iloc[idx[n:]]
                    t = crc_threshold(cal.conf, cal.wrong & ~cal.refused, alpha)
                    risks.append(((te.conf >= t) & te.wrong & ~te.refused).mean()); covs.append((te.conf >= t).mean())
                row[f"risk@fewshot{n}"] = float(np.mean(risks)); row[f"coverage@fewshot{n}"] = float(np.mean(covs))
            rows.append(row)
    return pd.DataFrame(rows)


def conformal_main(run):
    V = frame(run, "vanilla_rag")
    cs = conformal_shift(V)
    out = RESULTS / run
    cs.to_csv(out / "conformal_refusal.csv", index=False)
    print("\n=== B: conformal risk control (risk = answered AND wrong), general-domain vs few-shot in-domain calibration ===\n",
          cs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
    import sys
    conformal_main(sys.argv[sys.argv.index("--run") + 1])
