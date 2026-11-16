"""Is the token log-probability finding real? A pre-specified, multiplicity-controlled test (CPU only, reads scored runs).

python -m scripts.analyze_logprobs --runs v2_llama8b v2_qwen1.5b [--systems vanilla_rag improved_vanilla improved_trust]

Pre-specified (before the Qwen-14B / API-model runs are looked at):
  primary signal   lp_mean  (mean token log-probability of the whole answer)         -> the ONLY signal in the tested family
  control signal   the best NON-log-prob signal of each cell, chosen by AUROC on the same data (generous to the control)
  sensitivity      lp_nolow, lp_margin, lp_first_sent, lp_tok1 (reported, not tested)
Per cell (run x system x domain):
  AUROC for "the answer is good" with a bootstrap CI and a permutation p-value; the AUROC difference to the control (paired bootstrap).
  Value of acting on the signal at wrong-answer cost lambda in {0.5, 1, 2, 4}: utility of a threshold policy minus the best trivial policy
  (answer everything / refuse everything). The threshold is chosen by repeated 5-fold CROSS-FITTING (each question is decided with a
  threshold fitted on the other 80%), and the gain gets a paired bootstrap over questions and a one-sided p-value.
  Benjamini-Hochberg across all (cell, lambda) tests of one signal family, FDR 10%. The same procedure runs for the control.
"""
import argparse

import numpy as np
import pandas as pd

from scripts import eval_selective as ES
from src.common import RESULTS, RUNS
from src.evaluation import metrics as M

DOMS = ES.DOMS
COSTS = (0.5, 1.0, 2.0, 4.0)
PRIMARY = "lp_mean"
SENSITIVITY = ["lp_nolow", "lp_margin", "lp_first_sent", "lp_tok1"]
NON_LP = ["ret_top1", "sim_cov_mean", "nli_cov_mean", "nli_cov_min", "trust_no_contra"]


# ------------------------------------------------------------------ pure functions (unit-tested)
def crossfit_policy_utility(conf, a, r, k=5, reps=20, seed=0):
    """Expected utility of each question under a threshold chosen on the OTHER folds (endpoints allowed), averaged over
    `reps` random fold assignments. conf: higher = more confident; a/r: utility if answered/refused (policy_arrays)."""
    conf, a, r = np.asarray(conf, float), np.asarray(a, float), np.asarray(r, float)
    n = len(conf)
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    for _ in range(reps):
        folds = np.array_split(rng.permutation(n), k)
        for te in folds:
            tr = np.setdiff1d(np.arange(n), te)
            U = ES.utility_curve(conf[tr], a[tr], r[tr])                       # U[j] = answer the j most confident
            order = np.argsort(-conf[tr], kind="stable")
            ts = np.r_[np.inf, conf[tr][order]]                               # threshold j: answer iff conf >= ts[j]
            j = int(np.argmax(U))                                              # ties -> fewest answers (cautious)
            out[te] += np.where(conf[te] >= ts[j], a[te], r[te])
    return out / reps


def gain_test(conf, a, r, boots=2000, seed=0, **kw):
    """(gain over the best trivial policy, 95% CI, one-sided p) of the cross-fitted threshold policy; bootstrap over questions."""
    u = crossfit_policy_utility(conf, a, r, seed=seed, **kw)
    triv = a if a.mean() >= r.mean() else r
    d = u - triv
    rng = np.random.default_rng(seed + 1)
    bs = d[rng.integers(0, len(d), (boots, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975)), float(max((bs <= 0).mean(), 1 / boots))


def benjamini_hochberg(p, q=0.10):
    """Boolean mask of discoveries at FDR q."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    thresh = q * (np.arange(1, len(p) + 1) / len(p))
    ok = p[order] <= thresh
    k = int(np.max(np.nonzero(ok)[0])) + 1 if ok.any() else 0
    mask = np.zeros(len(p), bool)
    mask[order[:k]] = True
    return mask


def fast_auroc(conf, y):
    """Rank-based AUROC (ties count half), same value as sklearn's roc_auc_score but ~50x faster for resampling."""
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(conf)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_with_ci(conf, y, boots=1000, perms=2000, seed=0):
    rng = np.random.default_rng(seed)
    conf, y = np.asarray(conf, float), np.asarray(y)
    if y.min() == y.max():
        return float("nan"), float("nan"), float("nan"), float("nan")
    au = fast_auroc(conf, y)
    bs = []
    for _ in range(boots):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            bs.append(fast_auroc(conf[i], y[i]))
    null = [fast_auroc(conf, rng.permutation(y)) for _ in range(perms)]
    p = float((1 + sum(v >= au for v in null)) / (1 + perms))               # one-sided: better than chance
    return float(au), float(np.quantile(bs, .025)), float(np.quantile(bs, .975)), p


def selective_lift(conf, correct, coverage=0.5):
    """Accuracy among the `coverage` most confident answered answerable questions minus the overall accuracy."""
    order = np.argsort(-np.asarray(conf, float), kind="stable")
    k = max(1, int(round(coverage * len(order))))
    c = np.asarray(correct, float)
    return float(c[order[:k]].mean()), float(c.mean())


# ------------------------------------------------------------------ driver
def cell_arrays(x, signal):
    live = ~x.refused.values
    conf = np.where(live, x[signal].fillna(x.loc[live, signal].min()).values, -np.inf)
    return conf, live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--systems", nargs="+", default=["vanilla_rag", "improved_vanilla", "improved_trust"])
    ap.add_argument("--fdr", type=float, default=0.10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--qids-from", default=None, help="restrict every run to the questions of this run (paired comparison on a shared subset)")
    a_ = ap.parse_args()
    out = RESULTS / (a_.out or "logprob_analysis")
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 230)
    keep = None
    if a_.qids_from:
        ref = ES.load_frame(a_.qids_from, a_.systems[0])
        keep = set(zip(ref.domain, ref.qid))
    auc_rows, test_rows, sel_rows = [], [], []
    for run in a_.runs:
        for system in a_.systems:
            df = ES.load_frame(run, system)
            if keep is not None and not df.empty:
                df = df[[k in keep for k in zip(df.domain, df.qid)]].reset_index(drop=True)
            if df.empty or PRIMARY not in df:
                print(f"[skip] {run}/{system}: no scored rows with {PRIMARY}")
                continue
            for d in DOMS:
                x = df[df.domain == d].reset_index(drop=True)
                if len(x) < 60 or x[PRIMARY].notna().sum() < 30:
                    continue
                a1, r1 = ES.policy_arrays(x.answerable, x.correct, x.refused)
                live = ~x.refused.values
                good = (a1 > 0).astype(int)
                # control = best non-log-prob signal by AUROC on this very cell (generous to the control)
                aucs = {s: M.auroc(x.loc[live, s].values, good[live]) for s in NON_LP if s in x and x.loc[live, s].notna().all()}
                ctrl = max(aucs, key=lambda s: (aucs[s] if aucs[s] == aucs[s] else -1))
                for tag, sig in [("primary", PRIMARY), ("control", ctrl)] + [("sensitivity", s) for s in SENSITIVITY if s in x and x[s].notna().any()]:
                    conf, lv = cell_arrays(x, sig)
                    au, lo, hi, p_auc = auroc_with_ci(conf[lv], good[lv])
                    auc_rows.append({"run": run, "system": system, "domain": d, "role": tag, "signal": sig, "n_answered": int(lv.sum()),
                                     "auroc": au, "ci_lo": lo, "ci_hi": hi, "p_vs_chance": p_auc})
                    if tag in ("primary", "control"):
                        for lam in COSTS:
                            a, r = ES.policy_arrays(x.answerable, x.correct, x.refused, lam)
                            g, glo, ghi, p = gain_test(conf, a, r)
                            test_rows.append({"run": run, "system": system, "domain": d, "role": tag, "signal": sig, "wrong_cost": lam,
                                              "best_trivial": "answer" if a.mean() >= r.mean() else "refuse", "gain": g, "ci_lo": glo, "ci_hi": ghi, "p": p})
                # selective accuracy of the primary signal (answered, answerable questions only)
                m = live & x.answerable.values
                if m.sum() >= 40:
                    top, base = selective_lift(x.loc[m, PRIMARY].values, x.loc[m, "correct"].values, 0.5)
                    sel_rows.append({"run": run, "system": system, "domain": d, "n": int(m.sum()), "acc_all_answered": base,
                                     "acc_top50pct_by_lp_mean": top, "lift": top - base})
    auc, tests, sel = pd.DataFrame(auc_rows), pd.DataFrame(test_rows), pd.DataFrame(sel_rows)
    for role in ("primary", "control"):
        m = tests.role == role
        tests.loc[m, "discovery_fdr"] = benjamini_hochberg(tests.loc[m, "p"].values, a_.fdr)
    auc.to_csv(out / "logprob_auroc.csv", index=False); tests.to_csv(out / "logprob_gain_tests.csv", index=False)
    sel.to_csv(out / "logprob_selective_accuracy.csv", index=False)

    P = auc[auc.role == "primary"]; C = auc[auc.role == "control"]
    print(f"=== 1. AUROC of {PRIMARY} for 'answer is good' (permutation p vs chance; control = best non-log-prob signal per cell) ===")
    j = P.merge(C, on=["run", "system", "domain"], suffixes=("", "_ctrl"))
    j["lp_minus_control"] = j.auroc - j.auroc_ctrl
    print(j.groupby("domain").agg(cells=("auroc", "size"), lp_mean=("auroc", "mean"), control=("auroc_ctrl", "mean"), lp_minus_control=("lp_minus_control", "mean"),
                                  cells_p_lt_0_05=("p_vs_chance", lambda s: int((s < 0.05).sum()))).round(3).to_string())
    print(f"cells with {PRIMARY} AUROC significantly above chance (p<0.05): {(P.p_vs_chance < 0.05).sum()} of {len(P)}; "
          f"control: {(C.p_vs_chance < 0.05).sum()} of {len(C)}; {PRIMARY} above the control: {(j.lp_minus_control > 0).sum()} of {len(j)}")
    print(f"\n=== 2. Cross-fitted value of acting on the signal vs the best trivial policy; BH FDR {a_.fdr:.0%} within each signal family ===")
    for role in ("primary", "control"):
        t = tests[tests.role == role]
        print(f"{role:8s}: {len(t)} tests, {int(t.discovery_fdr.sum())} discoveries; by wrong-answer cost:",
              {float(k): int(v) for k, v in t.groupby('wrong_cost').discovery_fdr.sum().items()}, "| by domain:",
              {k: int(v) for k, v in t.groupby('domain').discovery_fdr.sum().items() if v})
    disc = tests[(tests.role == "primary") & (tests.discovery_fdr)]
    if len(disc):
        print("\nprimary-signal discoveries:")
        print(disc[["run", "system", "domain", "wrong_cost", "best_trivial", "gain", "ci_lo", "ci_hi", "p"]].round(3).to_string(index=False))
    hi = tests[(tests.role == "primary") & (tests.wrong_cost >= 2)]
    print(f"\nat wrong-answer cost >= 2: {int(hi.discovery_fdr.sum())} primary discoveries of {len(hi)} tests; "
          f"nominal p<0.05: {(hi.p < 0.05).sum()}; mean gain {hi.gain.mean():+.3f}")
    print("\n=== 3. Accuracy of the answers the model is most confident about (top 50% by lp_mean), answered answerable questions ===")
    print(sel.groupby("domain")[["acc_all_answered", "acc_top50pct_by_lp_mean", "lift"]].mean().round(3).to_string())
    # 4. every signal on the same footing: lift in accuracy from answering only the 50% most confident (no best-of selection)
    sigs = ["lp_mean", "lp_nolow", "lp_margin", "lp_min", "lp_first", "lp_first_sent", "lp_tok1", *NON_LP]
    rows = []
    for run in a_.runs:
        df = ES.load_frame(run, a_.systems[0])
        if keep is not None and not df.empty:
            df = df[[k in keep for k in zip(df.domain, df.qid)]].reset_index(drop=True)
        for d in DOMS:
            x = df[(df.domain == d) & df.answerable & ~df.refused]
            for sg in sigs:
                if sg in x and x[sg].notna().sum() >= 40 and len(x) >= 40:
                    top, base = selective_lift(x[sg].fillna(x[sg].min()).values, x.correct.values, 0.5)
                    rows.append({"run": run, "domain": d, "signal": sg, "lift": top - base})
    lift = pd.DataFrame(rows)
    lift.to_csv(out / "logprob_signal_lift.csv", index=False)
    tab = lift.pivot_table(index="signal", columns="domain", values="lift", aggfunc="mean")
    tab["mean"] = tab.mean(1)
    print(f"\n=== 4. Accuracy lift from answering only the 50% most confident, each signal separately ({a_.systems[0]}; 0 = no better than random) ===")
    print(tab.sort_values("mean", ascending=False).round(3).to_string())
    j = P.merge(C, on=["run", "system", "domain"], suffixes=("", "_c"))
    print(f"\n=== 5. {PRIMARY} AUROC by model (mean over systems) and by domain: does log-prob informativeness grow with model size? ===")
    print(j.groupby("run").agg(lp_mean=("auroc", "mean"), control=("auroc_c", "mean")).round(3).to_string())
    print(j.pivot_table(index="domain", columns="run", values="auroc").round(3).to_string())
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
