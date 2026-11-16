"""Phase 0: honest selective-answering evaluation (no GPU, no LLM; reads stored .scored.jsonl).

python -m scripts.eval_selective --run main_llama8b [--systems vanilla_rag trust_rag]

Why this exists: the reliability utility (+1 correct answer / correct refusal, 0 refusal of an answerable question,
-1 wrong answer) is maximised by "refuse everything" whenever accuracy is low, and plain RAG never refuses. A
confidence signal / gate is only worth something if it beats BOTH trivial policies on held-out questions, and only
if it ranks answers better than random abstention does. This script reports, per domain x system x signal:

  trivial policies   always-answer, always-refuse, best of the two, oracle selector (perfect confidence)
  AUROC              does the signal separate good answers from bad ones?
  E-AURC             risk-coverage area relative to random abstention (0) and the oracle (1), higher = better
  gain@oracle-t      best utility over all thresholds minus best trivial (optimistic: threshold tuned on the test set)
  gain@cal n         utility of a threshold picked from n labelled in-domain questions, on the remaining
                     questions, minus best trivial on the same questions (mean over random draws, P(gain<=0))
  cov@risk<=alpha    coverage reachable while keeping P(answered and wrong) <= alpha, vs random abstention
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.common import RESULTS, RUNS, read_jsonl
from src.evaluation import metrics as M
from src.pipeline.prompts import is_implicit_refusal

DOMS = ["squad2", "hotpotqa", "pubmedqa", "cuad", "financebench"]
GENERAL = ("squad2", "hotpotqa")
LR_FEATS = ["ret_top1", "ret_mean", "ret_gap", "sim_cov_mean", "nli_cov_mean", "nli_cov_min", "frac_chunks_support",
            "not_contra_chunk", "n_sent", "ans_len"]
LP_FEATS = ["lp_mean", "lp_min", "lp_first", "lp_nolow", "lp_margin"]   # token log-prob summaries (Stage 2 runs only)
LP_RAW_FEATS = ["lp_tok1", "lp_first5", "lp_first_sent"]   # from the raw per-token log-probs (runs generated after 26 Oct 2026)
SIGNALS = ["trust_v1", "trust_chunk_contra", "trust_no_contra", "ret_top1", "sim_cov_mean", "nli_cov_mean",
           "nli_cov_min", *LP_FEATS, *LP_RAW_FEATS, "checker", "lr_transfer", "lr_indomain_cv", "lr_lp_transfer",
           "lr_lp_indomain_cv", "lr_ck_transfer", "lr_ck_indomain_cv"]


def raw_lp_features(raw):
    """First-token, first-5-token and first-SENTENCE mean log-probability from the raw per-token record ({} if absent).
    The first sentence carries the direct answer (the prompts ask for it first), so it is the pre-specified refinement of lp_mean."""
    from src.common import _SENT_RE
    if not raw or not raw.get("lp"):
        return {}
    lp, toks = raw["lp"], raw.get("t") or []
    text = "".join(toks)
    m = _SENT_RE.search(text)
    cut, acc, n = (m.start() if m else len(text)), 0, 0
    for i, t in enumerate(toks):
        acc += len(t); n = i + 1
        if acc >= cut:
            break
    n = max(n, 1)
    return {"lp_tok1": float(lp[0]), "lp_first5": float(np.mean(lp[:5])), "lp_first_sent": float(np.mean(lp[:n]))}


# ------------------------------------------------------------------ policy algebra (pure numpy, unit-tested)
def policy_arrays(answerable, correct, refused, wrong_cost=1.0):
    """Per-question utility if the system ANSWERS (a) and if it REFUSES (r): +1 correct answer or correct refusal,
    0 refusal of an answerable question, -wrong_cost for a wrong answer or for answering an unanswerable question.
    A question the base system already refused cannot be answered by any policy, so a = r there."""
    answerable, refused = np.asarray(answerable, bool), np.asarray(refused, bool)
    correct = np.asarray(correct, float) > 0
    r = np.where(answerable, 0.0, 1.0)
    a = np.where(answerable & correct, 1.0, -float(wrong_cost))
    a = np.where(refused, r, a)
    return a, r


def utility_curve(conf, a, r):
    """U[k] = mean utility when the k most confident questions are answered and the rest refused (k = 0..n)."""
    order = np.argsort(-np.asarray(conf, float), kind="stable")
    n = len(order)
    return (r.sum() + np.r_[0.0, np.cumsum((a - r)[order])]) / n


def oracle_utility(a, r):
    return float((r.sum() + np.clip(a - r, 0, None).sum()) / len(a))


def trivial(a, r):
    return {"always_answer": float(a.mean()), "always_refuse": float(r.mean()), "best_trivial": float(max(a.mean(), r.mean()))}


def e_aurc(conf, wrong, answered):
    """(AURC_random - AURC_signal) / (AURC_random - AURC_oracle) on the answered questions; 0 = random, 1 = oracle."""
    conf, wrong = np.asarray(conf, float)[answered], np.asarray(wrong, float)[answered]
    if len(conf) < 10 or wrong.min() == wrong.max():
        return float("nan")
    risk_at = lambda c: (np.cumsum(wrong[np.argsort(-c, kind="stable")]) / np.arange(1, len(wrong) + 1)).mean()
    rand, sig, orc = wrong.mean(), risk_at(conf), risk_at(-wrong)
    return float((rand - sig) / (rand - orc))


def coverage_at_risk(conf, wrong_if_answered, answered, alpha):
    """Largest coverage with P(answered and wrong) <= alpha, signal vs random abstention (risk scales linearly)."""
    n = len(conf)
    ok = answered.astype(bool)
    c = np.where(ok, np.asarray(conf, float), -np.inf)
    order = np.argsort(-c, kind="stable")
    w = np.where(ok, wrong_if_answered, 0.0)[order]
    risk = np.cumsum(w) / n
    k = int(np.max(np.nonzero(risk <= alpha)[0], initial=-1)) + 1       # answers made
    k = min(k, int(ok.sum()))
    base = (np.where(ok, wrong_if_answered, 0.0).sum()) / n              # risk if everything answerable is answered
    rand_k = ok.sum() if base <= alpha else ok.sum() * alpha / base
    return k / n, float(rand_k) / n


def calibrated_gain(conf, a, r, n_cal, draws, rng):
    """Pick the utility-optimal threshold on n_cal random questions (endpoints 'answer all'/'refuse all' allowed),
    evaluate on the rest, subtract the best trivial policy on the same rest. Returns (mean gain, P(gain<=0), mean U)."""
    n = len(conf)
    gains, us = [], []
    for _ in range(draws):
        perm = rng.permutation(n)
        cal, te = perm[:n_cal], perm[n_cal:]
        ts = np.unique(np.r_[conf[cal], np.inf, -np.inf])
        best_t = max(ts, key=lambda t: (np.where(conf[cal] >= t, a[cal], r[cal]).mean(), -t))
        u = np.where(conf[te] >= best_t, a[te], r[te]).mean()
        gains.append(u - max(a[te].mean(), r[te].mean())); us.append(u)
    gains = np.asarray(gains)
    return float(gains.mean()), float((gains <= 0).mean()), float(np.mean(us))


# ------------------------------------------------------------------ data
def row_features(rec, implicit):
    f = dict(rec["features"])
    sc = rec["attribution"].get("scores", {})
    refused = bool(rec["refused"]) or (implicit and is_implicit_refusal(rec["answer"]))
    for k in ["nli_cov_mean", "nli_cov_min", "frac_chunks_support", "sim_cov_mean", "n_sent"]:
        f.setdefault(k, 0.0)
    contra_old = f.get("contra_max", 1.0)
    contra_chunk = contra_old
    if sc.get("nli_chunk_contra") is not None and sc.get("sim_sent") is not None:
        # whole-chunk contradiction against the chunk each sentence is about (proxy for the window-level fix)
        Cc, rel = np.asarray(sc["nli_chunk_contra"]), np.asarray(sc["sim_sent"]).argmax(1)
        contra_chunk = float(Cc[np.arange(len(Cc)), rel].max())
    f["not_contra_chunk"] = 1 - contra_chunk
    base = f["ret_top1"], f["nli_cov_mean"]
    f["trust_v1"] = (base[0] + base[1] + 1 - contra_old) / 3
    f["trust_chunk_contra"] = (base[0] + base[1] + 1 - contra_chunk) / 3
    f["trust_no_contra"] = (base[0] + base[1]) / 2
    f["ans_len"] = min(len(rec["answer"].split()), 200) / 200
    lp = rec.get("lp") or {}
    if lp:  # higher = more confident for every lp_* signal
        f.update(lp_mean=lp["mean"], lp_min=lp["min"], lp_first=lp["first_mean"], lp_nolow=1 - lp["frac_low"],
                 lp_margin=lp["margin_mean"] if lp.get("margin_mean") is not None else np.nan)
    f.update(raw_lp_features(rec.get("lp_raw")))
    if "p_halluc" in rec:   # supervised RAGTruth checker (Stage 2 Part B): higher = judged better supported
        f["checker"] = 1 - rec["p_halluc"]
    f["_refused"] = refused
    return f


def load_checker(run_dir):
    """{(domain, system, qid): p_halluc} written by scripts.apply_checker (Stage 2 Part B); empty until that has been run."""
    files = sorted(Path(run_dir).glob("checker_*.jsonl"))
    return {(c["domain"], c["system"], c["qid"]): c["p_halluc"] for c in read_jsonl(files[0])} if files else {}


def load_frame(run, system, implicit=True, runs_dir=None):
    rows = []
    ck = load_checker((runs_dir or RUNS) / run)
    for d in DOMS:
        p = (runs_dir or RUNS) / run / f"{d}__{system}.scored.jsonl"
        if not p.exists():
            continue
        for rec in read_jsonl(p):
            if (d, system, rec["qid"]) in ck:
                rec = dict(rec, p_halluc=ck[(d, system, rec["qid"])])
            f = row_features(rec, implicit)
            rows.append({"domain": d, "qid": rec["qid"], "answerable": bool(rec["answerable"]),
                         "refused": f.pop("_refused"), "correct": float(rec.get("correct") or 0.0),
                         **{k: float(v) for k, v in f.items() if isinstance(v, (int, float, np.floating))}})
    return pd.DataFrame(rows)


def add_learned(df, seed=0):
    """lr_transfer: logistic regression trained on OTHER general-domain questions (general -> everything else; the two
    general domains cross-fit each other). lr_indomain_cv: 5-fold CV inside the domain (optimistic: uses in-domain labels).
    lr_lp_*: the same with the token log-prob features added, only when the run has them (Stage 2)."""
    a, r = policy_arrays(df.answerable, df.correct, df.refused)
    df = df.assign(good=((a > 0) & ~df.refused).astype(int))
    live = ~df.refused
    variants = [("lr_transfer", "lr_indomain_cv", LR_FEATS)]
    has_lp = all(c in df and df.loc[live, c].notna().any() for c in LP_FEATS)
    has_ck = "checker" in df and df.loc[live, "checker"].notna().any()
    if has_lp:
        df = df.assign(**{c: df[c].fillna(df.loc[live, c].median()) for c in LP_FEATS})
        variants.append(("lr_lp_transfer", "lr_lp_indomain_cv", LR_FEATS + LP_FEATS))
    if has_ck:   # everything we have, including the supervised checker
        df = df.assign(checker=df["checker"].fillna(df.loc[live, "checker"].median()))
        variants.append(("lr_ck_transfer", "lr_ck_indomain_cv", LR_FEATS + (LP_FEATS if has_lp else []) + ["checker"]))

    for tname, cname, feats in variants:
        df[tname] = -np.inf
        df[cname] = -np.inf

        def fit(tr, feats=feats):
            sc = StandardScaler().fit(tr[feats])
            lr = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(tr[feats]), tr.good)
            return lambda x: lr.predict_proba(sc.transform(x[feats]))[:, 1]

        for d in DOMS:
            te_m = (df.domain == d) & live
            if not te_m.any():
                continue
            src = [g for g in GENERAL if g != d] or list(GENERAL)
            tr = df[df.domain.isin(src) & live]
            if tr.good.nunique() == 2:
                df.loc[te_m, tname] = fit(tr)(df[te_m])
            x = df[te_m]
            if x.good.nunique() == 2 and x.good.value_counts().min() >= 5:
                out = np.zeros(len(x))
                for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, x.good):
                    out[te_i] = fit(x.iloc[tr_i])(x.iloc[te_i])
                df.loc[te_m, cname] = out
    return df


# ------------------------------------------------------------------ evaluation
def evaluate(df, ncals=(20, 50), draws=200, alpha=0.2, boots=300, wrong_cost=1.0):
    rng = np.random.default_rng(0)
    rows = []
    for d in DOMS:
        x = df[df.domain == d].reset_index(drop=True)
        if x.empty:
            continue
        a, r = policy_arrays(x.answerable, x.correct, x.refused, wrong_cost)
        a1, _ = policy_arrays(x.answerable, x.correct, x.refused)      # outcome labels do not depend on the cost
        tv, orc = trivial(a, r), oracle_utility(a, r)
        live = ~x.refused.values
        wrong = (a1 < 0).astype(float)                       # answering this question would be wrong
        good = (a1 > 0).astype(int)
        for s in SIGNALS:
            if s not in x or not x.loc[live, s].notna().any():
                continue                                   # e.g. lp_* signals in runs generated without log-probs
            conf = np.where(live, x[s].fillna(x.loc[live, s].min()).values, -np.inf)
            if np.isinf(conf[live]).any() or not np.isfinite(conf[live]).any():
                continue
            U = utility_curve(conf, a, r)
            au = M.auroc(conf[live], good[live])
            lo = hi = float("nan")
            if live.sum() > 20 and good[live].min() != good[live].max():
                idx = np.nonzero(live)[0]
                bs = [M.auroc(conf[b], good[b]) for b in (rng.choice(idx, len(idx)) for _ in range(boots))]
                bs = [v for v in bs if v == v]
                lo, hi = (float(np.quantile(bs, .025)), float(np.quantile(bs, .975))) if bs else (np.nan, np.nan)
            cov_sig, cov_rand = coverage_at_risk(conf, wrong, live, alpha)
            row = {"domain": d, "signal": s, "n": len(x), "n_answered": int(live.sum()), **tv, "oracle_selector": orc,
                   "auroc": au, "auroc_lo": lo, "auroc_hi": hi, "e_aurc": e_aurc(conf, wrong, live),
                   "U_oracle_t": float(U.max()), "gain_oracle_t": float(U.max() - tv["best_trivial"]),
                   f"cov@risk<={alpha}": cov_sig, f"cov_random@risk<={alpha}": cov_rand}
            for nc in ncals:
                if len(x) > nc + 20:
                    g, p0, u = calibrated_gain(conf, a, r, nc, draws, rng)
                    row.update({f"gain@cal{nc}": g, f"P(gain<=0)@cal{nc}": p0, f"U@cal{nc}": u})
            rows.append(row)
    return pd.DataFrame(rows)


def reported_vs_trivial(run, df):
    """The previously reported systems (v1 full gate, v2) set against the trivial policies on the same questions."""
    out = []
    res = RESULTS / run
    v1 = pd.read_csv(res / "systems_ablation.csv") if (res / "systems_ablation.csv").exists() else None
    v2 = {}
    for f in res.glob("v2_system_*ncal*.csv"):
        v2[f.stem.replace("v2_system_", "")] = pd.read_csv(f)
    for d in DOMS:
        x = df[df.domain == d]
        if x.empty:
            continue
        a, r = policy_arrays(x.answerable, x.correct, x.refused)
        row = {"domain": d, "always_refuse": float(r.mean()), "plain_rag_incl_implicit_refusals": float(a.mean())}
        if v1 is not None:
            for s in ["vanilla_rag", "full"]:
                q = v1[(v1.domain == d) & (v1.system == s)]
                if len(q):
                    row[f"v1_{s}_utility"] = float(q.reliability_u.iloc[0])
        for k, t in v2.items():
            q = t[t.domain == d]
            if len(q) and "v2_safe_utility" in q:
                row[f"v2_safe_utility[{k}]"] = float(q.v2_safe_utility.iloc[0])
        best = max(row["always_refuse"], row["plain_rag_incl_implicit_refusals"])
        row["best_trivial"] = best
        for k in [c for c in row if c.endswith("utility") or c.startswith("v2_safe")]:
            row[f"{k} minus best_trivial"] = row[k] - best
        out.append(row)
    return pd.DataFrame(out)


def breakeven_cost(answerable, correct, refused):
    """Wrong-answer cost at which answering everything and refusing everything are equally good (utility is linear in the
    cost, so it is exact). NaN: refusing everything wins even at zero cost. inf: nothing is ever wrong."""
    a0, r = policy_arrays(answerable, correct, refused, 0.0)
    a1, _ = policy_arrays(answerable, correct, refused, 1.0)
    gain_at_zero, per_unit = a0.mean() - r.mean(), a0.mean() - a1.mean()
    if gain_at_zero <= 0:
        return float("nan")
    return float("inf") if per_unit <= 0 else float(gain_at_zero / per_unit)


def cost_sweep(df, costs=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0), n_cal=50, draws=150, signals=None):
    """For every wrong-answer cost: the trivial policies, what a perfect selector could add (the ceiling), and whether each
    signal's calibrated threshold beats the best trivial policy on held-out questions. Same splits for every cost."""
    rows = []
    for d in DOMS:
        x = df[df.domain == d].reset_index(drop=True)
        if x.empty:
            continue
        live = ~x.refused.values
        be = breakeven_cost(x.answerable, x.correct, x.refused)
        for c in costs:
            a, r = policy_arrays(x.answerable, x.correct, x.refused, c)
            tv, orc = trivial(a, r), oracle_utility(a, r)
            base = {"domain": d, "wrong_cost": c, "breakeven_cost": be, **tv,
                    "best_trivial_is": "answer" if tv["always_answer"] >= tv["always_refuse"] else "refuse",
                    "oracle_selector": orc, "oracle_gain": orc - tv["best_trivial"]}
            for s in (signals or SIGNALS):
                if s not in x or not x.loc[live, s].notna().any():
                    continue
                conf = np.where(live, x[s].fillna(x.loc[live, s].min()).values, -np.inf)
                if np.isinf(conf[live]).any() or not np.isfinite(conf[live]).any():
                    continue
                row = {**base, "signal": s, "deployable": "indomain_cv" not in s,
                       "gain_oracle_t": float(utility_curve(conf, a, r).max() - tv["best_trivial"])}
                if len(x) > n_cal + 20:
                    g, p0, _ = calibrated_gain(conf, a, r, n_cal, draws, np.random.default_rng(0))
                    row.update({f"gain@cal{n_cal}": g, f"P(gain<=0)@cal{n_cal}": p0})
                rows.append(row)
    return pd.DataFrame(rows)


def cost_matrix(sweep, n_cal=50, min_gain=0.01, max_p=0.1):
    """Domain x cost table. Cell = best trivial policy (A answer / R refuse), then the best DEPLOYABLE signal's held-out gain
    over it; '*' = gain > min_gain and P(gain<=0) < max_p. Second table: the ceiling a perfect selector could add."""
    g, p = f"gain@cal{n_cal}", f"P(gain<=0)@cal{n_cal}"
    dep = sweep[sweep.deployable & sweep[g].notna()]
    cells, ceil, wins = {}, {}, {}
    for (d, c), grp in dep.groupby(["domain", "wrong_cost"]):
        b = grp.sort_values(g, ascending=False).iloc[0]
        win = (grp[g] > min_gain) & (grp[p] < max_p)
        cells[(d, c)] = f"{b.best_trivial_is[0].upper()} {b[g]:+.3f}{'*' if win.any() else ''}"
        ceil[(d, c)] = f"{b.oracle_gain:.2f}"
        wins[c] = wins.get(c, 0) + int(win.any())
    mk = lambda cells_: pd.Series(cells_).unstack().reindex(index=[d for d in DOMS if d in sweep.domain.values])
    return mk(cells), mk(ceil), wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--systems", nargs="+", default=["vanilla_rag", "trust_rag"])
    ap.add_argument("--no-implicit", action="store_true", help="do not treat 'the passages do not mention...' as refusal")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--wrong-cost", type=float, default=1.0, help="penalty for a wrong answer (the mid-sem utility used 1)")
    ap.add_argument("--cost-sweep", action="store_true", help="also sweep the wrong-answer cost (see cost_sweep)")
    ap.add_argument("--costs", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    a = ap.parse_args()
    if a.wrong_cost <= 0 or min(a.costs) <= 0:
        raise SystemExit("costs must be > 0")
    out = RESULTS / a.run
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    allres = []
    suffix = "" if a.wrong_cost == 1.0 else f"_cost{a.wrong_cost:g}"
    for system in a.systems:
        df = load_frame(a.run, system, implicit=not a.no_implicit)
        if df.empty:
            continue
        df = add_learned(df)
        res = evaluate(df, alpha=a.alpha, wrong_cost=a.wrong_cost).assign(system=system)
        allres.append(res)
        res.to_csv(out / f"selective_{system}{suffix}.csv", index=False)
        cols = ["domain", "signal", "n_answered", "best_trivial", "oracle_selector", "auroc", "e_aurc", "gain_oracle_t",
                "gain@cal20", "gain@cal50", "P(gain<=0)@cal50", f"cov@risk<={a.alpha}", f"cov_random@risk<={a.alpha}"]
        print(f"\n=== {a.run} / {system}: selective answering vs trivial policies (wrong-answer cost {a.wrong_cost:g}) ===")
        print(res[[c for c in cols if c in res]].round(3).to_string(index=False))
        if a.wrong_cost == 1.0:      # the stored v1/v2 utilities were computed with cost 1 and cannot be recomputed
            rv = reported_vs_trivial(a.run, df)
            rv.to_csv(out / f"selective_reported_vs_trivial_{system}.csv", index=False)
            print(f"\n--- previously reported systems vs trivial policies ({system} answers) ---")
            print(rv.round(3).to_string(index=False))
        if a.cost_sweep:
            sw = cost_sweep(df, a.costs).assign(system=system)
            sw.to_csv(out / f"selective_cost_sweep_{system}.csv", index=False)
            cells, ceil, wins = cost_matrix(sw)
            be = sw.drop_duplicates("domain").set_index("domain").breakeven_cost
            print(f"\n=== {a.run} / {system}: wrong-answer cost sweep ===")
            print("Best trivial policy (A answer-all / R refuse-all) and best deployable signal's held-out gain over it"
                  " ('*' = significant):")
            print(cells.to_string())
            print("\nCeiling: what a PERFECT selector would add over the best trivial policy:")
            print(ceil.to_string())
            print("\nBreak-even cost (answer-all = refuse-all; nan = refuse-all wins even at cost 0):",
                  {d: (round(v, 2) if v == v else "nan") for d, v in be.items()})
            print("Domains where some deployable signal wins, per cost:", {c: f"{w}/{len(be)}" for c, w in sorted(wins.items())})
    if allres:
        pd.concat(allres).to_csv(out / f"selective_all{suffix}.csv", index=False)


if __name__ == "__main__":
    main()
