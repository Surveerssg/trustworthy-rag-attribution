"""Stage 2 Part A analysis: what the improved-retrieval / oracle / evidence-removal / log-prob / leave-one-out runs say.

python -m scripts.ingest_kaggle trag_results_stage2A.zip      # unpack first
python -m scripts.analyze_stage2 --model llama8b              # then qwen1.5b
Reads  experiments/runs/v2_<model>/*.scored.jsonl (Part A) and experiments/runs/main_<model>/ (mid-sem runs).
Writes experiments/results/v2_<model>/stage2_*.csv and selective_*.csv.  CPU only, no model downloads.

Sections (each skipped with a message if its systems are missing):
  0 replication   v2 vanilla_rag vs mid-sem vanilla_rag: same plans/prompt/model, so answers should (nearly) match
  1 retrieval     base -> improved retrieval: Hit@5, accuracy (paired bootstrap), wrong/unsupported rates, and attribution
                  accuracy on the MATCHED questions where gold was retrieved under both (no selection effect)
  2 oracle        gold forced into the context: generator ceiling, retrieval loss, attribution without the retrieval
                  confound, and position bias of the gold passage
  3 removal       gold evidence deleted -> the right behaviour is to refuse. Refusal/false-answer/leak rates, and how well
                  each confidence signal separates "evidence present" from "evidence removed", with the gate threshold
                  fitted on the general domains and transferred (a ground-truth refusal task in EVERY domain)
  4 selective     the Phase 0 protocol (vs always-answer / always-refuse) with the token log-prob signals added
  5 faithfulness  leave-one-out influence vs each attribution method: is the attributed passage actually used?
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts import eval_selective as ES
from src.attribution.methods import refresh_record
from src.common import RESULTS, RUNS, read_jsonl
from src.evaluation import metrics as M
from src.pipeline.prompts import is_implicit_refusal

DOMS, GENERAL = ES.DOMS, ES.GENERAL
ATTR = ["sim_sent", "nli_chunk", "nli_window", "nli_set"]
FILTER_T = 0.5
_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
REMOVAL_SIGNALS = ["ret_top1", "sim_cov_mean", "nli_cov_mean", "nli_cov_min", "trust_no_contra", "trust_v1", *ES.LP_FEATS,
                   *ES.LP_RAW_FEATS, "checker"]


# ------------------------------------------------------------------ record helpers
def strip_cites(s):
    return re.sub(r"\s+", " ", _CITE.sub("", s)).strip()


def load(runs_dir, run, system, tiebreak=True):
    """{domain: {qid: scored record}} (empty dict if the file does not exist).
    tiebreak=True applies the nli_set tie-break (paper/findings/stage2a_findings.md section 3) in memory to records scored before the fix; the files on
    disk are never rewritten, so the frozen mid-sem runs keep reproducing the numbers shown at the mid-sem."""
    out = {}
    ck = ES.load_checker(Path(runs_dir) / run)       # Stage 2 Part B scores, if that has been run for this run
    for d in DOMS:
        p = Path(runs_dir) / run / f"{d}__{system}.scored.jsonl"
        if p.exists():
            recs = (dict(r, p_halluc=ck[(d, system, r["qid"])]) if (d, system, r["qid"]) in ck else r for r in read_jsonl(p))
            out[d] = {r["qid"]: (refresh_record(r) if tiebreak else r) for r in recs}
    return out


def refused(r, implicit=True):
    return bool(r["refused"]) or (implicit and is_implicit_refusal(r["answer"]))


def hit(r):
    return (r.get("retrieval") or {}).get("hit")


def gold_rank(r):
    """1-based position of the first gold chunk in the context shown to the model, None if absent."""
    gold = set(r.get("gold_chunk_ids") or [])
    return next((i + 1 for i, c in enumerate(r["retrieved"]) if c["chunk_id"] in gold), None)


def attr_acc(r, m):
    return ((r.get("attr_metrics") or {}).get(m) or {}).get("attr_acc")


def unsup_rate(r):
    sc = (r.get("attribution") or {}).get("scores") or {}
    S = sc.get("nli_set", sc.get("nli_window"))
    return float((np.asarray(S).max(1) < FILTER_T).mean()) if S else None


def nm(x):
    x = [v for v in x if v is not None and v == v]
    return float(np.mean(x)) if x else float("nan")


def common(a, b, d):
    return [q for q in a.get(d, {}) if q in b.get(d, {})]


# ------------------------------------------------------------------ 0. replication
def replication(v2, main):
    rows = []
    for d in DOMS:
        q = common(v2, main, d)
        if not q:
            continue
        norm = lambda r: " ".join(r["answer"].lower().split())
        ans = [i for i in q if v2[d][i]["answerable"]]
        rows.append({"domain": d, "n": len(q),
                     "same_context": nm(float([c["chunk_id"] for c in v2[d][i]["retrieved"]] ==
                                              [c["chunk_id"] for c in main[d][i]["retrieved"]]) for i in q),
                     "same_answer": nm(float(norm(v2[d][i]) == norm(main[d][i])) for i in q),
                     "acc_v2": nm(v2[d][i]["correct"] for i in ans), "acc_main": nm(main[d][i]["correct"] for i in ans)})
    df = pd.DataFrame(rows)
    if len(df):
        df["acc_diff"] = df.acc_v2 - df.acc_main
    return df


# ------------------------------------------------------------------ 1. improved retrieval
def retrieval_effect(base, imp, label):
    rows = []
    for d in DOMS:
        q = common(base, imp, d)
        if not q:
            continue
        A = [i for i in q if base[d][i]["answerable"]]
        U = [i for i in q if not base[d][i]["answerable"]]
        cb = np.array([base[d][i]["correct"] for i in A], float)
        ci = np.array([imp[d][i]["correct"] for i in A], float)
        bs = M.paired_bootstrap(ci, cb) if len(A) > 1 else {"diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan}
        wrong = lambda S: nm(float((not refused(S[d][i])) and S[d][i]["correct"] < 1) for i in A)
        ranswered = lambda S, idx: [S[d][i] for i in idx if not refused(S[d][i])]
        matched = [i for i in A if attr_acc(base[d][i], "nli_set") is not None and attr_acc(imp[d][i], "nli_set") is not None]
        row = {"comparison": label, "domain": d, "n": len(q), "n_answerable": len(A),
               "hit_base": nm(hit(base[d][i]) for i in A), "hit_imp": nm(hit(imp[d][i]) for i in A),
               "acc_base": cb.mean() if len(A) else np.nan, "acc_imp": ci.mean() if len(A) else np.nan,
               "acc_delta": bs["diff"], "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"], "p": bs["p"],
               "wrong_base": wrong(base), "wrong_imp": wrong(imp),
               "answered_unanswerable_base": nm(float(not refused(base[d][i])) for i in U),
               "answered_unanswerable_imp": nm(float(not refused(imp[d][i])) for i in U),
               "unsup_base": nm(unsup_rate(r) for r in ranswered(base, q)), "unsup_imp": nm(unsup_rate(r) for r in ranswered(imp, q)),
               "n_matched_attr": len(matched)}
        for m in ATTR:
            ok = [i for i in matched if attr_acc(base[d][i], m) is not None and attr_acc(imp[d][i], m) is not None]
            row[f"attr_{m}_base"] = nm(attr_acc(base[d][i], m) for i in ok)
            row[f"attr_{m}_imp"] = nm(attr_acc(imp[d][i], m) for i in ok)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 2. oracle context
def oracle_effect(base, orc):
    rows = []
    for d in DOMS:
        q = [i for i in common(base, orc, d) if base[d][i]["answerable"] and base[d][i]["gold_chunk_ids"]]
        if not q:
            continue
        cb = np.array([base[d][i]["correct"] for i in q], float)
        co = np.array([orc[d][i]["correct"] for i in q], float)
        bs = M.paired_bootstrap(co, cb)
        got = np.array([bool(hit(base[d][i])) for i in q])
        row = {"domain": d, "n": len(q), "gold_retrieved_base": got.mean(), "acc_base": cb.mean(), "acc_oracle": co.mean(),
               "retrieval_loss": bs["diff"], "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
               "acc_base_gold_retrieved": cb[got].mean() if got.any() else np.nan,
               "acc_base_gold_missed": cb[~got].mean() if (~got).any() else np.nan,
               "acc_oracle_gold_missed": co[~got].mean() if (~got).any() else np.nan}
        for m in ["sim_sent", "nli_set"]:
            row[f"attr_{m}_oracle"] = nm(attr_acc(orc[d][i], m) for i in q)
            row[f"attr_{m}_base|gold_retrieved"] = nm(attr_acc(base[d][i], m) for i in q)
        rows.append(row)
    return pd.DataFrame(rows)


def oracle_position(orc):
    """Accuracy / attribution accuracy by the position of the gold passage in the shuffled oracle context."""
    rows, corr = [], []
    for d in DOMS:
        R = [r for r in orc.get(d, {}).values() if r["answerable"] and gold_rank(r)]
        if not R:
            continue
        pos = np.array([gold_rank(r) for r in R])
        cor = np.array([r["correct"] for r in R], float)
        for p in sorted(set(pos.tolist())):
            S = [r for r, x in zip(R, pos) if x == p]
            rows.append({"domain": d, "gold_position": p, "n": len(S), "acc": nm(r["correct"] for r in S),
                         "attr_nli_set": nm(attr_acc(r, "nli_set") for r in S), "attr_sim_sent": nm(attr_acc(r, "sim_sent") for r in S)})
        rho = p_val = np.nan
        if len(R) >= 20 and len(set(pos.tolist())) > 1 and cor.min() != cor.max():
            rho, p_val = spearmanr(pos, cor)
        corr.append({"domain": d, "n": len(R), "spearman_pos_vs_correct": rho, "p": p_val})
    return pd.DataFrame(rows), pd.DataFrame(corr)


# ------------------------------------------------------------------ 3. evidence removal
def _conf_vector(recs, sig, implicit):
    """Signal values with refused answers pushed below everything (a refusal is the least confident outcome)."""
    f = [ES.row_features(r, implicit) for r in recs]
    v = np.array([x.get(sig, np.nan) for x in f], float)
    ref = np.array([x["_refused"] for x in f])
    live = v[~ref & ~np.isnan(v)]
    if len(live) == 0:
        return None
    v = np.where(np.isnan(v), live.min(), v)
    return np.where(ref, live.min() - 1.0, v)


def removal_pairs(rem, withev, d):
    """Questions with gold retrieved in the with-evidence run: (removed record, with-evidence record) pairs."""
    q = [i for i in common(rem, withev, d) if hit(withev[d][i])]
    return [rem[d][i] for i in q], [withev[d][i] for i in q]


def removal_summary(rem, withev, label, implicit=True):
    rows = []
    for d in DOMS:
        R, W = removal_pairs(rem, withev, d)
        if not R:
            continue
        clean = [r for r in R if not (r.get("meta") or {}).get("residual_answer_in_context")]
        answered = [r for r in clean if not refused(r, implicit)]
        leak = []
        for r in answered:
            if r["answer_type"] == "label":
                continue                                  # a 3-way label matches by chance: not informative
            c, _ = M.correctness(strip_cites(r["answer"]), (r.get("meta") or {}).get("original_gold_answers") or [], r["answer_type"])
            if c is not None:
                leak.append(c)
        rows.append({"system": label, "domain": d, "n_pairs": len(R), "n_clean": len(clean),
                     "residual_share": 1 - len(clean) / len(R),
                     "refuse_removed": nm(float(refused(r, implicit)) for r in clean),
                     "refuse_removed_explicit": nm(float(r["refused"]) for r in clean),
                     "refuse_with_evidence": nm(float(refused(r, implicit)) for r in W),
                     "false_answer_when_removed": nm(float(not refused(r, implicit)) for r in clean),
                     "leak_original_answer": nm(leak), "n_leak_judged": len(leak),
                     "acc_with_evidence": nm(r["correct"] for r in W)})
    df = pd.DataFrame(rows)
    if len(df):
        df["refusal_gap"] = df.refuse_removed - df.refuse_with_evidence
    return df


def youden_threshold(v, should):
    """Threshold t (refuse iff v < t) maximising balanced accuracy = (TPR + TNR) / 2; 0.5 is the trivial level.
    F1 is not used for the choice: on balanced data 'refuse everything' already scores F1 = 0.667."""
    best_t, best_ba = float(v.max() + 1.0), 0.5
    for t in np.unique(v):
        pred = v < t
        ba = (pred[should].mean() + (~pred[~should]).mean()) / 2
        if ba > best_ba + 1e-12:
            best_t, best_ba = float(t), float(ba)
    return best_t, best_ba


def _balanced_acc(pred, should):
    return float((pred[should].mean() + (~pred[~should]).mean()) / 2)


def removal_signals(rem, withev, label, implicit=True):
    """Ground-truth refusal task: with-evidence answers must be given, evidence-removed answers must be refused.
    AUROC for 'evidence present' (higher confidence = present), balanced accuracy (0.5 = trivial) at the best in-domain
    threshold vs at the threshold fitted on the other general domain (the mid-sem calibration-transfer test, now with real
    unanswerable questions in every domain), F1 at the same thresholds and the F1 of refusing everything for reference."""
    data = {}
    for d in DOMS:
        R, W = removal_pairs(rem, withev, d)
        if len(R) < 10:
            continue
        data[d] = (W + R, np.r_[np.ones(len(W)), np.zeros(len(R))].astype(int))
    rows = []
    for sig in REMOVAL_SIGNALS:
        conf = {}
        for d, (recs, y) in data.items():
            if sig in ES.LP_FEATS and not all(r.get("lp") for r in recs):
                continue                                  # a side generated without log-probs: not comparable
            if sig in ES.LP_RAW_FEATS and not all(r.get("lp_raw") for r in recs):
                continue                                  # raw per-token log-probs exist only for newer runs
            if sig == "checker" and not all("p_halluc" in r for r in recs):
                continue                                  # checker not applied to one side (Part B not run for it)
            v = _conf_vector(recs, sig, implicit)
            if v is not None and np.isfinite(v).all():
                conf[d] = (v, y)
        for d, (v, y) in conf.items():
            should = y == 0
            t_in, ba_in = youden_threshold(v, should)
            row = {"system": label, "signal": sig, "domain": d, "n_pairs": int(y.sum()), "auroc_present": M.auroc(v, y),
                   "bal_acc_in_domain": ba_in, "f1_in_domain": M.prf(v < t_in, should)["f1"],
                   "f1_refuse_all": M.prf(np.ones_like(should), should)["f1"]}
            src = [g for g in GENERAL if g != d and g in conf] or [g for g in GENERAL if g in conf]
            if src:
                vv, yy = np.concatenate([conf[g][0] for g in src]), np.concatenate([conf[g][1] for g in src])
                tg, _ = youden_threshold(vv, yy == 0)
                pred = v < tg
                row.update({"t_general": tg, "bal_acc_transfer": _balanced_acc(pred, should),
                            "f1_transfer": M.prf(pred, should)["f1"], "refusal_rate_at_t_general": float(pred.mean())})
                row["transfer_loss"] = ba_in - row["bal_acc_transfer"]
            rows.append(row)
    for d, (recs, y) in data.items():                    # prompted self-refusal needs no threshold
        pred = np.array([refused(r, implicit) for r in recs])
        pr = M.prf(pred, y == 0)
        rows.append({"system": label, "signal": "self_refusal", "domain": d, "n_pairs": int(y.sum()),
                     "bal_acc_in_domain": _balanced_acc(pred, y == 0), "f1_in_domain": pr["f1"],
                     "f1_refuse_all": M.prf(np.ones(len(y), bool), y == 0)["f1"], "refusal_rate_at_t_general": float(pred.mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 5. leave-one-out faithfulness
def loo_faithfulness(recs, taus=(0.2, 0.5)):
    """influence_j = 1 - bidirectional NLI agreement between the full answer and the answer regenerated without chunk j.
    tau = how much the answer must change for the chunk to count as 'used'."""
    rows = []
    for d in DOMS:
        R = [r for r in recs.get(d, {}).values()
             if (r.get("loo") or {}).get("influence") and not r["refused"] and (r.get("attribution") or {}).get("scores")]
        if not R:
            continue
        for m in ATTR:
            for tau in taus:
                agree, unused, gold_used, faithful, rho = [], [], [], [], []
                for r in R:
                    S = (r["attribution"]["scores"] or {}).get(m)
                    infl = np.asarray(r["loo"]["influence"], float)
                    if S is None or len(S) == 0 or np.asarray(S).shape[1] != len(infl):
                        continue
                    cs = np.asarray(S, float).max(0)                 # answer-level support of each chunk
                    j = int(cs.argmax())
                    gold = set(r.get("gold_chunk_ids") or [])
                    gi = [k for k, c in enumerate(r["retrieved"]) if c["chunk_id"] in gold]
                    unused.append(float(infl[j] < tau))
                    if infl.max() >= tau:
                        agree.append(float(j == int(infl.argmax())))
                    if gi:
                        gold_used.append(float(infl[gi].max() >= tau))
                        if j in gi:
                            faithful.append(float(infl[j] >= tau))
                    if np.ptp(cs) > 0 and np.ptp(infl) > 0:
                        rho.append(spearmanr(cs, infl)[0])
                rows.append({"domain": d, "method": m, "tau": tau, "n": len(unused), "n_answer_depends_on_context": len(agree),
                             "top_attributed_is_top_influential": nm(agree), "top_attributed_unused": nm(unused),
                             "gold_chunk_used": nm(gold_used), "n_correct_attr": len(faithful),
                             "correct_attribution_is_faithful": nm(faithful), "spearman_score_vs_influence": nm(rho)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 4. selective answering with log-probs
def selective(runs_dir, run, systems, out):
    summ = []
    for s in systems:
        df = ES.load_frame(run, s, runs_dir=Path(runs_dir))
        if df.empty:
            print(f"[selective] {run}/{s}: no scored files, skipped")
            continue
        res = ES.evaluate(ES.add_learned(df)).assign(system=s)
        res.to_csv(out / f"selective_{s}.csv", index=False)
        has_lp = res.signal.isin(ES.LP_FEATS).any()
        for d, g in res.groupby("domain"):
            dep = g[~g.signal.str.contains("indomain_cv")]
            best = dep.sort_values("auroc", ascending=False).iloc[0]
            lp = dep[dep.signal.isin(ES.LP_FEATS + ["lr_lp_transfer"])]
            ck = dep[dep.signal == "checker"]
            wins = dep[(dep["gain@cal50"] > 0.01) & (dep["P(gain<=0)@cal50"] < 0.1)]
            summ.append({"system": s, "domain": d, "best_trivial": best.best_trivial, "best_signal": best.signal,
                         "best_auroc": best.auroc, "best_lp_signal": lp.sort_values("auroc", ascending=False).signal.iloc[0] if len(lp) else None,
                         "best_lp_auroc": lp.auroc.max() if len(lp) else np.nan,
                         "checker_auroc": ck.auroc.max() if len(ck) else np.nan,
                         "n_signals_beating_trivial": len(wins), "winners": ",".join(wins.signal) if len(wins) else ""})
    return pd.DataFrame(summ)


# ------------------------------------------------------------------ driver
def show(title, df, cols=None):
    print(f"\n=== {title} ===")
    print("(no data)" if df is None or df.empty else df[[c for c in (cols or df.columns) if c in df]].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="llama8b | qwen1.5b (reads v2_<model> and main_<model>)")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--no-implicit", action="store_true", help="count only explicit refusals")
    ap.add_argument("--skip-selective", action="store_true")
    ap.add_argument("--published-nli-set", action="store_true",
                    help="do NOT apply the nli_set tie-break (reproduces the pre-fix nli_set attribution numbers)")
    ap.add_argument("--qids-from", default=None,
                    help="restrict to the questions of this run (e.g. v2_qwen32b): paired cross-model comparison on a shared subset")
    a = ap.parse_args()
    runs_dir = Path(a.runs_dir) if a.runs_dir else RUNS
    v2n, mn = f"v2_{a.model}", f"main_{a.model}"
    out = Path(a.results_dir) if a.results_dir else RESULTS / (v2n + ("_subset" if a.qids_from else ""))
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    impl = not a.no_implicit
    keep = None
    if a.qids_from:
        ref = load(runs_dir, a.qids_from, "vanilla_rag", tiebreak=False)
        keep = {d: set(q) for d, q in ref.items()}

    def L(run, s):
        recs = load(runs_dir, run, s, tiebreak=not a.published_nli_set)
        if keep is None:
            return recs
        return {d: {q: r for q, r in qs.items() if q in keep.get(d, ())} for d, qs in recs.items() if d in keep}
    V2, MV, MT = L(v2n, "vanilla_rag"), L(mn, "vanilla_rag"), L(mn, "trust_rag")
    if not MT:                      # no mid-sem run for this model (e.g. Qwen-14B, API models): use its own trust_rag run
        MT = L(v2n, "trust_rag")
    IV, IT, OV = L(v2n, "improved_vanilla"), L(v2n, "improved_trust"), L(v2n, "oracle_vanilla")
    RV, RT = L(v2n, "removed_vanilla"), L(v2n, "removed_trust")

    def have(*s):
        ok = all(len(x) for x in s)
        return ok

    if have(V2, MV):
        df = replication(V2, MV)
        df.to_csv(out / "stage2_replication.csv", index=False)
        show("0. Replication: v2 vanilla_rag vs mid-sem vanilla_rag (same plans, prompt, model)", df)
    parts = []
    if have(V2, IV):
        parts.append(retrieval_effect(V2, IV, "vanilla: base -> improved retrieval"))
    if have(MT, IT):
        parts.append(retrieval_effect(MT, IT, "trust: base -> improved retrieval"))
    if parts:
        df = pd.concat(parts)
        df.to_csv(out / "stage2_retrieval_effect.csv", index=False)
        show("1. Improved retrieval (attribution on MATCHED questions where gold was retrieved under both)", df,
             ["comparison", "domain", "n", "hit_base", "hit_imp", "acc_base", "acc_imp", "acc_delta", "ci_lo", "ci_hi", "p",
              "wrong_base", "wrong_imp", "unsup_base", "unsup_imp", "n_matched_attr", "attr_nli_set_base", "attr_nli_set_imp",
              "attr_sim_sent_base", "attr_sim_sent_imp"])
    if have(V2, OV):
        df = oracle_effect(V2, OV)
        df.to_csv(out / "stage2_oracle.csv", index=False)
        show("2a. Oracle context: generator ceiling and retrieval loss", df)
        tab, corr = oracle_position(OV)
        tab.to_csv(out / "stage2_oracle_position.csv", index=False)
        corr.to_csv(out / "stage2_oracle_position_corr.csv", index=False)
        show("2b. Position bias: gold passage position in the shuffled oracle context", tab)
        show("2b'. Rank correlation (position vs correct)", corr)
    sums, sigs = [], []
    # "natural": gold present only if retrieval found it (small n for FinanceBench); "oracle": gold forced in (full n,
    # no retrieval confound). Trust-prompt log-probs are unavailable: Part A does not regenerate trust_rag with them.
    for label, rem, withev in [("vanilla|natural", RV, V2), ("vanilla|oracle", RV, OV), ("trust|natural", RT, MT)]:
        if have(rem, withev):
            sums.append(removal_summary(rem, withev, label, impl))
            sigs.append(removal_signals(rem, withev, label, impl))
    if sums:
        s, g = pd.concat(sums), pd.concat(sigs)
        s.to_csv(out / "stage2_removed_summary.csv", index=False)
        g.to_csv(out / "stage2_removed_signals.csv", index=False)
        show("3a. Evidence removed: should refuse (clean = gold answer not left in the context)", s)
        key = g[g.signal.isin(["self_refusal", "ret_top1", "nli_cov_mean", "trust_no_contra", "lp_mean", "lp_first"])]
        show("3b. 'Evidence present' vs 'removed': AUROC, balanced accuracy (0.5 = trivial) at in-domain vs general-fitted threshold", key,
             ["system", "signal", "domain", "n_pairs", "auroc_present", "bal_acc_in_domain", "bal_acc_transfer", "transfer_loss",
              "refusal_rate_at_t_general"])
    if not a.skip_selective:
        df = selective(runs_dir, v2n, ["vanilla_rag", "improved_vanilla", "improved_trust"], out)
        df.to_csv(out / "stage2_selective_summary.csv", index=False)
        show("4. Selective answering vs trivial policies, with token log-prob signals (details: selective_<system>.csv)", df)
    if have(V2):
        df = loo_faithfulness(V2)
        df.to_csv(out / "stage2_loo.csv", index=False)
        show("5. Leave-one-out faithfulness (is the attributed passage actually used?)", df[df.tau == 0.2] if len(df) else df)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
