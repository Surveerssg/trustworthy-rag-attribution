"""Stage-1 improvements on the saved answers (no new generation).

python -m scripts.improve_system --run main_llama8b
  1. Fused attribution  : score = w * sim_sent + (1-w) * nli_set, w tuned on general domains only.
  2. Soft filter        : drop a sentence only if BOTH NLI and similarity say unsupported, or it is contradicted.
  3. Learned confidence : logistic regression on all answer signals, trained on general domains only.
  4. Recalibrated gate  : per-domain threshold from n in-domain labelled questions (conformal risk control or
                          utility-optimal), evaluated on the remaining questions; 200 random draws.
Every comparison (plain RAG, v1 full system, v2) is on exactly the same held-out questions.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.analyze_refusal import crc_threshold
from src.attribution.methods import nli_probs
from src.common import RESULTS, RUNS, read_jsonl
from src.evaluation import metrics as M
from src.pipeline.prompts import is_implicit_refusal

DOMS = ["squad2", "hotpotqa", "pubmedqa", "cuad", "financebench"]
GENERAL = ("squad2", "hotpotqa")
FEATS = ["ret_top1", "ret_mean", "ret_gap", "sim_cov_mean", "nli_cov_mean", "nli_cov_min", "not_contra",
         "frac_chunks_support", "n_sent", "sim_min", "ans_len"]


def load(run, system):
    out = {d: read_jsonl(RUNS / run / f"{d}__{system}.scored.jsonl") for d in DOMS
           if (RUNS / run / f"{d}__{system}.scored.jsonl").exists()}
    ck = sorted((RUNS / run).glob("checker_*.jsonl"))
    if ck:  # attach the trained checker's hallucination probability
        p = {(c["domain"], c["system"], c["qid"]): c["p_halluc"] for c in read_jsonl(ck[0])}
        for d, recs in out.items():
            for r in recs:
                if (d, system, r["qid"]) in p:
                    r["p_halluc"] = p[(d, system, r["qid"])]
    return out


# ------------------------------------------------------------------ 1. fused attribution
def fused_attr_acc(recs, w):
    accs = []
    for r in recs:
        sc = r["attribution"].get("scores", {})
        if "sim_sent" not in sc or "nli_set" not in sc:
            continue
        gold = {c["chunk_id"] for c in r["retrieved"]} & set(r["gold_chunk_ids"])
        if not gold:
            continue
        S = w * np.asarray(sc["sim_sent"]) + (1 - w) * np.asarray(sc["nli_set"])
        top = S.argmax(1)
        accs.append(np.mean([r["retrieved"][j]["chunk_id"] in gold for j in top]))
    return float(np.mean(accs)) if accs else np.nan, len(accs)


def attribution_study(V):
    ws = np.round(np.linspace(0, 1, 21), 2)
    gen = [r for d in GENERAL for r in V.get(d, []) if r["answerable"] and not r["refused"]]
    best_w = max(ws, key=lambda w: fused_attr_acc(gen, w)[0])
    rows = []
    for d in DOMS:
        R = [r for r in V.get(d, []) if r["answerable"] and not r["refused"]]
        row = {"domain": d}
        for name, w in [("sim_sent", 1.0), ("nli_set", 0.0), (f"fused (w={best_w})", best_w)]:
            a, n = fused_attr_acc(R, w)
            row[name] = a; row["n"] = n
        rows.append(row)
    return pd.DataFrame(rows), float(best_w)


# ------------------------------------------------------------------ features for confidence
def feats(r):
    f = dict(r["features"])
    sc = r["attribution"].get("scores", {})
    for k in ["nli_cov_mean", "nli_cov_min", "frac_chunks_support", "sim_cov_mean"]:
        f.setdefault(k, 0.0)
    f["not_contra"] = 1 - f.get("contra_max", 1.0)
    f["n_sent"] = f.get("n_sent", 0)
    f["sim_min"] = float(np.asarray(sc["sim_sent"]).max(1).min()) if sc.get("sim_sent") else 0.0
    f["ans_len"] = min(len(r["answer"].split()), 200) / 200
    f["trust_score"] = 0.0 if r["refused"] else (f["ret_top1"] + f["nli_cov_mean"] + f["not_contra"]) / 3
    lp = r.get("lp") or {}
    if lp:  # token-probability confidence (Stage-2 runs generated with logprobs)
        f.update(lp_mean=lp["mean"], lp_min=lp["min"], lp_first=lp["first_mean"], lp_frac_low=lp["frac_low"],
                 lp_margin=lp.get("margin_mean") or 0.0)
    if "p_halluc" in r:
        f["checker"] = 1 - r["p_halluc"]
    return f


def frame(S):
    rows = []
    for d, recs in S.items():
        for r in recs:
            rows.append({"domain": d, "qid": r["qid"], "answerable": r["answerable"], "refused": r["refused"],
                         "correct": float(r["correct"]) if r["correct"] is not None else 0.0, **feats(r)})
    df = pd.DataFrame(rows)
    # "good answer" = answered and correct; for unanswerable questions answering is never good
    df["good"] = ((~df.refused) & df.answerable & (df.correct > 0)).astype(int)
    return df


OPTIONAL = ["lp_mean", "lp_min", "lp_first", "lp_frac_low", "lp_margin", "checker"]


def _cols(df):
    return FEATS + [c for c in OPTIONAL if c in df and df[c].notna().mean() > 0.9]


def _fit(tr):
    cols = _cols(tr)
    tr = tr.fillna({c: 0.0 for c in cols})
    sc = StandardScaler().fit(tr[cols])
    lr = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(tr[cols]), tr.good)
    return lambda x: lr.predict_proba(sc.transform(x[cols].fillna(0.0)))[:, 1]


def fit_confidence(df):
    """Cross-fitted: each general domain is scored by a model trained on the other general domain;
    specialised domains by a model trained on both general domains (never on themselves)."""
    models = {g: _fit(df[(df.domain == o) & ~df.refused]) for g, o in [(GENERAL[0], GENERAL[1]), (GENERAL[1], GENERAL[0])]}
    both = _fit(df[df.domain.isin(GENERAL) & ~df.refused])
    def conf(x):
        p = np.zeros(len(x))
        for d in x.domain.unique():
            m = (x.domain == d).values
            p[m] = models.get(d, both)(x[m])
        return np.where(x.refused, 0.0, p)
    return conf


# ------------------------------------------------------------------ 2. soft filter
def soft_filter_answer(r, t_nli=0.5, t_sim=0.55, t_con=0.8):
    """Returns (kept_sentence_texts, n_total). Drop only if contradicted, or unsupported by BOTH signals."""
    sc = r["attribution"].get("scores", {})
    sents = r["attribution"].get("sentences", [])
    if not sents or "nli_set" not in sc:
        return [s["text"] for s in sents], len(sents)
    E, Ssim = np.asarray(sc["nli_set"]), np.asarray(sc["sim_sent"])
    C = np.asarray(sc.get("nli_window_contra", np.zeros_like(E)))
    rel = Ssim.argmax(1)
    keep = []
    for i, s in enumerate(sents):
        contradicted = C[i, rel[i]] >= t_con and E[i].max() < 0.5
        unsupported = E[i].max() < t_nli and Ssim[i].max() < t_sim
        if not (contradicted or unsupported):
            keep.append(s["text"])
    return keep, len(sents)


def judge_text(text, r):
    c, _ = M.correctness(text, r["gold_answers"], r["answer_type"])
    if c is None:
        p = nli_probs([text, r["gold_answers"][0]], [r["gold_answers"][0], text])[:, 0]
        c = float(max(p) >= 0.5)
    return float(c)


def outcome(r, answered, text_correct):
    """reliability utility and flags for one question given the final decision."""
    if not answered:
        return {"answered": 0, "u": 0.0 if r["answerable"] else 1.0, "wrong": 0.0}
    if not r["answerable"]:
        return {"answered": 1, "u": -1.0, "wrong": 1.0}
    return {"answered": 1, "u": 1.0 if text_correct else -1.0, "wrong": 0.0 if text_correct else 1.0}


# ------------------------------------------------------------------ 3+4. systems on held-out questions
def build_candidates(base_recs, use_soft_filter):
    """Per question: whether an answer survives the filter, and whether that (filtered) answer is correct."""
    out = {}
    for r in base_recs:
        if r["refused"]:
            out[r["qid"]] = (False, 0.0); continue
        if use_soft_filter:
            keep, tot = soft_filter_answer(r)
            if not keep:
                out[r["qid"]] = (False, 0.0); continue
            ok = judge_text(" ".join(keep), r) if r["answerable"] else 0.0
        else:
            ok = float(r["correct"] or 0.0) if r["answerable"] else 0.0
        out[r["qid"]] = (True, ok)
    return out


def evaluate(run, n_cal=20, draws=200, alpha=0.2, base="vanilla_rag", filt="soft", implicit=True, margin=0.1):
    V, T = load(run, "vanilla_rag"), load(run, "trust_rag")
    B0 = V if base == "vanilla_rag" else T
    # v2: an answer that opens by saying the sources lack the information counts as a refusal
    B = {d: [dict(r, refused=r["refused"] or (implicit and is_implicit_refusal(r["answer"]))) for r in recs] for d, recs in B0.items()}
    dfB = frame(B)
    conf = fit_confidence(dfB)
    dfB["p"] = conf(dfB)
    # v1 decisions (from analyze_systems logic): trust answers + hard filter + SQuAD gate -> recompute quickly
    v1 = pd.read_csv(RESULTS / run / "systems_ablation.csv")
    rng = np.random.default_rng(0)
    rows, auc_rows = [], []
    for d in DOMS:
        if d not in B:
            continue
        recs = {r["qid"]: r for r in B[d]}
        vrecs = {r["qid"]: r for r in V[d]}
        cand = build_candidates(B[d], use_soft_filter=(filt == "soft"))
        x = dfB[dfB.domain == d].set_index("qid")
        qids = list(x.index)
        # learned-confidence quality vs the hand-made trust score (answerable, answered, vanilla)
        ans = x[x.answerable & ~x.refused]
        if ans.good.nunique() == 2:
            row_auc = {"domain": d, "AUROC_trust_score_v1": M.auroc(ans.trust_score, ans.good),
                       "AUROC_learned_v2": M.auroc(ans.p, ans.good)}
            for c in OPTIONAL:
                if c in ans and ans[c].notna().all():
                    row_auc[f"AUROC_{c}"] = M.auroc(ans[c], ans.good)
            auc_rows.append(row_auc)
        res = {k: [] for k in ["plain", "plain_implicit", "v2_general_t", "v2_quantile", "v2_crc", "v2_util", "v2_safe"]}
        cov = {k: [] for k in res}; risk = {k: [] for k in res}; selacc = {k: [] for k in res}
        # general-domain threshold (no in-domain labels): utility-optimal on general domains
        g = dfB[dfB.domain.isin(GENERAL)]
        def util_at(t, sub, cands, rr):
            us = []
            for q in sub:
                surv, ok = cands[q]
                answered = surv and x.loc[q, "p"] >= t if q in x.index else False
                us.append(outcome(rr[q], answered, ok)["u"])
            return np.mean(us)
        t_general = float(np.quantile(g.p, 0.5))
        gen_cands = {}
        for gd in GENERAL:
            gen_cands.update({r["qid"]: (not r["refused"], float(r["correct"] or 0)) for r in B[gd]})
        grecs = {r["qid"]: r for gd in GENERAL for r in B[gd]}
        gx = g.set_index("qid")
        best_u = -9
        for t in np.quantile(g.p, np.linspace(0, 0.95, 40)):
            u = np.mean([outcome(grecs[q], gen_cands[q][0] and gx.loc[q, "p"] >= t, gen_cands[q][1])["u"] for q in gx.index])
            if u > best_u:
                best_u, t_general = u, float(t)
        # label-free transfer: keep the general domain's REFUSAL RATE, not its threshold value
        q_src = float((g.p < t_general).mean())
        t_quant = float(np.quantile(x.p, q_src)) if q_src > 0 else -np.inf
        for _ in range(draws):
            perm = rng.permutation(len(qids))
            cal, test = [qids[i] for i in perm[:n_cal]], [qids[i] for i in perm[n_cal:]]
            # calibration labels: answered-and-wrong under the candidate answer
            cal_conf = np.array([x.loc[q, "p"] if cand[q][0] else -1 for q in cal])
            cal_wrong = np.array([outcome(recs[q], True, cand[q][1])["wrong"] if cand[q][0] else 0 for q in cal], bool)
            t_crc = crc_threshold(cal_conf, cal_wrong, alpha)
            ts = np.unique(np.r_[cal_conf, np.inf])
            cal_u = lambda t: np.mean([outcome(recs[q], cand[q][0] and x.loc[q, "p"] >= t, cand[q][1])["u"] for q in cal])
            t_util = max(ts, key=cal_u)
            # safe gate: only gate if it clearly beats answering everything on the calibration questions
            t_safe = t_util if cal_u(t_util) - cal_u(-np.inf) >= margin else -np.inf
            for name, dec in [("plain", lambda q: (not vrecs[q]["refused"], float(vrecs[q]["correct"] or 0) if vrecs[q]["answerable"] else 0.0, vrecs[q])),
                              ("plain_implicit", lambda q: (not recs[q]["refused"], float(recs[q]["correct"] or 0) if recs[q]["answerable"] else 0.0, recs[q])),
                              ("v2_general_t", lambda q: (cand[q][0] and x.loc[q, "p"] >= t_general, cand[q][1], recs[q])),
                              ("v2_quantile", lambda q: (cand[q][0] and x.loc[q, "p"] >= t_quant, cand[q][1], recs[q])),
                              ("v2_crc", lambda q: (cand[q][0] and x.loc[q, "p"] >= t_crc, cand[q][1], recs[q])),
                              ("v2_util", lambda q: (cand[q][0] and x.loc[q, "p"] >= t_util, cand[q][1], recs[q])),
                              ("v2_safe", lambda q: (cand[q][0] and x.loc[q, "p"] >= t_safe, cand[q][1], recs[q]))]:
                outs = [outcome(rr, a, ok) for a, ok, rr in (dec(q) for q in test)]
                res[name].append(np.mean([o["u"] for o in outs]))
                cov[name].append(np.mean([o["answered"] for o in outs]))
                risk[name].append(np.mean([o["wrong"] for o in outs]))
                answered_ans = [o for o, q in zip(outs, test) if o["answered"] and recs[q]["answerable"]]
                selacc[name].append(np.mean([o["u"] > 0 for o in answered_ans]) if answered_ans else np.nan)
        v1row = v1[(v1.domain == d) & (v1.system == "full")]
        row = {"domain": d, "n_test": len(qids) - n_cal,
               "v1_full_utility(all q)": float(v1row.reliability_u.iloc[0]) if len(v1row) else np.nan,
               "v1_coverage": float(v1row.coverage.iloc[0]) if len(v1row) else np.nan}
        for k in res:
            row[f"{k}_utility"] = np.mean(res[k]); row[f"{k}_coverage"] = np.mean(cov[k])
            row[f"{k}_risk"] = np.mean(risk[k]); row[f"{k}_sel_acc"] = np.nanmean(selacc[k])
        # paired significance on the final draw's test set is noisy; use per-draw utility difference distribution
        for k in ["v2_util", "v2_safe"]:
            diff = np.array(res[k]) - np.array(res["plain"])
            row[f"{k}_minus_plain"] = diff.mean(); row[f"P({k}<=plain)"] = float((diff <= 0).mean())
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(auc_rows), t_general


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-cal", type=int, default=20)
    ap.add_argument("--base", default="vanilla_rag")
    ap.add_argument("--filter", default="none", choices=["soft", "none"])
    a = ap.parse_args()
    out = RESULTS / a.run
    V = load(a.run, "vanilla_rag")
    att, w = attribution_study(V)
    att.to_csv(out / "v2_fused_attribution.csv", index=False)
    print(f"=== 1. Fused attribution (w tuned on general domains = {w}) — attribution accuracy | gold retrieved ===")
    print(att.round(3).to_string(index=False))
    sysdf, aucdf, tg = evaluate(a.run, n_cal=a.n_cal, base=a.base, filt=a.filter)
    aucdf.to_csv(out / "v2_confidence_auroc.csv", index=False)
    print("\n=== 3. Confidence quality: AUROC for 'answer is correct' (hand-made v1 vs learned v2) ===")
    print(aucdf.round(3).to_string(index=False))
    sysdf.to_csv(out / f"v2_system_{a.base}_{a.filter}_ncal{a.n_cal}.csv", index=False)
    cols = ["domain", "v1_full_utility(all q)", "v1_coverage", "plain_utility", "plain_coverage", "plain_implicit_utility", "plain_implicit_coverage",
            "v2_general_t_utility", "v2_quantile_utility", "v2_quantile_coverage",
            "v2_crc_utility", "v2_crc_risk", "v2_util_utility", "v2_util_coverage", "v2_safe_utility", "v2_safe_coverage",
            "v2_safe_sel_acc", "plain_sel_acc", "v2_safe_minus_plain", "P(v2_safe<=plain)"]
    print(f"\n=== 2+4. System v2 (soft filter + learned confidence + gate), base={a.base}, filter={a.filter}, n_cal={a.n_cal}, 200 draws ===")
    print(sysdf[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
