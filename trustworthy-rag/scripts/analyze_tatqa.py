"""TAT-QA (second financial dataset) analysis, pre-registered in paper/findings/analysis_plan_tatqa.md (hypotheses T1-T6).

python -m scripts.ingest_kaggle experiments/kaggle_raw/trag_results_stage2E_A.zip --no-analyses   # unpack first (every Part E zip)
python -m scripts.analyze_tatqa
Reads experiments/runs/tq_<model>/tatqa__{oracle_vanilla,oracle_trust,removed_trust}.scored.jsonl and, for comparison, the same
model's experiments/runs/v2_<model>/{hotpotqa,squad2,financebench}__oracle_vanilla.scored.jsonl. Models without a run are skipped.
Writes experiments/results/tatqa/*.csv. CPU only.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.analyze_stage2 import refused
from scripts.finance_attribution_diagnosis import has_number, key_numbers
from src.attribution.fusion import GRID, acc_by_weight, evaluate_selection, load_weights
from src.attribution.methods import refresh_record
from src.common import RESULTS, RUNS, read_jsonl

MODELS = [("qwen1.5b", "Qwen2.5-1.5B", 1.5), ("llama8b", "Llama-3.1-8B", 8), ("qwen14b", "Qwen2.5-14B", 14), ("phi4", "Phi-4-14B", 14),
          ("mistral24b", "Mistral-Small-24B", 24), ("gemma27b", "Gemma-3-27B", 27), ("qwen32b", "Qwen2.5-32B", 32)]
OUT = RESULTS / "tatqa"
G = list(GRID)
METHODS = {"sim_sent": 1.0, "nli_set": 0.0}
DEFAULT_W = 0.9


def load(run, domain, system):
    p = RUNS / run / f"{domain}__{system}.scored.jsonl"
    return [refresh_record(r) for r in read_jsonl(p)] if p.exists() else []


def attr_rows(recs):
    """[(record, accuracy-by-weight vector)] for records that answered and have a gold chunk in the context."""
    return [(r, a) for r in recs if r["answerable"] and (a := acc_by_weight(r)) is not None]


def mean_at(rows, w):
    return float(np.mean([a[G.index(w)] for _, a in rows])) if rows else np.nan


def num_sim_acc(r, base="sim_sent"):
    """T7: number-aware attribution (scripts/finance_attribution_diagnosis.py): share of the sentence's numbers found in the chunk,
    ties broken by the stored `base` scores (sim_sent for T7a, tie-broken nli_set for T7b)."""
    st = [s["text"] for s in r["attribution"]["sentences"]]
    texts = [c["text"] for c in r["retrieved"]]
    N = np.array([[np.mean([has_number(t, [x]) for x in key_numbers(s)]) if key_numbers(s) else 0.0 for t in texts] for s in st])
    S = N + 1e-3 * np.asarray(r["attribution"]["scores"][base])
    gold = set(r["gold_chunk_ids"])
    mask = np.array([c["chunk_id"] in gold for c in r["retrieved"]])
    return float(mask[S.argmax(1)].mean())


def cluster_boot(df, col, a_mask, b_mask, n=10000, seed=0):
    """Mean(col | a) - mean(col | b), records pooled over models; resampling unit = question (all its models' records move together)."""
    q = df.qid.unique()
    pos = {k: i for i, k in enumerate(q)}
    qi = df.qid.map(pos).to_numpy()
    v, A, B = df[col].to_numpy(float), a_mask.to_numpy(), b_mask.to_numpy()
    def stat(wt):
        w = wt[qi]
        return (v * w * A).sum() / (w * A).sum() - (v * w * B).sum() / (w * B).sum()
    obs = stat(np.ones(len(q)))
    rng = np.random.default_rng(seed)
    bs = np.array([stat(np.bincount(rng.integers(0, len(q), len(q)), minlength=len(q)).astype(float)) for _ in range(n)])
    bs = bs[~np.isnan(bs)]
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return {"diff": obs, "ci_lo": np.quantile(bs, .025), "ci_hi": np.quantile(bs, .975), "p": max(p, 1 / n),
            "n_a": int(A.sum()), "n_b": int(B.sum())}


def holm(ps):
    order, m, adj, run = np.argsort(ps), len(ps), np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    W = load_weights()
    fin_w = float(W.get("financebench", 0.0))
    have = [(m, label, size) for m, label, size in MODELS if (RUNS / f"tq_{m}" / "tatqa__oracle_vanilla.scored.jsonl").exists()]
    if not have:
        raise SystemExit("no TAT-QA runs yet: ingest the Part E zips first (python -m scripts.ingest_kaggle <zip> --no-analyses)")
    print("models with TAT-QA runs:", [l for _, l, _ in have])

    # ---------------- T1 replication + T5 mixture transfer, per model
    t1, recs_long, t5 = [], [], []
    for m, label, size in have:
        ov = load(f"tq_{m}", "tatqa", "oracle_vanilla")
        rows = attr_rows(ov)
        row = {"model": label, "n": len(ov), "acc_oracle": np.mean([r["correct"] for r in ov]), "n_attr": len(rows)}
        for meth, w in METHODS.items():
            row[f"tatqa_{meth}"] = mean_at(rows, w)
            gen = [mean_at(attr_rows(load(f"v2_{m}", d, "oracle_vanilla")), w) for d in ("hotpotqa", "squad2")]
            row[f"general_{meth}"] = np.nanmean(gen) if not all(np.isnan(gen)) else np.nan
            row[f"financebench_{meth}"] = mean_at(attr_rows(load(f"v2_{m}", "financebench", "oracle_vanilla")), w)
            row[f"gap_{meth}"] = row[f"tatqa_{meth}"] - row[f"general_{meth}"]
        t1.append(row)
        sel = evaluate_selection([r for r, _ in rows], n_labels=20) if len(rows) > 30 else {}
        t5.append({"model": label, "n": len(rows), "w_default_0.9": mean_at(rows, DEFAULT_W), f"w_financebench_{fin_w}": mean_at(rows, fin_w),
                   "w_from_20_tatqa_labels": sel.get("mix_from_labels", np.nan), "best_fixed_w": sel.get("best_fixed_w_on_test", np.nan),
                   "w_mode": sel.get("w_mode", np.nan)})
        for r, a in rows:
            mt = r["meta"]
            recs_long.append({"model": label, "qid": r["qid"], "answer_from": mt["answer_from"], "kind": mt["kind"],
                              "sim_sent": a[G.index(1.0)], "nli_set": a[G.index(0.0)], "num_sim": num_sim_acc(r),
                              "num_nli": num_sim_acc(r, "nli_set")})
    t1, t5, L = pd.DataFrame(t1), pd.DataFrame(t5), pd.DataFrame(recs_long)
    t1.to_csv(OUT / "t1_replication.csv", index=False); t5.to_csv(OUT / "t5_mixture_transfer.csv", index=False)
    L.to_csv(OUT / "attribution_records.csv", index=False)
    print("\n=== T1. Oracle attribution: TAT-QA vs general (HotpotQA+SQuAD mean) vs FinanceBench, same model ===")
    print(t1.round(3).to_string(index=False))
    for meth in METHODS:
        g = t1[f"gap_{meth}"].dropna()
        print(f"T1 {meth}: TAT-QA below general for {int((g < 0).sum())}/{len(g)} models (supported if >= 6 of 7)")
    print("\n=== T5. Mixture weight transferred from FinanceBench vs default vs picked from 20 TAT-QA labels ===")
    print(t5.round(3).to_string(index=False))

    # ---------------- T2-T4 mechanism, pooled over models, cluster bootstrap over questions, Holm over 5 tests
    tests = []
    for meth in METHODS:
        tests.append(("T2 table vs text", meth, cluster_boot(L, meth, L.answer_from == "table", L.answer_from == "text")))
        tab = L[L.answer_from.isin(["table", "table-text"])]
        tests.append(("T3 computed vs quoted (table-sourced)", meth, cluster_boot(tab, meth, tab.kind == "computed", tab.kind == "quoted")))
    L2 = pd.concat([L.assign(method="nli_set", v=L.nli_set), L.assign(method="sim_sent", v=L.sim_sent)], ignore_index=True)
    tests.append(("T4 nli_set vs sim_sent", "both", cluster_boot(L2, "v", L2.method == "nli_set", L2.method == "sim_sent")))
    T = pd.DataFrame([{"test": t, "method": m, **r} for t, m, r in tests])
    T["p_holm"] = holm(T.p.to_numpy())
    T.to_csv(OUT / "t2_t4_mechanism_tests.csv", index=False)
    print("\n=== T2-T4. Mechanism (pooled over models; resampling unit = question; Holm over 5 tests; |diff| < .05 is not an effect)")
    print("    diff = first group minus second: T2/T3 predict diff < 0, T4 predicts diff > 0 ===")
    print(T.round(4).to_string(index=False))
    t7 = []
    for name, a_col, b_col in (("T7a num_sim vs sim_sent", "num_sim", "sim_sent"), ("T7b num_nli vs nli_set", "num_nli", "nli_set")):
        L3 = pd.concat([L.assign(m=a_col, v=L[a_col]), L.assign(m=b_col, v=L[b_col])], ignore_index=True)
        t7.append({"test": name, **cluster_boot(L3, "v", L3.m == a_col, L3.m == b_col), "a": L[a_col].mean(), "b": L[b_col].mean()})
    t7 = pd.DataFrame(t7)
    t7["p_holm"] = holm(t7.p.to_numpy())
    t7.to_csv(OUT / "t7_number_aware.csv", index=False)
    print("\n=== T7. Number-aware attribution (a = with number match, b = without; Holm over T7a/T7b) ===")
    print(t7.round(4).to_string(index=False))
    cells = L.groupby(["answer_from", "kind"]).agg(n=("qid", "size"), n_questions=("qid", "nunique"), sim_sent=("sim_sent", "mean"),
                                                     nli_set=("nli_set", "mean")).reset_index()
    cells.to_csv(OUT / "attribution_by_cell.csv", index=False)
    print("\n--- attribution accuracy by answer source x kind (pooled over models)")
    print(cells.round(3).to_string(index=False))
    per_model = L.groupby(["model", "answer_from"])[["sim_sent", "nli_set"]].mean().unstack().round(3)
    per_model.to_csv(OUT / "attribution_by_source_per_model.csv")
    print("\n--- by answer source, per model\n" + per_model.to_string())

    # ---------------- accuracy by kind (oracle_vanilla)
    acc = []
    for m, label, _ in have:
        for r in load(f"tq_{m}", "tatqa", "oracle_vanilla"):
            acc.append({"model": label, "answer_from": r["meta"]["answer_from"], "kind": r["meta"]["kind"], "correct": r["correct"]})
    A = pd.DataFrame(acc).pivot_table(index="model", columns="kind", values="correct", aggfunc="mean").round(3)
    A.to_csv(OUT / "accuracy_by_kind.csv")
    print("\n=== Oracle accuracy by answer kind ===\n" + A.to_string())

    # ---------------- T6 refusal vs scale
    t6 = []
    for m, label, size in have:
        ot, rt = {r["qid"]: r for r in load(f"tq_{m}", "tatqa", "oracle_trust")}, {r["qid"]: r for r in load(f"tq_{m}", "tatqa", "removed_trust")}
        q = [k for k in rt if k in ot]
        if not q:
            continue
        clean = [k for k in q if not rt[k]["meta"].get("residual_answer_in_context")]
        refuse_removed, over = np.mean([refused(rt[k]) for k in q]), np.mean([refused(ot[k]) for k in q])
        t6.append({"model": label, "size_b": size, "n_pairs": len(q), "refuse_removed": refuse_removed, "over_refusal": over,
                   "bal_acc": (refuse_removed + 1 - over) / 2, "refuse_removed_clean": np.mean([refused(rt[k]) for k in clean]) if clean else np.nan,
                   "n_clean": len(clean), "answered_removed_wrong": np.mean([not refused(rt[k]) for k in q])})
    t6 = pd.DataFrame(t6)
    t6.to_csv(OUT / "t6_refusal.csv", index=False)
    print("\n=== T6. Refusal with the gold document removed (trust prompt) vs over-refusal with gold guaranteed ===")
    print(t6.round(3).to_string(index=False))
    if len(t6) >= 4:
        for col, alt in (("refuse_removed", "greater"), ("over_refusal", "less")):
            rho, p = spearmanr(t6.size_b, t6[col], alternative=alt)
            print(f"T6 Spearman(size, {col}) = {rho:.3f}, one-sided p ({alt}) = {p:.4f}")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
