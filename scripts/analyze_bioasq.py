"""BioASQ (second medical dataset) analysis, pre-registered in paper/findings/analysis_plan_bioasq.md (hypotheses B1-B5). CPU only.

python -m scripts.ingest_kaggle experiments/kaggle_raw/<Part G zip> --no-analyses   # unpack first
python -m scripts.analyze_bioasq
Reads experiments/runs/bq_<model>/bioasq__{oracle_vanilla,vanilla_rag,trust_rag,removed_trust}.scored.jsonl and, for B1, the same model's
experiments/runs/v2_<model>/pubmedqa__oracle_vanilla.scored.jsonl. Models without a run are skipped. Writes experiments/results/bioasq/.
"""
import numpy as np
import pandas as pd

from scripts.analyze_llm_attributor import boot_diff, holm
from scripts.analyze_stage2 import refused
from scripts.analyze_tatqa import MODELS, attr_rows, load, mean_at, num_sim_acc
from src.attribution.fusion import GRID
from src.common import RESULTS, RUNS

OUT = RESULTS / "bioasq"
G = list(GRID)
MARGIN = 0.05


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    have = [(m, l, s) for m, l, s in MODELS if (RUNS / f"bq_{m}" / "bioasq__oracle_vanilla.scored.jsonl").exists()]
    if not have:
        raise SystemExit("no BioASQ runs yet: ingest the Part G zips first")
    per_model, recs, acc_pairs, ref = [], [], [], []
    for m, label, size in have:
        ov = load(f"bq_{m}", "bioasq", "oracle_vanilla")
        rows = attr_rows(ov)
        pm = attr_rows(load(f"v2_{m}", "pubmedqa", "oracle_vanilla"))
        per_model.append({"model": label, "n_oracle": len(ov), "n_attr": len(rows), "bioasq_sim_sent": mean_at(rows, 1.0),
                          "bioasq_nli_set": mean_at(rows, 0.0), "bioasq_mixture_w0.9": mean_at(rows, 0.9), "pubmedqa_sim_sent": mean_at(pm, 1.0),
                          "pubmedqa_nli_set": mean_at(pm, 0.0)})
        for r, a in rows:
            recs.append({"model": label, "domain": "bioasq", "qid": r["qid"], "type": r["meta"]["type"], "sim_sent": a[G.index(1.0)],
                         "nli_set": a[G.index(0.0)], "num_sim": num_sim_acc(r)})
        nat = {r["qid"]: r for r in load(f"bq_{m}", "bioasq", "vanilla_rag")}
        for r in ov:
            if r["qid"] in nat:
                acc_pairs.append({"model": label, "domain": "bioasq", "qid": r["qid"], "type": r["meta"]["type"], "oracle": r["correct"],
                                  "natural": nat[r["qid"]]["correct"], "gold_retrieved": bool((nat[r["qid"]].get("retrieval") or {}).get("hit"))})
        tr = {r["qid"]: r for r in load(f"bq_{m}", "bioasq", "trust_rag")}
        rt = {r["qid"]: r for r in load(f"bq_{m}", "bioasq", "removed_trust")}
        pairs = [q for q in rt if q in tr and (tr[q].get("retrieval") or {}).get("hit")]
        if pairs:
            rr = np.mean([refused(rt[q]) for q in pairs]); over = np.mean([refused(tr[q]) for q in pairs])
            ref.append({"model": label, "size_b": size, "n_pairs": len(pairs), "refuse_removed": rr, "over_refusal": over,
                        "bal_acc": (rr + 1 - over) / 2, "leak_rate": np.mean([bool(rt[q]["meta"].get("residual_answer_in_context")) for q in pairs])})
    P, R, A, F = pd.DataFrame(per_model), pd.DataFrame(recs), pd.DataFrame(acc_pairs), pd.DataFrame(ref)
    for name, df in (("per_model", P), ("attribution_records", R), ("accuracy_pairs", A), ("refusal", F)):
        df.to_csv(OUT / f"{name}.csv", index=False)

    print("=== Per model: oracle attribution on BioASQ (snippet-level gold) vs PubMedQA (abstract-level gold) ===")
    print(P.round(3).to_string(index=False))
    b1 = int((P.bioasq_sim_sent < P.pubmedqa_sim_sent).sum())
    print(f"B1: BioASQ sim_sent below PubMedQA for {b1}/{len(P)} models -> {'supported' if b1 >= 6 else 'not supported'} (needs >= 6 of 7)")

    R2 = pd.concat([R.assign(m="sim_sent", v=R.sim_sent), R.assign(m="nli_set", v=R.nli_set)])
    R2 = R2.pivot_table(index=["model", "domain", "qid"], columns="m", values="v").reset_index()
    b2 = boot_diff(R2, "sim_sent", "nli_set")
    b3 = boot_diff(R, "num_sim", "sim_sent")
    A["loss"] = A.oracle - A.natural
    b4 = boot_diff(A.assign(zero=0.0), "loss", "zero")
    ph = holm([b2["p"], b4["p"]])
    rows = [{"test": "B2 sim_sent - nli_set (oracle)", **b2, "p_holm": ph[0],
             "verdict": "supported" if b2["diff"] >= MARGIN and ph[0] < 0.05 else "not supported"},
            {"test": "B3 num_sim - sim_sent (oracle), CI inside +-0.05", **b3, "p_holm": np.nan,
             "verdict": "supported" if -MARGIN < b3["ci_lo"] and b3["ci_hi"] < MARGIN else "not supported"},
            {"test": "B4 retrieval loss (oracle - natural accuracy)", **b4, "p_holm": ph[1],
             "verdict": "supported" if b4["diff"] >= MARGIN and ph[1] < 0.05 else "not supported"}]
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "tests.csv", index=False)
    print("\n=== B2-B4 (cluster bootstrap over questions) ===")
    print(T.round(4).to_string(index=False))

    print("\n=== Accuracy: natural retrieval vs oracle, by model and question type ===")
    print(A.groupby(["model", "type"])[["natural", "oracle", "gold_retrieved"]].mean().round(3).to_string())
    print("\n=== B5: refusal on the removal task (trust prompt; pairs with a gold chunk retrieved) ===")
    print(F.round(3).to_string(index=False))
    if len(F):
        big, small = F[F.size_b >= 8], F[F.size_b < 8]
        ok = (big.bal_acc >= 0.70).all() and (small.bal_acc < 0.60).all()
        print(f"B5: models >= 8B with balanced accuracy >= .70: {int((big.bal_acc >= .70).sum())}/{len(big)}; "
              f"1.5B below .60: {bool((small.bal_acc < .60).all())} -> {'supported' if ok else 'not supported'}")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
