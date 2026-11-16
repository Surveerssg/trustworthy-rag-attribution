"""Cross-model (scale) tables for the paper: one row per model, on the full question sets and on the shared Part D subset.

python -m scripts.analyze_stage2 --model <m> [--qids-from v2_qwen32b --skip-selective]   # first, for every model (writes the CSVs read here)
python -m scripts.scale_tables
Reads experiments/results/v2_<model>[_subset]/stage2_*.csv and the oracle runs; writes experiments/results/scale/*.csv. CPU only.
"""
import numpy as np
import pandas as pd

from src.attribution.fusion import GRID, acc_by_weight, load_weights
from src.attribution.methods import refresh_record
from src.common import RESULTS, RUNS, read_jsonl

MODELS = [("qwen1.5b", "Qwen2.5-1.5B"), ("llama8b", "Llama-3.1-8B"), ("qwen14b", "Qwen2.5-14B"), ("phi4", "Phi-4-14B"),
          ("mistral24b", "Mistral-Small-24B"), ("gemma27b", "Gemma-3-27B"), ("qwen32b", "Qwen2.5-32B")]
MODELS = [(m, l) for m, l in MODELS if (RUNS / f"v2_{m}").is_dir()]   # Part E models appear once their zips are ingested
DOMS = ["squad2", "hotpotqa", "pubmedqa", "cuad", "financebench"]
OUT = RESULTS / "scale"


def subset_keys():
    return {(r["domain"], r["qid"]) for d in DOMS for r in read_jsonl(RUNS / "v2_qwen32b" / f"{d}__vanilla_rag.scored.jsonl")}


def attribution(model, keep=None):
    """Oracle-condition attribution accuracy per domain for sim_sent (w=1), nli_set (w=0, tie-broken) and the fixed 20-label mixture."""
    W = load_weights()
    rows = []
    for d in DOMS:
        p = RUNS / f"v2_{model}" / f"{d}__oracle_vanilla.scored.jsonl"
        recs = [refresh_record(r) for r in read_jsonl(p) if keep is None or (d, r["qid"]) in keep]
        A = np.array([a for r in recs if (a := acc_by_weight(r)) is not None])
        g = list(GRID)
        rows.append({"model": model, "domain": d, "n": len(A), "sim_sent": A[:, g.index(1.0)].mean(), "nli_set": A[:, g.index(0.0)].mean(),
                     "mixture": A[:, g.index(round(float(W.get(d, 0.9)), 2))].mean()})
    return pd.DataFrame(rows)


def read(model, scope, name):
    p = RESULTS / (f"v2_{model}" + ("_subset" if scope == "subset" else "")) / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    keep = subset_keys()
    for scope in ("full", "subset"):
        acc, ref, att = [], [], []
        for m, label in MODELS:
            if scope == "full" and m in ("qwen32b", "gemma27b"):
                pass  # Part D models only have the subset; their 'full' rows equal the subset rows
            o = read(m, scope, "stage2_oracle.csv")
            for _, r in o.iterrows():
                acc.append({"model": label, "domain": r.domain, "n": r.n, "acc_base": r.acc_base, "acc_oracle": r.acc_oracle,
                            "retrieval_loss": r.retrieval_loss})
            s = read(m, scope, "stage2_removed_summary.csv")
            g = read(m, scope, "stage2_removed_signals.csv")
            for sysname in ("trust|natural", "vanilla|oracle"):
                gs = g[(g.signal == "self_refusal") & (g.system == sysname)] if len(g) else g
                ss = s[s.system == sysname] if len(s) else s
                for _, r in gs.iterrows():
                    sr = ss[ss.domain == r.domain]
                    ref.append({"model": label, "pairing": sysname, "domain": r.domain, "n_pairs": r.n_pairs, "bal_acc": r.bal_acc_in_domain,
                                "refuse_removed": sr.refuse_removed.iloc[0] if len(sr) else np.nan,
                                "over_refusal": sr.refuse_with_evidence.iloc[0] if len(sr) else np.nan})
            att.append(attribution(m, keep if scope == "subset" else None).assign(model=label))
        acc, ref, att = pd.DataFrame(acc), pd.DataFrame(ref), pd.concat(att)
        acc.to_csv(OUT / f"accuracy_{scope}.csv", index=False); ref.to_csv(OUT / f"refusal_{scope}.csv", index=False)
        att.to_csv(OUT / f"attribution_{scope}.csv", index=False)
        print(f"\n################ {scope.upper()} ({'shared Part D subset: 100 questions per domain' if scope == 'subset' else 'all questions'})")
        print("\n=== Accuracy (vanilla prompt): base retrieval / oracle context ===")
        print(acc.pivot(index="model", columns="domain", values="acc_base").reindex([l for _, l in MODELS])[DOMS].round(3).to_string())
        print(acc.pivot(index="model", columns="domain", values="acc_oracle").reindex([l for _, l in MODELS])[DOMS].round(3).to_string())
        print("\n=== Refusal balanced accuracy, trust prompt, natural pairing (and over-refusal) ===")
        t = ref[ref.pairing == "trust|natural"]
        print(t.pivot(index="model", columns="domain", values="bal_acc").reindex([l for _, l in MODELS])[DOMS].round(3).to_string())
        print(t.pivot(index="model", columns="domain", values="over_refusal").reindex([l for _, l in MODELS])[DOMS].round(3).to_string())
        print(t.pivot(index="model", columns="domain", values="n_pairs").reindex([l for _, l in MODELS])[DOMS].to_string())
        print("\n=== Oracle attribution accuracy: sim_sent / nli_set / fixed mixture, and macro / worst ===")
        for col in ("sim_sent", "nli_set", "mixture"):
            p = att.pivot(index="model", columns="domain", values=col).reindex([l for _, l in MODELS])[DOMS]
            p["macro"], p["worst"] = p.mean(1), p.min(1)
            print(f"-- {col}\n" + p.round(3).to_string())


if __name__ == "__main__":
    main()
