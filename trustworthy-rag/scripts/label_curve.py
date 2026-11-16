"""How many labelled questions does the per-domain attribution mix need? (exploratory; the 20-label recipe was fixed on 26 Oct 2026)

python -m scripts.label_curve [--draws 500]
Reads experiments/runs/v2_<model>/<domain>__oracle_vanilla.scored.jsonl (gold passage guaranteed) for every model that has them.
For each model x domain and each draw: shuffle the usable questions, keep everything after the first 50 as the TEST set, and pick the
weight w of `w * sim_sent + (1 - w) * nli_set` from the first n questions (n = 0, 2, 5, 10, 20, 50; n = 0 means the default w = 0.9
used for domains without labels). The test set is the same for every n within a draw, so the curve is a paired comparison.
Reported: held-out attribution accuracy (mean and 10th percentile over draws), next to pure similarity (w = 1), pure NLI (w = 0) and
the best fixed w chosen on the test set itself (a ceiling no label budget can beat). Writes experiments/results/label_curve/. CPU only.
"""
import argparse

import numpy as np
import pandas as pd

from scripts.analyze_stage2 import DOMS, load
from src.attribution import fusion as F
from src.common import RESULTS, RUNS

MODELS = [("qwen1.5b", "Qwen2.5-1.5B"), ("llama8b", "Llama-3.1-8B"), ("qwen14b", "Qwen2.5-14B"), ("phi4", "Phi-4-14B"),
          ("mistral24b", "Mistral-Small-24B"), ("gemma27b", "Gemma-3-27B"), ("qwen32b", "Qwen2.5-32B")]
NS = [0, 2, 5, 10, 20, 50]
POOL, MIN_TEST = 50, 20
OUT = RESULTS / "label_curve"


def curve(A, draws, seed):
    """A: [n_records x len(GRID)] accuracy by weight. Returns per-draw held-out accuracy for every n, plus the fixed baselines."""
    g = list(F.GRID)
    rng = np.random.default_rng(seed)
    acc = {n: [] for n in NS}
    base = {"sim_sent": [], "nli_set": [], "ceiling": []}
    for _ in range(draws):
        perm = rng.permutation(len(A))
        lab, test = A[perm[:POOL]], A[perm[POOL:]]
        T = test.mean(0)
        for n in NS:
            w = F.DEFAULT_W if n == 0 else F.pick_weight(lab[:n].mean(0))
            acc[n].append(T[g.index(round(w, 2))])
        base["sim_sent"].append(T[g.index(1.0)]); base["nli_set"].append(T[g.index(0.0)]); base["ceiling"].append(T.max())
    return acc, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=500)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    rows = []
    for m, label in MODELS:
        if not (RUNS / f"v2_{m}").is_dir():
            continue
        data = load(RUNS, f"v2_{m}", "oracle_vanilla")
        for d in DOMS:
            A = np.array([x for r in data.get(d, {}).values() if r["answerable"] and not r["refused"] and (x := F.acc_by_weight(r)) is not None])
            if len(A) < POOL + MIN_TEST:
                print(f"skip {label}/{d}: {len(A)} usable questions (< {POOL + MIN_TEST})")
                continue
            acc, base = curve(A, a.draws, seed=0)
            row = {"model": label, "domain": d, "n_usable": len(A), "n_test": len(A) - POOL,
                   **{k: float(np.mean(v)) for k, v in base.items()}}
            for n in NS:
                row[f"n{n}"] = float(np.mean(acc[n]))
                row[f"n{n}_p10"] = float(np.quantile(acc[n], .10))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "label_curve_by_model_domain.csv", index=False)
    cols = ["sim_sent", "nli_set"] + [f"n{n}" for n in NS] + ["ceiling"]
    by_dom = df.groupby("domain")[cols + [f"n{n}_p10" for n in NS]].mean().reindex([d for d in DOMS if d in set(df.domain)])
    by_dom.to_csv(OUT / "label_curve_by_domain.csv")
    print(f"=== Held-out attribution accuracy vs number of labelled questions (mean over models and {a.draws} draws; gold guaranteed)")
    print("    n0 = default w=0.9 (no labels) | ceiling = best fixed w chosen on the test set itself ===")
    print(by_dom[cols].round(3).to_string())
    print("\n--- 10th percentile over draws (an unlucky label draw)")
    print(by_dom[[f"n{n}_p10" for n in NS]].round(3).to_string())
    per_model = df.groupby("model")[cols].agg(["mean", "min"])   # mean = macro over domains, min = worst domain
    summ = pd.DataFrame({c: per_model[(c, "mean")] for c in cols}).round(3)
    worst = pd.DataFrame({c: per_model[(c, "min")] for c in cols}).round(3)
    summ.to_csv(OUT / "label_curve_macro.csv"); worst.to_csv(OUT / "label_curve_worst_domain.csv")
    print("\n=== Per model: macro over domains\n" + summ.to_string())
    print("\n=== Per model: worst domain\n" + worst.to_string())
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
