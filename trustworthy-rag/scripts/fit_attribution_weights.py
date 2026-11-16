"""Calibrate the per-domain similarity/NLI attribution weight from a few labelled questions (paper/findings/stage2a_findings.md section 8).

python -m scripts.fit_attribution_weights --run v2_llama8b --system oracle_vanilla --n-labels 20 [--write]
                                          [--check v2_qwen1.5b:oracle_vanilla v2_llama8b:vanilla_rag ...]

For every domain: draw n labelled questions (seeded), pick w, then report on the REMAINING questions the attribution accuracy of
pure similarity (w=1), pure evidence-set NLI (w=0) and the picked w, plus the average over 200 random label draws. `--check`
applies the weights fitted on --run to other runs (another generator, or natural retrieval order) to see whether they transfer.
`--write` stores the weights in configs/attribution_weights.json (read by src/attribution/explain.py and the dashboard).
nli_set carries the tie-break (paper/findings/stage2a_findings.md section 3) because records are loaded through analyze_stage2.load.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from scripts.analyze_stage2 import DOMS, load
from src.attribution import fusion as F
from src.common import RUNS


def usable(recs):
    return [r for r in recs.values() if r["answerable"] and not r["refused"] and F.acc_by_weight(r) is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--system", default="oracle_vanilla")
    ap.add_argument("--n-labels", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", nargs="*", default=[], help="run:system pairs to evaluate the fitted weights on")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    data = load(RUNS, a.run, a.system)
    weights, rows, grid = {}, [], list(F.GRID)
    for d in DOMS:
        recs = usable(data.get(d, {}))
        if len(recs) <= a.n_labels + 10:
            continue
        w, lab = F.fit_weight(recs, a.n_labels, a.seed)
        weights[d] = w
        lab_set = set(lab)
        held = [r for i, r in enumerate(recs) if i not in lab_set]
        A = np.array([F.acc_by_weight(r) for r in held])
        ev = F.evaluate_selection(recs, a.n_labels, seed=a.seed)
        rows.append({"domain": d, "n_usable": len(recs), "w_picked": w, "heldout_sim_sent": A[:, grid.index(1.0)].mean(),
                     "heldout_nli_set": A[:, grid.index(0.0)].mean(), "heldout_picked": A[:, grid.index(w)].mean(),
                     "avg_over_200_draws": ev["mix_from_labels"], "best_fixed_on_test": ev["best_fixed_w_on_test"],
                     "w_mode_over_draws": ev["w_mode"]})
    df = pd.DataFrame(rows)
    print(f"=== weights fitted on {a.run}/{a.system} with {a.n_labels} labelled questions per domain (seed {a.seed}) ===")
    print(df.round(3).to_string(index=False))
    print("\nmacro average over domains:", df[["heldout_sim_sent", "heldout_nli_set", "heldout_picked", "avg_over_200_draws"]].mean().round(3).to_dict())
    print("worst domain             :", df[["heldout_sim_sent", "heldout_nli_set", "heldout_picked", "avg_over_200_draws"]].min().round(3).to_dict())
    for spec in a.check:
        run, system = spec.split(":")
        other = load(RUNS, run, system)
        out = []
        for d in DOMS:
            recs = usable(other.get(d, {}))
            if d not in weights or not recs:
                continue
            A = np.array([F.acc_by_weight(r) for r in recs])
            out.append({"domain": d, "n": len(recs), "w": weights[d], "sim_sent": A[:, grid.index(1.0)].mean(),
                        "nli_set": A[:, grid.index(0.0)].mean(), "with_fitted_w": A[:, grid.index(weights[d])].mean()})
        o = pd.DataFrame(out)
        print(f"\n--- the same weights applied to {run}/{system} (no refitting) ---")
        print(o.round(3).to_string(index=False))
        print("macro:", o[["sim_sent", "nli_set", "with_fitted_w"]].mean().round(3).to_dict(), "| worst domain:",
              o[["sim_sent", "nli_set", "with_fitted_w"]].min().round(3).to_dict())
    if a.write:
        F.WEIGHTS_PATH.parent.mkdir(exist_ok=True)
        meta = {"weights": weights, "default_for_other_domains": F.DEFAULT_W, "n_labels": a.n_labels, "seed": a.seed,
                "fitted_on": f"{a.run}/{a.system}", "score": "w * sim_sent + (1 - w) * nli_set (nli_set with tie-break)",
                "written": time.strftime("%Y-%m-%d")}
        F.WEIGHTS_PATH.write_text(json.dumps(meta, indent=2), encoding="utf8")
        print(f"\nwritten {F.WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
