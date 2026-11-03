"""Paired comparison on the RAGTruth QA test split (900 responses): the supervised ModernBERT checker (Stage 2 Part B) vs the
mid-sem zero-shot NLI detectors. CPU only, uses the saved per-response scores; nothing is re-run.

python -m scripts.compare_ragtruth [--n-boot 10000]
Primary test (one, fixed in advance): checker vs mid-sem nli_set (DeBERTa-v3-base), F1 at each method's own pre-chosen threshold
(checker: dev-tuned; NLI: train-tuned, exactly as eval_ragtruth.py). Secondary (reported, not corrected): the other zero-shot methods
and the large NLI model, and threshold-free AUROC. Paired bootstrap: the same resampled responses for both methods.
Writes experiments/results/ragtruth/paired_comparison.csv
"""
import argparse

import numpy as np
import pandas as pd

from src.common import RESULTS, ROOT
from src.evaluation import metrics as M

D = RESULTS / "ragtruth"
LARGE = "_DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def best_threshold(x, y):
    """Same rule as scripts/eval_ragtruth.py (copied: that script needs tqdm and the NLI stack at import)."""
    ts = np.unique(np.quantile(x, np.linspace(0, 1, 201)))
    return max(ts, key=lambda t: M.prf(x >= t, y)["f1"])


def f1_boot(pred, y, idx):
    """F1 for every bootstrap replicate at once; pred, y: bool arrays; idx: [B, n] resample indices."""
    p, t = pred[idx], y[idx]
    tp, fp, fn = (p & t).sum(1), (p & ~t).sum(1), (~p & t).sum(1)
    return np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)


def auroc_boot(s, y, idx):
    return np.array([M.auroc(s[i], y[i]) for i in idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    ck = pd.read_csv(ROOT / "experiments/checker/ModernBERT-base/test_predictions.csv")
    t_ck = float(pd.read_csv(D / "checker_ModernBERT-base.csv").threshold.iloc[0])
    arms = {"checker ModernBERT-base (supervised)": (ck.set_index("id").p, t_ck)}
    for tag, name in (("", "base"), (LARGE, "large")):
        tr, te = pd.read_csv(D / f"train_scores{tag}.csv"), pd.read_csv(D / f"test_scores{tag}.csv").set_index("id")
        for m in ("nli_set", "nli_window", "nli_chunk", "sim_sent"):
            arms[f"{m} zero-shot DeBERTa-v3-{name}"] = (te[m], best_threshold(tr[m].values, tr.label.values))
    ids = ck.id.values
    y = ck.set_index("id").label.loc[ids].values.astype(bool)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ids), (a.n_boot, len(ids)))
    idx_auc = idx[:2000]                                   # AUROC is slower; 2,000 replicates are enough for its interval

    ref = "checker ModernBERT-base (supervised)"
    s_ref, t_ref = arms[ref][0].loc[ids].values, arms[ref][1]
    f_ref, a_ref = f1_boot(s_ref >= t_ref, y, idx), auroc_boot(s_ref, y, idx_auc)
    rows = []
    for name, (s, t) in arms.items():
        s = s.loc[ids].values
        f1 = M.prf(s >= t, y)["f1"]
        row = {"method": name, "threshold": t, "f1": 100 * f1, "auroc": M.auroc(s, y)}
        if name != ref:
            df = f_ref - f1_boot(s >= t, y, idx)
            da = a_ref - auroc_boot(s, y, idx_auc)
            row.update({"checker_minus_f1": 100 * (M.prf(s_ref >= t_ref, y)["f1"] - f1),
                        "f1_diff_ci_lo": 100 * np.quantile(df, .025), "f1_diff_ci_hi": 100 * np.quantile(df, .975),
                        "p_f1_two_sided": min(1.0, 2 * min((df <= 0).mean(), (df >= 0).mean())),
                        "checker_minus_auroc": M.auroc(s_ref, y) - row["auroc"],
                        "auroc_diff_ci_lo": np.quantile(da, .025), "auroc_diff_ci_hi": np.quantile(da, .975),
                        "p_auroc_two_sided": min(1.0, 2 * min((da <= 0).mean(), (da >= 0).mean())),
                        "role": "PRIMARY" if name == "nli_set zero-shot DeBERTa-v3-base" else "secondary"})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(D / "paired_comparison.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"RAGTruth QA test: n={len(ids)}, hallucinated={y.sum()}, bootstrap {a.n_boot} (AUROC {len(idx_auc)})")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
