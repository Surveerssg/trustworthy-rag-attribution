"""Domain-calibrated attribution: similarity by default, mixed with evidence-set NLI by a weight picked from a few labelled questions.

Why (paper/findings/stage2a_findings.md, README section "Attribution"): with the gold passage guaranteed, sentence-level similarity
(`sim_sent`) picks the gold chunk best on average (0.71 vs 0.68 for nli_set, and 12-36x cheaper) but fails in FinanceBench
(0.33 vs 0.52 for NLI). Verification ("is the claim supported?") stays with NLI; this module only decides WHICH CHUNK a
sentence is attributed to.

    score = w * sim_sent + (1 - w) * nli_set          w in {0, 0.1, ..., 1}, one w per domain, chosen on n labelled questions
"""
import json
from pathlib import Path

import numpy as np

from src.common import ROOT

GRID = tuple(round(float(x), 2) for x in np.linspace(0, 1, 11))
DEFAULT_W = 0.9    # tuned on the two general domains only: nearly pure similarity; used when a domain has no calibrated weight
WEIGHTS_PATH = ROOT / "configs" / "attribution_weights.json"


def fused(sim, nli, w):
    return w * np.asarray(sim, float) + (1 - w) * np.asarray(nli, float)


def _gold_idx(rec):
    gold = set(rec.get("gold_chunk_ids") or [])
    return {i for i, c in enumerate(rec["retrieved"]) if c["chunk_id"] in gold}


def acc_by_weight(rec, grid=GRID):
    """Attribution accuracy (share of answer sentences whose top chunk is gold) for every weight in the grid, or None if the
    record cannot be used (no gold chunk in the context, or the scores are missing). `nli_set` should already carry the
    tie-break (src.attribution.methods.refresh_record)."""
    sc = (rec.get("attribution") or {}).get("scores") or {}
    gold = _gold_idx(rec)
    if not gold or "sim_sent" not in sc or "nli_set" not in sc:
        return None
    sim, nli = np.asarray(sc["sim_sent"], float), np.asarray(sc["nli_set"], float)
    if sim.size == 0 or sim.shape != nli.shape:
        return None
    return np.array([np.mean([int(j) in gold for j in fused(sim, nli, w).argmax(1)]) for w in grid])


def pick_weight(curve, grid=GRID):
    """Best weight on a mean-accuracy curve; ties go to the LARGER weight (more similarity: cheaper, no NLI model needed)."""
    return float(max(zip(np.round(curve, 12), grid))[1])


def fit_weight(records, n_labels=20, seed=0, grid=GRID):
    """Choose the weight from `n_labels` randomly drawn labelled questions of ONE domain.
    Returns (w, indices of the labelled questions within the usable records)."""
    usable = [(i, a) for i, r in enumerate(records) if (a := acc_by_weight(r, grid)) is not None]
    if len(usable) < 2:
        raise ValueError("not enough usable records to fit an attribution weight")
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(usable))[:min(n_labels, len(usable))]
    curve = np.mean([usable[k][1] for k in pick], axis=0)
    return pick_weight(curve, grid), [usable[k][0] for k in pick]


def evaluate_selection(records, n_labels=20, draws=200, seed=0, grid=GRID):
    """How good is 'pick w from n labels'? Held-out attribution accuracy over random draws, next to the fixed alternatives."""
    A = np.array([a for r in records if (a := acc_by_weight(r, grid)) is not None])
    n = len(A)
    rng = np.random.default_rng(seed)
    accs, ws = [], []
    for _ in range(draws):
        perm = rng.permutation(n)
        cal, te = perm[:n_labels], perm[n_labels:]
        w = pick_weight(A[cal].mean(0), grid)
        accs.append(A[te, list(grid).index(w)].mean()); ws.append(w)
    return {"n": n, "sim_sent": float(A[:, list(grid).index(1.0)].mean()), "nli_set": float(A[:, list(grid).index(0.0)].mean()),
            "mix_from_labels": float(np.mean(accs)), "best_fixed_w_on_test": float(A.mean(0).max()),
            "w_mean": float(np.mean(ws)), "w_mode": float(max(set(ws), key=ws.count))}


def load_weights(path=None):
    p = Path(path or WEIGHTS_PATH)
    return json.loads(p.read_text(encoding="utf8")).get("weights", {}) if p.exists() else {}


def fuse_weight(domain, weights=None):
    """The calibrated weight of a domain, DEFAULT_W if it has none."""
    return float((weights if weights is not None else load_weights()).get(domain, DEFAULT_W))


def fused_matrix(scores, domain, weights=None):
    """[n_sentences x n_chunks] attribution matrix, or None when the needed score matrices are missing."""
    if "sim_sent" not in scores or "nli_set" not in scores:
        return scores.get("sim_sent")
    return fused(scores["sim_sent"], scores["nli_set"], fuse_weight(domain, weights))
