"""All evaluation metrics as plain functions (unit-tested in tests/test_metrics.py)."""
import math
import re
import string
from collections import Counter

import numpy as np


# ------------------------------------------------------------------ retrieval
def retrieval_metrics(retrieved_ids, gold_ids, k=None):
    """Chunk-level Recall@k (fraction of gold retrieved), Hit@k, Precision@k, MRR, nDCG@k."""
    r = retrieved_ids[:k] if k else retrieved_ids
    g = set(gold_ids)
    if not g:
        return None
    rel = [1 if x in g else 0 for x in r]
    dcg = sum(v / math.log2(i + 2) for i, v in enumerate(rel))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(g), len(r))))
    first = next((i for i, v in enumerate(rel) if v), None)
    return {"recall": sum(rel) / len(g), "hit": float(any(rel)), "precision": sum(rel) / len(r) if r else 0.0,
            "mrr": 1 / (first + 1) if first is not None else 0.0, "ndcg": dcg / idcg if idcg else 0.0}


# ------------------------------------------------------------------ answer correctness
def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def token_recall(pred, gold):
    p, g = Counter(normalize(pred).split()), normalize(gold).split()
    if not g:
        return 0.0
    return sum((Counter(g) & p).values()) / len(g)


def exact_match(pred, gold):
    return float(normalize(pred) == normalize(gold))


def contains_answer(pred, gold):
    """STR-EM (ALCE): normalised gold string appears in the prediction."""
    ng = normalize(gold)
    return float(bool(ng) and f" {ng} " in f" {normalize(pred)} ")  # whole-word: gold "no" must not match "not"


def parse_label(pred):
    m = re.match(r"\W*(yes|no|maybe)\b", pred.strip().lower())
    return m.group(1) if m else None


_NUM = re.compile(r"-?\(?\$?\s?\d[\d,]*\.?\d*\)?\s?%?")


def extract_numbers(s):
    out = []
    for m in _NUM.findall(s or ""):
        neg = m.strip().startswith("-") or (m.strip().startswith("(") and m.strip().endswith(")"))
        v = re.sub(r"[^\d.]", "", m)
        if v and v != "." and v.count(".") <= 1:
            try:
                x = float(v)
                out.append(-x if neg else x)
            except ValueError:
                pass
    return out


def numeric_match(pred, gold, rel_tol=0.01):
    """Any key gold number (years excluded) appears in pred within tolerance, allowing x1000 unit shifts."""
    g = [x for x in extract_numbers(gold) if not (1900 <= abs(x) <= 2100 and float(x).is_integer())]
    p = extract_numbers(pred)
    if not g:
        return None
    for gv in g:
        for pv in p:
            for scale in (1, 1e3, 1e-3, 1e6, 1e-6):
                if math.isclose(abs(pv) * scale, abs(gv), rel_tol=rel_tol, abs_tol=0.006):
                    return 1.0
    return 0.0


def correctness(pred, gold_answers, answer_type):
    """Returns (binary_correct, continuous_score). Domain-appropriate but pipeline-independent."""
    if not gold_answers:
        return None, None
    if answer_type == "short":
        c = max(contains_answer(pred, g) for g in gold_answers)
        f = max(token_f1(pred, g) for g in gold_answers)
        return c, f
    if answer_type == "label":
        lab = parse_label(pred)
        c = float(lab == gold_answers[0].lower())
        return c, c
    if answer_type == "span":
        r = max(token_recall(pred, g) for g in gold_answers)
        return float(r >= 0.5), r
    if answer_type == "free":
        nm = numeric_match(pred, gold_answers[0])
        f = token_f1(pred, gold_answers[0])
        if nm is not None:
            return nm, max(nm, f)
        return None, f  # qualitative financial answer: resolved by NLI judge in scoring
    raise ValueError(answer_type)


# ------------------------------------------------------------------ attribution
def attribution_metrics(scores, gold_mask, threshold, skip_empty_rows=False):
    """scores: [n_sent x k] for one method; gold_mask: list[bool] len k (retrieved chunk is gold).
    AttrAcc  : fraction of sentences whose top-scored chunk is gold (only meaningful if any gold retrieved)
    EAS      : fraction of sentences with max score >= threshold (Evidence Attribution Score)
    AttrP/R  : answer-level attributed chunk set (argmax per sentence, above threshold) vs gold retrieved set."""
    S = np.asarray(scores, dtype=float)
    if S.size == 0:
        return None
    if skip_empty_rows:  # e.g. self-citation: a sentence with no citation has no attribution (not "chunk 1")
        S = S[S.max(1) > 0]
        if S.size == 0:
            return None
    top = S.argmax(1)
    supported = S.max(1) >= threshold
    out = {"eas": float(supported.mean())}
    if any(gold_mask):
        gm = np.asarray(gold_mask)
        out["attr_acc"] = float(gm[top].mean())
        attributed = {int(j) for j, s in zip(top, supported) if s}
        gold = {j for j, g in enumerate(gold_mask) if g}
        out["attr_p"] = len(attributed & gold) / len(attributed) if attributed else 0.0
        out["attr_r"] = len(attributed & gold) / len(gold)
    return out


# ------------------------------------------------------------------ refusal
def prf(pred, gold):
    """Binary P/R/F1 for the positive class (here: 'refuse')."""
    pred, gold = np.asarray(pred, bool), np.asarray(gold, bool)
    tp = (pred & gold).sum(); fp = (pred & ~gold).sum(); fn = (~pred & gold).sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": float(p), "recall": float(r), "f1": float(f)}


def threshold_sweep(conf, should_refuse, n=101):
    """Refuse iff conf < t. Returns list of dicts over thresholds; best F1 threshold."""
    conf = np.asarray(conf, float)
    ts = np.quantile(conf, np.linspace(0, 1, n))
    rows = [{"t": float(t), **prf(conf < t, should_refuse), "refusal_rate": float((conf < t).mean())} for t in ts]
    best = max(rows, key=lambda r: r["f1"])
    return rows, best


# ------------------------------------------------------------------ calibration
def auroc(scores, labels):
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def ece(probs, labels, bins=10):
    probs, labels = np.asarray(probs, float), np.asarray(labels, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if m.any():
            e += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(e)


def brier(probs, labels):
    return float(np.mean((np.asarray(probs, float) - np.asarray(labels, float)) ** 2))


def aurc(conf, correct):
    """Area under risk-coverage curve (lower is better) for selective answering."""
    order = np.argsort(-np.asarray(conf, float))
    c = np.asarray(correct, float)[order]
    risks = 1 - np.cumsum(c) / np.arange(1, len(c) + 1)
    return float(risks.mean())


# ------------------------------------------------------------------ significance
def paired_bootstrap(a, b, n=10000, seed=0):
    """Mean diff (a-b), 95% CI and two-sided p-value for paired per-question scores."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return {"diff": float(d.mean()), "ci_lo": float(np.quantile(bs, .025)), "ci_hi": float(np.quantile(bs, .975)),
            "p": float(min(max(p, 1 / n), 1.0))}   # two-sided p cannot exceed 1 (it did when every difference was 0)


def paired_ttest(a, b):
    from scipy.stats import ttest_rel
    t = ttest_rel(a, b)
    return {"t": float(t.statistic), "p": float(t.pvalue)}


def mcnemar(a, b):
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return {"p": 1.0, "n01": 0, "n10": 0}
    return {"p": float(binomtest(n10, n01 + n10, 0.5).pvalue), "n01": n01, "n10": n10}
