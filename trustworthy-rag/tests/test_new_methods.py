"""Tests for the methods added after the pilot."""
import numpy as np

from src.evaluation import metrics as M
from src.pipeline.prompts import is_refusal
from scripts.analyze_refusal import crc_threshold


def test_self_cite_rows_without_citation_are_skipped():
    S = [[0, 0, 0], [0, 1, 0]]                      # sentence 1 has no citation
    a = M.attribution_metrics(S, [False, True, False], 0.5, skip_empty_rows=True)
    assert a["attr_acc"] == 1.0                      # only the cited sentence counts (was 0.5 when row 1 -> "chunk 1")
    assert M.attribution_metrics([[0, 0]], [True, False], 0.5, skip_empty_rows=True) is None


def test_refusal_detection():
    assert is_refusal("INSUFFICIENT EVIDENCE")
    assert is_refusal("The passages do not contain enough information to answer.")
    assert not is_refusal("The treaty was signed in 1648 [2].")


def test_conformal_risk_control_holds_on_exchangeable_data():
    rng = np.random.default_rng(0)
    conf = rng.uniform(size=4000)
    wrong = rng.uniform(size=4000) > conf           # higher confidence -> less often wrong
    cal, test = slice(0, 500), slice(500, None)
    for alpha in (0.05, 0.1, 0.2):
        t = crc_threshold(conf[cal], wrong[cal], alpha)
        risk = np.mean((conf[test] >= t) & wrong[test])
        assert risk <= alpha + 0.02, (alpha, risk)   # guarantee (up to sampling noise)
    assert crc_threshold(conf[:50], np.ones(50, bool), 0.01) == np.inf   # impossible target -> refuse all


def test_evidence_set_nli_recognises_joint_support():
    from src.attribution.methods import attribute
    chunks = [{"text": "Castor et Pollux is an opera by Jean-Philippe Rameau, first performed in 1737."},
              {"text": "Manon Lescaut is an opera in four acts by Giacomo Puccini, premiered in 1893."},
              {"text": "The Eiffel Tower was completed in 1889."}]
    r = attribute("Castor et Pollux and Manon Lescaut were composed by different people.", chunks)
    single, joint = np.asarray(r["scores"]["nli_window"]).max(), np.asarray(r["scores"]["nli_set"]).max()
    assert joint > single and joint >= 0.5           # multi-hop claim only supported jointly
    assert np.asarray(r["scores"]["nli_set"])[0, 2] < 0.5   # irrelevant passage gets no credit
