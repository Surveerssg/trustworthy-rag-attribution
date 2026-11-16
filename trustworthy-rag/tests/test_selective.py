"""Tests for the selective-answering evaluation (scripts/eval_selective.py): pure numpy, no models."""
import numpy as np

from scripts.eval_selective import (calibrated_gain, coverage_at_risk, e_aurc, oracle_utility, policy_arrays,
                                    trivial, utility_curve)


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    answerable = rng.uniform(size=n) < 0.5
    correct = answerable & (rng.uniform(size=n) < 0.6)
    refused = np.zeros(n, bool)
    return policy_arrays(answerable, correct, refused), answerable, correct


def test_policy_arrays_definition():
    a, r = policy_arrays([True, True, False, False], [1, 0, 0, 0], [False, False, False, True])
    assert a.tolist() == [1.0, -1.0, -1.0, 1.0]       # last: base system already refused an unanswerable -> a = r
    assert r.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_curve_endpoints_are_the_trivial_policies():
    (a, r), _, _ = _toy()
    U = utility_curve(np.random.default_rng(1).uniform(size=len(a)), a, r)
    t = trivial(a, r)
    assert np.isclose(U[0], t["always_refuse"]) and np.isclose(U[-1], t["always_answer"])


def test_oracle_selector_dominates_everything():
    (a, r), _, _ = _toy()
    orc = oracle_utility(a, r)
    assert orc >= trivial(a, r)["best_trivial"]
    assert orc >= utility_curve(np.random.default_rng(2).uniform(size=len(a)), a, r).max() - 1e-12
    perfect = utility_curve(a - r, a, r)               # confidence = true gain of answering
    assert np.isclose(perfect.max(), orc)


def test_random_confidence_has_no_value():
    (a, r), _, _ = _toy(n=2000)
    rng = np.random.default_rng(3)
    conf = rng.uniform(size=len(a))
    wrong = (a < 0).astype(float)
    assert abs(e_aurc(conf, wrong, np.ones(len(a), bool))) < 0.1          # ~0 for noise
    assert e_aurc(-wrong + 1e-6 * conf, wrong, np.ones(len(a), bool)) > 0.95   # ~1 for a perfect signal
    gains = [calibrated_gain(conf, a, r, 50, 100, np.random.default_rng(s))[0] for s in range(3)]
    assert max(gains) < 0.02                           # a noise threshold never beats the best trivial policy


def test_informative_signal_beats_trivial_and_random_abstention():
    rng = np.random.default_rng(4)
    n = 3000
    good = rng.uniform(size=n) < 0.4
    conf = good * 1.0 + rng.normal(0, 0.8, n)          # informative but noisy
    a = np.where(good, 1.0, -1.0)
    r = np.zeros(n)                                    # all answerable: refusing is worth 0
    assert calibrated_gain(conf, a, r, 100, 100, np.random.default_rng(5))[0] > 0.05
    cov_sig, cov_rand = coverage_at_risk(conf, (a < 0).astype(float), np.ones(n, bool), alpha=0.2)
    assert cov_sig > cov_rand + 0.1


def test_forced_refusals_cannot_be_answered():
    a, r = policy_arrays([True, False], [1, 0], [True, True])
    assert (a == r).all()                              # whatever the policy, a refused question stays refused


# ------------------------------------------------------------------ wrong-answer cost sensitivity
def test_wrong_cost_scales_only_the_penalty():
    from scripts.eval_selective import policy_arrays as pa
    a, r = pa([True, True, False], [1, 0, 0], [False, False, False], wrong_cost=3.0)
    assert a.tolist() == [1.0, -3.0, -3.0] and r.tolist() == [0.0, 0.0, 1.0]   # answering an unanswerable q also costs 3
    a1, _ = pa([True, True, False], [1, 0, 0], [False, False, False])
    assert a1.tolist() == [1.0, -1.0, -1.0]                                    # default stays the mid-sem utility


def test_breakeven_cost_is_where_the_trivial_policies_tie():
    from scripts.eval_selective import breakeven_cost, policy_arrays as pa
    ans, cor, ref = [True] * 10, [1] * 6 + [0] * 4, [False] * 10             # answer-all = 0.6 - 0.4 c, refuse-all = 0
    be = breakeven_cost(ans, cor, ref)
    assert np.isclose(be, 1.5)
    a, r = pa(ans, cor, ref, be)
    assert np.isclose(a.mean(), r.mean())
    # half the questions unanswerable and answered: refuse-all (0.5) beats answer-all (0.4) even at cost 0
    assert np.isnan(breakeven_cost([True] * 5 + [False] * 5, [1] * 4 + [0] * 6, [False] * 10))
    assert breakeven_cost([True] * 4, [1] * 4, [False] * 4) == float("inf")   # never wrong


def test_cost_sweep_flips_the_best_trivial_policy_and_finds_value_when_refusing_is_best():
    import pandas as pd
    from scripts.eval_selective import cost_matrix, cost_sweep
    rng = np.random.default_rng(0)
    n = 400
    good = rng.uniform(size=n) < 0.6
    df = pd.DataFrame({"domain": "squad2", "answerable": True, "correct": good.astype(float), "refused": False,
                       "ret_top1": good + rng.normal(0, 0.6, n)})              # informative signal
    sw = cost_sweep(df, costs=(0.1, 4.0), signals=["ret_top1"])
    lo, hi = sw[sw.wrong_cost == 0.1].iloc[0], sw[sw.wrong_cost == 4.0].iloc[0]
    assert lo.best_trivial_is == "answer" and hi.best_trivial_is == "refuse"
    assert np.isclose(lo.breakeven_cost, good.mean() / (1 - good.mean()))      # all answerable: p / (1 - p)
    assert hi.oracle_gain > lo.oracle_gain                                     # selecting matters more when errors are dear
    assert hi["gain@cal50"] > 0.02 and hi["P(gain<=0)@cal50"] < 0.1            # refuse-all is beatable by answering the confident
    cells, ceil, wins = cost_matrix(sw)
    assert cells.loc["squad2", 4.0].startswith("R +") and cells.loc["squad2", 4.0].endswith("*") and wins[4.0] == 1
