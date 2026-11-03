"""scripts/analyze_logprobs.py: the statistics, on synthetic data where the answer is known."""
import numpy as np

from scripts import analyze_logprobs as L
from scripts import eval_selective as ES
from src.evaluation import metrics as M


def synth(n=300, signal=1.0, p_good=0.5, seed=0):
    rng = np.random.default_rng(seed)
    good = rng.uniform(size=n) < p_good
    conf = signal * good + rng.normal(0, 1, n)
    answerable = np.ones(n, bool)
    return conf, answerable, good.astype(float)


def test_fast_auroc_matches_sklearn_including_ties():
    rng = np.random.default_rng(0)
    c = np.round(rng.normal(size=200), 1)                    # many ties
    y = rng.uniform(size=200) < 0.4
    assert np.isclose(L.fast_auroc(c, y), M.auroc(c, y.astype(int)))
    assert np.isnan(L.fast_auroc(c, np.ones(200, bool)))


def test_benjamini_hochberg_known_case():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    m = L.benjamini_hochberg(p, 0.05)
    assert m.tolist() == [True, True, False, False, False, False, False, False, False, False]
    assert not L.benjamini_hochberg(np.array([0.5, 0.6, 0.7]), 0.1).any() and L.benjamini_hochberg(np.array([0.001]), 0.1).all()


def test_noise_signal_has_no_value_and_informative_signal_does():
    conf, ans, good = synth(300, signal=0.0, seed=1)
    a, r = ES.policy_arrays(ans, good, np.zeros(300, bool), 2.0)             # expensive errors: refuse-all is the trivial policy
    gain, lo, hi, p = L.gain_test(conf, a, r, boots=500, reps=5)
    assert lo < 0.02 and p > 0.1                                             # nothing to find in noise
    conf, ans, good = synth(300, signal=2.5, seed=2)
    a, r = ES.policy_arrays(ans, good, np.zeros(300, bool), 2.0)
    gain, lo, hi, p = L.gain_test(conf, a, r, boots=500, reps=5)
    assert gain > 0.1 and lo > 0 and p < 0.01


def test_crossfit_never_uses_a_questions_own_label():
    """Utilities are decided from OTHER folds: shuffling the labels of the held-out fold cannot change that fold's decisions."""
    conf, ans, good = synth(200, 1.5, seed=3)
    a, r = ES.policy_arrays(ans, good, np.zeros(200, bool), 1.0)
    u1 = L.crossfit_policy_utility(conf, a, r, k=5, reps=1, seed=7)
    a2 = a.copy()
    fold0 = np.array_split(np.random.default_rng(7).permutation(200), 5)[0]
    rng = np.random.default_rng(9)
    a2[fold0] = rng.permutation(a[fold0])                                    # scramble fold 0's outcomes only
    u2 = L.crossfit_policy_utility(conf, a2, r, k=5, reps=1, seed=7)
    decided_same = (u1 != u2)[fold0] & (a[fold0] == a2[fold0])               # where the outcome is unchanged, so must the decision be
    assert not decided_same.any()


def test_auroc_with_ci_covers_the_truth_and_detects_chance():
    conf, _, good = synth(400, 1.0, seed=4)                                  # true AUROC = Phi(1/sqrt(2)) = 0.76
    au, lo, hi, p = L.auroc_with_ci(conf, good.astype(int), boots=300, perms=300)
    assert lo < 0.76 < hi and p < 0.01
    conf, _, good = synth(400, 0.0, seed=5)
    assert L.auroc_with_ci(conf, good.astype(int), boots=300, perms=300)[3] > 0.05


def test_selective_lift():
    conf = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.0]); correct = np.array([1, 1, 1, 0, 0, 0.0])
    top, base = L.selective_lift(conf, correct, 0.5)
    assert top == 1.0 and base == 0.5
