"""Domain-calibrated attribution weights (src/attribution/fusion.py): pure numpy, synthetic records."""
import json

import numpy as np

from src.attribution import fusion as F


def record(sim_correct, nli_correct, k=4, gold=1):
    """One answer sentence over k chunks. sim/nli each either put their highest score on the gold chunk or on chunk 0."""
    sim, nli = np.full((1, k), 0.2), np.full((1, k), 0.2)
    sim[0, gold if sim_correct else 0] = 0.9
    nli[0, gold if nli_correct else 0] = 0.9
    return {"retrieved": [{"chunk_id": f"c{i}"} for i in range(k)], "gold_chunk_ids": [f"c{gold}"],
            "attribution": {"scores": {"sim_sent": sim.tolist(), "nli_set": nli.tolist()}}}


def domain(n, sim_acc, nli_acc, seed=0):
    rng = np.random.default_rng(seed)
    return [record(rng.uniform() < sim_acc, rng.uniform() < nli_acc) for _ in range(n)]


def test_acc_by_weight_endpoints_are_the_two_methods():
    r = record(sim_correct=True, nli_correct=False)
    a = F.acc_by_weight(r)
    assert a[list(F.GRID).index(1.0)] == 1.0 and a[list(F.GRID).index(0.0)] == 0.0
    assert F.acc_by_weight({"retrieved": [{"chunk_id": "c0"}], "gold_chunk_ids": [], "attribution": {"scores": {}}}) is None


def test_weight_follows_the_better_method_per_domain():
    similarity_domain = domain(120, sim_acc=0.9, nli_acc=0.5, seed=1)       # like HotpotQA / PubMedQA
    nli_domain = domain(120, sim_acc=0.35, nli_acc=0.9, seed=2)             # like FinanceBench
    assert F.fit_weight(similarity_domain, 20, seed=0)[0] > 0.5       # similarity-dominant
    assert F.fit_weight(nli_domain, 20, seed=0)[0] < 0.5              # NLI-dominant (every w < 0.5 decides identically here:
                                                                      # a genuine tie, which goes to the largest such w)


def test_ties_go_to_the_larger_weight_and_selection_is_deterministic():
    both_right = [record(True, True) for _ in range(30)]
    assert F.fit_weight(both_right, 20, seed=0)[0] == 1.0                    # cheaper method wins a tie
    recs = domain(80, 0.8, 0.6, seed=3)
    assert F.fit_weight(recs, 20, seed=5) == F.fit_weight(recs, 20, seed=5)


def test_mix_from_20_labels_beats_either_single_method_across_mixed_domains():
    doms = {"general": domain(200, 0.9, 0.5, 4), "finance": domain(200, 0.3, 0.9, 5)}
    ev = {d: F.evaluate_selection(r, 20, draws=100, seed=0) for d, r in doms.items()}
    macro = lambda k: np.mean([e[k] for e in ev.values()])
    assert macro("mix_from_labels") > max(macro("sim_sent"), macro("nli_set")) + 0.05
    assert min(e["mix_from_labels"] for e in ev.values()) > min(e["sim_sent"] for e in ev.values()) + 0.2   # protects the worst domain


def test_weight_file_lookup_and_default(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"weights": {"financebench": 0.0, "cuad": 0.9}}), encoding="utf8")
    w = F.load_weights(p)
    assert F.fuse_weight("financebench", w) == 0.0 and F.fuse_weight("cuad", w) == 0.9
    assert F.fuse_weight("some_new_domain", w) == F.DEFAULT_W                 # unseen domain: similarity-dominant default
    assert F.load_weights(tmp_path / "missing.json") == {}


def test_fused_matrix_uses_the_domain_weight_and_degrades_gracefully():
    sc = {"sim_sent": [[1.0, 0.0]], "nli_set": [[0.0, 1.0]]}
    assert np.allclose(F.fused_matrix(sc, "d", {"d": 1.0}), [[1.0, 0.0]])
    assert np.allclose(F.fused_matrix(sc, "d", {"d": 0.0}), [[0.0, 1.0]])
    assert F.fused_matrix({"sim_sent": [[0.3, 0.4]]}, "d", {}) == [[0.3, 0.4]]   # no NLI scores: plain similarity


def test_explain_uses_the_domain_weight_for_the_attributed_source(tmp_path, monkeypatch):
    """Similarity points at passage 1, NLI at passage 2. A similarity-weighted domain must cite [1], an NLI-weighted one [2]."""
    from src.attribution import explain as EX
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"weights": {"squad2": 1.0, "financebench": 0.0}}), encoding="utf8")
    monkeypatch.setattr(F, "WEIGHTS_PATH", p)
    monkeypatch.setattr(EX, "embed", lambda texts, *a, **k: np.ones((len(texts), 3)) / np.sqrt(3))
    rec = {"refused": False, "answer": "It is X.", "retrieved": [{"text": "Passage one.", "score": 0.6}, {"text": "Passage two.", "score": 0.5}],
           "attribution": {"sentences": [{"text": "It is X.", "cites": []}],
                           "scores": {"sim_sent": [[0.9, 0.3]], "nli_set": [[0.2, 0.8]], "nli_window_contra": [[0.0, 0.0]]}}}
    assert EX.explain(dict(rec, domain="squad2"))["sentences"][0]["source"] == 1
    assert EX.explain(dict(rec, domain="financebench"))["sentences"][0]["source"] == 2
    assert EX.explain(dict(rec, domain="unseen"))["sentences"][0]["source"] == 1     # default weight 0.9: similarity-dominant
