"""nli_set tie-break (paper/findings/stage2a_findings.md section 3). Runs without torch: the embedder and the NLI model are replaced by tiny stand-ins."""
import numpy as np

from src.attribution import methods as MT


def fake_embed(texts, name=None, batch_size=64):
    out = []
    for t in texts:
        t = t.lower()
        v = (np.array([1.0, 1.0, 0.0]) if "claim" in t else np.array([1.0, 0.2, 0.0]) if "alpha" in t
             else np.array([0.2, 1.0, 0.0]) if "beta" in t else np.array([0.0, 0.0, 1.0]))
        out.append(v / np.linalg.norm(v))
    return np.array(out)


def fake_nli(premises, hypotheses, batch_size=16):
    """Alpha alone entails 0.3, beta alone 0.6, together 0.9, anything else 0.1."""
    rows = []
    for p in premises:
        p = p.lower()
        e = 0.9 if ("alpha" in p and "beta" in p) else 0.3 if "alpha" in p else 0.6 if "beta" in p else 0.1
        rows.append([e, 1 - e, 0.0])
    return np.array(rows)


CHUNKS = [{"text": "Alpha text."}, {"text": "Beta text."}, {"text": "Gamma text."}]


def test_break_set_ties_prefers_the_chunk_with_more_own_evidence():
    ns, nw = np.array([[0.9, 0.9, 0.1]]), np.array([[0.3, 0.6, 0.1]])
    out = MT.break_set_ties(ns, nw)
    assert ns.argmax(1)[0] == 0 and out.argmax(1)[0] == 1        # before: lower index wins the tie; after: the stronger chunk
    assert out[0, 2] == 0.1                                       # not jointly credited: untouched
    assert np.abs(out - ns).max() <= MT.SET_TIE_EPS               # support values move by at most eps


def test_break_set_ties_changes_nothing_without_joint_credit():
    ns = nw = np.array([[0.2, 0.7, 0.4]])
    assert np.array_equal(MT.break_set_ties(ns, nw), ns)


def test_attribute_end_to_end_credits_the_individually_stronger_chunk(monkeypatch):
    monkeypatch.setattr(MT, "embed", fake_embed)
    monkeypatch.setattr(MT, "nli_probs", fake_nli)
    MT._SENT_EMB.clear()
    r = MT.attribute("The claim holds.", CHUNKS, methods=("sim_sent", "nli_window", "nli_set"))
    win, st = np.array(r["scores"]["nli_window"])[0], np.array(r["scores"]["nli_set"])[0]
    assert np.allclose(win, [0.3, 0.6, 0.1])
    assert st[0] > 0.9 and st[1] > 0.9 and np.isclose(st[2], 0.1)  # both chunks of the pair carry the joint support ...
    assert st.argmax() == 1                                        # ... but attribution goes to the stronger one (was index 0)
    assert r["set_tiebreak"] == MT.SET_TIE_EPS
    assert st.max() - 0.9 <= MT.SET_TIE_EPS                        # the sentence-level support value is unchanged


def test_attribution_no_longer_depends_on_where_the_gold_chunk_sits(monkeypatch):
    monkeypatch.setattr(MT, "embed", fake_embed)
    monkeypatch.setattr(MT, "nli_probs", fake_nli)
    for order in ([0, 1, 2], [1, 0, 2], [2, 1, 0]):               # beta (the individually stronger chunk) at every position
        MT._SENT_EMB.clear()
        chunks = [CHUNKS[i] for i in order]
        s = np.array(MT.attribute("The claim holds.", chunks, methods=("sim_sent", "nli_window", "nli_set"))["scores"]["nli_set"])[0]
        assert order[int(s.argmax())] == 1


def _old_style_record():
    return {"retrieved": [{"chunk_id": "c0"}, {"chunk_id": "c1"}, {"chunk_id": "c2"}], "gold_chunk_ids": ["c1"],
            "attribution": {"scores": {"nli_set": [[0.9, 0.9, 0.1]], "nli_window": [[0.3, 0.6, 0.1]]}},
            "attr_metrics": {"nli_set": {"eas": 1.0, "attr_acc": 0.0, "attr_p": 0.0, "attr_r": 0.0}}}


def test_refresh_record_corrects_old_records_in_memory_and_is_idempotent():
    old = _old_style_record()
    new = MT.refresh_record(old)
    assert old["attr_metrics"]["nli_set"]["attr_acc"] == 0.0 and "set_tiebreak" not in old["attribution"]   # input not mutated
    assert new["attr_metrics"]["nli_set"]["attr_acc"] == 1.0 and new["attribution"]["set_tiebreak"] == MT.SET_TIE_EPS
    assert MT.refresh_record(new) is new                                                                   # second call is a no-op
    assert MT.refresh_record({"attribution": {}}) == {"attribution": {}}                                    # records without scores pass through
