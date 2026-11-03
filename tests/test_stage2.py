"""Tests for scripts/analyze_stage2.py on tiny synthetic scored records (same fields score_run writes)."""
import numpy as np

from scripts import analyze_stage2 as A


def rec(qid, domain="squad2", answerable=True, correct=1.0, refused=False, answer="The answer is Paris.", chunks=("a", "b", "c"),
        gold=("a",), ret_top1=0.6, hit=None, residual=False, orig=None, lp=None, nli_set=None, loo=None, attr=None, atype="short"):
    r = {"qid": qid, "domain": domain, "answerable": answerable, "correct": correct, "refused": refused, "answer": answer,
         "answer_type": atype, "gold_chunk_ids": list(gold) if answerable else [],
         "retrieved": [{"chunk_id": c, "score": ret_top1 - 0.05 * i, "text": "t"} for i, c in enumerate(chunks)],
         "features": {"ret_top1": ret_top1, "ret_mean": ret_top1 - 0.05, "ret_gap": 0.05, "nli_cov_mean": 0.9, "nli_cov_min": 0.8,
                      "frac_chunks_support": 0.4, "sim_cov_mean": 0.7, "n_sent": 1},
         "attribution": {"scores": ({"nli_set": nli_set, "sim_sent": nli_set} if nli_set is not None else {})},
         "attr_metrics": attr or {}, "meta": {"residual_answer_in_context": residual, "original_gold_answers": orig or []}}
    if hit is not None:
        r["retrieval"] = {"hit": float(hit)}
    if lp:
        r["lp"] = lp
    if loo:
        r["loo"] = {"influence": loo}
    return r


def by(domain, recs):
    return {domain: {r["qid"]: r for r in recs}}


def test_gold_rank():
    assert A.gold_rank(rec("q", chunks=("x", "y", "a"), gold=("a",))) == 3
    assert A.gold_rank(rec("q", chunks=("x", "y", "z"), gold=("a",))) is None


def test_oracle_position_detects_primacy_bias():
    R = []
    for pos in (1, 2, 3):                                   # accuracy falls with gold position
        for k in range(30):
            chunks = ["x", "y", "z"]; chunks[pos - 1] = "a"
            R.append(rec(f"{pos}-{k}", chunks=chunks, correct=float(k < {1: 28, 2: 20, 3: 10}[pos])))
    tab, corr = A.oracle_position(by("squad2", R))
    assert tab.gold_position.tolist() == [1, 2, 3] and tab.n.tolist() == [30, 30, 30]
    assert np.allclose(tab.acc, [28 / 30, 20 / 30, 10 / 30])
    assert corr.spearman_pos_vs_correct.iloc[0] < -0.3 and corr.p.iloc[0] < 0.01


def test_removal_summary_rates_leaks_and_residual_exclusion():
    W = [rec(f"q{i}", hit=1, correct=1.0) for i in range(10)]
    R = ([rec(f"q{i}", answerable=False, refused=True, answer="INSUFFICIENT EVIDENCE") for i in range(4)] +   # refuses
         [rec(f"q{i}", answerable=False, answer="It is Paris.", orig=["Paris"]) for i in range(4, 7)] +       # leaks gold answer
         [rec(f"q{i}", answerable=False, answer="Probably Rome.", orig=["Paris"]) for i in range(7, 9)] +    # wrong guess
         [rec("q9", answerable=False, answer="Paris", orig=["Paris"], residual=True)])                        # not a true removal
    s = A.removal_summary(by("squad2", R), by("squad2", W), "vanilla")
    row = s.iloc[0]
    assert row.n_pairs == 10 and row.n_clean == 9 and np.isclose(row.residual_share, 0.1)
    assert np.isclose(row.refuse_removed, 4 / 9) and np.isclose(row.false_answer_when_removed, 5 / 9)
    assert np.isclose(row.leak_original_answer, 3 / 5) and row.n_leak_judged == 5
    assert np.isclose(row.refuse_with_evidence, 0.0) and np.isclose(row.refusal_gap, 4 / 9)


def test_removal_pairs_require_gold_retrieved_with_evidence():
    W = [rec("q1", hit=1), rec("q2", hit=0)]                  # q2: retrieval missed the gold even WITH evidence
    R = [rec("q1", answerable=False), rec("q2", answerable=False)]
    r, w = A.removal_pairs(by("squad2", R), by("squad2", W), "squad2")
    assert [x["qid"] for x in r] == ["q1"]


def test_removal_signal_auroc_orientation_and_refusal_is_lowest_confidence():
    n = 24
    W = [rec(f"q{i}", hit=1, ret_top1=0.7 + 0.001 * i) for i in range(n)]
    R = [rec(f"q{i}", answerable=False, ret_top1=0.3 + 0.001 * i) for i in range(n)]
    g = A.removal_signals(by("squad2", R), by("squad2", W), "vanilla")
    row = g[(g.signal == "ret_top1") & (g.domain == "squad2")].iloc[0]
    assert row.auroc_present > 0.99 and row.f1_in_domain > 0.95
    # a removed answer that refuses must rank below everything, whatever its (high) raw signal
    R2 = [rec(f"q{i}", answerable=False, refused=True, answer="INSUFFICIENT EVIDENCE", ret_top1=0.95) for i in range(n)]
    g2 = A.removal_signals(by("squad2", R2), by("squad2", W), "trust")
    assert g2[(g2.signal == "ret_top1")].auroc_present.iloc[0] > 0.99
    assert g2[(g2.signal == "self_refusal")].f1_in_domain.iloc[0] > 0.99


def test_retrieval_effect_uses_matched_questions_for_attribution():
    m = lambda acc: {"nli_set": {"attr_acc": acc}, "sim_sent": {"attr_acc": acc}}
    base = [rec("q1", hit=1, correct=1, attr=m(1.0)), rec("q2", hit=0, correct=0), rec("q3", hit=1, correct=0, attr=m(0.0))]
    imp = [rec("q1", hit=1, correct=1, attr=m(1.0)), rec("q2", hit=1, correct=1, attr=m(0.0)), rec("q3", hit=1, correct=1, attr=m(1.0))]
    df = A.retrieval_effect(by("squad2", base), by("squad2", imp), "t")
    row = df.iloc[0]
    assert np.isclose(row.hit_base, 2 / 3) and np.isclose(row.hit_imp, 1.0)
    assert np.isclose(row.acc_base, 1 / 3) and np.isclose(row.acc_imp, 1.0) and row.acc_delta > 0
    assert row.n_matched_attr == 2                           # q2 has no attribution under the base run -> excluded
    assert np.isclose(row.attr_nli_set_base, 0.5) and np.isclose(row.attr_nli_set_imp, 1.0)


def test_loo_faithfulness_agreement_unused_and_unfaithful_correct_attribution():
    # chunk a (gold) is attributed and used; second record: gold attributed but removing it changes nothing
    S = [[0.9, 0.2, 0.1]]
    used = rec("q1", nli_set=S, loo=[0.9, 0.05, 0.0], gold=("a",))
    unused = rec("q2", nli_set=S, loo=[0.0, 0.8, 0.0], gold=("a",))
    df = A.loo_faithfulness(by("squad2", [used, unused]), taus=(0.5,))
    row = df[df.method == "nli_set"].iloc[0]
    assert row.n == 2 and np.isclose(row.top_attributed_unused, 0.5)
    assert np.isclose(row.top_attributed_is_top_influential, 0.5)
    assert np.isclose(row.gold_chunk_used, 0.5) and np.isclose(row.correct_attribution_is_faithful, 0.5)


def test_replication_counts_identical_answers():
    a = [rec("q1", answer="Paris"), rec("q2", answer="Rome")]
    b = [rec("q1", answer="paris"), rec("q2", answer="Berlin", correct=0.0)]
    df = A.replication(by("squad2", a), by("squad2", b))
    assert np.isclose(df.same_answer.iloc[0], 0.5) and np.isclose(df.same_context.iloc[0], 1.0)
    assert np.isclose(df.acc_diff.iloc[0], 0.5)


def test_youden_threshold_is_not_fooled_by_refuse_everything():
    rng = np.random.default_rng(0)
    t, ba = A.youden_threshold(rng.uniform(size=300), rng.uniform(size=300) < 0.5)   # no signal at all
    assert ba < 0.65                                                              # trivial level is 0.5
    v = np.r_[np.full(50, 0.9), np.full(50, 0.1)]
    t, ba = A.youden_threshold(v, np.r_[np.zeros(50, bool), np.ones(50, bool)])
    assert ba == 1.0 and 0.1 < t <= 0.9


def test_logprob_signals_skipped_when_one_side_has_none():
    lp = {"mean": -0.1, "min": -1.0, "first_mean": -0.2, "frac_low": 0.1, "margin_mean": 3.0}
    W = [rec(f"q{i}", hit=1) for i in range(12)]                                  # e.g. the mid-sem trust run: no log-probs
    R = [rec(f"q{i}", answerable=False, lp=lp) for i in range(12)]
    g = A.removal_signals(by("squad2", R), by("squad2", W), "trust")
    assert "lp_mean" not in set(g.signal) and "ret_top1" in set(g.signal)
    g2 = A.removal_signals(by("squad2", R), by("squad2", [dict(w, lp=lp) for w in W]), "vanilla")
    assert "lp_mean" in set(g2.signal)


def test_removal_reports_the_refuse_everything_reference():
    n = 20
    W = [rec(f"q{i}", hit=1) for i in range(n)]
    R = [rec(f"q{i}", answerable=False, refused=True, answer="INSUFFICIENT EVIDENCE") for i in range(n)]
    row = A.removal_signals(by("squad2", R), by("squad2", W), "x").query("signal == 'self_refusal'").iloc[0]
    assert np.isclose(row.f1_refuse_all, 2 / 3) and np.isclose(row.bal_acc_in_domain, 1.0)


# ------------------------------------------------------------------ supervised checker (Stage 2 Part B) plumbing
def test_checker_scores_are_attached_and_become_a_signal(tmp_path):
    import json
    from scripts import eval_selective as ES
    run = tmp_path / "r1"; run.mkdir()
    recs = [rec(f"q{i}", correct=float(i % 2)) for i in range(30)]
    (run / "squad2__vanilla_rag.scored.jsonl").write_text("\n".join(json.dumps(r) for r in recs), encoding="utf8")
    ck = [{"domain": "squad2", "system": "vanilla_rag", "qid": f"q{i}", "p_halluc": 0.9 if i % 2 == 0 else 0.1} for i in range(30)]
    (run / "checker_ModernBERT-base.jsonl").write_text("\n".join(json.dumps(c) for c in ck), encoding="utf8")
    df = ES.load_frame("r1", "vanilla_rag", runs_dir=tmp_path)
    assert np.allclose(df.set_index("qid").loc["q0", "checker"], 0.1) and np.allclose(df.set_index("qid").loc["q1", "checker"], 0.9)
    assert A.load(tmp_path, "r1", "vanilla_rag")["squad2"]["q0"]["p_halluc"] == 0.9
    res = ES.evaluate(ES.add_learned(df), draws=20, boots=20)
    assert res[res.signal == "checker"].auroc.iloc[0] > 0.99          # checker perfectly separates correct from wrong here


def test_no_checker_file_changes_nothing(tmp_path):
    import json
    from scripts import eval_selective as ES
    run = tmp_path / "r1"; run.mkdir()
    (run / "squad2__vanilla_rag.scored.jsonl").write_text(json.dumps(rec("q0")), encoding="utf8")
    assert ES.load_checker(run) == {} and "checker" not in ES.load_frame("r1", "vanilla_rag", runs_dir=tmp_path)


def test_checker_removal_signal_skipped_when_one_side_lacks_it():
    n = 12
    W = [dict(rec(f"q{i}", hit=1), p_halluc=0.1) for i in range(n)]
    R = [rec(f"q{i}", answerable=False) for i in range(n)]            # checker not applied to the removed run
    g = A.removal_signals(by("squad2", R), by("squad2", W), "x")
    assert "checker" not in set(g.signal)
    g2 = A.removal_signals(by("squad2", [dict(r, p_halluc=0.9) for r in R]), by("squad2", W), "x")
    assert g2[g2.signal == "checker"].auroc_present.iloc[0] > 0.99     # low p_halluc with evidence, high without
