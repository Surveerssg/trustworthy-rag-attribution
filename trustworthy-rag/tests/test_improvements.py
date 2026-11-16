"""Tests for the Stage-1/2 improvements: implicit refusals, logprob summaries, explanation reason codes."""
from src.pipeline.llm import _summarise_logprobs
from src.pipeline.prompts import is_implicit_refusal, refusal_kind


def test_implicit_refusal_detection():
    assert refusal_kind("INSUFFICIENT EVIDENCE") == "explicit"
    assert is_implicit_refusal("Unfortunately, the provided passages do not include the cash flow statement for FY2018.")
    assert is_implicit_refusal('There is no mention of "Price Restrictions" in the provided contract passages.')
    assert is_implicit_refusal("The contract does not explicitly mention a non-compete clause. However, ...")
    # real answers that merely contain a negation must NOT count as refusals
    assert not is_implicit_refusal("No, Adobe does not have an improving operating margin profile as of FY2022.")
    assert not is_implicit_refusal("Castor et Pollux was not composed by Puccini; it is by Rameau [1].")
    # only the opening sentence decides
    assert not is_implicit_refusal("The capex was $1,577 million [2]. The passages do not mention 2019.")


def test_logprob_summary():
    toks = [{"logprob": -0.01, "top_logprobs": [{"logprob": -0.01}, {"logprob": -5.0}]},
            {"logprob": -2.0, "top_logprobs": [{"logprob": -2.0}, {"logprob": -2.1}]}]
    s = _summarise_logprobs(toks)
    assert s["n"] == 2 and abs(s["mean"] + 1.005) < 1e-9 and s["min"] == -2.0
    assert abs(s["frac_low"] - 0.5) < 1e-9 and s["margin_mean"] > 2
    assert _summarise_logprobs([]) is None


def test_explanation_reason_codes():
    from src.attribution.explain import explain
    chunks = [{"text": "Marie Curie was born in Warsaw in 1867. She moved to Paris.", "score": 0.7},
              {"text": "Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1911.", "score": 0.6}]
    rec = {"answer": "Marie Curie was born in Warsaw. She won 5 Nobel Prizes. She invented the telephone.",
           "retrieved": chunks, "refused": False,
           "attribution": {"sentences": [{"text": "Marie Curie was born in Warsaw.", "cites": []},
                                         {"text": "She won 5 Nobel Prizes.", "cites": []},
                                         {"text": "She invented the telephone.", "cites": []}],
                           "scores": {"sim_sent": [[0.95, 0.4], [0.4, 0.7], [0.2, 0.2]],
                                      "nli_set": [[0.99, 0.0], [0.0, 0.1], [0.0, 0.0]],
                                      "nli_window_contra": [[0.0, 0.0], [0.0, 0.9], [0.0, 0.0]]}}}
    e = explain(rec)
    codes = [r["code"] for r in e["reasons"]]
    assert [s["status"] for s in e["sentences"]] == ["supported", "contradicted", "unsupported"]
    assert "Warsaw" in e["sentences"][0]["quote"]
    assert "CONTRADICTION" in codes and "UNSUPPORTED_SENTENCE" in codes and "NUMBER_NOT_IN_SOURCE" in codes
    e2 = explain(rec, gate_threshold=0.9, confidence=0.3)
    assert e2["decision"] == "refuse" and e2["reasons"][0]["code"] == "LOW_CONFIDENCE"
