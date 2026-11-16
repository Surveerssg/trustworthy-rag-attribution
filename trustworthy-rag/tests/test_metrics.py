"""Toy/synthetic checks for every metric and the chunker. Run: python -m pytest tests -q"""
import numpy as np

from src.evaluation import metrics as M
from src.pipeline.chunking import chunk_document, gold_chunk_ids
from src.attribution.methods import answer_sentences


def test_retrieval():
    r = M.retrieval_metrics(["a", "b", "c"], ["b", "z"])
    assert r["recall"] == 0.5 and r["hit"] == 1 and abs(r["precision"] - 1 / 3) < 1e-9 and r["mrr"] == 0.5
    assert M.retrieval_metrics(["a"], []) is None


def test_correctness():
    assert M.correctness("It was Paris, France.", ["Paris"], "short")[0] == 1
    assert M.correctness("They are not both rock bands.", ["no"], "short")[0] == 0
    assert M.correctness("No, they are not.", ["no"], "short")[0] == 1
    assert M.correctness("Maybe. The evidence is mixed.", ["maybe"], "label")[0] == 1
    assert M.correctness("No, it doesn't.", ["yes"], "label")[0] == 0
    assert M.correctness("Capex was $1,577 million in FY2018.", ["$1577.00"], "free")[0] == 1
    assert M.correctness("Capex was $1.577 billion.", ["$1577.00"], "free")[0] == 1
    assert M.correctness("Capex was $2,000 million.", ["$1577.00"], "free")[0] == 0
    assert M.correctness("The agreement terminates after five years.",
                         ["This Agreement terminates five years after the Effective Date"], "span")[0] == 1


def test_attribution():
    S = [[0.9, 0.1, 0.2], [0.2, 0.3, 0.1]]
    a = M.attribution_metrics(S, [True, False, False], threshold=0.5)
    assert a["eas"] == 0.5 and a["attr_acc"] == 0.5 and a["attr_p"] == 1.0 and a["attr_r"] == 1.0
    assert "attr_acc" not in M.attribution_metrics(S, [False] * 3, 0.5)


def test_refusal_and_calibration():
    assert M.prf([1, 1, 0, 0], [1, 0, 1, 0])["f1"] == 0.5
    rows, best = M.threshold_sweep([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0])
    assert best["f1"] == 1.0
    assert M.ece([1.0, 0.0], [1, 0]) == 0.0 and M.brier([1, 0], [1, 0]) == 0
    assert M.auroc([0.9, 0.1], [1, 0]) == 1.0
    assert M.aurc([0.9, 0.1], [1, 0]) < M.aurc([0.1, 0.9], [1, 0])


def test_stats():
    a, b = np.ones(50), np.zeros(50)
    assert M.paired_bootstrap(a, b)["p"] < 0.01
    assert M.mcnemar(a, b)["p"] < 0.01


def test_chunker_and_gold():
    text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(100))
    doc = {"doc_id": "d", "title": "T", "text": text}
    ch = chunk_document(doc, chunk_words=40)
    assert all(len(c["text"].split()) <= 40 for c in ch) and len(ch) > 5
    assert all(text[c["start"]:c["end"]] == c["text"] for c in ch)
    s = text.index("Sentence number 50 ")
    q = {"domain": "x", "gold_doc_ids": ["d"],
         "gold_evidence": [{"doc_id": "d", "text": "Sentence number 50 talks about topic 50.", "start": s, "end": s + 20}]}
    g = gold_chunk_ids(q, {"d": ch})
    assert g and all("number 50" in next(c for c in ch if c["chunk_id"] == x)["text"] for x in g)
    q2 = {"domain": "x", "gold_doc_ids": ["d"], "gold_evidence": [{"doc_id": "d", "text": "number 77 talks about topic 77"}]}
    assert gold_chunk_ids(q2, {"d": ch})


def test_answer_sentence_citations():
    s = answer_sentences("Paris is the capital [1]. It has 2 million people. [2, 3]")
    assert [x["cites"] for x in s] == [[1], [2, 3]]
    assert s[0]["text"] == "Paris is the capital ."  or s[0]["text"].startswith("Paris is the capital")


def test_paired_bootstrap_p_is_a_probability_even_with_zero_differences():
    p = M.paired_bootstrap(np.ones(8), np.ones(8))["p"]
    assert 0 < p <= 1.0
