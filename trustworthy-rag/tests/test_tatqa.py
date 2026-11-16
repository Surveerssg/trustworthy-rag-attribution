"""TAT-QA preparation: gold answer strings, and gold evidence that maps to the table / paragraphs it names."""
import pytest

from src.common import PROC, read_jsonl
from src.evaluation.metrics import correctness
from src.pipeline.data import _tatqa_answer


def test_answer_strings_carry_the_unit():
    assert _tatqa_answer(-22.22, "percent") == "-22.22%"
    assert _tatqa_answer(["$1,496.5"], "million") == "$1,496.5 million"
    assert _tatqa_answer(["fixed-price type", "cost-plus type"], "") == "fixed-price type; cost-plus type"
    assert _tatqa_answer(["4.5%"], "percent") == "4.5%"


def test_numeric_gold_is_matched_with_unit_shift():
    assert correctness("Total sales were $1.4965 billion.", [_tatqa_answer(["$1,496.5"], "million")], "free")[0] == 1.0
    assert correctness("The change was -22.2%.", [_tatqa_answer(-22.22, "percent")], "free")[0] == 1.0


@pytest.mark.skipif(not (PROC / "tatqa" / "questions.jsonl").exists(), reason="run python -m scripts.prepare_tatqa first")
def test_gold_chunks_cover_the_labelled_evidence():
    qs = read_jsonl(PROC / "tatqa" / "questions.jsonl")
    chunks = {c["chunk_id"]: c for c in read_jsonl(PROC / "tatqa" / "chunks_w180.jsonl")}
    assert len(qs) == 200 and all(q["gold_chunk_ids"] for q in qs)
    for q in qs:
        gold_text = " ".join(chunks[c]["text"] for c in q["gold_chunk_ids"])
        if q["meta"]["answer_from"] == "table":   # the table's first row must sit in a gold chunk
            assert q["gold_evidence"][0]["text"].split("\n")[0][:40] in gold_text
        assert all(chunks[c]["doc_id"] == q["gold_doc_ids"][0] for c in q["gold_chunk_ids"])
