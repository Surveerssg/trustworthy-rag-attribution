"""BioASQ preparation: balanced sample, snippet-level gold that sits inside the gold chunks, removal covers every listed document."""
import re
from collections import Counter

import pytest

from src.common import PROC, read_jsonl

pytestmark = pytest.mark.skipif(not (PROC / "bioasq" / "questions.jsonl").exists(), reason="run python -m scripts.prepare_bioasq first")


def test_sample_is_balanced_and_every_question_has_gold():
    qs = read_jsonl(PROC / "bioasq" / "questions.jsonl")
    kinds = Counter((q["meta"]["type"], q["gold_answers"][0] if q["answer_type"] == "label" else "-") for q in qs)
    assert kinds == {("yesno", "yes"): 50, ("yesno", "no"): 50, ("factoid", "-"): 100}
    assert all(q["gold_chunk_ids"] for q in qs)


def test_snippets_are_inside_gold_chunks_and_gold_docs_are_removable():
    qs = read_jsonl(PROC / "bioasq" / "questions.jsonl")
    chunks = {c["chunk_id"]: c for c in read_jsonl(PROC / "bioasq" / "chunks_w180.jsonl")}
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    for q in qs:
        gold_text = norm(" ".join(chunks[c]["text"] for c in q["gold_chunk_ids"]))
        assert any(norm(e["text"])[:60] in gold_text for e in q["gold_evidence"])
        assert {chunks[c]["doc_id"] for c in q["gold_chunk_ids"]} <= set(q["gold_doc_ids"])
