"""Recursive sentence-packing chunker with character offsets (needed for span-level gold matching).

Assumption: chunk_words=180 because all-MiniLM-L6-v2 truncates at 256 word-pieces; larger chunks
would be silently cut at embedding time. Same config for every domain (controlled design).
"""
import re

_BOUND = re.compile(r"(?<=[.!?;])\s+|\n\s*\n+")


def sentence_spans(text):
    spans, start = [], 0
    for m in _BOUND.finditer(text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _hard_split(text, s, e, max_words):
    """Split an over-long 'sentence' (tables, run-on legal clauses) at word boundaries."""
    out, words = [], list(re.finditer(r"\S+", text[s:e]))
    for i in range(0, len(words), max_words):
        grp = words[i:i + max_words]
        out.append((s + grp[0].start(), s + grp[-1].end()))
    return out


def chunk_document(doc, chunk_words=180, overlap_sents=1):
    text = doc["text"]
    units = []
    for s, e in sentence_spans(text):
        n = len(text[s:e].split())
        units.extend(_hard_split(text, s, e, chunk_words) if n > chunk_words else [(s, e)])
    chunks, cur, cur_words = [], [], 0
    for (s, e) in units:
        n = len(text[s:e].split())
        if cur and cur_words + n > chunk_words:
            chunks.append((cur[0][0], cur[-1][1]))
            cur = cur[-overlap_sents:] if overlap_sents else []
            cur_words = sum(len(text[a:b].split()) for a, b in cur)
            if cur_words + n > chunk_words:  # overlap would overflow; drop it
                cur, cur_words = [], 0
        cur.append((s, e))
        cur_words += n
    if cur:
        chunks.append((cur[0][0], cur[-1][1]))
    out = []
    for i, (s, e) in enumerate(chunks):
        out.append({"chunk_id": f"{doc['doc_id']}::{i}", "doc_id": doc["doc_id"], "title": doc.get("title", ""),
                    "text": text[s:e], "start": s, "end": e})
    return out


_TOK = re.compile(r"[a-z0-9]+")


def _toks(s):
    return _TOK.findall(s.lower())


def gold_chunk_ids(question, chunks_by_doc, min_token_recall=0.5):
    """Which chunks contain the gold evidence of a question.
    span evidence -> char overlap; text evidence -> normalised substring or token recall >= threshold."""
    gold = set()
    for ev in question["gold_evidence"]:
        cands = chunks_by_doc.get(ev["doc_id"], [])
        if "start" in ev:
            for c in cands:
                if c["start"] < ev["end"] and ev["start"] < c["end"]:
                    gold.add(c["chunk_id"])
            continue
        et = _toks(ev["text"])
        if not et:
            continue
        et_str, best, best_r = " ".join(et), None, 0.0
        for c in cands:
            ct = _toks(c["text"])
            if et_str in " ".join(ct):
                gold.add(c["chunk_id"])
                best = None
                break
            cs = set(ct)
            r = sum(t in cs for t in et) / len(et)
            if r > best_r:
                best, best_r = c["chunk_id"], r
        if best and best_r >= min_token_recall:
            gold.add(best)
    # documents fully given as evidence (PubMedQA abstracts): every chunk of the doc is gold
    if not gold:
        for d in question["gold_doc_ids"]:
            if question["domain"] == "pubmedqa":
                gold.update(c["chunk_id"] for c in chunks_by_doc.get(d, []))
    return sorted(gold)
