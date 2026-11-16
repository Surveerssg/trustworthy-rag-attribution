"""Black-box evidence attribution: for every answer sentence, score every retrieved chunk.

Methods (all model-internals-free):
  sim_chunk   cosine(answer sentence, whole chunk)                    [MiniLM]
  sim_sent    max cosine(answer sentence, any sentence of the chunk)  [MiniLM]
  nli_chunk   P(entail | premise=chunk, hypothesis=answer sentence)   [DeBERTa-v3 NLI]
  nli_window  max P(entail) over 2-sentence windows of the chunk      [ours: for dense/long text]
  nli_set     evidence-set NLI (ours): also tests the sentence against the concatenation of its top-2
              most similar chunks, so multi-hop claims supported only jointly are recognised.
              Attribution credit goes to both chunks of the best-supporting set.
  self_cite   the generator's own [n] citations (trust prompt only)
  loo         leave-one-out perturbation: drop chunk j, regenerate, measure answer change (causal)
"""
import re
import time
from functools import lru_cache

import numpy as np

from src.common import split_sentences
from src.pipeline.chunking import sentence_spans
from src.pipeline.index import device, embed

import os
NLI_MODEL = os.environ.get("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def strip_citations(s):
    return re.sub(r"\s+", " ", _CITE.sub("", s)).strip()


def answer_sentences(answer):
    """Split answer into sentences; keep each sentence's own citations."""
    out = []
    for s in split_sentences(answer):
        cites = sorted({int(x) for m in _CITE.findall(s) for x in m.split(",")})
        clean = strip_citations(s)
        if len(clean) > 2:
            out.append({"text": clean, "cites": cites})
        elif cites and out:  # "... 1990. [2]" -> citation belongs to the preceding sentence
            out[-1]["cites"] = sorted(set(out[-1]["cites"]) | set(cites))
    return out


@lru_cache(maxsize=1)
def _nli():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device()).eval()
    if device() == "cuda":
        model = model.half()
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    return tok, model, labels


def nli_probs(premises, hypotheses, batch_size=16):
    """Returns array [n, 3] of (entail, neutral, contradiction) probabilities.
    Pairs are processed in length-sorted order (less padding) and returned in the original order."""
    import torch
    tok, model, lab = _nli()
    n = len(premises)
    if n == 0:
        return np.zeros((0, 3))
    order = sorted(range(n), key=lambda i: len(premises[i]) + len(hypotheses[i]))
    out = np.zeros((n, 3))
    for i in range(0, n, batch_size):
        idx = order[i:i + batch_size]
        enc = tok([premises[j] for j in idx], [hypotheses[j] for j in idx], truncation="only_first",
                  max_length=512, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits.float(), -1).cpu().numpy()
        ent = p[:, lab["entailment"]]
        con = p[:, lab["contradiction"]] if "contradiction" in lab else np.zeros(len(idx))
        out[idx] = np.stack([ent, 1 - ent - con, con], 1)
    return out


_SENT_EMB = {}


def _chunk_sentence_embs(text):
    """Sentence embeddings of a chunk, cached (the same retrieved chunks recur across systems/questions)."""
    if text not in _SENT_EMB:
        cs = [text[a:b] for a, b in sentence_spans(text)] or [text]
        _SENT_EMB[text] = embed(cs)
    return _SENT_EMB[text]


def _windows(text, size=2):
    sents = [text[s:e] for s, e in sentence_spans(text)] or [text]
    if len(sents) <= size:
        return [" ".join(sents)]
    return [" ".join(sents[i:i + size]) for i in range(len(sents) - size + 1)]


SET_TIE_EPS = 1e-4   # far below any real score difference; only ever decides ties between jointly credited chunks


def break_set_ties(nli_set, nli_window, eps=SET_TIE_EPS):
    """nli_set credits the joint (two-chunk) entailment to BOTH chunks of the best pair, so the later per-sentence argmax
    always picked the lower-indexed one: attribution accuracy depended on where the gold chunk sat in the context
    (paper/findings/stage2a_findings.md section 3; gold in positions 1-2 vs 4-5 differed by 0.15 for nli_set against ~0.04 for the other methods).
    Break the tie with each chunk's OWN single-chunk evidence: the jointly credited chunk that supports the sentence more
    on its own ranks higher. Only entries raised by the joint score are touched, and by at most eps."""
    ns, nw = np.asarray(nli_set, dtype=float), np.asarray(nli_window, dtype=float)
    return np.where(ns > nw + 1e-12, ns + eps * nw, ns)


def refresh_record(rec):
    """Apply break_set_ties to a record scored BEFORE the fix, in memory (files are not touched; idempotent).
    Recomputes attr_metrics['nli_set'] from the corrected scores; features are left alone (they move by < eps)."""
    from src.evaluation import metrics as M
    att = rec.get("attribution") or {}
    sc = att.get("scores") or {}
    if att.get("set_tiebreak") or "nli_set" not in sc or "nli_window" not in sc:
        return rec
    rec = dict(rec)
    att = dict(att)
    sc = dict(sc)
    sc["nli_set"] = break_set_ties(sc["nli_set"], sc["nli_window"]).tolist()
    att["scores"], att["set_tiebreak"] = sc, SET_TIE_EPS
    rec["attribution"] = att
    gold = set(rec.get("gold_chunk_ids") or [])
    mask = [c["chunk_id"] in gold for c in rec["retrieved"]]
    am = dict(rec.get("attr_metrics") or {})
    am["nli_set"] = M.attribution_metrics(sc["nli_set"], mask, 0.5)   # same threshold score_run uses for NLI methods
    rec["attr_metrics"] = am
    return rec


def attribute(answer, chunks, methods=("sim_chunk", "sim_sent", "nli_chunk", "nli_window", "nli_set", "self_cite")):
    """Returns {"sentences": [...], "scores": {method: [n_sent x n_chunk]}, "seconds": {method: s}}."""
    sents = answer_sentences(answer)
    n, k = len(sents), len(chunks)
    res = {"sentences": sents, "scores": {}, "seconds": {}}
    if n == 0 or k == 0:
        return res
    st = [s["text"] for s in sents]
    ct = [c["text"] for c in chunks]

    if "sim_chunk" in methods:
        t0 = time.time()
        res["scores"]["sim_chunk"] = (embed(st) @ embed(ct).T).tolist()
        res["seconds"]["sim_chunk"] = time.time() - t0

    if "sim_sent" in methods:
        t0 = time.time()
        sv, m = embed(st), np.zeros((n, k))
        for j, c in enumerate(ct):
            m[:, j] = (sv @ _chunk_sentence_embs(c).T).max(1)
        res["scores"]["sim_sent"] = m.tolist()
        res["seconds"]["sim_sent"] = time.time() - t0

    if "nli_chunk" in methods:
        t0 = time.time()
        p = nli_probs([c for _ in st for c in ct], [s for s in st for _ in ct])
        res["scores"]["nli_chunk"] = p[:, 0].reshape(n, k).tolist()
        res["scores"]["nli_chunk_contra"] = p[:, 2].reshape(n, k).tolist()
        res["seconds"]["nli_chunk"] = time.time() - t0

    if "nli_window" in methods:
        t0 = time.time()
        prem, hyp, owner = [], [], []
        for i, s in enumerate(st):
            for j, c in enumerate(ct):
                for w in _windows(c):
                    prem.append(w); hyp.append(s); owner.append((i, j))
        p = nli_probs(prem, hyp, batch_size=32)
        ent, con = np.zeros((n, k)), np.zeros((n, k))
        for (i, j), row in zip(owner, p):
            ent[i, j] = max(ent[i, j], row[0])
            con[i, j] = max(con[i, j], row[2])
        res["scores"]["nli_window"] = ent.tolist()
        res["scores"]["nli_window_contra"] = con.tolist()
        res["seconds"]["nli_window"] = time.time() - t0

    if "nli_set" in methods and "nli_window" in methods and k >= 2:
        t0 = time.time()
        sim = np.asarray(res["scores"]["sim_sent"]) if "sim_sent" in res["scores"] else (embed(st) @ embed(ct).T)
        ent = np.asarray(res["scores"]["nli_window"]).copy()
        pairs = [tuple(np.argsort(-sim[i])[:2]) for i in range(n)]
        p = nli_probs([ct[a] + " " + ct[b] for a, b in pairs], st)
        for i, (a, b) in enumerate(pairs):
            # joint support: both chunks of the set receive the joint entailment if it beats single-chunk support
            ent[i, a] = max(ent[i, a], p[i, 0])
            ent[i, b] = max(ent[i, b], p[i, 0])
        res["scores"]["nli_set"] = break_set_ties(ent, res["scores"]["nli_window"]).tolist()
        res["set_tiebreak"] = SET_TIE_EPS
        res["seconds"]["nli_set"] = time.time() - t0 + res["seconds"].get("nli_window", 0)

    if "self_cite" in methods:
        m = np.zeros((n, k))
        for i, s in enumerate(sents):
            for c in s["cites"]:
                if 1 <= c <= k:
                    m[i, c - 1] = 1.0
        res["scores"]["self_cite"] = m.tolist()
        res["seconds"]["self_cite"] = 0.0
    return res


def leave_one_out(llm, prompt_fn, question, chunks, answer_type, full_answer):
    """Causal attribution: influence_j = 1 - NLI-bidirectional-agreement(full answer, answer without chunk j)."""
    t0 = time.time()
    infl, alts = [], []
    for j in range(len(chunks)):
        sub = chunks[:j] + chunks[j + 1:]
        alt = llm.generate(prompt_fn(question, sub, answer_type))["text"]
        alts.append(alt)
    a, b = strip_citations(full_answer), [strip_citations(x) for x in alts]
    if not b:
        return {"influence": [], "alts": [], "seconds": 0.0}
    p1 = nli_probs([a] * len(b), b)[:, 0]
    p2 = nli_probs(b, [a] * len(b))[:, 0]
    infl = (1 - np.minimum(p1, p2)).tolist()   # answer changed meaningfully => chunk was influential
    return {"influence": infl, "alts": alts, "seconds": time.time() - t0}
