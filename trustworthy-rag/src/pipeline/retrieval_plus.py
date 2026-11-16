"""Retrieval variants for the 'can we fix the retrieval bottleneck?' experiment.

dense     : MiniLM + Chroma (baseline, used everywhere else)
bm25      : Okapi BM25 over the same chunks
hybrid    : reciprocal-rank fusion (RRF, k=60) of dense top-100 and BM25 top-100
rerank    : hybrid top-50 re-scored by cross-encoder/ms-marco-MiniLM-L-6-v2
scoped    : entity-scoped retrieval -- restrict candidates to the document(s) whose title best matches the
            question (company+year for FinanceBench, contract name for CUAD), then dense + rerank inside it.
oracle    : gold chunks guaranteed in the top-k (rest filled by best non-gold dense hits, deterministic shuffle).
entity    : entity-scoped hybrid (ours, improved arm): legal -> restrict to the contract named in the question,
            then dense + BM25 fused by RRF; financial -> restrict to the filings of the company (and fiscal year /
            filing type) named in the question, dense + BM25 RRF, boost pages holding the financial statement the
            question names, then cross-encoder rerank inside the filing. Finance chunk Hit@5 0.188 -> 0.309.
removed   : evidence-removal counterfactual -- the gold evidence is deleted from the index (whole gold documents;
            for CUAD only the gold clause chunks, so the contract remains but the clause is absent). The same
            answerable question should now be REFUSED. Gives realistic, wording-controlled unanswerables in any domain.
"""
import random
import re
from collections import defaultdict
from functools import lru_cache

import numpy as np

from src.common import PROC, read_jsonl
from src.pipeline.index import Retriever, device, embed

_TOK = re.compile(r"[a-z0-9]+")


def toks(s):
    return _TOK.findall(s.lower())


@lru_cache(maxsize=8)
def _chunks(domain, chunk_words=180):
    ch = read_jsonl(PROC / domain / f"chunks_w{chunk_words}.jsonl")
    return ch, {c["chunk_id"]: i for i, c in enumerate(ch)}


@lru_cache(maxsize=8)
def _bm25(domain):
    from rank_bm25 import BM25Okapi
    ch, _ = _chunks(domain)
    return BM25Okapi([toks(f"{c['title']} {c['text']}") for c in ch])


@lru_cache(maxsize=4)
def _chunk_matrix(domain, chunk_words=180):
    """All stored chunk embeddings of a domain (from ChromaDB), row-aligned with _chunks(domain)."""
    from src.pipeline.index import _client, collection_name
    ch, idx = _chunks(domain, chunk_words)
    col = _client().get_collection(collection_name(domain, chunk_words))
    M = np.zeros((len(ch), 384), dtype=np.float32)
    ids = [c["chunk_id"] for c in ch]
    for b in range(0, len(ids), 2000):
        got = col.get(ids=ids[b:b + 2000], include=["embeddings"])
        for cid, e in zip(got["ids"], got["embeddings"]):
            M[idx[cid]] = e
    return M


FIN_STATEMENTS = {
    "cash flow": ["cash flows", "cash flow statement", "statements of cash flows"],
    "balance sheet": ["balance sheet", "statement of financial position", "total assets", "total liabilities"],
    "income statement": ["income statement", "statements of income", "statement of operations", "statements of operations", "net income"],
    "p&l": ["statements of income", "statement of operations", "gross profit"],
}


@lru_cache(maxsize=1)
def _fb_meta():
    import json
    from src.common import RAW
    return {d["doc_name"]: d for d in map(json.loads, open(RAW / "financebench_document_information.jsonl", encoding="utf8"))}


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device())


def _as_hit(c, score):
    return {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "title": c["title"], "text": c["text"], "score": float(score)}


class PlusRetriever:
    def __init__(self, domain, mode="hybrid"):
        self.domain, self.mode = domain, mode
        self.dense = Retriever(domain)
        self.ch, self.idx = _chunks(domain)

    def _bm25_top(self, q, n):
        s = _bm25(self.domain).get_scores(toks(q))
        top = np.argsort(-s)[:n]
        return [(int(i), float(s[i])) for i in top]

    def _rrf(self, q, dense_hits, n=100, K=60):
        sc = defaultdict(float)
        for r, h in enumerate(dense_hits):
            sc[self.idx[h["chunk_id"]]] += 1 / (K + r + 1)
        for r, (i, _) in enumerate(self._bm25_top(q, n)):
            sc[i] += 1 / (K + r + 1)
        return sorted(sc, key=sc.get, reverse=True)

    def _rerank(self, q, cand_idx, k):
        pairs = [(q, self.ch[i]["text"]) for i in cand_idx]
        s = _reranker().predict(pairs, batch_size=32, show_progress_bar=False)
        order = np.argsort(-s)[:k]
        # keep dense cosine as the reported 'score' so confidence features stay comparable across modes
        qv, E = embed([q])[0], _chunk_matrix(self.domain)
        return [_as_hit(self.ch[cand_idx[j]], float(E[cand_idx[j]] @ qv)) | {"rerank": float(s[j])} for j in order]

    def _scope_docs(self, q, top_docs=2):
        """Pick the source document(s) whose title tokens best overlap the question (company/contract/year)."""
        qt = set(toks(q))
        by_doc = defaultdict(list)
        for i, c in enumerate(self.ch):
            by_doc[c["title"]].append(i)
        scored = sorted(by_doc, key=lambda t: -len(qt & set(toks(t))) / (1 + 0.1 * len(set(toks(t)))))
        return [i for t in scored[:top_docs] for i in by_doc[t]]

    def _entity_scope(self, q):
        """Indices of chunks belonging to the document(s) the question names; None = no confident match."""
        ql = q.lower()
        if self.domain == "cuad":
            idx = [i for i, c in enumerate(self.ch) if c["title"] and c["title"].lower() in ql]
            return idx or None
        if self.domain == "financebench":
            meta = _fb_meta()
            docs = {c["doc_id"].rsplit("_p", 1)[0][3:] for c in self.ch}
            ql = re.sub(r"[-']", " ", ql).replace("amex", "american express")
            comp = [dn for dn in docs if dn in meta and re.search(r"\b" + re.escape(re.sub(r"[-']", " ", meta[dn]["company"].lower())) + r"\b", ql)]
            if not comp:
                return None
            years = {int(y) for y in re.findall(r"(?:fy\s?|fiscal\s|\b)(20\d\d)\b", ql)}
            quarter = bool(re.search(r"\bq[1-4]\b", ql))
            keep = [dn for dn in comp if (not years or meta[dn]["doc_period"] in years)]
            if quarter:
                keep = [dn for dn in keep if meta[dn]["doc_type"] == "10q"] or keep
            elif keep:
                keep = [dn for dn in keep if meta[dn]["doc_type"] != "10q"] or keep
            keep = set(keep or comp)
            return [i for i, c in enumerate(self.ch) if c["doc_id"].rsplit("_p", 1)[0][3:] in keep]
        return None

    def _entity(self, q, k, K=60):
        scope = self._entity_scope(q)
        qv, E = embed([q])[0], _chunk_matrix(self.domain)
        if scope is None:
            scope = list(range(len(self.ch)))
        scope = np.asarray(scope)
        dense = scope[np.argsort(-(E[scope] @ qv))[:100]]
        bm = _bm25(self.domain).get_scores(toks(q))[scope]
        sparse = scope[np.argsort(-bm)[:100]]
        sc = defaultdict(float)
        for r, i in enumerate(dense):
            sc[int(i)] += 1 / (K + r + 1)
        for r, i in enumerate(sparse):
            sc[int(i)] += 1 / (K + r + 1)
        if self.domain == "financebench":
            # statement boost: pages containing the financial statement the question names
            ql = q.lower()
            keys = [kw for trig, kws in FIN_STATEMENTS.items() if trig in ql for kw in kws]
            if keys:
                for i in sc:
                    if any(kw in self.ch[i]["text"].lower() for kw in keys):
                        sc[i] += 1 / (K + 1)
            # rerank within the scoped filing (helps here, unlike whole-corpus legal search)
            cand = sorted(sc, key=sc.get, reverse=True)[:50]
            return self._rerank(q, cand, k)
        top = sorted(sc, key=sc.get, reverse=True)[:k]
        return [_as_hit(self.ch[i], float(E[i] @ qv)) for i in top]

    def retrieve_many(self, queries, k=5, gold=None, gold_docs=None):
        dense = self.dense.retrieve_many(queries, k=100 if self.mode != "dense" else k)
        out = []
        for qi, (q, dh) in enumerate(zip(queries, dense)):
            if self.mode == "dense":
                out.append(dh[:k])
            elif self.mode == "bm25":
                out.append([_as_hit(self.ch[i], s) for i, s in self._bm25_top(q, k)])
            elif self.mode == "hybrid":
                fused = self._rrf(q, dh)[:k]
                qv, E = embed([q])[0], _chunk_matrix(self.domain)
                out.append([_as_hit(self.ch[i], float(E[i] @ qv)) for i in fused])
            elif self.mode == "rerank":
                out.append(self._rerank(q, self._rrf(q, dh)[:50], k))
            elif self.mode == "scoped":
                cand = self._scope_docs(q)
                s = _chunk_matrix(self.domain)[cand] @ embed([q])[0]
                top = [cand[j] for j in np.argsort(-s)[:50]]
                out.append(self._rerank(q, top, k))
            elif self.mode == "entity":
                out.append(self._entity(q, k) if self.domain in ("cuad", "financebench") else dh[:k])
            elif self.mode == "removed":
                if self.domain == "cuad":
                    banned = lambda h: h["chunk_id"] in set(gold[qi] or [])
                else:
                    gd = set(gold_docs[qi] or [])
                    banned = lambda h: h["doc_id"] in gd
                out.append([h for h in dh if not banned(h)][:k])
            elif self.mode == "oracle":
                g = [c for c in (gold[qi] or [])]
                qv, E = embed([q])[0], _chunk_matrix(self.domain)
                gh = [_as_hit(self.ch[self.idx[c]], float(E[self.idx[c]] @ qv)) for c in g[:2] if c in self.idx]
                rest = [h for h in dh if h["chunk_id"] not in set(g)][:k - len(gh)]
                hits = gh + rest
                random.Random(q).shuffle(hits)  # gold position must not leak
                out.append(hits)
        return out
