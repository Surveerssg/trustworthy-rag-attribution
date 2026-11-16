"""Embedding model + per-domain ChromaDB collections + dense retriever."""
from collections import defaultdict
from functools import lru_cache

import numpy as np

from src.common import INDICES, PROC, read_jsonl, write_jsonl
from src.pipeline.chunking import chunk_document, gold_chunk_ids

DEFAULT_EMBED = "sentence-transformers/all-MiniLM-L6-v2"


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def get_embedder(name=DEFAULT_EMBED):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device=device())


def embed(texts, name=DEFAULT_EMBED, batch_size=64):
    return get_embedder(name).encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                     show_progress_bar=len(texts) > 2000, convert_to_numpy=True)


def _client():
    import chromadb
    return chromadb.PersistentClient(path=str(INDICES / "chroma"))


def collection_name(domain, chunk_words):
    return f"{domain}_w{chunk_words}"


def build_index(domain, chunk_words=180, embed_model=DEFAULT_EMBED):
    corpus = read_jsonl(PROC / domain / "corpus.jsonl")
    chunks = [c for d in corpus for c in chunk_document(d, chunk_words)]
    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_id"]].append(c)
    # attach gold chunk ids to questions
    qs = read_jsonl(PROC / domain / "questions.jsonl")
    for q in qs:
        q["gold_chunk_ids"] = gold_chunk_ids(q, by_doc)
    write_jsonl(PROC / domain / "questions.jsonl", qs)
    write_jsonl(PROC / domain / f"chunks_w{chunk_words}.jsonl", chunks)

    # title is prepended for embedding only (standard 'contextual header' practice, same for all domains)
    vecs = embed([f"{c['title']}. {c['text']}" if c["title"] else c["text"] for c in chunks], embed_model)
    client = _client()
    name = collection_name(domain, chunk_words)
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name, metadata={"hnsw:space": "cosine", "embed_model": embed_model})
    B = 5000
    for i in range(0, len(chunks), B):
        part = chunks[i:i + B]
        col.add(ids=[c["chunk_id"] for c in part], embeddings=vecs[i:i + B].tolist(),
                documents=[c["text"] for c in part],
                metadatas=[{"doc_id": c["doc_id"], "title": c["title"]} for c in part])
    n_gold = [len(q["gold_chunk_ids"]) for q in qs if q["answerable"]]
    stats = {"domain": domain, "docs": len(corpus), "chunks": len(chunks),
             "avg_chunk_words": float(np.mean([len(c["text"].split()) for c in chunks])),
             "questions": len(qs), "answerable_with_gold_chunk": sum(n > 0 for n in n_gold),
             "answerable": len(n_gold)}
    print(stats)
    return stats


class Retriever:
    def __init__(self, domain, chunk_words=180, embed_model=DEFAULT_EMBED):
        self.col = _client().get_collection(collection_name(domain, chunk_words))
        self.embed_model = embed_model

    def retrieve_many(self, queries, k=5, batch=16):
        qv = embed(queries, self.embed_model)
        res = {"ids": [], "documents": [], "metadatas": [], "distances": []}
        for b in range(0, len(queries), batch):  # chroma/sqlite caps variables per query call
            r = self.col.query(query_embeddings=qv[b:b + batch].tolist(), n_results=k,
                               include=["documents", "metadatas", "distances"])
            for key in res:
                res[key] += r[key]
        out = []
        for i in range(len(queries)):
            out.append([{"chunk_id": cid, "doc_id": m["doc_id"], "title": m["title"], "text": d,
                         "score": float(1 - dist)}
                        for cid, d, m, dist in zip(res["ids"][i], res["documents"][i],
                                                    res["metadatas"][i], res["distances"][i])])
        return out
