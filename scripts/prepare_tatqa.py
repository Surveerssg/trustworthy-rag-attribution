"""TAT-QA (second financial dataset): processed data, chunks, gold chunks and the precomputed retrieval plan, without ChromaDB.

python -m scripts.prepare_tatqa              # data + chunks + gold chunk ids, then the plan (needs sentence-transformers)
python -m scripts.prepare_tatqa --skip-plan  # data + chunks + gold chunk ids only (no model needed)
Needs data/raw/tatqa/tatqa_dataset_dev.json (https://github.com/NExTplusplus/TAT-QA, dataset_raw/tatqa_dataset_dev.json).

Writes data/processed/tatqa/{corpus,questions,chunks_w180}.jsonl and data/plans/tatqa.jsonl with the same three conditions and
the same rules as scripts.export_plan (src.pipeline.index.Retriever + src.pipeline.retrieval_plus.PlusRetriever):
  main     dense top-k (MiniLM cosine; chunk title prepended for embedding only, empty here)
  oracle   up to 2 gold chunks + best non-gold dense hits from the dense top-100, shuffled with random.Random(question)
  removed  dense top-100 with every chunk of the gold document deleted, first k kept
Search is exact (numpy) instead of ChromaDB's HNSW: identical up to HNSW's approximation on a corpus this small.
The plan is built ONCE and shipped in the Kaggle bundle, so every model sees byte-identical contexts.
"""
import argparse
import random
from collections import defaultdict

import numpy as np

from src.common import DATA, INDICES, PROC, read_jsonl, write_jsonl
from src.pipeline.chunking import chunk_document, gold_chunk_ids
from src.pipeline.data import prepare_tatqa

D, CHUNK_WORDS, EMBED = "tatqa", 180, "sentence-transformers/all-MiniLM-L6-v2"


def build_chunks(domain=D):
    corpus = read_jsonl(PROC / domain / "corpus.jsonl")
    chunks = [c for doc in corpus for c in chunk_document(doc, CHUNK_WORDS)]
    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_id"]].append(c)
    qs = read_jsonl(PROC / domain / "questions.jsonl")
    for q in qs:
        q["gold_chunk_ids"] = gold_chunk_ids(q, by_doc)
    write_jsonl(PROC / domain / "questions.jsonl", qs)
    write_jsonl(PROC / domain / f"chunks_w{CHUNK_WORDS}.jsonl", chunks)
    n_gold = [len(q["gold_chunk_ids"]) for q in qs]
    print(f"[{domain}] docs={len(corpus)} chunks={len(chunks)} avg_chunk_words={np.mean([len(c['text'].split()) for c in chunks]):.0f} "
          f"questions={len(qs)} with_gold_chunk={sum(n > 0 for n in n_gold)} gold_chunks_per_q={np.mean(n_gold):.2f}")
    return chunks, qs


def hit(c, score):
    return {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "title": c["title"], "text": c["text"], "score": float(score)}


def build_plan(chunks, qs, k=5, domain=D, note="low by design: questions do not name the company"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED, device="cpu")
    enc = lambda t: model.encode(t, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    E = enc([f"{c['title']}. {c['text']}" if c["title"] else c["text"] for c in chunks]).astype(np.float32)
    Q = enc([q["question"] for q in qs]).astype(np.float32)
    INDICES.mkdir(parents=True, exist_ok=True)
    np.save(INDICES / f"emb_{EMBED.replace('/', '_')}_{domain}.npy", E)
    idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    rows, hits_main = [], []
    for q, qv in zip(qs, Q):
        s = E @ qv
        dh = [hit(chunks[i], s[i]) for i in np.argsort(-s)[:100]]
        g = q["gold_chunk_ids"]
        gh = [hit(chunks[idx[c]], s[idx[c]]) for c in g[:2] if c in idx]
        orc = gh + [h for h in dh if h["chunk_id"] not in set(g)][:k - len(gh)]
        random.Random(q["question"]).shuffle(orc)   # gold position must not leak (same rule as PlusRetriever 'oracle')
        gd = set(q["gold_doc_ids"])
        rows.append({"qid": q["qid"], "main": dh[:k], "oracle": orc, "removed": [h for h in dh if h["doc_id"] not in gd][:k]})
        hits_main.append(any(h["chunk_id"] in set(g) for h in dh[:k]))
    write_jsonl(DATA / "plans" / f"{domain}.jsonl", rows)
    gold_in_oracle = np.mean([any(h["chunk_id"] in set(q["gold_chunk_ids"]) for h in r["oracle"]) for q, r in zip(qs, rows)])
    leak = np.mean([any(h["doc_id"] in set(q["gold_doc_ids"]) for h in r["removed"]) for q, r in zip(qs, rows)])
    print(f"[{domain}] plan: {len(rows)} questions | main Hit@{k} {np.mean(hits_main):.3f}" + (f" ({note})" if note else "") +
          f" | gold in oracle context {gold_in_oracle:.3f} | gold document left in removed context {leak:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-plan", action="store_true")
    a = ap.parse_args()
    prepare_tatqa()
    chunks, qs = build_chunks()
    if not a.skip_plan:
        build_plan(chunks, qs)
