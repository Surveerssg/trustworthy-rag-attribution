"""Precompute retrieval for every question and condition (main / oracle / removed) -> data/plans/<domain>.jsonl.
Generation on any machine (laptop, Colab) then uses byte-identical contexts, and needs no index.
python -m scripts.export_plan [--config configs/main_qwen1.5b.yaml]"""
import argparse
from src.common import DATA, load_config, write_jsonl
from scripts.run_generation import select_questions
from src.pipeline.index import Retriever
from src.pipeline.retrieval_plus import PlusRetriever

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/main_qwen1.5b.yaml")
    cfg = load_config(ap.parse_args().config); k = cfg.get("k", 5)
    for d in cfg["domains"]:
        qs = select_questions(d, cfg); Q = [q["question"] for q in qs]
        main = Retriever(d).retrieve_many(Q, k=k)
        orc = PlusRetriever(d, "oracle").retrieve_many(Q, k=k, gold=[q.get("gold_chunk_ids", []) for q in qs])
        rem = PlusRetriever(d, "removed").retrieve_many(Q, k=k, gold=[q.get("gold_chunk_ids", []) for q in qs],
                                                        gold_docs=[q.get("gold_doc_ids", []) for q in qs])
        write_jsonl(DATA / "plans" / f"{d}.jsonl", [{"qid": q["qid"], "main": a, "oracle": b, "removed": c}
                                                    for q, a, b, c in zip(qs, main, orc, rem)])
        print(d, len(qs))
