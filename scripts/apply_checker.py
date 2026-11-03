"""Score our own answers with the trained hallucination checker -> a new confidence signal.

python -m scripts.apply_checker --run main_llama8b --checker ModernBERT-base [--systems vanilla_rag trust_rag]
Writes experiments/runs/<run>/checker_<name>.jsonl : {domain, system, qid, p_halluc}
"""
import argparse
import glob
import os

import torch

from scripts.train_checker import predict
from src.attribution.methods import strip_citations
from src.common import ROOT, RUNS, append_jsonl, read_jsonl

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checker", default="ModernBERT-base")
    ap.add_argument("--systems", nargs="+", default=None)
    ap.add_argument("--max-len", type=int, default=2048)
    a = ap.parse_args()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = ROOT / "experiments" / "checker" / a.checker / "model"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to("cuda" if torch.cuda.is_available() else "cpu")
    dst = RUNS / a.run / f"checker_{a.checker}.jsonl"
    done = {(r["domain"], r["system"], r["qid"]) for r in read_jsonl(dst)} if dst.exists() else set()
    for f in sorted(glob.glob(str(RUNS / a.run / "*__*.jsonl"))):
        if f.endswith(".scored.jsonl"):
            continue
        d, s = os.path.basename(f)[:-6].split("__")
        if s == "closed_book" or (a.systems and s not in a.systems):
            continue
        rows = [r for r in read_jsonl(f) if (d, s, r["qid"]) not in done and r["retrieved"]]
        if not rows:
            continue
        items = [{"question": r["question"], "context": " ".join(f"passage {i + 1}: {c['text']}" for i, c in enumerate(r["retrieved"])),
                  "answer": strip_citations(r["answer"])} for r in rows]
        p = predict(model, tok, items, a.max_len)
        for r, pi in zip(rows, p):
            append_jsonl(dst, {"domain": d, "system": s, "qid": r["qid"], "p_halluc": float(pi)})
        print(d, s, len(rows))
