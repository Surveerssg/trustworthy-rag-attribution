"""Supervised hallucination checker trained on RAGTruth (the route to the published 68.2 F1).

python -m scripts.train_checker --model answerdotai/ModernBERT-base --max-len 2048 --epochs 3
Input  : "Question: ... Context: ... Answer: ..."   Label: response contains a hallucination (RAGTruth labels).
Train  : RAGTruth QA train split minus a 10% development split (threshold + model selection).
Test   : official RAGTruth QA test split (900 responses), evaluated ONCE with the dev-tuned threshold.
Writes : experiments/checker/<name>/ (model, test predictions) and experiments/results/ragtruth/checker_<name>.csv
Reported separately from zero-shot results, because this model is trained on the benchmark's own training data.
"""
import argparse
import json
import random
import re

import numpy as np
import pandas as pd
import torch

from src.common import RAW, RESULTS, ROOT
from src.evaluation import metrics as M


def load(split, tasks=("QA",)):
    src = {}
    for l in open(RAW / "ragtruth" / "source_info.jsonl", encoding="utf8"):
        s = json.loads(l)
        if s["task_type"] in tasks:
            src[s["source_id"]] = s
    rows = []
    for l in open(RAW / "ragtruth" / "response.jsonl", encoding="utf8"):
        r = json.loads(l)
        if r["source_id"] in src and r["split"] == split:
            s = src[r["source_id"]]
            info = s["source_info"]
            if isinstance(info, dict):
                q, ctx = info.get("question", ""), info.get("passages", "")
            else:
                q, ctx = "", str(info)
            rows.append({"id": r["id"], "task": s["task_type"], "question": q, "context": ctx,
                         "answer": r["response"], "label": int(bool(r["labels"]))})
    return rows


def fmt(r):
    ctx = re.sub(r"\s+", " ", r["context"])   # outside the f-string: a backslash inside {} is a SyntaxError before Python 3.12
    return f"Question: {r['question']}\nContext: {ctx}\nAnswer: {r['answer']}"


class DS(torch.utils.data.Dataset):
    def __init__(self, rows, tok, max_len):
        self.enc = tok([fmt(r) for r in rows], truncation=True, max_length=max_len)
        self.y = [r["label"] for r in rows]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {**{k: v[i] for k, v in self.enc.items()}, "labels": self.y[i]}


def predict(model, tok, rows, max_len, bs=8):
    model.eval()
    out = []
    dev = next(model.parameters()).device
    for i in range(0, len(rows), bs):
        enc = tok([fmt(r) for r in rows[i:i + bs]], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            out.append(torch.softmax(model(**enc).logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-acc", type=int, default=4)
    a = ap.parse_args()
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                              TrainingArguments)
    name = a.model.split("/")[-1]
    out = ROOT / "experiments" / "checker" / name
    tr_all, te = load("train"), load("test")
    random.Random(0).shuffle(tr_all)
    ndev = len(tr_all) // 10
    dev, tr = tr_all[:ndev], tr_all[ndev:]
    print(f"train {len(tr)} (pos {sum(r['label'] for r in tr)}), dev {len(dev)}, test {len(te)} (pos {sum(r['label'] for r in te)})")
    tok = AutoTokenizer.from_pretrained(a.model)
    # reference_compile=False: ModernBERT otherwise torch.compiles its layers, a needless failure risk on a Kaggle T4
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2, reference_compile=False)
    args = TrainingArguments(output_dir=str(out / "ckpt"), num_train_epochs=a.epochs, learning_rate=a.lr,
                             per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.grad_acc,
                             warmup_ratio=0.1, weight_decay=0.01, fp16=torch.cuda.is_available(), logging_steps=50,
                             save_strategy="no", report_to=[], seed=0, dataloader_num_workers=2)
    Trainer(model=model, args=args, train_dataset=DS(tr, tok, a.max_len),
            data_collator=DataCollatorWithPadding(tok)).train()
    model.save_pretrained(out / "model"); tok.save_pretrained(out / "model")

    pdev, pte = predict(model, tok, dev, a.max_len), predict(model, tok, te, a.max_len)
    ydev, yte = np.array([r["label"] for r in dev]), np.array([r["label"] for r in te])
    ts = np.unique(np.quantile(pdev, np.linspace(0, 1, 201)))
    t = max(ts, key=lambda t: M.prf(pdev >= t, ydev)["f1"])
    s = M.prf(pte >= t, yte)
    rng = np.random.default_rng(0)
    f1s = [M.prf(pte[i] >= t, yte[i])["f1"] for i in (rng.integers(0, len(yte), len(yte)) for _ in range(1000))]
    res = {"method": f"ours: {name} fine-tuned on RAGTruth QA train (supervised)", "precision": 100 * s["precision"],
           "recall": 100 * s["recall"], "f1": 100 * s["f1"],
           "f1_ci": f"[{100 * np.quantile(f1s, .025):.1f},{100 * np.quantile(f1s, .975):.1f}]",
           "auroc": M.auroc(pte, yte), "threshold": float(t), "source": "this work (supervised)"}
    pd.DataFrame([res]).to_csv(RESULTS / "ragtruth" / f"checker_{name}.csv", index=False)
    pd.DataFrame({"id": [r["id"] for r in te], "label": yte, "p": pte}).to_csv(out / "test_predictions.csv", index=False)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
