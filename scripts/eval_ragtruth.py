"""External benchmark: response-level hallucination detection on RAGTruth (QA task, official test split).

python -m scripts.eval_ragtruth [--n-train 600]
Detector score for a response = 1 - min over its sentences of the best support by a method (a response is
hallucinated if ANY sentence is unsupported). Threshold is tuned for F1 on a sample of the TRAIN split only,
then applied once to the full QA test split (900 responses). Published baselines (RAGTruth, Table 5, QA F1):
GPT-4-turbo prompt 45.6, GPT-3.5 prompt 30.8, SelfCheckGPT 43.7, LMvLM 30.1, fine-tuned Llama-2-13B 68.2.
"""
import argparse
import json
import random
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.attribution.methods import attribute
from src.common import RAW, RESULTS
from src.evaluation import metrics as M

METHODS = ["sim_sent", "nli_chunk", "nli_window", "nli_set"]
PUBLISHED = {"Prompt gpt-3.5-turbo": (18.8, 84.4, 30.8), "Prompt gpt-4-turbo": (33.2, 90.6, 45.6),
             "SelfCheckGPT gpt-3.5-turbo": (35.0, 58.0, 43.7), "LMvLM gpt-4-turbo": (18.7, 76.9, 30.1),
             "Finetuned Llama-2-13B (supervised)": (61.6, 76.3, 68.2)}


def load(split):
    src = {}
    for l in open(RAW / "ragtruth" / "source_info.jsonl", encoding="utf8"):
        s = json.loads(l)
        if s["task_type"] == "QA":
            src[s["source_id"]] = s["source_info"]
    out = []
    for l in open(RAW / "ragtruth" / "response.jsonl", encoding="utf8"):
        r = json.loads(l)
        if r["source_id"] in src and r["split"] == split:
            si = src[r["source_id"]]
            passages = [p.strip() for p in re.split(r"passage \d+:", si["passages"]) if p.strip()]
            out.append({"id": r["id"], "model": r["model"], "response": r["response"], "question": si["question"],
                        "chunks": [{"text": p} for p in passages], "label": int(bool(r["labels"]))})
    return out


def scores(items):
    rows = []
    for it in tqdm(items):
        a = attribute(it["response"], it["chunks"], methods=("sim_sent", "nli_chunk", "nli_window", "nli_set"))
        row = {"id": it["id"], "model": it["model"], "label": it["label"]}
        for m in METHODS:
            S = a["scores"].get(m)
            row[m] = 1 - float(np.asarray(S).max(1).min()) if S else 0.0  # hallucination score
        rows.append(row)
    return pd.DataFrame(rows)


def best_threshold(x, y):
    ts = np.unique(np.quantile(x, np.linspace(0, 1, 201)))
    return max(ts, key=lambda t: M.prf(x >= t, y)["f1"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=600)
    a = ap.parse_args()
    out = RESULTS / "ragtruth"
    out.mkdir(parents=True, exist_ok=True)
    tr = load("train")
    random.Random(0).shuffle(tr)
    te = load("test")
    print(f"train sample {a.n_train} of {len(tr)}, test {len(te)} (hallucinated {sum(x['label'] for x in te)})")
    dtr, dte = scores(tr[:a.n_train]), scores(te)
    from src.attribution.methods import NLI_MODEL as _M
    _t = "" if _M.endswith("nli-deberta-v3-base") else "_" + _M.split("/")[-1]
    dtr.to_csv(out / f"train_scores{_t}.csv", index=False)
    dte.to_csv(out / f"test_scores{_t}.csv", index=False)
    res = [{"method": k, "precision": p, "recall": r, "f1": f, "source": "published (RAGTruth Tab.5)"} for k, (p, r, f) in PUBLISHED.items()]
    for m in METHODS:
        t = best_threshold(dtr[m].values, dtr.label.values)
        s = M.prf(dte[m].values >= t, dte.label.values)
        auc = M.auroc(dte[m].values, dte.label.values)
        # bootstrap CI on test F1
        rng = np.random.default_rng(0)
        f1s = []
        for _ in range(1000):
            i = rng.integers(0, len(dte), len(dte))
            f1s.append(M.prf(dte[m].values[i] >= t, dte.label.values[i])["f1"])
        res.append({"method": f"ours: {m} (zero-shot, train-tuned t={t:.3f})", "precision": 100 * s["precision"],
                    "recall": 100 * s["recall"], "f1": 100 * s["f1"], "f1_ci": f"[{100*np.quantile(f1s,.025):.1f},{100*np.quantile(f1s,.975):.1f}]",
                    "auroc": auc, "source": "this work"})
    df = pd.DataFrame(res)
    from src.attribution.methods import NLI_MODEL
    tag = "" if NLI_MODEL.endswith("nli-deberta-v3-base") else "_" + NLI_MODEL.split("/")[-1]
    df.to_csv(out / f"ragtruth_qa_response_level{tag}.csv", index=False)
    print(df.round(3).to_string(index=False))
