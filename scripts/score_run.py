"""Stage 2 (cheap, no LLM except LOO): retrieval metrics, correctness, attribution, confidence features.

python -m scripts.score_run --run main [--loo 40 --loo-system vanilla_rag]
Writes <domain>__<system>.scored.jsonl next to the generation outputs.
"""
import argparse
import glob
import os

import numpy as np
from tqdm import tqdm

from src.attribution.methods import attribute, leave_one_out, nli_probs, strip_citations
from src.common import RUNS, load_config, read_jsonl, write_jsonl
from src.evaluation import metrics as M
from src.pipeline import prompts as P
from src.pipeline.llm import from_config

SIM_T, NLI_T = 0.5, 0.5
METHODS = ("sim_chunk", "sim_sent", "nli_chunk", "nli_window", "nli_set", "self_cite")


def features(rec, attr):
    s = sorted((c["score"] for c in rec["retrieved"]), reverse=True)  # oracle contexts are shuffled
    f = {"ret_top1": s[0] if s else 0.0, "ret_mean": float(np.mean(s)) if s else 0.0,
         "ret_gap": (s[0] - s[1]) if len(s) > 1 else 0.0, "self_refused": float(rec["refused"])}
    sc = attr.get("scores", {})
    if "nli_window" in sc:
        E, C = np.array(sc.get("nli_set", sc["nli_window"])), np.array(sc["nli_window_contra"])
        # contradiction is only meaningful against the chunk the sentence is about (NLI marks unrelated
        # passages as 'contradiction' for pronoun-led hypotheses) -> use the most similar chunk per sentence
        rel = np.array(sc["sim_sent"]).argmax(1) if "sim_sent" in sc else E.argmax(1)
        f.update(nli_cov_mean=float(E.max(1).mean()), nli_cov_min=float(E.max(1).min()),
                 contra_max=float(C[np.arange(len(C)), rel].max()),
                 frac_chunks_support=float((E.max(0) >= NLI_T).mean()), n_sent=len(E))
    if "sim_sent" in sc:
        f["sim_cov_mean"] = float(np.array(sc["sim_sent"]).max(1).mean())
    return f


def _refresh_scores(rec):
    """Recompute every retrieved chunk's cosine from the stored (title-prefixed) index embeddings, so the
    retrieval-confidence features have one definition across dense / hybrid / oracle conditions."""
    if not rec["retrieved"] or rec["domain"] not in DOMAIN_SET:
        return rec
    from src.common import INDICES
    if not (INDICES / "chroma").exists():  # Colab bundle: plan scores already come from the stored index
        return rec
    from src.pipeline.index import embed
    from src.pipeline.retrieval_plus import _chunk_matrix, _chunks
    E, (_, idx) = _chunk_matrix(rec["domain"]), _chunks(rec["domain"])
    qv = embed([rec["question"]])[0]
    rec = dict(rec)
    rec["retrieved"] = [dict(c, score=float(E[idx[c["chunk_id"]]] @ qv)) if c["chunk_id"] in idx else c
                        for c in rec["retrieved"]]
    return rec


DOMAIN_SET = {"hotpotqa", "squad2", "pubmedqa", "cuad", "financebench"}


def score_record(rec):
    rec = _refresh_scores(rec)
    out = dict(rec)
    ans = rec["answer"]
    # retrieval
    if rec["answerable"] and rec["gold_chunk_ids"] and rec["retrieved"]:
        out["retrieval"] = M.retrieval_metrics([c["chunk_id"] for c in rec["retrieved"]], rec["gold_chunk_ids"])
    # correctness (a refusal on an answerable question is incorrect)
    if rec["answerable"]:
        c, cont = M.correctness(strip_citations(ans), rec["gold_answers"], rec["answer_type"])
        if c is None:  # qualitative FinanceBench answers: NLI judge (gold entailed by answer, or vice versa)
            p = nli_probs([strip_citations(ans), rec["gold_answers"][0]], [rec["gold_answers"][0], strip_citations(ans)])[:, 0]
            c = float(max(p) >= 0.5)
            out["judge"] = "nli"
        out["correct"] = 0.0 if rec["refused"] else c
        out["correct_cont"] = 0.0 if rec["refused"] else cont
    else:
        out["correct"] = float(rec["refused"])
    # attribution (against the same retrieved chunks, for every system incl. closed-book)
    attr = {} if rec["refused"] or not rec["retrieved"] else attribute(ans, rec["retrieved"], METHODS)
    out["attribution"] = attr
    gold = set(rec["gold_chunk_ids"])
    mask = [c["chunk_id"] in gold for c in rec["retrieved"]]
    out["attr_metrics"] = {}
    for m, S in attr.get("scores", {}).items():
        if m.endswith("_contra"):
            continue
        t = SIM_T if m.startswith("sim") else NLI_T if m.startswith("nli") else 0.5
        if m == "self_cite" and "trust" not in rec["system"]:
            continue  # only the trust prompt asks for citations
        out["attr_metrics"][m] = M.attribution_metrics(S, mask, t, skip_empty_rows=(m == "self_cite"))
    out["features"] = features(rec, attr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--only", default=None, help="substring filter on file names")
    ap.add_argument("--loo", type=int, default=0, help="#answerable questions per domain for leave-one-out")
    ap.add_argument("--loo-system", default="vanilla_rag")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run = RUNS / a.run
    cfg = load_config(run / "config.yaml")
    for f in sorted(glob.glob(str(run / "*.jsonl"))):
        if f.endswith(".scored.jsonl") or (a.only and a.only not in os.path.basename(f)):
            continue
        dst = f.replace(".jsonl", ".scored.jsonl")
        if os.path.exists(dst) and not a.force and os.path.getmtime(dst) > os.path.getmtime(f):
            continue
        rows = [score_record(r) for r in tqdm(read_jsonl(f), desc=os.path.basename(f))]
        write_jsonl(dst, rows)

    if a.loo:
        llm = from_config(cfg)
        pfn = P.vanilla_rag if a.loo_system == "vanilla_rag" else P.trust_rag
        for f in sorted(glob.glob(str(run / f"*__{a.loo_system}.scored.jsonl"))):
            rows = read_jsonl(f)
            n = 0
            for r in tqdm(rows, desc=f"LOO {os.path.basename(f)}"):
                if n >= a.loo:
                    break
                if not r["answerable"] or r["refused"] or not r["retrieved"]:
                    continue
                if "loo" not in r:
                    r["loo"] = leave_one_out(llm, pfn, r["question"], r["retrieved"], r["answer_type"], r["answer"])
                n += 1
            write_jsonl(f, rows)


if __name__ == "__main__":
    main()
