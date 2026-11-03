"""LLM-as-attributor baseline: the pre-registered tests L1-L4 (paper/findings/analysis_plan_llm_attributor.md). CPU only.

python -m scripts.ingest_kaggle experiments/kaggle_raw/trag_results_stage2E_F.zip --no-analyses   # unpack first
python -m scripts.analyze_llm_attributor
Reads experiments/runs/attr_<judge>/<domain>__judge.jsonl and the scored oracle_vanilla records they were built from. Sentences without a judge
output (a run cut off by its time budget) are left out, and the counts say so. Writes experiments/results/llm_attributor/.
"""
import re

import numpy as np
import pandas as pd

from scripts.finance_attribution_diagnosis import has_number, key_numbers
from scripts.llm_attribute import ITEMS
from src.attribution.fusion import fused, load_weights
from src.attribution.methods import refresh_record
from src.common import RESULTS, RUNS, read_jsonl

JUDGES = [("qwen32b", "Qwen2.5-32B judge"), ("llama8b", "Llama-3.1-8B judge")]
DOMS = ["financebench", "tatqa", "cuad", "pubmedqa", "hotpotqa", "squad2"]
OUT = RESULTS / "llm_attributor"
MARGIN = 0.05


def judge_outputs():
    out = {}
    for j, _ in JUDGES:
        rows = {}
        for d in DOMS:
            p = RUNS / f"attr_{j}" / f"{d}__judge.jsonl"
            if p.exists():
                rows.update({r["id"]: r for r in read_jsonl(p)})
        out[j] = rows
    return out


def reading(o, k):
    """Amendment 1 (analysis_plan_llm_attributor.md): the judge's choice is the passage number written in its reply; the first-token
    probabilities serve as its scores only when they hold at least half of the mass (Llama writes '[3]', whose first token is '[')."""
    p = np.asarray(o["probs"], float)
    if p.sum() >= 0.5:
        return p
    m = re.search(r"[1-%d]" % k, o.get("text") or "")
    out = np.zeros(k)
    if m:
        out[int(m.group()) - 1] = 1.0
    return out


def records():
    """Scored oracle records keyed like the judge items (run|domain|qid)."""
    keys = {"|".join(it["id"].split("|")[:3]) for it in read_jsonl(ITEMS)}
    recs = {}
    for k in keys:
        run, d, _ = k.split("|")
        if (run, d) not in recs:
            recs[(run, d)] = {r["qid"]: r for r in read_jsonl(RUNS / run / f"{d}__oracle_vanilla.scored.jsonl")}
    return {k: refresh_record(recs[(k.split("|")[0], k.split("|")[1])][k.split("|")[2]]) for k in keys}


def per_record(recs, J):
    W = load_weights()
    rows = []
    for key, r in recs.items():
        run, d, qid = key.split("|")
        sents = r["attribution"]["sentences"]
        ids = [f"{key}|{i}" for i in range(len(sents))]
        gold = set(r["gold_chunk_ids"])
        mask = np.array([c["chunk_id"] in gold for c in r["retrieved"]])
        sc = r["attribution"]["scores"]
        sim, nli = np.asarray(sc["sim_sent"]), np.asarray(sc["nli_set"])
        mix = fused(sim, nli, float(W.get(d, 0.9)))
        texts = [c["text"] for c in r["retrieved"]]
        N = np.array([[np.mean([has_number(t, [x]) for x in key_numbers(s["text"])]) if key_numbers(s["text"]) else 0.0 for t in texts] for s in sents])
        acc = lambda S: float(mask[np.asarray(S).argmax(1)].mean())
        row = {"key": key, "run": run, "generator": run.split("_", 1)[1], "domain": d, "qid": qid, "n_sent": len(sents),
               "sim_sent": acc(sim), "nli_set": acc(nli), "mixture": acc(mix), "num_sim": acc(N + 1e-3 * sim), "num_mixture": acc(N + 1e-3 * mix)}
        for j, _ in JUDGES:
            out = [J[j].get(i) for i in ids]
            if any(o is None for o in out):
                row[f"judge_{j}"] = row[f"num_judge_{j}"] = np.nan        # not (fully) judged: left out of every comparison with this judge
                continue
            P = np.array([reading(o, len(mask)) for o in out], float)          # Amendment 1: the number the judge wrote
            row[f"judge_{j}"] = float(np.mean([P[i].max() > 0 and mask[P[i].argmax()] for i in range(len(out))]))
            row[f"judge_literal_{j}"] = float(np.mean([bool(o["pick"]) and mask[o["pick"] - 1] for o in out]))   # pre-registered first-token rule
            row[f"num_judge_{j}"] = acc(N + 1e-3 * P)
            row[f"format_ok_{j}"] = float(np.mean([o["ok"] for o in out]))
            row[f"seconds_{j}"] = float(np.sum([o["seconds"] for o in out]))
        rows.append(row)
    return pd.DataFrame(rows)


def boot_diff(df, a, b, n=10000, seed=0, cluster=("domain", "qid")):
    """Paired mean difference a - b over records, cluster bootstrap over questions."""
    x = df[[*cluster, a, b]].dropna()
    if not len(x):
        return {"diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan, "n": 0}
    x = x.assign(d=x[a] - x[b])
    g = x.groupby(list(cluster)).d.agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n, len(g)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return {"diff": float(s.sum() / c.sum()), "ci_lo": float(np.quantile(bs, .025)), "ci_hi": float(np.quantile(bs, .975)),
            "p": max(p, 1 / n), "n": int(len(x))}


def holm(ps):
    ps = np.asarray(ps, float)
    order, m, adj, run = np.argsort(ps), len(ps), np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def verdict(r, sign=+1):
    return "supported" if (np.sign(r.diff) == sign and abs(r.diff) >= MARGIN and r.p_holm < 0.05) else "not supported"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    J = judge_outputs()
    print({j: f"{len(v)} sentences judged" for j, v in J.items()})
    D = per_record(records(), J)
    D.to_csv(OUT / "records.csv", index=False)
    cols = ["sim_sent", "nli_set", "mixture", "num_sim", "num_mixture"] + [c for j, _ in JUDGES for c in (f"judge_{j}", f"judge_literal_{j}", f"num_judge_{j}")]
    cols = [c for c in cols if c in D]
    t = D.groupby("domain")[cols].mean().reindex(DOMS)
    t.insert(0, "n_records", D.groupby("domain").size())
    for j, _ in JUDGES:
        t.insert(1, f"n_judged_{j}", D.groupby("domain")[f"judge_{j}"].count())
    t.to_csv(OUT / "by_domain.csv")
    print("\n=== Attribution accuracy, gold passage guaranteed (records pooled over the 7 generators) ===")
    print(t.round(3).to_string())

    # L1: 32B judge vs sim_sent, per domain (Holm over 6)
    L1 = pd.DataFrame([{"test": "L1 judge32 - sim_sent", "domain": d, **boot_diff(D[D.domain == d], "judge_qwen32b", "sim_sent")} for d in DOMS])
    L1["p_holm"] = holm(L1.p.fillna(1))
    L1["verdict"] = [verdict(r) if r.n else "no data" for r in L1.itertuples()]
    # L2: non-inferiority of num_sim to the 32B judge on the two financial datasets (CI bound only)
    L2 = pd.DataFrame([{"test": "L2 judge32 - num_sim", "domain": d, **boot_diff(D[D.domain == d], "judge_qwen32b", "num_sim")}
                       for d in ("financebench", "tatqa")])
    L2["verdict"] = ["supported (num_sim within 0.05)" if r.ci_hi < MARGIN else "not supported" if r.n else "no data" for r in L2.itertuples()]
    # L3: number + judge vs judge, financial datasets (Holm over 2)
    L3 = pd.DataFrame([{"test": "L3 num_judge32 - judge32", "domain": d, **boot_diff(D[D.domain == d], "num_judge_qwen32b", "judge_qwen32b")}
                       for d in ("financebench", "tatqa")])
    L3["p_holm"] = holm(L3.p.fillna(1))
    L3["verdict"] = [verdict(r) if r.n else "no data" for r in L3.itertuples()]
    # L4: 32B vs 8B judge, pooled
    L4 = pd.DataFrame([{"test": "L4 judge32 - judge8", "domain": "all", **boot_diff(D, "judge_qwen32b", "judge_llama8b")}])
    L4["p_holm"] = L4.p
    L4["verdict"] = [verdict(r) if r.n else "no data" for r in L4.itertuples()]
    T = pd.concat([L1, L2, L3, L4], ignore_index=True)
    T.to_csv(OUT / "tests.csv", index=False)
    print("\n=== Pre-registered tests (cluster bootstrap over questions; supported = predicted sign, |diff| >= .05, Holm p < .05) ===")
    print(T.round(4).to_string(index=False))

    print("\n--- reported, not tested: per generator (financial datasets)")
    fin = D[D.domain.isin(["financebench", "tatqa"])]
    print(fin.groupby(["domain", "generator"])[["sim_sent", "nli_set", "num_sim", "num_mixture", "judge_qwen32b", "num_judge_qwen32b",
                                                 "judge_llama8b"]].mean().round(3).to_string())
    own = D[~D.generator.eq("qwen32b")]
    print("\n--- sensitivity: without the 32B judge's own answers (generator qwen32b)")
    print(own.groupby("domain")[["sim_sent", "num_sim", "judge_qwen32b", "num_judge_qwen32b"]].mean().reindex(DOMS).round(3).to_string())
    print("\n--- cost and format")
    for j, label in JUDGES:
        if f"seconds_{j}" in D:
            n = D[f"judge_{j}"].notna()
            print(f"{label}: {D.loc[n, 'n_sent'].sum()} sentences, {D.loc[n, f'seconds_{j}'].sum() / 3600:.2f} request-hours "
                  f"(parallel requests overlap), format ok {D.loc[n, f'format_ok_{j}'].mean():.3f}")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
