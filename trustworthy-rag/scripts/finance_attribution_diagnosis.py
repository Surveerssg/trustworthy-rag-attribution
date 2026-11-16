"""Why does attribution collapse on FinanceBench? (exploratory, 28 Oct 2026; write-up: paper/findings/finance_diagnosis_findings.md)

python -m scripts.finance_attribution_diagnosis diagnose   # no model: failure features from the stored scores (laptop Python)
python -m scripts.finance_attribution_diagnosis rescore    # re-scores sim_sent under passage transformations (needs sentence-transformers)

Records: oracle_vanilla (gold passage guaranteed) of every v2 model; the answers and contexts are NOT regenerated. Only the passage text
shown to the attribution scorer changes, identically for gold and distractor passages:
  original     the stored chunk text (reproduces the stored sim_sent)
  strip        boilerplate removed: 'Table of Contents', bare page numbers, and lines that recur on >= 20% of the pages of the same filing
  rows         PDF tables rebuilt row by row: pdf text puts every cell on its own line, so the sentence splitter feeds the scorer fragments
               such as '1,488' or '$'; a row label and the value cells after it are joined into one unit 'label: v1 | v2 | v3'
  rows_header  as rows, and each value is paired with its column header when a header row (all years) precedes it: 'label: 2018 = 1,488; ...'
  rows_header+strip
plus one scorer that ignores wording: num_match = share of the sentence's numbers (years excluded) found in the passage, ties broken by
sim_sent on the same text. Control domains (prose, no tables): HotpotQA and SQuAD 2.0, where the transformations should change nothing.
"""
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from src.attribution.fusion import acc_by_weight
from src.attribution.methods import refresh_record
from src.common import PROC, RESULTS, RUNS, read_jsonl
from src.evaluation.metrics import extract_numbers

MODELS = ["qwen1.5b", "llama8b", "qwen14b", "phi4", "mistral24b", "gemma27b", "qwen32b"]
OUT = RESULTS / "finance_diagnosis"
_CELL = re.compile(r"^[\(\-–—]?\s*[$€£]?\s*\(?-?[\d][\d,]*\.?\d*\)?\s*%?\)?$|^[-–—$%]$")
_YEAR = re.compile(r"^(19|20)\d\d$")


def records(domain, system="oracle_vanilla"):
    out = []
    for m in MODELS:
        p = RUNS / f"v2_{m}" / f"{domain}__{system}.scored.jsonl"
        if not p.exists():
            continue
        for r in map(refresh_record, read_jsonl(p)):
            sents = (r.get("attribution") or {}).get("sentences") or []
            if r["answerable"] and not r["refused"] and sents and acc_by_weight(r) is not None:
                out.append(dict(r, model=m))
    return out


def gold_mask(r):
    g = set(r["gold_chunk_ids"])
    return np.array([c["chunk_id"] in g for c in r["retrieved"]])


def acc(S, mask):
    return float(mask[np.asarray(S).argmax(1)].mean())


def key_numbers(s):
    return [x for x in extract_numbers(s) if not (1900 <= abs(x) <= 2100 and float(x).is_integer()) and abs(x) > 0]


def has_number(text, nums, tol=0.01):
    t = extract_numbers(text)
    return any(any(abs(abs(a) * sc - abs(b)) <= tol * abs(b) + 0.006 for a in t for sc in (1, 1e3, 1e-3, 1e6, 1e-6)) for b in nums)


def is_cell(line):
    return bool(_CELL.match(line.strip())) if line.strip() else False


# ------------------------------------------------------------------ transformations
def boilerplate_lines():
    """Lines that recur on >= 20% of the pages of one filing (running headers/footers), per filing."""
    pages = defaultdict(list)
    for d in read_jsonl(PROC / "financebench" / "corpus.jsonl"):
        pages[d["meta"]["doc"]].append({l.strip() for l in d["text"].split("\n") if l.strip()})
    rep = {}
    for doc, ps in pages.items():
        cnt = Counter(l for p in ps for l in p)
        rep[doc] = {l for l, c in cnt.items() if c >= max(3, 0.2 * len(ps)) and not is_cell(l)}
    return rep


def strip(text, rep):
    keep = [l for l in text.split("\n") if l.strip() and l.strip() not in rep and l.strip().lower() != "table of contents"
            and not re.fullmatch(r"\s*\d{1,3}\s*", l)]
    return "\n".join(keep)


def rows(text, header=False):
    """Rebuild table rows from one-cell-per-line PDF text. Prose lines pass through (joined by spaces)."""
    units, prose, label, vals, head = [], [], [], [], None

    def end_prose():
        if prose:
            units.append(" ".join(prose)); prose.clear()

    def end_row():
        nonlocal head
        lab, v = " ".join(label).strip(), [x for x in vals if x not in ("$", "%", "-", "–", "—")]
        if v:
            end_prose()
            if all(_YEAR.match(x) for x in v):            # a header row of years: remember it, keep it as a unit
                head = v
                units.append((f"{lab}: " if lab else "") + "columns " + " | ".join(v))
            elif header and head and len(head) == len(v):
                units.append(f"{lab}: " + "; ".join(f"{h} = {x}" for h, x in zip(head, v)))
            else:
                units.append(f"{lab}: " + " | ".join(v))
        elif lab:
            prose.append(lab)                              # a short line with no numbers after it is ordinary text
        label.clear(); vals.clear()

    for l in (x.strip() for x in text.split("\n")):
        if not l:
            continue
        if is_cell(l):
            vals.append(l)
            continue
        if vals:
            end_row()
        if len(l.split()) > 12:                            # a long line is prose, never a row label
            if label:
                prose.append(" ".join(label)); label.clear()
            prose.append(l)
        else:
            label.append(l)
    end_row()
    end_prose()
    return "\n\n".join(units)


# ------------------------------------------------------------------ 1. diagnosis from stored scores
def diagnose():
    R = records("financebench")
    rows_ = []
    for r in R:
        m = gold_mask(r)
        sc = r["attribution"]["scores"]
        nums = key_numbers(r["gold_answers"][0])
        golds = [c for c, g in zip(r["retrieved"], m) if g]
        dists = [c for c, g in zip(r["retrieved"], m) if not g]
        glines = [l for c in golds for l in c["text"].split("\n") if l.strip()]
        reasoning = r["meta"].get("reasoning") or ""
        kind = "quoted" if reasoning == "Information extraction" else "computed" if "umerical" in reasoning else "other/novel"
        pick = np.asarray(sc["sim_sent"]).argmax(1)
        gfil = {c["doc_id"].rsplit("_p", 1)[0] for c in golds}
        rows_.append({"model": r["model"], "qid": r["qid"], "kind": kind,
                      "gold_tableness": np.mean([is_cell(l) for l in glines]) if glines else 0.0,
                      "num_in_gold": bool(nums) and any(has_number(c["text"], nums) for c in golds),
                      "num_in_distractor": bool(nums) and any(has_number(c["text"], nums) for c in dists),
                      "same_filing_distractors": sum(c["doc_id"].rsplit("_p", 1)[0] in gfil for c in dists),
                      "sim_picked_has_number": bool(nums) and np.mean([has_number(r["retrieved"][j]["text"], nums) for j in pick]) > 0,
                      **{k: acc(sc[k], m) for k in ("sim_sent", "sim_chunk", "nli_window", "nli_set")}})
    D = pd.DataFrame(rows_)
    D["table_gold"] = np.where(D.gold_tableness >= 0.3, "gold mostly table cells (>=30% of lines)", "gold mostly prose")
    D["answer_number"] = np.where(~D.num_in_gold & D.kind.eq("computed"), "computed, not in passage",
                                  np.where(D.num_in_gold, "number quoted in gold passage", "no number / other"))
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "diagnosis_records.csv", index=False)
    meth = ["sim_sent", "sim_chunk", "nli_window", "nli_set"]
    print(f"FinanceBench, gold passage guaranteed: {len(D)} answer records over {D.model.nunique()} models, {D.qid.nunique()} questions")
    for by in ("table_gold", "answer_number", "kind", "num_in_distractor"):
        t = D.groupby(by)[meth].mean().round(3)
        t.insert(0, "n", D.groupby(by).size())
        print(f"\n--- attribution accuracy by {by}\n{t.to_string()}")
    f = D[D.sim_sent < 1]
    print(f"\n--- sim_sent failures ({len(f)} records): the chunk it picked contains the gold answer's number in "
          f"{f.sim_picked_has_number.mean():.1%} (a near-duplicate of the evidence, i.e. arguably not an error)")
    print(f"    records with a distractor holding the gold number: {D.num_in_distractor.mean():.1%}; "
          f"mean same-filing distractors: {D.same_filing_distractors.mean():.2f} of 4")
    D.groupby(["table_gold", "answer_number"])[meth].agg(["mean", "size"]).round(3).to_csv(OUT / "diagnosis_cells.csv")


# ------------------------------------------------------------------ 2. re-scoring
def rescore():
    from src.pipeline.index import embed
    from src.pipeline.chunking import sentence_spans
    rep = boilerplate_lines()
    cache = {}

    def units_emb(text):
        if text not in cache:
            u = [text[a:b] for a, b in sentence_spans(text)] or [text or " "]
            cache[text] = embed(u)
        return cache[text]

    def transform(c, how):
        t = c["text"]
        doc = c["doc_id"].rsplit("_p", 1)[0][3:] if c["doc_id"].startswith("fb_") else None
        if "strip" in how and doc:
            t = strip(t, rep.get(doc, set()))
        if how.startswith("rows_header"):
            t = rows(t, header=True)
        elif how.startswith("rows"):
            t = rows(t)
        return t

    HOW = ["original", "strip", "rows", "rows_header", "rows_header+strip"]
    res = []
    for domain in ("financebench", "hotpotqa", "squad2"):
        R = records(domain)
        for r in R:
            m = gold_mask(r)
            st = [s["text"] for s in r["attribution"]["sentences"]]
            sv = embed(st)
            row = {"domain": domain, "model": r["model"], "qid": r["qid"], "stored_sim_sent": acc(r["attribution"]["scores"]["sim_sent"], m)}
            sims = {}
            for how in HOW:
                S = np.stack([(sv @ units_emb(transform(c, how)).T).max(1) for c in r["retrieved"]], 1)
                sims[how] = S
                row[f"sim_sent[{how}]"] = acc(S, m)
            for base in ("original", "rows_header+strip"):
                texts = [transform(c, base) for c in r["retrieved"]]
                N = np.array([[np.mean([has_number(t, [x]) for x in key_numbers(s)]) if key_numbers(s) else 0.0 for t in texts] for s in st])
                row[f"num_match+sim[{base}]"] = acc(N + 1e-3 * sims[base], m)
            res.append(row)
        print(domain, len(R), "records", flush=True)
    D = pd.DataFrame(res)
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "rescore_records.csv", index=False)
    cols = [c for c in D.columns if c.startswith(("stored", "sim_sent[", "num_match"))]
    t = D.groupby("domain")[cols].mean().round(3).T
    t.to_csv(OUT / "rescore_summary.csv")
    print("\n=== Attribution accuracy (gold guaranteed), same answers and contexts, passage text transformed for the scorer only ===")
    print(t.to_string())
    print("\n--- FinanceBench per model")
    print(D[D.domain == "financebench"].groupby("model")[cols].mean().round(3).to_string())
    fb = D[D.domain == "financebench"]
    q = fb.groupby("qid")[cols].mean()   # question-level means for a paired cluster bootstrap
    rng = np.random.default_rng(0)
    print("\n--- FinanceBench, change vs sim_sent[original], 95% CI (bootstrap over questions)")
    for c in cols[2:]:
        d = (q[c] - q["sim_sent[original]"]).to_numpy()
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        print(f"  {c:28s} {d.mean():+.3f}  [{np.quantile(bs, .025):+.3f}, {np.quantile(bs, .975):+.3f}]")


# ------------------------------------------------------------------ 3. number-aware scorer combined with the stored scores (no model)
def num_matrix(r):
    st = [s["text"] for s in r["attribution"]["sentences"]]
    texts = [c["text"] for c in r["retrieved"]]
    return np.array([[np.mean([has_number(t, [x]) for x in key_numbers(s)]) if key_numbers(s) else 0.0 for t in texts] for s in st])


def combos():
    """Every domain, gold guaranteed (oracle_vanilla) and natural retrieval when the gold chunk was retrieved (vanilla_rag).
    num+X = number match first, X breaks ties (and decides alone for sentences without numbers)."""
    from src.attribution.fusion import fused, load_weights
    W = load_weights()
    doms = ["squad2", "hotpotqa", "pubmedqa", "cuad", "financebench"]
    res = []
    for system in ("oracle_vanilla", "vanilla_rag"):
        for d in doms:
            for r in records(d, system):
                m, sc, N = gold_mask(r), r["attribution"]["scores"], num_matrix(r)
                if not m.any():
                    continue
                sim, nli = np.asarray(sc["sim_sent"]), np.asarray(sc["nli_set"])
                mix = fused(sim, nli, float(W.get(d, 0.9)))
                has_num = [bool(key_numbers(s["text"])) for s in r["attribution"]["sentences"]]
                res.append({"system": system, "domain": d, "model": r["model"], "qid": r["qid"], "frac_sent_with_numbers": np.mean(has_num),
                            "sim_sent": acc(sim, m), "nli_set": acc(nli, m), "mixture": acc(mix, m),
                            "num+sim": acc(N + 1e-3 * sim, m), "num+nli": acc(N + 1e-3 * nli, m), "num+mixture": acc(N + 1e-3 * mix, m)})
    D = pd.DataFrame(res)
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "combos_records.csv", index=False)
    cols = ["sim_sent", "nli_set", "mixture", "num+sim", "num+nli", "num+mixture"]
    for system in ("oracle_vanilla", "vanilla_rag"):
        S = D[D.system == system]
        t = S.groupby("domain")[cols + ["frac_sent_with_numbers"]].mean().reindex(doms)
        t.insert(0, "n", S.groupby("domain").size())
        per_model = S.groupby(["model", "domain"])[cols].mean().groupby("model")
        mac, worst = per_model.mean(), per_model.min()
        print(f"\n=== {system}: attribution accuracy by domain (records pooled over models"
              f"{'' if system == 'oracle_vanilla' else '; only answers whose context holds a gold chunk'}) ===")
        print(t.round(3).to_string())
        print("--- per model: macro over domains\n" + mac.round(3).to_string())
        print("--- per model: worst domain\n" + worst.round(3).to_string())
        t.to_csv(OUT / f"combos_{system}_by_domain.csv"); mac.to_csv(OUT / f"combos_{system}_macro.csv"); worst.to_csv(OUT / f"combos_{system}_worst.csv")
    rng = np.random.default_rng(0)
    print("\n--- paired differences, 95% CI (bootstrap over questions within domain)")
    for system in ("oracle_vanilla", "vanilla_rag"):
        for d in doms:
            q = D[(D.system == system) & (D.domain == d)].groupby("qid")[cols].mean()
            for a, b in (("num+nli", "nli_set"), ("num+mixture", "mixture")):
                x = (q[a] - q[b]).to_numpy()
                bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(3000)]
                print(f"  {system:15s} {d:13s} {a} - {b}: {x.mean():+.3f} [{np.quantile(bs, .025):+.3f}, {np.quantile(bs, .975):+.3f}]")


if __name__ == "__main__":
    {"diagnose": diagnose, "rescore": rescore, "combos": combos}[sys.argv[1] if len(sys.argv) > 1 else "diagnose"]()
