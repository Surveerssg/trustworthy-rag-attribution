"""Read the filled verification sheets -> agreement, adjudication list, and validation of the automatic methods.

Protocol (scripts/make_annotation_sheet.py): Claude proposed a label + supporting passages for each of 100 answer sentences; team members
verify every item (Agree / Disagree + correction / Unsure). Report it as "LLM-proposed, human-verified" labels.

python -m scripts.analyze_human_labels --a paper/human_eval/verify_A_<name>.xlsx [--b paper/human_eval/verify_B_<name>.xlsx]
python -m scripts.analyze_human_labels --claude-only      # PRELIMINARY: validate against Claude's unverified labels
Outputs (paper/human_eval/):
  agreement.csv          each verifier vs Claude (% agree, kappa) and verifier vs verifier, overall and per domain
  to_adjudicate.csv      items where the two verifiers' final labels differ (fill 'final', rerun with --adjudicated)
  method_validation.csv  each method's sentence-support score vs the labels: AUROC, acc@0.5, and whether its top passage is in the
                         labelled supporting set; plus how often the automatic gold chunk is among the labelled supporting passages
"""
import argparse
import re

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.attribution.methods import refresh_record
from src.common import PAPER, RUNS, read_jsonl
from src.evaluation import metrics as M

OUT = PAPER / "human_eval"
BIN = {"Supported": 1, "Partially supported": 0, "Not supported": 0}
METHODS = ["sim_chunk", "sim_sent", "nli_chunk", "nli_window", "nli_set"]


def parse_passages(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return set()
    # pandas reads a numbers-only column as floats ("1.0"), so parse numerically
    return {int(float(t)) for t in re.findall(r"\d+(?:\.\d+)?", str(x)) if 1 <= float(t) <= 5}


def read_claude():
    d = pd.read_csv(OUT / "claude_labels.csv", dtype=str).fillna("")
    return d.rename(columns={"label": "claude_label", "passages": "claude_passages"})[["item_id", "claude_label", "claude_passages"]]


def read_cards(path):
    """The 'Items' tab is a column of cards (scripts/make_annotation_sheet.py): a header row whose column A holds the item id (S001...),
    then one row per field, field name in column A and value in column B."""
    from openpyxl import load_workbook
    from scripts.make_annotation_sheet import F_LABEL, F_PASSAGES, F_VERDICT
    ws = load_workbook(path, data_only=True)["Items"]
    rows, cur = {}, None
    field = {F_VERDICT: "verdict", F_LABEL: "fix_label", F_PASSAGES: "fix_passages"}
    for a, b in ws.iter_rows(min_col=1, max_col=2, values_only=True):
        if isinstance(a, str) and re.fullmatch(r"S\d{3}", a):
            cur = rows.setdefault(a, {"item_id": a, "verdict": None, "fix_label": None, "fix_passages": None})
        elif cur is not None and a in field:
            cur[field[a]] = b if b not in ("", None) else None
    return pd.DataFrame(list(rows.values()))


def read_sheet(path, claude):
    """One verifier's final label/passages: Claude's proposal unless they disagreed and gave a correction."""
    df = read_cards(path).merge(claude, on="item_id")
    dis = df.verdict.eq("Disagree")
    df["label"] = np.where(dis & df.fix_label.notna(), df.fix_label, df.claude_label)
    df["passages"] = [f if d and pd.notna(f) else c for d, f, c in zip(dis, df.fix_passages, df.claude_passages)]
    df.loc[df.verdict.isna(), ["label", "passages"]] = np.nan          # not verified yet
    return df[["item_id", "verdict", "label", "passages"]]


def kappa(a, b):
    return cohen_kappa_score(a, b) if len(set(a) | set(b)) > 1 else float("nan")


def agreement_rows(name, d, x, y):
    rows = []
    for dom, g in [("ALL", d)] + list(d.groupby("domain")):
        rows.append({"pair": name, "domain": dom, "n": len(g), "pct_same_label": float((g[x] == g[y]).mean()),
                     "kappa_3way": kappa(g[x], g[y]), "kappa_binary": kappa(g[x].map(BIN), g[y].map(BIN))})
    return rows


def method_scores(key):
    """Each method's stored attribution scores for the annotated sentence (from the scored source runs)."""
    cache, rows = {}, []
    for _, k in key.iterrows():
        f = RUNS / k["run"] / f"{k['domain']}__{k['system']}.scored.jsonl"
        if f not in cache:
            cache[f] = {r["qid"]: r for r in read_jsonl(f)} if f.exists() else {}
        r = cache[f].get(k["qid"])
        r = refresh_record(r) if r else r   # nli_set tie-break (stage2a_findings section 3), applied in memory
        row = {"item_id": k["item_id"]}
        sc = (r or {}).get("attribution", {}).get("scores", {})
        for m in METHODS:
            S = sc.get(m)
            if S and k["sent_idx"] < len(S):
                v = np.asarray(S[k["sent_idx"]])
                row[m] = float(v.max()); row[m + "_top"] = int(v.argmax()) + 1
        rows.append(row)
    return pd.DataFrame(rows)


def validate(d):
    """d: one row per item with 'gold' (label), 'gold_passages' (set) and the key columns."""
    d = d[d.gold.notna()].copy()
    d["y"] = d.gold.map(BIN)
    d = d.merge(method_scores(d[["item_id", "run", "system", "domain", "qid", "sent_idx"]]), on="item_id", how="left")
    val = []
    for m in METHODS:
        v = d[d[m].notna()] if m in d else pd.DataFrame()
        if len(v) < 10:
            continue
        sup = v[(v.y == 1) & v.gold_passages.map(bool)]
        val.append({"method": m, "n": len(v), "AUROC_vs_labels": M.auroc(v[m], v.y), "acc@0.5": float(((v[m] >= 0.5) == (v.y == 1)).mean()),
                    "top_passage_in_labelled_set": float(np.mean([t in g for t, g in zip(sup[m + "_top"], sup.gold_passages)])) if len(sup) else np.nan})
    # is the automatic gold chunk (what attribution accuracy is scored against) among the labelled supporting passages?
    gp = d.gold_positions.fillna("").astype(str).map(lambda s: {int(float(t)) for t in s.split()})
    sup = d[(d.y == 1) & gp.map(bool) & d.gold_passages.map(bool)]
    val.append({"method": "automatic gold chunk", "n": len(sup), "AUROC_vs_labels": np.nan, "acc@0.5": np.nan,
                "top_passage_in_labelled_set": float(np.mean([bool(a & b) for a, b in zip(gp[sup.index], sup.gold_passages)])) if len(sup) else np.nan})
    return pd.DataFrame(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=None)
    ap.add_argument("--b", default=None)
    ap.add_argument("--adjudicated", default=None, help="to_adjudicate.csv with a filled 'final' column")
    ap.add_argument("--claude-only", action="store_true", help="PRELIMINARY: use Claude's unverified labels as the reference")
    x = ap.parse_args()
    key, claude = pd.read_csv(OUT / "annotation_key.csv"), read_claude()
    base = key.merge(claude, on="item_id")
    pd.set_option("display.width", 200)

    if x.claude_only:
        d = base.assign(gold=base.claude_label, gold_passages=base.claude_passages.map(parse_passages))
        val = validate(d)
        val.to_csv(OUT / "method_validation_claude_only.csv", index=False)
        print("PRELIMINARY (Claude's labels, not yet verified by people)\n", val.round(3).to_string(index=False))
        return

    assert x.a, "give --a (and --b if a second verifier finished), or --claude-only"
    sheets = {"A": read_sheet(x.a, claude)}
    if x.b:
        sheets["B"] = read_sheet(x.b, claude)
    d = base
    for w, s in sheets.items():
        d = d.merge(s.rename(columns={c: f"{c}_{w}" for c in ("verdict", "label", "passages")}), on="item_id")
    done = np.logical_and.reduce([d[f"label_{w}"].notna() for w in sheets])
    print(f"verified by {'+'.join(sheets)}: {done.sum()} / {len(d)}")
    d = d[done].copy()

    rows = []
    for w in sheets:
        v = d[f"verdict_{w}"]
        print(f"verifier {w}: Agree {v.eq('Agree').mean():.1%}, Disagree {v.eq('Disagree').mean():.1%}, Unsure {v.eq('Unsure').mean():.1%}")
        rows += agreement_rows(f"Claude vs {w}", d, "claude_label", f"label_{w}")
    if "B" in sheets:
        rows += agreement_rows("A vs B", d, "label_A", "label_B")
    ag = pd.DataFrame(rows)
    ag.to_csv(OUT / "agreement.csv", index=False)
    print("\n=== Agreement ===\n", ag.round(3).to_string(index=False))

    if "B" in sheets:
        same = d.label_A == d.label_B
        if x.adjudicated:
            adj = pd.read_csv(x.adjudicated)[["item_id", "final"]]
            d = d.merge(adj, on="item_id", how="left")
            d["gold"] = np.where(same, d.label_A, d.final)
        else:
            d[~same][["item_id", "domain", "sentence", "claude_label", "label_A", "label_B", "passages_A", "passages_B"]] \
                .assign(final="").to_csv(OUT / "to_adjudicate.csv", index=False)
            print(f"\n{(~same).sum()} disagreements written to to_adjudicate.csv (until adjudicated, only agreed items are used)")
            d["gold"] = np.where(same, d.label_A, None)
        d["gold_passages"] = [parse_passages(a) | parse_passages(b) for a, b in zip(d.passages_A, d.passages_B)]
    else:
        d["gold"], d["gold_passages"] = d.label_A, d.passages_A.map(parse_passages)
    val = validate(d)
    val.to_csv(OUT / "method_validation.csv", index=False)
    print("\n=== Methods vs verified labels ===\n", val.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
