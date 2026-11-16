"""Spot-check of the Qwen-32B judge's declines (exploratory, plan: paper/findings/analysis_plan_judge_declines.md). CPU only.

python -m scripts.judge_declines sheet     -> paper/human_eval/judge_declines/{declines_sheet.xlsx, declines_key.csv}
python -m scripts.judge_declines analyze --sheet paper/human_eval/judge_declines/declines_sheet_<name>.xlsx

The judge (Part F) declined to name any passage for about 800 sentences although the gold passage was in the context. The sheet mixes 30 of
those sentences (5 per domain) with 15 sentences where the judge did name a passage (controls), shuffled, with no hint which is which. The
labeller judges each sentence against the 5 passages from scratch (no proposed label). Do NOT give the labeller declines_key.csv.
"""
import argparse
import random
import re

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

from scripts.analyze_llm_attributor import DOMS, judge_outputs, reading
from scripts.llm_attribute import ITEMS
from scripts.make_annotation_sheet import FONT, HEAD, LABEL_W, SENT, TEXT_W, THIN, YELLOW, _height
from src.common import PAPER, RUNS, read_jsonl

OUT = PAPER / "human_eval" / "judge_declines"
N_DECLINE, N_CONTROL = 5, {"financebench": 3, "tatqa": 3, "pubmedqa": 3, "cuad": 2, "hotpotqa": 2, "squad2": 2}   # 30 + 15
LABELS = ["Supported", "Partially supported", "Not supported", "No factual claim"]
DOM_NAME = {"squad2": "General", "hotpotqa": "General (multi-hop)", "pubmedqa": "Medical", "cuad": "Legal", "financebench": "Financial",
            "tatqa": "Financial (TAT-QA)"}
F_LABEL, F_PASSAGES, F_NOTES = "YOUR LABEL", "SUPPORTING PASSAGES", "NOTES (optional)"


def declined(o, k):
    """The judge named no passage (same reading as the main analysis, Amendment 1)."""
    return reading(o, k).max() == 0


def sample(seed=0):
    J = judge_outputs()["qwen32b"]
    items = [it for it in read_jsonl(ITEMS) if it["id"] in J]
    rng = random.Random(seed)
    chosen, counts = [], {}
    for d in DOMS:
        dec = [it for it in items if it["domain"] == d and declined(J[it["id"]], len(it["chunk_ids"]))]
        ctl = [it for it in items if it["domain"] == d and not declined(J[it["id"]], len(it["chunk_ids"]))]
        counts[d] = (len(dec), len(dec) + len(ctl))
        chosen += [dict(it, declined=True) for it in rng.sample(dec, N_DECLINE)]
        chosen += [dict(it, declined=False) for it in rng.sample(ctl, N_CONTROL[d])]
    rng.shuffle(chosen)
    recs = {}
    for it in chosen:
        if (it["run"], it["domain"]) not in recs:
            recs[(it["run"], it["domain"])] = {r["qid"]: r for r in read_jsonl(RUNS / it["run"] / f"{it['domain']}__oracle_vanilla.scored.jsonl")}
        r = recs[(it["run"], it["domain"])][it["qid"]]
        gold = set(r["gold_chunk_ids"])
        it.update(question=r["question"], answer=r["answer"], passages=[c["text"] for c in r["retrieved"]],
                  gold_positions=[i + 1 for i, c in enumerate(r["retrieved"]) if c["chunk_id"] in gold], judge_reply=J[it["id"]].get("text", ""))
    for n, it in enumerate(chosen, 1):
        it["item_id"] = f"D{n:02d}"
    return chosen, counts


def guidelines(ws):
    ws.title = "Guidelines"
    rows = [
        ("Spot-check: is each sentence supported by the passages?", True),
        ("", False),
        ("Each card on the 'Items' tab is one sentence from an AI system's answer (blue row) and the 5 passages the system was given. "
         "Decide from scratch whether the passages support the sentence. Judge only that sentence, only against those 5 passages; ignore your own "
         "knowledge: a true fact that the passages do not state is NOT supported.", False),
        ("", False),
        ("FILL IN ONLY THE YELLOW CELLS (the label is a dropdown):", True),
        ("  YOUR LABEL: one of the four below.", False),
        ("  SUPPORTING PASSAGES: the passage number(s) that support it, e.g. 2 or 1,3 (leave empty for Not supported / No factual claim).", False),
        ("  NOTES: optional, one line (e.g. 'number is computed', 'year wrong', 'needs expert knowledge').", False),
        ("", False),
        ("THE LABELS", True),
        ("  Supported: everything the sentence claims can be verified from the passages (one passage, or several together).", False),
        ("  Partially supported: some claims are verifiable, at least one is not (e.g. right company, wrong figure).", False),
        ("  Not supported: the main claim is absent from, or contradicted by, the passages.", False),
        ("  No factual claim: filler or a bare verdict with nothing to check ('Based on the passages, the answer is as follows.', 'Maybe.').", False),
        ("", False),
        ("RULES OF THUMB", True),
        ("  - Paraphrase, rounding (1,577 ~ 1.58 billion) and consistent unit changes are fine.", False),
        ("  - A calculation (e.g. a difference of two figures) is Supported only if all its inputs appear in the passages and the arithmetic is right.", False),
        ("  - Yes/No sentences: judge the stated conclusion. 'Yes, the drug reduced mortality' needs a passage that shows it.", False),
        ("  - Do not discuss items with anyone until the sheet is finished. Do not look at any other project file about these items.", False),
        ("", False),
        ("Expected time: about 1 minute per item (45 items). Save as declines_sheet_<yourname>.xlsx.", False),
    ]
    for i, (t, bold) in enumerate(rows, 1):
        c = ws.cell(row=i, column=1, value=t)
        c.font = Font(name=FONT, bold=bold, size=13 if i == 1 else 10)
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 140


def items_sheet(ws, items):
    ws.title = "Items"
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = LABEL_W, TEXT_W
    ws.sheet_view.zoomScale = 110
    n = len(items)
    top = ws.cell(row=1, column=1, value="PROGRESS")
    top.font = Font(name=FONT, bold=True, color="FFFFFF"); top.fill = HEAD
    done = ws.cell(row=1, column=2, value="=" + "+".join(f'COUNTIF(B:B,"{x}")' for x in LABELS) + f'&" of {n} items done (fill only the yellow cells)"')
    done.font = Font(name=FONT, bold=True, color="FFFFFF"); done.fill = HEAD
    ws.freeze_panes = "A2"
    dv = DataValidation(type="list", formula1='"' + ",".join(LABELS) + '"', allow_blank=True, showErrorMessage=True, error="Choose a label from the list")
    ws.add_data_validation(dv)

    def put(r, name, value, fill=None, bold=False, size=10):
        a = ws.cell(row=r, column=1, value=name)
        a.font = Font(name=FONT, bold=True, size=9, color="44546A"); a.alignment = Alignment(wrap_text=True, vertical="top")
        b = ws.cell(row=r, column=2, value=value)
        b.font = Font(name=FONT, size=size, bold=bold); b.alignment = Alignment(wrap_text=True, vertical="top")
        for c in (a, b):
            c.border = THIN
            if fill:
                c.fill = fill
        ws.row_dimensions[r].height = _height(value)
        return b

    r = 3
    for k, it in enumerate(items, 1):
        for col, v in ((1, it["item_id"]), (2, f"{DOM_NAME[it['domain']]}   ·   item {k} of {n}")):
            c = ws.cell(row=r, column=col, value=v)
            c.font = Font(name=FONT, bold=True, color="FFFFFF", size=11); c.fill = HEAD
        r += 1
        put(r, "QUESTION", it["question"]); r += 1
        put(r, "FULL ANSWER", it["answer"]); r += 1
        put(r, "SENTENCE TO JUDGE", it["sentence"], fill=SENT, bold=True, size=11); r += 1
        dv.add(put(r, F_LABEL, None, fill=YELLOW).coordinate); r += 1
        put(r, F_PASSAGES, None, fill=YELLOW); r += 1
        put(r, F_NOTES, None, fill=YELLOW); r += 1
        for j, p in enumerate((it["passages"] + [""] * 5)[:5], 1):
            put(r, f"PASSAGE {j}", re.sub(r"\s*\n\s*", " | ", p.strip())[:3000], size=9); r += 1   # table rows on one line
        r += 1


def make_sheet():
    items, counts = sample()
    OUT.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    guidelines(wb.active)
    items_sheet(wb.create_sheet(), items)
    wb.save(OUT / "declines_sheet.xlsx")
    pd.DataFrame([{"item_id": it["item_id"], "id": it["id"], "domain": it["domain"], "declined": it["declined"],
                   "gold_positions": ",".join(map(str, it["gold_positions"])), "judge_reply": it["judge_reply"]} for it in items]
                 ).to_csv(OUT / "declines_key.csv", index=False)
    pd.DataFrame([{"domain": d, "declines": a, "sentences": b} for d, (a, b) in counts.items()]).to_csv(OUT / "decline_counts.csv", index=False)
    print({d: f"{a}/{b} declined" for d, (a, b) in counts.items()}, f"total {sum(a for a, _ in counts.values())}")
    print(f"{len(items)} items -> {OUT / 'declines_sheet.xlsx'} (send this one only); key -> declines_key.csv (keep)")


def read_sheet(path):
    ws = load_workbook(path, data_only=True)["Items"]
    out, cur = {}, None
    for a, b in ws.iter_rows(min_row=3, max_col=2, values_only=True):
        if isinstance(a, str) and re.fullmatch(r"D\d{2}", a):
            cur = out.setdefault(a, {})
        elif cur is not None and a in (F_LABEL, F_PASSAGES, F_NOTES):
            cur[a] = "" if b is None else str(b).strip()
    return out


def analyze(sheet):
    from scipy.stats import fisher_exact
    key = pd.read_csv(OUT / "declines_key.csv", dtype={"gold_positions": str})
    lab = read_sheet(sheet)
    key["label"] = key.item_id.map(lambda i: lab.get(i, {}).get(F_LABEL, ""))
    key["passages"] = key.item_id.map(lambda i: lab.get(i, {}).get(F_PASSAGES, ""))
    missing = key[~key.label.isin(LABELS)]
    if len(missing):
        print(f"WARNING: {len(missing)} items without a valid label: {list(missing.item_id)} (left out)")
    k = key[key.label.isin(LABELS)].copy()
    k["unsupported"] = k.label.isin(["Not supported", "No factual claim"])
    k["gold_supports"] = [bool(set(re.findall(r"\d", p)) & set(str(g).split(","))) for p, g in zip(k.passages, k.gold_positions)]
    print("\n=== Labels: judge declined vs judge named a passage (counts) ===")
    print(pd.crosstab(k.declined.map({True: "declined", False: "named"}), k.label).reindex(columns=LABELS, fill_value=0).to_string())
    a, b = k[k.declined], k[~k.declined]
    tbl = [[a.unsupported.sum(), (~a.unsupported).sum()], [b.unsupported.sum(), (~b.unsupported).sum()]]
    print(f"\nNot supported or no claim: declined {a.unsupported.mean():.2f} ({a.unsupported.sum()}/{len(a)}), "
          f"named {b.unsupported.mean():.2f} ({b.unsupported.sum()}/{len(b)}); Fisher exact p = {fisher_exact(tbl)[1]:.3f} (exploratory)")
    cnt = pd.read_csv(OUT / "decline_counts.csv").set_index("domain").declines
    per = a.groupby("domain").unsupported.mean()
    print(f"Declined, weighted by each domain's number of declines: {(per * cnt[per.index]).sum() / cnt[per.index].sum():.2f}")
    print("Declined, per domain:", per.round(2).to_dict())
    print(f"Declined but Supported by the gold passage (the judge missed it): {(a.label.eq('Supported') & a.gold_supports).sum()}/{len(a)}")
    k.to_csv(OUT / "declines_labelled.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sheet", "analyze"])
    ap.add_argument("--sheet")
    a = ap.parse_args()
    make_sheet() if a.mode == "sheet" else analyze(a.sheet)


if __name__ == "__main__":
    main()
