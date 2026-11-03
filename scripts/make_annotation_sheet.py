"""Human validation set: is each answer sentence supported by the passages the model saw, and by which passage(s)?

Protocol (decided 27 Oct 2026; the team are not domain experts and have little time): an LLM (Claude) proposes a label, the supporting
passage numbers and a one-line reason with a quote for every item; two team members then VERIFY every item independently (Agree /
Disagree + their own label). The paper must describe it exactly so ("LLM-proposed, human-verified"), not as independent expert annotation.

python -m scripts.make_annotation_sheet items [--per-domain 20]   -> paper/human_eval/items.jsonl + annotation_key.csv
python -m scripts.make_annotation_sheet show --start 1 --n 8       -> prints items for labelling (no method scores are shown)
python -m scripts.make_annotation_sheet sheets                     -> verify_A.xlsx, verify_B.xlsx (needs claude_labels.csv)
Then scripts/analyze_human_labels.py reads the filled sheets.
"""
import argparse
import json
import random
import re

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

from src.attribution.methods import answer_sentences
from src.common import PAPER, RUNS, read_jsonl

OUT = PAPER / "human_eval"
DOMS = ["squad2", "hotpotqa", "pubmedqa", "cuad", "financebench"]
# Stage 2 runs (current pipeline): natural retrieval and gold-guaranteed (oracle) contexts, a small and a mid-size model
SOURCES = [(run, s) for run in ("v2_llama8b", "v2_qwen1.5b") for s in ("vanilla_rag", "oracle_vanilla")]
LABELS = ["Supported", "Partially supported", "Not supported"]
VERDICTS = ["Agree", "Disagree", "Unsure"]
DOM_NAME = {"squad2": "General", "hotpotqa": "General (multi-hop)", "pubmedqa": "Medical", "cuad": "Legal", "financebench": "Financial"}
FONT = "Arial"
YELLOW = PatternFill("solid", start_color="FFF2CC")
GREY = PatternFill("solid", start_color="EDEDED")
HEAD = PatternFill("solid", start_color="1F3864")
THIN = Border(*(Side(style="thin", color="BFBFBF"),) * 4)


def collect(per_domain, seed=0):
    """Sentences of answered questions, balanced over the four sources, at most 2 sentences per answer."""
    rng = random.Random(seed)
    items = []
    for d in DOMS:
        groups = {}
        for run, system in SOURCES:
            p = RUNS / run / f"{d}__{system}.jsonl"
            if not p.exists():
                continue
            for r in read_jsonl(p):
                if r["refused"] or not r["retrieved"] or not r["answerable"]:
                    continue
                gold = set(r.get("gold_chunk_ids") or [])
                gold_pos = [i + 1 for i, c in enumerate(r["retrieved"]) if c["chunk_id"] in gold]
                for i, s in enumerate(answer_sentences(r["answer"])):
                    if len(s["text"].split()) >= 4:  # skip fragments like "Yes."
                        groups.setdefault((run, system), []).append(
                            {"run": run, "system": system, "domain": d, "qid": r["qid"], "sent_idx": i, "sentence": s["text"],
                             "question": r["question"], "answer": r["answer"], "passages": [c["text"] for c in r["retrieved"]],
                             "gold_positions": gold_pos})
        for g in groups.values():
            rng.shuffle(g)
        seen, chosen = {}, []
        while len(chosen) < per_domain and any(groups.values()):
            for g in sorted(groups):  # round-robin over the sources
                while groups[g]:
                    it = groups[g].pop()
                    key = (it["run"], it["system"], it["qid"])
                    if seen.get(key, 0) < 2:
                        seen[key] = seen.get(key, 0) + 1
                        chosen.append(it)
                        break
                if len(chosen) == per_domain:
                    break
        items += chosen
    rng.shuffle(items)
    for n, it in enumerate(items, 1):
        it["item_id"] = f"S{n:03d}"
    return items


def show(start, n):
    items = read_jsonl(OUT / "items.jsonl")[start - 1:start - 1 + n]
    for it in items:
        print(f"\n######## {it['item_id']} | {it['domain']}\nQ: {it['question']}\nANSWER: {it['answer']}\nSENTENCE: {it['sentence']}")
        for k, p in enumerate(it["passages"], 1):
            print(f"  [{k}] " + re.sub(r"\s*\n\s*", " | ", p.strip()))   # table rows on one line


def guidelines(ws, who):
    ws.title = "Guidelines"
    rows = [
        (f"Verification of proposed sentence-support labels: Verifier {who}", True),
        ("Work alone: do not discuss items with the other verifier until both sheets are finished.", False),
        ("", False),
        ("WHAT YOU SEE. The 'Items' tab is one long column of cards; scroll down only. Each card (dark blue header) is one sentence from an AI "
         "system's answer (blue 'SENTENCE TO JUDGE') and, at the bottom of the card, the 5 passages the system was given. An AI assistant (Claude) "
         "has PROPOSED a label, the supporting passage number(s) and a reason with a short quote (grey rows). Your job is to CHECK that proposal "
         "against the passages. Judge only that sentence, only against those 5 passages; ignore your own knowledge: a true fact that the passages "
         "do not state is NOT supported. The top row counts how many items you have finished.", False),
        ("", False),
        ("FILL IN ONLY THE YELLOW CELLS (the dropdowns appear when you click the cell):", True),
        ("  YOUR VERDICT: Agree (the proposed label AND passages are right) / Disagree / Unsure (cannot tell, e.g. needs expert knowledge)", False),
        ("  CORRECT LABEL and CORRECT PASSAGES: only when you Disagree. NOTES: optional (why).", False),
        ("", False),
        ("THE LABELS", True),
        ("  Supported: everything the sentence claims can be verified from the passages (one passage, or several together).", False),
        ("  Partially supported: some claims are verifiable, at least one is not (e.g. right name, wrong year).", False),
        ("  Not supported: the main claim is absent from, or contradicted by, the passages.", False),
        ("", False),
        ("RULES OF THUMB", True),
        ("  - Paraphrase is fine. Rounding is fine (1,577 is about 1.58 billion). Unit changes are fine (thousands vs millions) if consistent.", False),
        ("  - Combining two passages is fine (multi-hop): 'X was born in the city where Y is located' is Supported if one passage "
         "gives X's birthplace and another gives Y's location.", False),
        ("  - Yes/No sentences: judge the stated conclusion. 'Yes, the drug reduced mortality' needs a passage that shows it.", False),
        ("  - Numbers must match. A calculation (e.g. a difference of two figures) is Supported only if both inputs appear in the passages.", False),
        ("  - Generic filler ('Based on the passages,...') with no factual claim: Supported.", False),
        ("  - Do not just click Agree: open the quoted passage and check that the quote is really there and really says that.", False),
        ("", False),
        ("Expected time: about 30-45 s per item. Save as verify_<A|B>_<yourname>.xlsx and send it back.", False),
    ]
    for i, (t, bold) in enumerate(rows, 1):
        c = ws.cell(row=i, column=1, value=t)
        c.font = Font(name=FONT, bold=bold, size=13 if i == 1 else 10)
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 140


LABEL_W, TEXT_W = 26, 120          # column widths (A = field name, B = content): only vertical scrolling is needed
SENT = PatternFill("solid", start_color="DDEBF7")
# field names in column A that the verifier fills in (scripts/analyze_human_labels.py reads them back by these names)
F_VERDICT, F_LABEL, F_PASSAGES, F_NOTES = "YOUR VERDICT", "CORRECT LABEL (if Disagree)", "CORRECT PASSAGES (if Disagree)", "NOTES (optional)"


def _height(text, width=TEXT_W):
    """Row height (points) that shows all wrapped text: Excel does not auto-fit rows written by openpyxl."""
    lines = sum(max(1, -(-len(part) // int(width * 1.05))) for part in str(text or "").split("\n"))
    return min(409, max(16, 13.5 * lines + 4))


def items_sheet(ws, items, labels):
    """One item per vertical card: header, question, answer, the sentence, the proposal, the yellow cells to fill, then the passages.
    Two columns only, so the sheet scrolls straight down; the top row (progress counter) stays frozen."""
    ws.title = "Items"
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = LABEL_W, TEXT_W
    ws.sheet_view.zoomScale = 110
    n = len(items)
    top = ws.cell(row=1, column=1, value="PROGRESS")
    top.font = Font(name=FONT, bold=True, color="FFFFFF"); top.fill = HEAD
    done = ws.cell(row=1, column=2, value=f'=COUNTIF(B:B,"Agree")+COUNTIF(B:B,"Disagree")+COUNTIF(B:B,"Unsure")&" of {n} items done   '
                                           f'(fill only the yellow cells; scroll down)"')
    done.font = Font(name=FONT, bold=True, color="FFFFFF"); done.fill = HEAD
    ws.row_dimensions[1].height = 20
    ws.freeze_panes = "A2"
    dv_v = DataValidation(type="list", formula1='"' + ",".join(VERDICTS) + '"', allow_blank=True, showErrorMessage=True,
                          error="Choose Agree, Disagree or Unsure")
    dv_l = DataValidation(type="list", formula1='"' + ",".join(LABELS) + '"', allow_blank=True, showErrorMessage=True,
                          error="Choose a label from the list")
    ws.add_data_validation(dv_v); ws.add_data_validation(dv_l)

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
        lab = labels[it["item_id"]]
        h1 = ws.cell(row=r, column=1, value=it["item_id"])
        h2 = ws.cell(row=r, column=2, value=f"{DOM_NAME[it['domain']]}   ·   item {k} of {n}")
        for c in (h1, h2):
            c.font = Font(name=FONT, bold=True, color="FFFFFF", size=11); c.fill = HEAD
        ws.row_dimensions[r].height = 20
        r += 1
        put(r, "QUESTION", it["question"]); r += 1
        put(r, "FULL ANSWER", it["answer"]); r += 1
        put(r, "SENTENCE TO JUDGE", it["sentence"], fill=SENT, bold=True, size=11); r += 1
        put(r, "PROPOSED LABEL", lab["label"], fill=GREY, bold=True); r += 1
        put(r, "PROPOSED PASSAGES", str(lab["passages"]) or "(none)", fill=GREY); r += 1
        put(r, "PROPOSED REASON", lab["why"], fill=GREY); r += 1
        dv_v.add(put(r, F_VERDICT, None, fill=YELLOW).coordinate); r += 1
        dv_l.add(put(r, F_LABEL, None, fill=YELLOW).coordinate); r += 1
        put(r, F_PASSAGES, None, fill=YELLOW); r += 1
        put(r, F_NOTES, None, fill=YELLOW); r += 1
        for j, p in enumerate((it["passages"] + [""] * 5)[:5], 1):
            put(r, f"PASSAGE {j}", re.sub(r"\s*\n\s*", " | ", p.strip())[:3000], size=9); r += 1   # table rows on one line
        r += 1   # blank row between cards


def load_labels():
    df = pd.read_csv(OUT / "claude_labels.csv", dtype=str).fillna("")
    bad = set(df.label) - set(LABELS)
    assert not bad, f"unknown labels {bad}"
    return {r.item_id: {"label": r.label, "passages": r.passages, "why": r.why} for r in df.itertuples()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["items", "show", "sheets"])
    ap.add_argument("--per-domain", type=int, default=20)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tag", default="", help="suffix for the sheet file names, e.g. _v2 (when an older copy is open in Excel)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "items":
        items = collect(a.per_domain)
        with open(OUT / "items.jsonl", "w", encoding="utf8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        pd.DataFrame([{k: it[k] for k in ("item_id", "run", "system", "domain", "qid", "sent_idx", "sentence")} |
                      {"gold_positions": " ".join(map(str, it["gold_positions"]))} for it in items]) \
            .to_csv(OUT / "annotation_key.csv", index=False)
        print(len(items), "items"); print(pd.DataFrame(items).groupby(["domain", "run", "system"]).size().to_string())
    elif a.cmd == "show":
        show(a.start, a.n)
    else:
        items, labels = read_jsonl(OUT / "items.jsonl"), load_labels()
        missing = [it["item_id"] for it in items if it["item_id"] not in labels]
        assert not missing, f"no proposed label for {missing[:10]}"
        for who in ("A", "B"):
            wb = Workbook()
            guidelines(wb.active, who)
            items_sheet(wb.create_sheet(), items, labels)
            wb.active = 0  # open on the guidelines
            wb.save(OUT / f"verify_{who}{a.tag}.xlsx")
        print(f"written verify_A{a.tag}.xlsx, verify_B{a.tag}.xlsx:", len(items), "items")


if __name__ == "__main__":
    main()
