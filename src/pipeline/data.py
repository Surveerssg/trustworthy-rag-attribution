"""Convert each raw dataset into one unified format.

corpus.jsonl    {doc_id, title, text, meta}
questions.jsonl {qid, domain, question, gold_answers, answer_type, answerable,
                 gold_doc_ids, gold_evidence, meta}

gold_evidence is a list of {doc_id, text, start?, end?} used to decide which chunks are "gold"
(for retrieval metrics and attribution accuracy).
"""
import json
import random
import re
from collections import defaultdict

from src.common import PROC, RAW, write_jsonl

SEED = 13


def _save(domain, corpus, questions):
    write_jsonl(PROC / domain / "corpus.jsonl", corpus)
    write_jsonl(PROC / domain / "questions.jsonl", questions)
    print(f"[{domain}] corpus={len(corpus)} docs, questions={len(questions)} "
          f"(answerable={sum(q['answerable'] for q in questions)})")


def _stratified(items, key, n, rng):
    groups = defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    for g in groups.values():
        rng.shuffle(g)
    out, keys = [], sorted(groups)
    while len(out) < n and any(groups.values()):
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


# ---------------------------------------------------------------- HotpotQA (general)
def prepare_hotpotqa(n_questions=200, n_corpus_questions=1000):
    """Distractor dev set. Corpus = all paragraphs of `n_corpus_questions` questions
    (gold + distractors of many questions => realistic retrieval difficulty)."""
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    rng = random.Random(SEED)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    pool = [ds[i] for i in idx[:n_corpus_questions]]

    corpus, seen = [], {}
    for ex in pool:
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            if title not in seen:
                seen[title] = f"hp_{len(seen)}"
                corpus.append({"doc_id": seen[title], "title": title,
                               "text": "".join(sents), "meta": {}})

    chosen = _stratified(pool, lambda e: (e["type"], e["level"]), n_questions, rng)
    qs = []
    for ex in chosen:
        ctx = dict(zip(ex["context"]["title"], ex["context"]["sentences"]))
        ev = []
        for t, sid in zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"]):
            if t in ctx and sid < len(ctx[t]):
                ev.append({"doc_id": seen[t], "text": ctx[t][sid].strip()})
        qs.append({"qid": ex["id"], "domain": "hotpotqa", "question": ex["question"],
                   "gold_answers": [ex["answer"]], "answer_type": "short", "answerable": True,
                   "gold_doc_ids": sorted({e["doc_id"] for e in ev}), "gold_evidence": ev,
                   "meta": {"type": ex["type"], "level": ex["level"]}})
    _save("hotpotqa", corpus, qs)


# ---------------------------------------------------------------- SQuAD 2.0 (refusal)
def prepare_squad2(n_answerable=250, n_unanswerable=250):
    data = json.load(open(RAW / "squad2_dev.json", encoding="utf8"))["data"]
    corpus, ans, unans = [], [], []
    for art in data:
        for pi, para in enumerate(art["paragraphs"]):
            did = f"sq_{len(corpus)}"
            corpus.append({"doc_id": did, "title": art["title"].replace("_", " "),
                           "text": para["context"], "meta": {"para": pi}})
            for qa in para["qas"]:
                if qa["is_impossible"]:
                    unans.append({"qid": qa["id"], "domain": "squad2", "question": qa["question"],
                                  "gold_answers": [], "answer_type": "short", "answerable": False,
                                  "gold_doc_ids": [did], "gold_evidence": [],
                                  "meta": {"plausible": [a["text"] for a in qa.get("plausible_answers", [])]}})
                else:
                    a = qa["answers"]
                    ans.append({"qid": qa["id"], "domain": "squad2", "question": qa["question"],
                                "gold_answers": sorted({x["text"] for x in a}), "answer_type": "short",
                                "answerable": True, "gold_doc_ids": [did],
                                "gold_evidence": [{"doc_id": did, "text": a[0]["text"],
                                                   "start": a[0]["answer_start"],
                                                   "end": a[0]["answer_start"] + len(a[0]["text"])}],
                                "meta": {}})
    rng = random.Random(SEED)
    rng.shuffle(ans)
    rng.shuffle(unans)
    _save("squad2", corpus, ans[:n_answerable] + unans[:n_unanswerable])


# ---------------------------------------------------------------- PubMedQA (medical)
def prepare_pubmedqa(n_questions=200):
    data = json.load(open(RAW / "pubmedqa_ori_pqal.json", encoding="utf8"))
    corpus, items = [], []
    for pmid, ex in data.items():
        did = f"pm_{pmid}"
        corpus.append({"doc_id": did, "title": ex["QUESTION"],  # abstract title == question in PQA-L
                       "text": " ".join(ex["CONTEXTS"]), "meta": {"labels": ex["LABELS"]}})
        items.append({"qid": pmid, "domain": "pubmedqa", "question": ex["QUESTION"],
                      "gold_answers": [ex["final_decision"]], "answer_type": "label",
                      "answerable": True, "gold_doc_ids": [did],
                      "gold_evidence": [{"doc_id": did, "text": c} for c in ex["CONTEXTS"]],
                      "meta": {"long_answer": ex["LONG_ANSWER"], "label": ex["final_decision"]}})
    # NOTE: PQA-L abstracts are stored without their conclusion section (the long answer),
    # so the answer must be inferred from results -- a genuinely hard attribution setting.
    for c in corpus:
        c["title"] = ""  # do not leak the question verbatim into the indexed text
    chosen = _stratified(items, lambda q: q["meta"]["label"], n_questions, random.Random(SEED))
    _save("pubmedqa", corpus, chosen)


# ---------------------------------------------------------------- CUAD (legal)
def _clean_contract_name(title):
    name = re.sub(r"_\d{2}_\d{2}_\d{4}", "", title)            # drop filing dates
    name = re.sub(r"-EX-[\w.]+-", " ", name)                   # drop exhibit codes
    return re.sub(r"[_\-]+", " ", name).strip().title()


def prepare_cuad(n_contracts=60, n_answerable=200, n_unanswerable=100):
    """Corpus = n_contracts full contracts. Questions are rewritten to name the contract,
    since in a pooled corpus the original 'this contract' is ambiguous. CUAD's own empty
    annotations give legally-grounded unanswerable questions (clause absent from contract)."""
    data = json.load(open(RAW / "cuad" / "CUADv1.json", encoding="utf8"))["data"]
    rng = random.Random(SEED)
    rng.shuffle(data)
    data = data[:n_contracts]
    skip = {"Document Name", "Parties", "Agreement Date", "Effective Date"}  # trivial header lookups
    corpus, ans, unans = [], [], []
    for ci, c in enumerate(data):
        did = f"cu_{ci}"
        para = c["paragraphs"][0]
        name = _clean_contract_name(c["title"])
        corpus.append({"doc_id": did, "title": name, "text": para["context"], "meta": {"raw_title": c["title"]}})
        for qa in para["qas"]:
            m = re.search(r'related to "(.+?)".*Details: (.*)$', qa["question"], re.S)
            cat, details = (m.group(1), m.group(2).strip()) if m else (qa["id"].split("__")[-1], "")
            if cat in skip:
                continue
            q = f'In the {name}, what does the contract say about "{cat}" ({details.rstrip(".")})?'
            base = {"qid": qa["id"], "domain": "cuad", "question": q, "answer_type": "span",
                    "gold_doc_ids": [did], "meta": {"category": cat, "contract": name}}
            if qa["answers"]:
                ev = [{"doc_id": did, "text": a["text"], "start": a["answer_start"],
                       "end": a["answer_start"] + len(a["text"])} for a in qa["answers"]]
                ans.append({**base, "gold_answers": [a["text"] for a in qa["answers"]],
                            "answerable": True, "gold_evidence": ev})
            else:
                unans.append({**base, "gold_answers": [], "answerable": False, "gold_evidence": []})
    chosen = _stratified(ans, lambda q: q["meta"]["category"], n_answerable, rng)
    rng.shuffle(unans)
    _save("cuad", corpus, chosen + unans[:n_unanswerable])


# ---------------------------------------------------------------- FinanceBench (financial)
def prepare_financebench():
    """Corpus = every page of the 84 full 10-K/10-Q/8-K PDFs (one doc per page, so page-level
    gold evidence is exact). All 150 open-source questions are used."""
    import fitz  # pymupdf
    rows = [json.loads(l) for l in open(RAW / "financebench_open_source.jsonl", encoding="utf8")]
    pdf_dir = RAW / "financebench_pdfs"
    corpus, missing = [], set()
    for doc in sorted({r["doc_name"] for r in rows}):
        p = pdf_dir / f"{doc}.pdf"
        if not p.exists() or p.stat().st_size < 1000:
            missing.add(doc)
            continue
        try:
            pdf = fitz.open(p)
        except Exception:
            missing.add(doc)
            continue
        company_year = doc.replace("_", " ")
        for pno, page in enumerate(pdf):
            txt = re.sub(r"[ \t]+", " ", page.get_text()).strip()
            if len(txt) > 50:
                corpus.append({"doc_id": f"fb_{doc}_p{pno}", "title": company_year, "text": txt,
                               "meta": {"doc": doc, "page": pno}})
    qs = []
    for r in rows:
        if r["doc_name"] in missing:
            continue
        ev = []
        for e in r["evidence"]:
            pno = int(e["evidence_page_num"])  # FinanceBench page numbers are 0-indexed PDF pages
            ev.append({"doc_id": f"fb_{e['doc_name']}_p{pno}", "text": e["evidence_text"]})
        qs.append({"qid": r["financebench_id"], "domain": "financebench", "question": r["question"],
                   "gold_answers": [r["answer"]], "answer_type": "free", "answerable": True,
                   "gold_doc_ids": sorted({e["doc_id"] for e in ev}), "gold_evidence": ev,
                   "meta": {"question_type": r["question_type"], "reasoning": r["question_reasoning"],
                            "doc": r["doc_name"], "justification": r["justification"]}})
    if missing:
        print(f"[financebench] WARNING missing/corrupt PDFs: {sorted(missing)}")
    _save("financebench", corpus, qs)


# ---------------------------------------------------------------- TAT-QA (second financial dataset)
def _tatqa_answer(ans, scale):
    """Gold answer string: list -> '; '-joined, number -> its value, plus the unit TAT-QA stores separately in `scale`."""
    s = "; ".join(str(a) for a in ans) if isinstance(ans, list) else str(ans)
    if scale == "percent":
        return s if s.rstrip().endswith("%") else s + "%"
    return f"{s} {scale}" if scale else s


def prepare_tatqa(n_questions=200):
    """TAT-QA dev set (hybrid table + text excerpts of annual reports). Corpus = all 278 dev contexts, one document each: the table
    linearised row by row (cells joined by ' | ') followed by its paragraphs. Gold evidence comes from TAT-QA's own labels: the
    table if the answer is drawn from the table, the `rel_paragraphs` if from the text, both for table-text. Questions are a
    stratified sample over answer source (table / text / table-text) x answer kind (computed = arithmetic or count; quoted =
    span or multi-span), so the mechanism analysis has every cell populated. Answer type 'free', as for FinanceBench."""
    data = json.load(open(RAW / "tatqa" / "tatqa_dataset_dev.json", encoding="utf8"))
    clean = lambda s: re.sub(r"\s+", " ", s or "").strip()
    corpus, items = [], []
    for ci, ctx in enumerate(data):
        did = f"tq_{ci}"
        rows = [" | ".join(clean(c) for c in row).strip() for row in ctx["table"]["table"] if any(clean(c) for c in row)]
        table = "\n".join(rows)
        text, spans = table, {"table": (0, len(table))}
        for p in sorted(ctx["paragraphs"], key=lambda p: p["order"]):
            pt = clean(p["text"])
            if pt:
                text += "\n\n"
                spans[str(p["order"])] = (len(text), len(text) + len(pt))
                text += pt
        corpus.append({"doc_id": did, "title": "", "text": text, "meta": {"uid": ctx["table"]["uid"]}})
        for q in ctx["questions"]:
            parts = (["table"] if q["answer_from"] in ("table", "table-text") else []) + \
                    ([str(o) for o in q["rel_paragraphs"]] if q["answer_from"] in ("text", "table-text") else [])
            ev = [{"doc_id": did, "text": text[spans[k][0]:spans[k][1]], "start": spans[k][0], "end": spans[k][1]}
                  for k in parts if k in spans]
            kind = "computed" if q["answer_type"] in ("arithmetic", "count") else "quoted"
            items.append({"qid": q["uid"], "domain": "tatqa", "question": clean(q["question"]),
                          "gold_answers": [_tatqa_answer(q["answer"], q["scale"])], "answer_type": "free",
                          "answerable": True, "gold_doc_ids": [did], "gold_evidence": ev,
                          "meta": {"answer_from": q["answer_from"], "tatqa_type": q["answer_type"], "kind": kind,
                                   "scale": q["scale"], "derivation": q["derivation"], "context": did}})
    items = [q for q in items if q["gold_evidence"]]
    chosen = _stratified(items, lambda q: (q["meta"]["answer_from"], q["meta"]["kind"]), n_questions, random.Random(SEED))
    _save("tatqa", corpus, chosen)


# ---------------------------------------------------------------- BioASQ (second medical dataset, snippet-level gold)
def _pubmed_abstracts(pmids, cache):
    """Title + abstract for PubMed ids via NCBI E-utilities (free, no key; <= 3 requests per second), cached in a jsonl file."""
    import time
    import urllib.request
    import xml.etree.ElementTree as ET
    have = {r["pmid"]: r for r in map(json.loads, open(cache, encoding="utf8"))} if cache.exists() else {}
    todo = sorted(set(pmids) - set(have))
    with open(cache, "a", encoding="utf8") as f:
        for i in range(0, len(todo), 200):
            url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&tool=btp_trustworthy_rag&id="
                   + ",".join(todo[i:i + 200]))
            root = ET.fromstring(urllib.request.urlopen(url, timeout=120).read())
            for art in root.iter("PubmedArticle"):
                t = art.find(".//ArticleTitle")
                r = {"pmid": art.findtext(".//PMID"), "title": "".join(t.itertext()).strip() if t is not None else "",
                     "abstract": " ".join("".join(a.itertext()).strip() for a in art.iter("AbstractText")).strip()}
                have[r["pmid"]] = r
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[bioasq] fetched {min(i + 200, len(todo))}/{len(todo)} abstracts", flush=True)
            time.sleep(0.5)
    return have


def prepare_bioasq(n_yes=50, n_no=50, n_factoid=100, n_distractor_questions=800):
    """BioASQ Task B training 13b (Krithara et al.), yes/no and factoid questions. Gold evidence = the expert snippets (one or two sentences)
    inside PubMed abstracts, so medical attribution is measured at sentence level (PubMedQA counts the whole abstract). Corpus = title + abstract
    of every document listed for the sampled questions plus the documents of `n_distractor_questions` other questions. gold_doc_ids = every
    document listed for the question, so the removal condition deletes all of them. Snippets are matched by text (offsets are kept only when
    they still point at the same text in the current abstract)."""
    raw = json.load(open(RAW / "bioasq" / "BioASQ-training13b" / "training13b.json", encoding="utf8"))["questions"]
    pmid = lambda url: url.rsplit("/", 1)[-1]
    rng = random.Random(SEED)
    pool = {"yes": [], "no": [], "factoid": []}
    for q in raw:
        if not any(s["beginSection"] == "abstract" and s["endSection"] == "abstract" for s in q["snippets"]):
            continue
        if q["type"] == "yesno":
            pool[str(q["exact_answer"]).strip().lower()].append(q)
        elif q["type"] == "factoid":
            pool["factoid"].append(q)
    for v in pool.values():
        rng.shuffle(v)
    cand = pool["yes"][:3 * n_yes] + pool["no"][:3 * n_no] + pool["factoid"][:3 * n_factoid]   # 3x: room for questions whose snippets are lost
    chosen_ids = {q["id"] for q in cand}
    others = [q for q in raw if q["id"] not in chosen_ids]
    rng.shuffle(others)
    distract = others[:n_distractor_questions]
    abstracts = _pubmed_abstracts({pmid(d) for q in cand + distract for d in q["documents"]}, RAW / "bioasq" / "pubmed_abstracts.jsonl")
    corpus = {p: {"doc_id": f"bq_{p}", "title": a["title"], "text": a["abstract"], "meta": {"pmid": p}}
              for p, a in abstracts.items() if len(a["abstract"].split()) >= 20}
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    picked = {"yes": [], "no": [], "factoid": []}
    for q in cand:
        key = "factoid" if q["type"] == "factoid" else str(q["exact_answer"]).strip().lower()
        if len(picked[key]) >= {"yes": n_yes, "no": n_no, "factoid": n_factoid}[key]:
            continue
        ev = []
        for s in q["snippets"]:
            p = pmid(s["document"])
            if s["beginSection"] != "abstract" or s["endSection"] != "abstract" or p not in corpus:
                continue
            text = corpus[p]["text"]
            if norm(s["text"]) not in norm(text):
                continue
            e = {"doc_id": f"bq_{p}", "text": s["text"].strip()}
            a, b = s.get("offsetInBeginSection"), s.get("offsetInEndSection")
            if a is not None and text[a:b].strip() == s["text"].strip():
                e.update(start=a, end=b)
            ev.append(e)
        if not ev:
            continue
        if q["type"] == "factoid":
            gold = sorted({a.strip() for a in (q["exact_answer"] if isinstance(q["exact_answer"], list) else [q["exact_answer"]])
                           for a in (a if isinstance(a, list) else [a]) if a.strip()})
            atype = "short"
        else:
            gold, atype = [key], "label"
        picked[key].append({"qid": q["id"], "domain": "bioasq", "question": q["body"].strip(), "gold_answers": gold, "answer_type": atype,
                            "answerable": True, "gold_doc_ids": sorted({f"bq_{pmid(d)}" for d in q["documents"] if pmid(d) in corpus}),
                            "gold_evidence": ev, "meta": {"type": q["type"], "ideal_answer": (q.get("ideal_answer") or [""])[0]}})
    qs = picked["yes"] + picked["no"] + picked["factoid"]
    _save("bioasq", list(corpus.values()), qs)


PREPARERS = {"hotpotqa": prepare_hotpotqa, "squad2": prepare_squad2, "pubmedqa": prepare_pubmedqa,
             "cuad": prepare_cuad, "financebench": prepare_financebench, "tatqa": prepare_tatqa, "bioasq": prepare_bioasq}
