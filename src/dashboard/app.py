"""Explainability dashboard.  streamlit run src/dashboard/app.py

Browse mode : inspect any scored experiment row (answer, per-sentence support, attribution heatmap per method,
              retrieved chunks with gold markers, confidence signals, refusal decision).
Live mode   : ask a question against any domain index through the full trustworthy pipeline.
"""
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import streamlit as st

from src.attribution.explain import explain, render_text
from src.attribution.fusion import fuse_weight, fused_matrix
from src.common import RUNS, read_jsonl

st.set_page_config(page_title="Trustworthy RAG", layout="wide")
GATE_T = 0.45   # default; overwritten by the slider
# "fused" = domain-calibrated mix of similarity and NLI (the default attribution); the others are the single methods
METHODS = ["fused", "sim_sent", "nli_set", "nli_window", "nli_chunk", "sim_chunk", "self_cite"]


def trust_score(f, refused):
    return 0.0 if refused else (f["ret_top1"] + f.get("nli_cov_mean", 0) + 1 - f.get("contra_max", 1)) / 3


def color(v):
    # red (unsupported) -> green (supported)
    r, g = int(220 - 150 * v), int(90 + 120 * v)
    return f"rgba({r},{g},90,0.28)"


def render(rec, gate_t, method):
    attr = rec.get("attribution") or {}
    sents, scores = attr.get("sentences", []), dict(attr.get("scores", {}))
    if "sim_sent" in scores and "nli_set" in scores:
        scores["fused"] = fused_matrix(scores, rec.get("domain")).tolist()
        st.sidebar.caption(f"fused attribution weight for {rec.get('domain')}: {fuse_weight(rec.get('domain')):.1f} on similarity")
    f = rec.get("features", {})
    ts = trust_score(f, rec["refused"])
    exp = explain(rec, gate_threshold=gate_t, confidence=ts)
    gated = exp["decision"] == "refuse"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trust score", f"{ts:.2f}")
    c2.metric("Decision", "REFUSE" if gated else "ANSWER")
    c3.metric("Retrieval top-1 cosine", f"{f.get('ret_top1', 0):.2f}")
    c4.metric("Evidence coverage (NLI)", f"{f.get('nli_cov_mean', 0):.2f}")
    if "correct" in rec:
        st.caption(f"Gold: {rec.get('gold_answers')} · answerable={rec['answerable']} · auto-judged correct={rec['correct']}")

    st.subheader("Why this decision")
    tag = {"high": "🔴", "medium": "🟠", "low": "🟡", "info": "🟢"}
    for r in exp["reasons"] or [{"severity": "info", "message": "No problems detected: every sentence is supported by a source."}]:
        st.markdown(f"{tag.get(r['severity'], '•')} {html.escape(r['message'])}")

    if exp["sentences"]:
        st.subheader("Each claim and its evidence")
        st.dataframe(pd.DataFrame([{"#": i + 1, "status": s["status"], "support": s["support"], "claim": s["text"],
                                    "source": f"[{s['source']}]", "evidence quote": s["quote"]}
                                   for i, s in enumerate(exp["sentences"])]), use_container_width=True, hide_index=True)
    st.download_button("Download explanation report", render_text(exp, rec.get("answer")), file_name="explanation.txt")

    st.subheader("Answer (sentences coloured by best NLI support)")
    if sents and method in scores:
        S = np.asarray(scores["nli_window"] if "nli_window" in scores else scores[method])
        parts = []
        for i, s in enumerate(sents):
            best = int(np.asarray(scores[method])[i].argmax()) + 1
            parts.append(f"<span style='background:{color(S[i].max())};padding:2px 4px;border-radius:4px'>"
                         f"{html.escape(s['text'])} <sup>[{best}]</sup></span>")
        st.markdown(" ".join(parts), unsafe_allow_html=True)
    else:
        st.write(rec["answer"])

    if scores.get(method):
        st.subheader(f"Attribution matrix — {method}")
        df = pd.DataFrame(scores[method], columns=[f"[{i + 1}]" for i in range(len(rec["retrieved"]))],
                          index=[f"S{i + 1}: {s['text'][:60]}" for i, s in enumerate(sents)])
        st.dataframe(df.style.background_gradient(cmap="Greens", vmin=0, vmax=1).format("{:.2f}"), use_container_width=True)

    st.subheader("Retrieved evidence")
    gold = set(rec.get("gold_chunk_ids", []))
    for i, c in enumerate(rec["retrieved"]):
        tag = " ✅ gold" if c["chunk_id"] in gold else ""
        with st.expander(f"[{i + 1}] {c.get('title', '')[:60]}  ·  cos={c['score']:.2f}{tag}"):
            st.write(c["text"])
    with st.expander("All confidence features"):
        st.json(f)


mode = st.sidebar.radio("Mode", ["Browse experiments", "Live question"])
gate_t = st.sidebar.slider("Refusal threshold (trust score)", 0.0, 1.0, GATE_T, 0.01)
method = st.sidebar.selectbox("Attribution method", METHODS)

if mode == "Browse experiments":
    runs = sorted(p.name for p in RUNS.iterdir() if p.is_dir()) if RUNS.exists() else []
    run = st.sidebar.selectbox("Run", runs)
    files = sorted(p.name for p in (RUNS / run).glob("*.scored.jsonl")) if run else []
    fname = st.sidebar.selectbox("Domain / system", files)
    if fname:
        rows = read_jsonl(RUNS / run / fname)
        only_wrong = st.sidebar.checkbox("Only incorrect answers")
        if only_wrong:
            rows = [r for r in rows if not r.get("correct")]
        i = st.sidebar.number_input("Row", 0, max(0, len(rows) - 1), 0)
        if rows:
            st.title(rows[i]["question"])
            render(rows[i], gate_t, method)
else:
    from scripts.score_run import score_record
    from src.pipeline import prompts as P
    from src.pipeline.index import Retriever
    from src.pipeline.llm import LLM

    dom = st.sidebar.selectbox("Domain index", ["hotpotqa", "squad2", "pubmedqa", "cuad", "financebench"])
    model = st.sidebar.text_input("Ollama model", "qwen2.5:1.5b")
    q = st.text_input("Ask a question")
    if q:
        with st.spinner("retrieving + generating + attributing ..."):
            hits = Retriever(dom).retrieve_many([q], k=5)[0]
            ans = LLM(model=model).generate(P.trust_rag(q, hits, "short"))["text"]
            rec = score_record({"qid": "live", "domain": dom, "system": "trust_rag", "question": q, "answer_type": "short",
                                "answerable": False, "gold_answers": [], "gold_chunk_ids": [], "retrieved": hits,
                                "answer": ans, "refused": P.is_refusal(ans), "gen_seconds": 0, "cached": False})
            rec.pop("correct", None)
        render(rec, gate_t, method)
