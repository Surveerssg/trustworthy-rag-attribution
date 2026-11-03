"""Human-readable explanations for one answer, built only from the signals the system actually uses
(so the explanation is faithful to the decision, not a post-hoc story).

explain(record) -> {
  "sentences": [{text, source, quote, support, status}],     # per answer sentence
  "reasons":   [{code, severity, message}],                   # why confidence is high/low, why refused
  "decision":  "answer" | "refuse",
}
record = one scored row (scripts/score_run.py output) or the dict returned in the dashboard's live mode.
"""
import re

import numpy as np

from src.attribution.fusion import fuse_weight, fused
from src.evaluation import metrics as M
from src.pipeline.chunking import sentence_spans
from src.pipeline.index import embed

# attribution (WHICH passage a sentence comes from) = w * similarity + (1 - w) * NLI, w calibrated per domain from 20 labelled
# questions (configs/attribution_weights.json, scripts/fit_attribution_weights.py); support/verification below stays NLI-based
T_SUPPORT = 0.5       # NLI support threshold
T_SIM = 0.55          # similarity threshold used together with NLI
T_CONTRA = 0.8        # contradiction threshold (vs the sentence's most relevant passage)
T_RETRIEVAL = 0.45    # best question-passage cosine below which retrieval is flagged as weak


def _quote(sentence, passage):
    """The single sentence inside the passage that best matches the answer sentence (locating = similarity)."""
    sents = [passage[a:b].strip() for a, b in sentence_spans(passage)] or [passage]
    sims = embed(sents) @ embed([sentence])[0]
    j = int(np.argmax(sims))
    return sents[j], float(sims[j])


def _numbers_missing(answer, passages):
    nums = [x for x in M.extract_numbers(answer) if not (1900 <= abs(x) <= 2100 and float(x).is_integer())]
    ev = M.extract_numbers(" ".join(passages))
    missing = []
    for n in nums:
        if not any(np.isclose(abs(n) * s, abs(e), rtol=0.005, atol=0.006) for e in ev for s in (1, 1e3, 1e-3, 1e6, 1e-6)):
            missing.append(n)
    return missing


def explain(rec, gate_threshold=None, confidence=None):
    chunks = rec.get("retrieved", [])
    attr = rec.get("attribution") or {}
    sc, sents = attr.get("scores", {}), attr.get("sentences", [])
    reasons, out_sents = [], []

    from src.pipeline.prompts import is_implicit_refusal
    if rec.get("refused"):
        reasons.append({"code": "MODEL_REFUSED", "severity": "high",
                        "message": "The model itself said the passages do not contain enough information."})
    elif is_implicit_refusal(rec.get("answer", "")):
        reasons.append({"code": "ANSWER_ADMITS_MISSING_EVIDENCE", "severity": "high",
                        "message": "The answer itself begins by saying the sources do not contain the needed information."})

    scores = [c["score"] for c in chunks]
    if chunks and max(scores) < T_RETRIEVAL:
        reasons.append({"code": "WEAK_RETRIEVAL", "severity": "high",
                        "message": f"No retrieved passage closely matches the question (best similarity {max(scores):.2f})."})

    if sents and "nli_set" in sc and "sim_sent" in sc:
        E, S = np.asarray(sc["nli_set"]), np.asarray(sc["sim_sent"])
        C = np.asarray(sc.get("nli_window_contra", np.zeros_like(E)))
        attr_scores = fused(S, E, fuse_weight(rec.get("domain")))
        for i, s in enumerate(sents):
            # hybrid attribution: the model's own citation when it gives one (most accurate for capable models),
            # otherwise the domain-calibrated similarity+NLI score; among several cited passages take the best-supported one
            cites = [c - 1 for c in s.get("cites", []) if 1 <= c <= len(chunks)]
            j = max(cites, key=lambda c: attr_scores[i, c]) if cites else int(attr_scores[i].argmax())
            quote, qsim = _quote(s["text"], chunks[j]["text"])
            sup = float(E[i].max())
            rel = int(S[i].argmax())
            if C[i, rel] >= T_CONTRA and sup < T_SUPPORT:
                status = "contradicted"
                reasons.append({"code": "CONTRADICTION", "severity": "high",
                                "message": f"Passage [{rel + 1}] contradicts sentence {i + 1}."})
            elif sup >= T_SUPPORT:
                status = "supported"
            elif S[i].max() >= T_SIM:
                status = "weak"      # topically matched but not logically proven (often an inference)
            else:
                status = "unsupported"
                reasons.append({"code": "UNSUPPORTED_SENTENCE", "severity": "high",
                                "message": f"Sentence {i + 1} is not supported by any retrieved passage."})
            out_sents.append({"text": s["text"], "source": j + 1, "quote": quote, "quote_similarity": round(qsim, 3),
                              "support": round(sup, 3), "status": status, "model_cited": s.get("cites", [])})
            if s.get("cites") and (j + 1) not in s["cites"] and sup >= T_SUPPORT:
                reasons.append({"code": "CITATION_MISMATCH", "severity": "low",
                                "message": f"Sentence {i + 1} cites {s['cites']} but the evidence is in passage [{j + 1}]."})
        n_sup = sum(x["status"] == "supported" for x in out_sents)
        if out_sents:
            share = n_sup / len(out_sents)
            reasons.append({"code": "SUPPORT_COVERAGE", "severity": "info" if share >= 0.75 else "medium",
                            "message": f"{n_sup} of {len(out_sents)} sentences are directly supported by the sources."})

    missing = _numbers_missing(rec.get("answer", ""), [c["text"] for c in chunks])
    if missing:
        shown = ", ".join(f"{x:g}" for x in missing[:3])
        reasons.append({"code": "NUMBER_NOT_IN_SOURCE", "severity": "high",
                        "message": f"Number(s) {shown} in the answer do not appear in the retrieved passages."})

    decision = "refuse" if rec.get("refused") or is_implicit_refusal(rec.get("answer", "")) else "answer"
    if confidence is not None and gate_threshold is not None and decision == "answer":
        if confidence < gate_threshold:
            decision = "refuse"
            reasons.insert(0, {"code": "LOW_CONFIDENCE", "severity": "high",
                               "message": f"Confidence {confidence:.2f} is below this domain's threshold {gate_threshold:.2f}."})
    return {"sentences": out_sents, "reasons": reasons, "decision": decision}


def render_text(exp, answer=None):
    """Plain-text explanation report (used by the dashboard export and case studies)."""
    lines = [f"Decision: {exp['decision'].upper()}"]
    if answer:
        clean = re.sub(r"\s+", " ", answer).strip()   # not inside the f-string: SyntaxError before Python 3.12
        lines.append(f"Answer: {clean}")
    for i, s in enumerate(exp["sentences"], 1):
        lines.append(f"  S{i} [{s['status']}, support {s['support']:.2f}] {s['text']}")
        lines.append(f"      evidence [{s['source']}]: \"{s['quote']}\"")
    if exp["reasons"]:
        lines.append("Why:")
        lines += [f"  - ({r['severity']}) {r['message']}" for r in exp["reasons"]]
    return "\n".join(lines)
