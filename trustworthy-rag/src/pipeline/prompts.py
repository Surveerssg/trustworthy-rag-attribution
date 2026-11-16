"""Prompt templates. The same templates are used for every domain; only a short
answer-FORMAT hint differs by answer type (short / yes-no label / clause span / financial value),
which is task format, not domain tuning."""

REFUSAL = "INSUFFICIENT EVIDENCE"

FORMAT_HINT = {
    "short": "Answer in 1-2 sentences. State the direct answer in the first sentence.",
    "label": "Begin your answer with exactly one of: Yes, No, or Maybe. Then justify it in 1-2 sentences.",
    "span": "Answer in 1-3 sentences, quoting or closely paraphrasing the relevant contract language.",
    "free": "Answer in 1-3 sentences. Give the exact figure(s) with units where applicable.",
}


def format_passages(chunks):
    return "\n\n".join(f"[{i + 1}] ({c['title']}) {c['text']}" if c.get("title") else f"[{i + 1}] {c['text']}"
                       for i, c in enumerate(chunks))


def closed_book(q, answer_type):
    return f"Question: {q}\n\n{FORMAT_HINT[answer_type]}\nAnswer:"


def vanilla_rag(q, chunks, answer_type):
    return (f"Use the following passages to answer the question.\n\n{format_passages(chunks)}\n\n"
            f"Question: {q}\n\n{FORMAT_HINT[answer_type]}\nAnswer:")


def trust_rag(q, chunks, answer_type):
    return ("Answer the question using ONLY the information in the passages below. "
            "After each sentence, cite the supporting passage number(s) in brackets, e.g. [2]. "
            f"If the passages do not contain enough information to answer, reply exactly: {REFUSAL}\n\n"
            f"{format_passages(chunks)}\n\nQuestion: {q}\n\n{FORMAT_HINT[answer_type]}\nAnswer:")


PARAPHRASE = ("Rewrite the following question so that it asks for exactly the same information "
              "using different wording. Keep all names, numbers and dates. Output only the rewritten question.\n\n"
              "Question: {q}\nRewritten question:")


def is_refusal(text):
    t = (text or "").strip().upper()
    return REFUSAL in t or t.startswith("I DON'T KNOW") or t.startswith("I CANNOT ANSWER") \
        or "NOT ENOUGH INFORMATION" in t or "DO NOT CONTAIN ENOUGH" in t or "DOES NOT CONTAIN ENOUGH" in t \
        or "DO NOT PROVIDE ENOUGH" in t or "NOT MENTIONED IN THE PASSAGES" in t


# ---------------------------------------------------------------- implicit refusals
# Models often decline in their own words ("The passages do not mention X...") instead of the exact phrase.
# Manually checked: 25/25 random matches were genuine statements that the sources lack the information.
import re as _re
SRC = r"(passages?|context|provided (text|information|documents?)|documents?|sources?|contract|agreement|report|filing|text|information provided|given information|excerpts?)"
_IMPLICIT = _re.compile(
    SRC + r"[^.]{0,40}?\b(do(es)?n't|do(es)? not|did not|fails? to)\s+(explicitly\s+|directly\s+|specifically\s+|clearly\s+)?(include|contain|mention|provide|state|specify|discuss|address|cover|describe)"
    r"|\bnot\s+(explicitly\s+|directly\s+|specifically\s+)?(stated|mentioned|provided|specified|included|given|discussed|found)\s+(in|within)\s+(the\s+)?" + SRC +
    r"|\bno\s+(information|mention|data|details?)\s+(about|of|on|regarding|is|are)"
    r"|\b(cannot|can't|unable to|not possible to)\s+(be\s+)?(determine|determined|answer|answered|calculate|calculated|find|found)"
    r"|\binsufficient\s+(information|evidence)", _re.I)


def is_implicit_refusal(text):
    """True if the answer OPENS by saying the sources do not contain the needed information."""
    first = _re.split(r"(?<=[.!?])\s+", (text or "").strip())[0]
    return bool(_IMPLICIT.search(first))


def refusal_kind(text):
    return "explicit" if is_refusal(text) else "implicit" if is_implicit_refusal(text) else None
