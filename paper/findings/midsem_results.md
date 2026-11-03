# Mid-semester results — Towards Trustworthy & Explainable RAG

Setup: one fixed pipeline (180-word chunks · MiniLM embeddings · ChromaDB · top-5), five corpora, two open generators
run on Kaggle T4×2 (Qwen2.5-1.5B, Llama-3.1-8B; temperature 0; identical retrieved contexts). 1,350 questions ×
3 systems × 2 models = 8,100 generated answers, all scored. Raw tables: `experiments/results/main_*/`.

## 1. Retrieval differs enormously by domain (the confound we control for)
| Domain | Hit@5 | Recall@5 | MRR | Best variant tried |
|---|---|---|---|---|
| General — HotpotQA | 0.965 | 0.760 | 0.896 | hybrid+rerank 1.000 |
| General — SQuAD 2.0 | 0.872 | 0.872 | 0.733 | hybrid+rerank 0.992 |
| Medical — PubMedQA | 0.990 | 0.920 | 0.972 | dense (baseline) best |
| Legal — CUAD | 0.585 | 0.458 | 0.450 | hybrid 0.660; **reranker hurts: 0.315** |
| Financial — FinanceBench | 0.188 | 0.178 | 0.102 | nothing helps (≤ 0.195) |

Answer accuracy is much higher when the gold passage is retrieved (Llama-8B, vanilla RAG):
CUAD 0.76 vs 0.07, SQuAD 0.93 vs 0.22, FinanceBench 0.39 vs 0.22 → attribution must be analysed *conditioned on retrieval*.

## 2. Direction A — attribution accuracy degrades in specialised domains (conditioned on gold retrieved)
AttrAcc = fraction of answer sentences whose top-attributed passage is a gold passage. Vanilla RAG, 95% bootstrap CI.

| Domain | sim-sent (Llama-8B) | sim-sent (Qwen-1.5B) | NLI-chunk (Llama) | **evidence-set NLI (ours, Llama)** |
|---|---|---|---|---|
| SQuAD 2.0 | 0.888 [.85,.92] | 0.904 | 0.807 | 0.875 |
| HotpotQA (multi-hop) | 0.799 [.75,.84] | 0.837 | 0.644 | **0.715** (+7.1 vs chunk NLI) |
| PubMedQA | 0.714 [.68,.75] | 0.893 | 0.582 | 0.586 |
| CUAD | 0.773 [.71,.83] | 0.734 | 0.511 | 0.580 |
| FinanceBench (n=28) | 0.458 [.30,.61] | 0.426 | 0.440 | 0.387 |

Findings: (i) sentence-level similarity is the most accurate *locator* of evidence in every domain;
(ii) our evidence-set NLI beats chunk-level NLI in 5/5 domains for Qwen and 4/5 for Llama (not FinanceBench, n=28), largest on multi-hop HotpotQA (+7 to +13 points);
(iii) accuracy drops from ~0.8–0.9 (general) to ~0.7 (medical/legal) and ~0.45 (financial) even when the right passage
was retrieved. Unsupported-claim rate (NLI) rises from 0.34 (SQuAD) to 0.63 (CUAD) and 0.82 (FinanceBench) (Llama-8B).
Attribution cost: similarity ≈ 0.01–0.02 s/answer, NLI 0.04–0.31 s/answer on a T4 (vs 5–11 s generation).

## 3. External benchmark — RAGTruth QA hallucination detection (official test split, 900 responses)
| Detector | F1 |
|---|---|
| Prompted GPT-3.5-turbo (published) | 30.8 |
| SelfCheckGPT (published) | 43.7 |
| **Prompted GPT-4-turbo (published)** | **45.6** |
| Ours, zero-shot 184M NLI — chunk / window / **evidence-set** | 40.3 / 42.8 / **43.6** [37.9, 48.4] |
| Fine-tuned Llama-2-13B, supervised (published) | 68.2 |

A free, laptop-runnable zero-shot detector is statistically indistinguishable from GPT-4-turbo prompting; our NLI
variants improve monotonically (AUROC 0.72 → 0.74 → 0.76). Supervised fine-tuning remains clearly stronger.

## 4. Direction C — refusal and calibration
**Prompted refusal depends on the model.** "Reply INSUFFICIENT EVIDENCE" — refusal rate on unanswerable SQuAD /
CUAD questions: Qwen-1.5B 2% / 0%, Llama-8B 42% / 88%. Small models ignore the instruction → post-hoc gating is needed.

**Grounded refusal F1 (SQuAD 2.0; threshold fitted there):** best post-hoc gate 0.70–0.72 (both models), self-refusal
0.54 (Llama) / 0.04 (Qwen). **Transferred to CUAD** (lawyer-annotated absent clauses): 0.50–0.59 vs 0.52–0.64 with an
in-domain threshold → thresholds do not transfer cleanly.

**Conformal refusal (headline).** Target: answered-and-wrong rate ≤ α = 0.2. Threshold calibrated on general questions:
| Domain | Llama-8B risk | Qwen-1.5B risk | with 20 in-domain labels (Llama / Qwen) |
|---|---|---|---|
| SQuAD 2.0 | 0.124 ✓ | 0.140 ✓ | 0.119 / 0.151 |
| HotpotQA | **0.290 ✗** | **0.270 ✗** | 0.177 / 0.168 |
| PubMedQA | **0.335 ✗** | **0.220 ✗** | 0.168 / 0.157 |
| CUAD | **0.300 ✗** | **0.250 ✗** | 0.173 / 0.176 |
| FinanceBench | **0.333 ✗** | **0.320 ✗** | 0.147 / 0.152 |
The general-domain guarantee is violated in every specialised domain for both generators; 20 labelled in-domain
questions restore it. Calibration error of a general-domain calibrator (Llama-8B): ECE 0.08 general → 0.21 legal,
0.25 financial; 50 in-domain labels bring it to ≈ 0.08–0.10.

## 5. Baseline vs full system + ablation (paired bootstrap, per-question reliability utility)
Utility: +1 correct answer or correct refusal · 0 refusal of an answerable question · −1 wrong answer.
| Domain | Full − vanilla RAG (Llama-8B) | Full − vanilla RAG (Qwen-1.5B) | Full − closed-book |
|---|---|---|---|
| SQuAD 2.0 | **+0.68** (p<1e-4) | **+0.69** (p<1e-4) | +1.27 / +1.29 |
| CUAD | **+0.73** (p<1e-4) | **+0.68** (p<1e-4) | +1.33 / +1.24 |
| FinanceBench | **+0.48** (p<1e-4) | **+0.53** (p<1e-4) | +0.78 / +0.71 |
| HotpotQA | −0.17 (p=0.013) | +0.04 (n.s.) | +0.23 / +0.41 |
| PubMedQA | −0.11 (n.s.) | +0.06 (n.s.) | +0.34 / +0.24 |
Ablation: removing the confidence gate costs up to −0.48 (CUAD, Qwen) and is significant in 4/5 domains for Qwen and 2/5 for Llama; the
hallucination filter adds nothing on top of the gate; the trust prompt adds nothing for Qwen (it never refuses).
**Honest caveat:** with the SQuAD-fitted threshold the full system over-abstains (coverage 3–34%); it wins where wrong
answers are frequent and loses on multi-hop/medical questions whose correct answers are inferential and therefore rarely
NLI-entailed. This is exactly the calibration-transfer problem of §4 — per-domain recalibration is the next step.

## 6. Does retrieval help or hurt? (closed-book → vanilla RAG, answerable questions)
Retrieval helps 12–62% of questions and hurts only 0–4% — except PubMedQA with Qwen-1.5B (harms 9.5%).

## 7. Why answers fail — mechanism by domain (Llama-8B, vanilla RAG)
| Domain | correct | retrieval miss | evidence ignored | evidence misread | answered unanswerable |
|---|---|---|---|---|---|
| FinanceBench | 0.25 | **0.63** | 0.08 | 0.03 | – |
| CUAD | 0.32 | 0.26 | 0.05 | 0.04 | **0.33** |
| PubMedQA | 0.56 | 0.01 | **0.34** | 0.10 | – |
| HotpotQA | 0.61 | 0.02 | 0.15 | **0.21** | – |
| SQuAD 2.0 | 0.42 | 0.05 | 0.01 | 0.02 | **0.50** |
Each domain fails differently: finance fails at retrieval, medicine at drawing conclusions from the evidence,
multi-hop at combining evidence, and legal/SQuAD at knowing when to refuse.
Numeric grounding complements NLI in finance (error-detection AUROC 0.59 → 0.67 combined, Llama-8B).

## What we tried (the "journey" slide)
Llama-3.2-3B / 3.1-8B locally (270 s/query on a 2 GB GPU — infeasible) → Qwen-1.5B locally → Kaggle T4×2 ·
Gemma-4 as generator (always-on reasoning, empty answers) → used as judge · Gemini 2.5 retired → 3.5-Flash-Lite
(500 req/day free cap; running as 3rd generator) · chunk NLI → window NLI → evidence-set NLI · contradiction vs all
passages (false alarms) → vs most-relevant passage · substring answer match bug ("no" ⊂ "not") → whole-word ·
reranker (hurts legal) · entity-scoped retrieval (failed, cause identified) · RAGAS (needs strong judge) → NLI + LLM judge.

## Next (after mid-sem)
Human validation (2 annotators × 150 sentences) · Gemma judge for FinanceBench/CUAD correctness · few-shot recalibrated
full system · Kaggle "extensions" stage (evidence-removal refusal benchmark, oracle context, position bias) ·
Gemini-3.5 as third generator · paper results + discussion sections.
