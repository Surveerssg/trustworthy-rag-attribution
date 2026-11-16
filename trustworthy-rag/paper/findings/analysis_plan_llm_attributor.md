# LLM-as-attributor baseline: analysis plan (Part F)

Written 31 Oct 2026, BEFORE any judge output exists. Do not change the hypotheses, prompt, tests or thresholds after the results are seen; anything
added later is labelled exploratory. Scripts: `scripts/llm_attribute.py` (Kaggle), `scripts/analyze_llm_attributor.py` (laptop, CPU only).

## Why
Reviewers will ask: "why not just ask an LLM which passage supports each sentence?" The paper's number-aware scorer is free and model-free; its value
depends on how close it comes to an LLM judge. The judge is also a stronger baseline for attribution in every domain.

## Design
- **Records:** every `oracle_vanilla` answer (gold passage guaranteed) of the seven generators that answered, was not a refusal and has a gold chunk
  in its context: 6,059 answers, 12,008 sentences (FinanceBench 1,945; TAT-QA 2,517; CUAD 1,922; PubMedQA 2,494; HotpotQA 1,598; SQuAD 1,532).
  Answers, contexts and sentence splits are exactly those already scored (`attribution.sentences`), so every method is compared on the same sentences.
- **Judges:** Qwen2.5-32B-Instruct (strong) and Llama-3.1-8B-Instruct (cheap), 4-bit, Ollama, temperature 0, at most 4 output tokens.
- **Prompt (fixed):**
  ```
  Below are numbered passages and one sentence taken from an answer. Which passage supports the sentence?
  Reply with the passage number only.

  [1] (title) text
  ...
  [5] (title) text

  Sentence: <sentence>
  Passage number:
  ```
  Passages are formatted exactly as in the generation prompts (`src.pipeline.prompts.format_passages`).
- **Score:** the probability of each passage number in the first output token (top-10 log-probs); the attributed passage is the highest. If no
  passage number is among the first token's alternatives, the first digit 1-5 in the reply is used; if there is none, the sentence counts as a miss
  (reported as the format-failure rate).
- **Number + judge:** number matching first (as in the paper), the judge's probabilities break ties.
- **Attribution accuracy** as everywhere in the paper: share of a record's sentences whose top passage is a gold chunk, averaged over records.

## Hypotheses (primary)
- **L1:** the 32B judge beats `sim_sent` in each of the six domains (predicted sign: positive).
- **L2 (cost-effectiveness):** on FinanceBench and on TAT-QA, the number-aware scorer (`num_sim`, no model) is non-inferior to the 32B judge with a
  margin of 0.05: supported if the upper end of the 95% CI of (judge - num_sim) is below 0.05.
- **L3:** on FinanceBench and on TAT-QA, number + judge beats the judge alone (predicted sign: positive).
- **L4:** the 32B judge beats the 8B judge, pooled over the six domains.

## Tests
- Cluster bootstrap over questions (10,000 resamples; all generators' records of a question move together), paired differences.
- L1 (6 tests), L3 (2 tests), L4 (1 test): Holm correction within each hypothesis; supported only with the predicted sign, |difference| >= 0.05 and
  a corrected p < 0.05. L2 is decided by its confidence bound only.
- Reported, not tested: the judge vs `nli_set`, the 20-label mixture and `num_mixture`; per-generator results; the judge's cost (GPU seconds per
  sentence); results without the 32B judge's own answers (`v2_qwen32b`, `tq_qwen32b`), since a model may prefer its own text.

## Amendment 1 (31 Oct 2026, after inspecting the Llama-8B reply format only, BEFORE any accuracy was computed)
Llama-3.1-8B answers `[3]` rather than `3` (11,000+ of 12,008 replies), so its first output token is `[` and the passage numbers appear only as
low-probability alternatives (median total probability 0.05). The first-token rule above then reads those alternatives instead of the model's
answer: it disagrees with the number the model actually wrote for 1,138 sentences (9.5%). Change: the **primary pick is the passage number written
in the reply** (first digit 1-5); the first-token probabilities are used as the judge's scores only when they hold at least half of the probability
mass (otherwise the written number gets probability 1). This only affects how a reply is read, not the prompt, models, records or tests. The
original literal rule is still reported next to it for every hypothesis (`judge_literal`). The Qwen2.5-32B run uses the same reading.

## Known limits
- One prompt, one format; no prompt tuning (tuning on these data would make the comparison unfair to the cheap methods).
- The judge sees the five passages of the oracle context only (as every other method does).
- Gold labels are automatic (one evidence page for FinanceBench), so every accuracy is a lower bound for every method alike.
