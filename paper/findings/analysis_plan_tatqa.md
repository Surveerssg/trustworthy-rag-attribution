# Second financial dataset (TAT-QA) and two new model families: analysis plan

Written 28 Oct 2026, BEFORE any TAT-QA, Phi-4 or Mistral-Small answer exists. Do not change the hypotheses, tests or thresholds below
after the results are seen; anything added later is labelled exploratory. Script: `scripts/analyze_tatqa.py` (CPU only).

## Why
The headline result rests on ONE financial dataset: with the gold passage guaranteed, `sim_sent` attribution is .32-.36 on FinanceBench for
every model (general text .84-.93) and `nli_set` .51-.55. A reviewer can say this is a FinanceBench quirk (long, noisy PDF pages), and the
paper does not say WHY it happens. TAT-QA answers both: it is a different financial dataset (hybrid table + text excerpts of annual reports),
and every question is labelled by its answer source (table / text / table-text) and kind (computed = arithmetic or count; quoted = span or
multi-span). Two new model families (Phi-4 14B, Mistral-Small 24B) check that the result is not specific to Qwen, Llama and Gemma.

## Design
- **Data:** TAT-QA dev, all 278 contexts as one pooled corpus (one document each: the table linearised row by row with cells joined by
  ` | `, then its paragraphs). Same chunker (180 words), embedder (MiniLM), k = 5 and prompts as every other domain; answer type `free`
  (as FinanceBench), so correctness is the same numeric-match-then-NLI-judge rule.
- **Questions:** 200, stratified over answer source x kind (37 per cell; text-computed has only 16 in the whole dev set). Seed 13.
- **Gold evidence:** TAT-QA's own labels (the table for table answers, `rel_paragraphs` for text answers, both for table-text), mapped to
  chunks by character overlap. 200/200 questions have a gold chunk (1.24 per question).
- **Conditions:** TAT-QA questions do not name the company, so natural retrieval over the pooled corpus is ill-posed (reported, not used).
  Systems: `oracle_vanilla` (gold guaranteed: accuracy and attribution), `oracle_trust` (answerable side of the refusal pair) and
  `removed_trust` (gold document deleted: should refuse). Contexts are precomputed once (`data/plans/tatqa.jsonl`) and identical for all models.
- **Models:** Qwen2.5-1.5B, Llama-3.1-8B, Qwen2.5-14B, Phi-4-14B, Mistral-Small-24B, Gemma-3-27B, Qwen2.5-32B (all 4-bit, Ollama, Kaggle).
  Phi-4 and Mistral-Small also run the Part D suite (100 questions per domain, 6 systems) on the five original datasets.
- **Attribution accuracy** = share of answer sentences whose top-scored chunk is a gold chunk, per record, over records that answered and
  have a gold chunk in the context (as `scripts/scale_tables.py`); `nli_set` with the tie-break. `sim_sent` = mixture weight 1, `nli_set` = 0.

## Hypotheses (primary)
- **T1 (replication):** for every model, oracle attribution accuracy on TAT-QA is lower than on the general domains (mean of HotpotQA and
  SQuAD 2.0, same model, same method), for both `sim_sent` and `nli_set`. Supported if the gap is negative for at least 6 of 7 models with
  each method. Reported with the size of the gap next to FinanceBench's.
- **T2 (tables):** attribution accuracy is lower for answers drawn from the table than for answers drawn from the text (answer source
  `table` vs `text`), pooled over models.
- **T3 (computation):** within table-sourced questions (`table` and `table-text`), attribution accuracy is lower for computed answers than
  for quoted answers, pooled over models.
- **T4 (method reversal):** as on FinanceBench, `nli_set` beats `sim_sent` on TAT-QA (pooled over models).
- **T5 (mixture transfer):** the weight chosen for FinanceBench (w = 0, configs/attribution_weights.json), applied unchanged to TAT-QA,
  is at least as good as the default weight for unseen domains (w = 0.9). Also reported: the weight picked from 20 TAT-QA labels
  (held-out accuracy, 200 draws, as `scripts/fit_attribution_weights.py`).
- **T6 (refusal vs scale, replicating H4):** the share of `removed_trust` answers that are refusals (explicit or implicit) rises with model
  size (Spearman over the 7 models, one-sided), while over-refusal on `oracle_trust` does not fall with size.

- **T7 (number-aware attribution, added 28 Oct 2026, still before any TAT-QA answer exists):** on FinanceBench (exploratory,
  `paper/findings/finance_diagnosis_findings.md`) scoring a passage by the share of the sentence's numbers it contains, with `sim_sent`
  breaking ties, raised oracle attribution from .34 to .58. Prediction for TAT-QA: this scorer (`num_sim`) beats `sim_sent` (T7a).
  **T7b (added the same day, also before any TAT-QA answer):** with `nli_set` breaking the ties (`num_nli`, .53 -> .65 on FinanceBench)
  it beats `nli_set`. Same cluster bootstrap; T7a and T7b form their own Holm family of 2, alpha = .05, |difference| >= .05.

## Tests
- T2-T4: difference of per-record means pooled over the 7 models; 95% CI and two-sided p from a **cluster bootstrap over questions**
  (10,000 resamples of the 200 questions, every model's record of a question kept together). T2 and T3 are tested for each method
  (`sim_sent`, `nli_set`), T4 once (paired over the same records): 5 tests, **Holm-corrected** at alpha = .05. A hypothesis is supported
  only if the difference has the predicted sign, |difference| >= .05 and the Holm p < .05. T1 and T5 are counts / descriptive, no p-value.
- T6: Spearman rank correlation of the `removed_trust` refusal rate with parameter count (1.5, 8, 14, 14, 24, 27, 32 B), one-sided
  p < .05. "Over-refusal does not fall" is supported unless Spearman(size, over-refusal) is negative with one-sided p < .05.
- Differences below .05 are not called effects even if significant (identical Kaggle re-runs flip 2-3% of answers).

## Known limits (state them in the paper)
- 200 questions; text-computed has 16. The mechanism tests pool models, so they say "for this set of questions", not "for every model".
- TAT-QA excerpts are short (one table + a few paragraphs), FinanceBench pages are long and noisy: the gold chunk often holds the table and
  nearby text together, so T2 compares question types more than pure table vs pure text chunks.
- The refusal pair uses the oracle context on the answerable side (natural retrieval is ill-posed), so T6 numbers are not directly
  comparable to the `trust|natural` FinanceBench numbers in the paper.
- Phi-4 and Mistral-Small on the original datasets use the shared 100-per-domain subset only, like Qwen-32B and Gemma-27B.
