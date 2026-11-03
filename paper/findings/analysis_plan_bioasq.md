# Second medical dataset (BioASQ): analysis plan (Part G)

Written 31 Oct 2026, BEFORE any BioASQ answer exists. Do not change the hypotheses, tests or thresholds after the results are seen; anything added
later is labelled exploratory. Script: `scripts/analyze_bioasq.py` (CPU only).

## Why
PubMedQA counts the whole source abstract as gold, so medical attribution is measured coarsely and its removal negatives are easy (paper,
limitation iii). BioASQ gives expert snippets (one or two sentences) inside PubMed abstracts, so medical attribution can be measured at sentence level,
and its questions are self-contained, so all conditions (natural, oracle, removal) apply.

## Design
- **Data:** BioASQ Task B training 13b (Krithara et al.; registered download). 200 questions: 50 yes/no with answer yes, 50 with answer no, 100
  factoid (seed 13), each with at least one abstract snippet found in the current PubMed abstract. Corpus: title + abstract of every document listed
  for these questions plus the documents of 800 other BioASQ questions (12,837 abstracts, 22,118 chunks), fetched once from PubMed (E-utilities).
- **Gold evidence:** the snippets, matched by text (stored offsets kept when they still match: 190 of 200 questions). Every question has a gold chunk.
  `gold_doc_ids` = every document listed for the question, so the removal condition deletes all of them.
- **Answers:** yes/no as `label` (first word), factoid as `short` (any listed synonym, whole word). Same chunker, embedder, prompts, k = 5.
- **Systems:** `oracle_vanilla`, `vanilla_rag`, `removed_trust`, `trust_rag` (priority order). Models: the same seven generators.

## Hypotheses (primary)
- **B1 (PubMedQA's coarse gold inflated medical attribution):** with the gold passage guaranteed, `sim_sent` attribution on BioASQ is lower than the
  same model's PubMedQA value, for at least 6 of 7 models.
- **B2 (no finance-style method reversal in medicine):** `sim_sent` > `nli_set` on BioASQ (oracle), pooled over models.
- **B3 (the number-aware scorer is harmless off finance):** on BioASQ (oracle), `num_sim` - `sim_sent` lies inside (-0.05, +0.05): supported if the
  whole 95% CI is inside that interval.
- **B4 (retrieval matters in medicine once the corpus is realistic):** accuracy loss due to retrieval (oracle - natural, vanilla prompt) >= 0.05,
  pooled over models.
- **B5 (refusal transfers):** trust-prompt balanced accuracy on the removal task (natural answer with a gold chunk retrieved vs the same question with
  its documents removed) >= 0.70 for each of the six models of 8B or more; below 0.60 for Qwen2.5-1.5B.

## Tests
- B2, B3, B4: cluster bootstrap over questions (10,000 resamples; all models' records of a question move together), paired differences. B2 and B4 are
  supported only with the predicted sign, |difference| >= 0.05 and p < 0.05 after Holm over {B2, B4}. B3 is decided by its CI.
- B1 and B5 are counts over models (descriptive, no p-value); per-model values are reported with bootstrap CIs.
- Reported, not tested: accuracy by question type (yes/no vs factoid), the 20-label mixture (default weight 0.9, BioASQ has no fitted weight),
  `nli_set`, leak rate of the answer into the removed context.

## Known limits
- 200 questions (100 factoid, 100 yes/no); 7 models answer all of them.
- BioASQ lists several relevant documents per question (median 6) and their snippets; a distractor abstract that also answers the question but
  is not listed counts as non-gold, so attribution accuracy is a lower bound.
- Snippets come from BioASQ editions 1-12; PubMed abstracts were fetched on 31 Oct 2026, and 10 of the 200 questions are matched by text only.
