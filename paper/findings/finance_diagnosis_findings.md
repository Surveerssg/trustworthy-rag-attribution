# Why attribution collapses on FinanceBench (28 Oct 2026, exploratory)

**Update 31 Oct 2026:** re-run with all 7 models (Phi-4 and Mistral-Small added; they were not used to design the scorer). The CSVs in
`experiments/results/finance_diagnosis/` and the paper (Section V-C, Table "why") use the 7-model numbers; logs `experiments/logs/finance_*_7models.log`
and `combos_7models.log`. Every conclusion below holds: table-heavy gold .278 vs prose .537 (sim); rows + headers +.103 [.060, .147];
boilerplate -.035; number + sim .351 -> .579; number + mixture .528 -> .645 (+.123 [.084, .163]); other domains within +-.005. Held-out
confirmation on TAT-QA: `tatqa_findings.md` (T7a/T7b supported). The tables below are the original 5-model run.

Script: `scripts/finance_attribution_diagnosis.py` (`diagnose`: laptop Python; `rescore`: needs sentence-transformers, ran on CPU in a
scratch environment). Tables: `experiments/results/finance_diagnosis/`. Records: `oracle_vanilla` (gold passage guaranteed), 5 models,
644 FinanceBench answers over 149 questions. No answer or context was regenerated: only the passage text given to the scorer changes,
identically for gold and distractor passages. Exploratory (chosen after seeing the FinanceBench data); the held-out confirmation is T7 on
TAT-QA (`analysis_plan_tatqa.md`, registered before any TAT-QA answer exists).

## 1. Diagnosis from the stored scores (no model)
| Gold passage | n | sim_sent | nli_set |
|---|---|---|---|
| mostly prose | 190 | .532 | .546 |
| mostly table cells (>= 30% of its lines are bare numbers) | 454 | **.262** | .519 |

| Answer | n | sim_sent | nli_set |
|---|---|---|---|
| number quoted in the gold passage | 286 | .336 | .541 |
| computed (numerical reasoning, number not in the passage) | 170 | .263 | .588 |

- **Tables, not computation, break similarity.** PDF text puts every table cell on its own line, so the sentence splitter hands the
  similarity scorer fragments such as `1,488` or `$`. Similarity halves on table-heavy gold passages; NLI is barely affected.
  Computed answers are not harder for NLI (.59).
- Prose finance passages are still hard (.53 vs .85-.93 on general text): 1.7 of the 4 distractors come from the same filing.
- **Part of the "error" is near-duplicate evidence:** in 25.5% of records a distractor also contains the gold answer's number, and 28.9%
  of similarity failures pick such a passage. The automatic gold label (one evidence page) under-counts correct attributions. The number
  check allows a 1% tolerance and unit shifts, so a few of these matches may be coincidences.

## 2. Controlled re-scoring (similarity scorer; same answers and contexts)
| Passage text given to the scorer | FinanceBench | HotpotQA | SQuAD 2.0 |
|---|---|---|---|
| original (reproduces stored .342) | .340 | .888 | .885 |
| boilerplate stripped | .305 | .888 | .885 |
| table rows rebuilt (`label: v1 \| v2`) | .399 | .888 | .885 |
| rows + column headers (`label: 2018 = 1,488; ...`) | **.440** | .888 | .885 |
| number-aware scorer + similarity tie-break (`num_sim`) | **.582** | .888 | .887 |
| `num_sim` on rows + headers + stripped | .592 | .888 | .887 |

Change vs original on FinanceBench, 95% CI over questions: boilerplate -.034 [-.068, -.002]; rows +.062 [+.023, +.103];
rows + headers +.106 [+.062, +.151]; `num_sim` +.248 [+.200, +.298]. Every model moves the same way (per-model table in the CSV).

## What it says
1. **Table formatting is a cause.** Rebuilding rows with their column headers recovers about a third of the gap (.34 → .44) with no
   change to the model, the answer or the context.
2. **Boilerplate is not a cause.** Stripping it slightly hurts (it also removes statement titles that help matching).
3. **Numbers are the missing signal.** A scorer that checks which passage contains the sentence's numbers, with similarity only breaking
   ties, reaches .58: above `nli_set` (.53), with no NLI model, and with no change on prose (HotpotQA, SQuAD).
4. **Candidate method for the paper:** number-aware attribution plus table-row rebuilding. It is cheap, needs no labels, and targets the
   diagnosed failure. It was found on FinanceBench, so it must be confirmed on held-out data before it is claimed: TAT-QA (T7).

## 3. Number match combined with the stored scores, every domain (`combos`, no model)
`num+X`: rank passages by the share of the sentence's numbers they contain; X breaks ties (and decides alone for sentences without
numbers). Natural retrieval = `vanilla_rag` answers whose context holds a gold chunk (FinanceBench: only 122 records).

| Gold guaranteed | sim_sent | nli_set | mixture | num+sim | num+nli | num+mixture |
|---|---|---|---|---|---|---|
| SQuAD 2.0 | .886 | .872 | .886 | .889 | .878 | .889 |
| HotpotQA | .888 | .809 | .888 | .888 | .831 | .888 |
| PubMedQA | .826 | .614 | .826 | .821 | .654 | .821 |
| CUAD | .671 | .620 | .698 | .674 | .663 | .697 |
| FinanceBench | .342 | .527 | .527 | .582 | **.649** | **.649** |
| per model: macro | .70-.75 | .67-.70 | .74-.81 | .74-.81 | .71-.76 | **.76-.83** |
| per model: worst domain | .32-.36 | .51-.55 | .51-.55 | .54-.62 | .59-.69 | **.59-.71** |

| Natural retrieval | sim_sent | nli_set | mixture | num+sim | num+nli | num+mixture |
|---|---|---|---|---|---|---|
| FinanceBench (n = 122) | .476 | .439 | .439 | **.618** | .584 | .584 |
| per model: macro | .72-.80 | .61-.67 | .71-.78 | **.74-.83** | .70-.72 | .74-.81 |
| per model: worst domain | .45-.53 | .33-.52 | .33-.52 | **.58-.68** | .55-.61 | .55-.61 |

Paired 95% CIs (questions): `num+mixture - mixture` on FinanceBench +.125 [+.084, +.167] (gold guaranteed) and +.143 [+.054, +.242]
(natural); on the four other domains between -.004 and +.008, every CI containing 0. `num+nli - nli_set` is positive in every domain.

**Reading.**
- The number-match step is the fix: it lifts FinanceBench by .12-.25 in both conditions, for all five models, and leaves prose
  domains unchanged (it only acts on sentences with numbers, and those numbers rarely occur in the wrong passage).
- With the gold passage guaranteed, `num+mixture` is best (macro .76-.83, worst domain .59-.71, up from .74-.81 / .51-.55).
- In natural retrieval order `num+sim` is best and needs **no labels and no NLI model**; the 20-label weight for FinanceBench (pure NLI)
  was fitted in the oracle condition and does not carry over to natural order there (NLI .44 < similarity .48).
- Simplest defensible method for the paper: number match first, similarity second (`num+sim`), with the per-domain NLI mix as an
  optional refinement when the gold passage is present. Both are confirmed or refuted on TAT-QA (T7a, T7b).

## Caveats
- Exploratory: the transformations and the scorer were designed after looking at FinanceBench failures.
- Gold labels are automatic (one evidence page); near-duplicate pages mean every FinanceBench attribution number is a lower bound.
- The row rebuilder is a heuristic for one-cell-per-line PDF text; it does nothing on prose by design (the control domains are prose).
- Not yet tried: `num_sim` with `nli_set` as the tie-break, NLI on rebuilt rows (needs a GPU pass), and CUAD / PubMedQA.
