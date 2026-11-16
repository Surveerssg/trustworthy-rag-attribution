# Stage 2 Part A findings (Kaggle run, analysed 26 Oct 2026)

Data: `trag_results_stage2A.zip` (117.7 MB, intact): Llama-3.1-8B and Qwen2.5-1.5B, 1,350 questions x 4 systems
(`vanilla_rag`, `improved_vanilla`, `improved_trust`, `oracle_vanilla`) and 1,000 answerable questions x 2 evidence-removal
systems (`removed_vanilla`, `removed_trust`), all with token log-probabilities, all scored. Generation: 0 retries. Logs:
`experiments/kaggle_logs/`. Reproduce:
```bash
python -m scripts.ingest_kaggle trag_results_stage2A.zip --no-analyses
python -m scripts.analyze_stage2 --model llama8b        # then qwen1.5b; writes experiments/results/v2_<model>/stage2_*.csv
python -m scripts.eval_selective --run v2_llama8b --systems vanilla_rag improved_vanilla improved_trust --cost-sweep
```
All numbers below come from those CSVs. Labels are automatic (not human-validated). Section 5 rests on 40 answers per domain.
nli_set values include the tie-break fix described in section 3 (applied in memory when the analysis loads the data).

## 0. How reproducible is a re-run? (read this before interpreting any small difference)
Re-running the identical config (same plans, prompt, temperature 0, seed 0) on Kaggle gave the same retrieved context for
100% of questions but different answer text for 38% (Llama) and 45% (Qwen) of them, mostly wording. Per-question correctness
flipped for **1.7% of questions (Llama) and 3.3% (Qwen)**; accuracy moved by 0.000 to 0.013 (Llama) and up to 0.016 (Qwen)
per domain with the vanilla prompt. The worst case found: Qwen, trust prompt, PubMedQA, where both arms use byte-identical
contexts and prompts, still differs by **+0.040 (0.395 -> 0.435), and a paired bootstrap calls it significant (p = 0.01)**.
**Treat accuracy differences below about 4-5 points as noise, even when the bootstrap says p < 0.05**: the bootstrap only
resamples questions, it does not know that two identical runs disagree.

## 1. Improved retrieval helps accuracy; it does not hurt attribution
Hit@5 base -> improved: SQuAD 0.872 -> 0.992, HotpotQA 0.965 -> 1.000, CUAD 0.585 -> 0.710, FinanceBench 0.188 -> 0.309
(PubMedQA is plain dense retrieval in both arms, so its two arms are identical by construction).

Accuracy, vanilla prompt, base -> improved, paired bootstrap (95% CI):

| Domain | Llama-8B | Qwen-1.5B |
|---|---|---|
| SQuAD 2.0 | 0.840 -> 0.924, +0.084 [0.040, 0.132] | 0.752 -> 0.824, +0.072 [0.020, 0.124] |
| HotpotQA | 0.605 -> 0.665, +0.060 [-0.005, 0.125] | 0.510 -> 0.580, +0.070 [0.015, 0.125] |
| CUAD | 0.475 -> 0.550, +0.075 [0.015, 0.140] | 0.470 -> 0.515, +0.045 [-0.020, 0.110] |
| FinanceBench | 0.240 -> 0.320, +0.080 [0.007, 0.153] | 0.240 -> 0.253, +0.013 [-0.040, 0.067] |

The trust prompt gives the same picture (Llama SQuAD +0.096, FinanceBench +0.100 [0.033, 0.173]). Given the noise floor of
section 0, the solid effects are the ones of about 7 points or more with a CI clear of zero: **SQuAD (both models), HotpotQA
(Qwen; Llama borderline p = 0.07), CUAD and FinanceBench (Llama only)**. CUAD (Qwen) and FinanceBench (Qwen) show no
reliable gain. FinanceBench still has Hit@5 of only 0.31, so most of its failures remain retrieval failures.

Attribution on the **matched** questions (gold retrieved under both arms, so there is no selection effect), `sim_sent`
accuracy base -> improved: Llama SQuAD 0.886 -> 0.872, HotpotQA 0.807 -> 0.839, CUAD 0.788 -> 0.841, FinanceBench
0.491 -> 0.499 (n = 19); Qwen 0.899 -> 0.894, 0.845 -> 0.801, 0.724 -> 0.791, 0.481 -> 0.488. Changes are within about
+/-0.07 with no consistent sign loss, so better retrieval did not make attribution worse (answers the proposal's Dir E
question for these two retrieval changes; chunk size and chunking were not varied). FinanceBench n = 19 is too small to read.

## 2. Oracle context: where the failures are, with retrieval taken out
Accuracy with the gold passage forced into the (shuffled) context, vanilla prompt, vs the normal retrieval run:

| Domain | Llama base -> oracle (retrieval loss) | Qwen base -> oracle |
|---|---|---|
| SQuAD 2.0 | 0.840 -> 0.940 (+0.100 [0.056, 0.144]) | 0.752 -> 0.808 (+0.056) |
| HotpotQA | 0.605 -> 0.765 (+0.160 [0.100, 0.220]) | 0.510 -> 0.605 (+0.095) |
| PubMedQA | 0.555 -> 0.540 (-0.015, n.s.) | 0.470 -> 0.380 (**-0.090 [-0.145, -0.040]**) |
| CUAD | 0.475 -> 0.680 (+0.205 [0.130, 0.280]) | 0.470 -> 0.610 (+0.140) |
| FinanceBench | 0.242 -> 0.530 (**+0.289 [0.201, 0.376]**) | 0.242 -> 0.336 (+0.094) |

- Retrieval is the largest loss in FinanceBench (+0.29, Llama) and CUAD (+0.21); in HotpotQA it is +0.16 even though 96.5% of
  questions retrieve at least one gold paragraph (a multi-hop question needs both); in SQuAD +0.10. In **PubMedQA retrieval
  is not the problem at all** (gold is retrieved 99% of the time and giving it for sure does not help). PubMedQA is a
  generator inference limit, which is what the mid-sem error taxonomy said ("evidence ignored").
- **Attribution without the retrieval confound.** Accuracy of picking the gold chunk (Llama / Qwen), `sim_sent`:
  SQuAD 0.869 / 0.897, HotpotQA 0.878 / 0.850, PubMedQA 0.741 / 0.883, CUAD 0.706 / 0.581, **FinanceBench 0.347 / 0.322**
  (chance is about 0.2). With n = 149 this replaces the mid-sem FinanceBench figure that rested on 28 questions.
  The best method **differs by domain**: in FinanceBench evidence-set NLI (0.51 / 0.53) is clearly better than similarity,
  while in SQuAD, HotpotQA and PubMedQA similarity is as good or better. No single attribution method is best everywhere.
- **Position of the gold passage.** Spearman(position, correct): Qwen PubMedQA +0.193 (p = 0.006; accuracy 0.27 with gold
  first, 0.52 with gold last, a recency effect of the small model); Llama HotpotQA -0.135 (p = 0.056), Llama CUAD -0.119
  (p = 0.093), Qwen CUAD -0.133 (p = 0.060) hint at primacy; everything else is n.s. PubMedQA "gold" is every chunk of an
  abstract, so its position is the first of several.

## 3. A bug in nli_set was real: quantified, and fixed on 26 Oct 2026
**What was changed.** `src/attribution/methods.py`: `break_set_ties()` is now applied inside `attribute()`, so every newly scored
record (Kaggle Part B, future runs, the dashboard) uses the corrected `nli_set`. `refresh_record()` applies the same correction in
memory to records scored earlier, and `scripts/analyze_stage2.py` does so by default (`--published-nli-set` reproduces the old
numbers). **No file on disk was rewritten**: the mid-sem runs (`main_*`) and the legacy analysis scripts still give the numbers
shown at the mid-sem, and re-scoring a mid-sem run with `score_run --force` would now give the corrected nli_set instead.
Effect on the Part A tables: only the nli_set columns moved (checked column by column; the other methods' rows are identical).
The numbers below describe the artifact and its size; the rest of this document uses the corrected values.

`nli_set` credits the joint (two-chunk) entailment score to BOTH chunks of the pair, and the argmax then breaks the tie toward
the lower index. In the shuffled oracle condition this makes attribution accuracy depend on where the gold chunk sits:
accuracy with gold in positions 1-2 minus positions 4-5 is **0.152 on average for nli_set (positive in all 10 model-domain
cells; significant Spearman trend in 6 of 10)** against 0.038 for nli_window, 0.051 for sim_sent and 0.028 for nli_chunk
(significant in 0 of 10 for nli_window and sim_sent). Breaking ties by the chunk's own single-chunk NLI score (computable from the
stored scores, no re-scoring) removes the trend (mean Spearman -0.163 -> -0.049):

Attribution accuracy (mean of the 10 model-domain cells):

| Condition | nli_set as published | nli_set, ties broken by own evidence | nli_window | sim_sent |
|---|---|---|---|---|
| oracle (gold at a random position) | 0.653 | 0.681 | 0.641 | 0.707 |
| natural retrieval order | 0.640 | 0.632 | 0.579 | 0.744 |

So the artifact deflates nli_set when gold is not first (-0.028 on average, up to -0.051) and slightly inflates it in natural
retrieval order (+0.008 on average, up to +0.056 for Llama PubMedQA and +0.030 for Llama HotpotQA; Qwen FinanceBench goes
the other way, -0.071), which is where all mid-sem nli_set numbers come from. **The joint-support idea survives** (tie-broken nli_set 0.632 vs single-chunk window NLI 0.579, +0.05),
but **plain `sim_sent` (0.744) beats every NLI variant on average**, so the mid-sem claim that NLI-based attribution is
preferable should not be made. The paper should report nli_set with the tie-break and state that the mid-sem version had this
order effect. Tests: `tests/test_set_tiebreak.py` runs the real `attribute()` with stand-in models and shows the old rule
attributing to the wrong chunk when the stronger chunk comes later, and the fixed rule being order-invariant.

## 4. Evidence removal: a ground-truth refusal task in every domain
Gold evidence deleted, so the right behaviour is to refuse. "natural" pairs each removed question with its normal run (gold
retrieved: SQuAD 218, HotpotQA 193, PubMedQA 198, CUAD 117, FinanceBench 28 questions); "oracle" pairs it with the
gold-forced run (full n; FinanceBench 149).

**Prompted refusal is a scale effect.** Balanced accuracy of the model's own refusal decision (0.5 = never refuses):

| Prompt / model | SQuAD | HotpotQA | PubMedQA | CUAD | FinanceBench (n = 28) |
|---|---|---|---|---|---|
| trust prompt, Llama-8B | 0.739 | 0.749 | 0.742 | 0.778 | 0.768 |
| trust prompt, Qwen-1.5B | 0.521 | 0.560 | 0.530 | 0.598 | 0.518 |
| vanilla prompt, Llama-8B | 0.592 | 0.674 | 0.710 | 0.748 | 0.571 |
| vanilla prompt, Qwen-1.5B | 0.505 | 0.503 | 0.513 | 0.603 | 0.500 |

The 8B model refuses 54-91% of the time when evidence is missing and refuses it far less when present, with a **similar
balanced accuracy (0.74-0.78) in all five domains**: where the capability exists, it transfers. The 1.5B model almost ignores
the instruction (refuses 4-21% of the time when evidence is missing). The 8B trust prompt over-refuses: it declines 34% of
HotpotQA, 22% of CUAD and 14% of FinanceBench questions that did have the evidence. Without the instruction (vanilla prompt)
the models answer anyway in 43-86% (Llama) and 75-100% (Qwen) of the cases.

**Confidence signals detect missing evidence far better than they detect wrong answers.** Best AUROC per domain for
"evidence present" (oracle vs removed) against best AUROC for "answer is correct" (Part A vanilla run), Llama:
HotpotQA 0.862 vs 0.730, PubMedQA 0.950 vs 0.642, SQuAD 0.844 vs 0.762, CUAD 0.773 vs 0.728, FinanceBench 0.767 vs 0.752
(Qwen: 0.802 vs 0.702, 0.943 vs 0.575, 0.850 vs 0.788, 0.635 vs 0.712, 0.587 vs 0.675). This is the same fact the cost sweep
showed from another side: signals help where there are unanswerable questions, not for catching wrong answers among
answerable ones. Caution: removing the whole gold PubMedQA abstract leaves unrelated abstracts, an easy negative
(`ret_top1` alone reaches AUROC 0.95), so these are topic-absent, not subtly-insufficient, negatives.

**Thresholds fitted on the general domains transfer on this task.** Balanced accuracy lost by using the threshold fitted on
the other general domain instead of an in-domain one: mean 0.035-0.050, median 0.022-0.035 (max 0.16-0.21, PubMedQA). Mean
over models and domains, oracle pairing: `trust_no_contra` AUROC 0.779, `lp_mean` 0.744, `ret_top1` 0.747, `nli_cov_mean` 0.751.
Log-probabilities are as good as retrieval and NLI signals here, not better. (This contrasts with the mid-sem result that
a general-domain threshold does not keep the wrong-answer RISK under control; that risk depends on predicting correctness,
which is much harder.)

Leak check (the removed-evidence answer still equals the original gold): 7-8% SQuAD, 20-25% HotpotQA for Llama; CUAD 22-28%
and FinanceBench 20-32% are **not** evidence of memorisation: the "clean" filter only checks that the gold string is absent
from the context, which cannot catch the same fact appearing in another contract clause or filing page.

## 5. Leave-one-out faithfulness (Gap B), 40 answers per domain, vanilla prompt
`correct attribution is faithful` = when the method points at the gold chunk, removing that chunk changes the answer
(tau 0.5): 0.73-1.00 in nearly every cell (Llama CUAD 0.78-0.90, PubMedQA 0.86-0.95; Qwen HotpotQA 0.73-0.93). So
"cited but not used" is **rare** for the plain prompt: a correct attribution is almost always a used chunk. What the methods
do poorly is pick the MOST influential chunk: `top_attributed_is_top_influential` (tau 0.5, chance 0.2) is 0.35-0.60 in
SQuAD/HotpotQA/PubMedQA for Llama but only 0.18-0.55 in CUAD and 0.28-0.38 in FinanceBench; for Qwen CUAD 0.10-0.28.
`sim_sent` has the highest rank correlation with the causal influence in **9 of the 10 model-domain cells** (Llama CUAD is the
exception, where nli_set and nli_window are higher); its Spearman values are 0.11-0.51, the NLI variants' -0.03 to 0.45.
Caveat: n = 40, tau is a judgement, and influence is measured with the same NLI model family.

## 6. Token log-probabilities as a confidence signal: what survives a pre-specified, multiplicity-controlled test
`python -m scripts.analyze_logprobs --runs v2_llama8b v2_qwen1.5b` (statistics unit-tested on synthetic data). **Design, fixed before
the Qwen-14B and API runs are looked at:** one primary signal, `lp_mean` (mean token log-probability of the answer); a control that is
generous to the competition (the best NON-log-prob signal of each cell, chosen by AUROC on the same data); 30 cells (2 models x 3 systems
x 5 domains). AUROC for "the answer is good" with a bootstrap CI and a permutation test; the value of acting on the signal at
wrong-answer cost lambda in {0.5, 1, 2, 4} as the utility of a threshold policy minus the best trivial policy, with the threshold chosen by
repeated 5-fold cross-fitting (every question is decided with a threshold fitted on the other 80%), a paired bootstrap over questions, and
Benjamini-Hochberg at FDR 10% within each signal family (120 tests each). Results (vanilla prompt unless stated):

1. **Log-probs carry real information, but are not generally better than retrieval/NLI signals.** `lp_mean` AUROC is significantly above
   chance (p < 0.05) in 24 of 30 cells, and so is the control (24 of 30); `lp_mean` beats the control in 12 of 30. By domain, `lp_mean` vs
   control: HotpotQA 0.662 vs 0.593 (+0.069, the only clear win), PubMedQA 0.574 vs 0.566, CUAD 0.670 vs 0.700, SQuAD 0.690 vs 0.725,
   FinanceBench 0.617 vs 0.667.
2. **Answering only what the model is most sure of works, and log-probs are the best single signal for it.** Accuracy gain from keeping
   only the 50% most confident answered questions (random abstention gives 0): `lp_mean` +0.088 on average (HotpotQA +0.138, SQuAD
   +0.105, CUAD +0.080, PubMedQA +0.075, FinanceBench +0.042) and `lp_nolow` +0.085, against +0.051 (`ret_top1`), +0.050
   (`sim_cov_mean`), +0.047, +0.036, +0.035 for the retrieval/NLI signals. **The two families are complementary**: in PubMedQA and HotpotQA
   (retrieval almost always succeeds) the retrieval/NLI signals give about 0 to +0.05 while log-probs give +0.075 to +0.14; in CUAD and
   FinanceBench (retrieval failures matter) `ret_top1` is as good or better (+0.105 / +0.091 vs +0.080 / +0.042). Retrieval signals detect a
   retrieval failure, log-probs detect generator uncertainty.
3. **No robust evidence that any of this beats the trivial policies when a wrong answer is expensive.** `lp_mean`: 5 discoveries in 120
   tests (by lambda 0.5 / 1 / 2 / 4: 3 / 1 / 1 / 0); the non-log-prob control: 11 (7 / 2 / 2 / 0), mostly SQuAD (detecting unanswerable
   questions). At lambda >= 2: 1 of 60 for `lp_mean` (Llama vanilla HotpotQA, lambda 2, +0.155 [0.068, 0.238] over refuse-all) and 2 of 60 for
   the control (Llama improved_trust: SQuAD `nli_cov_mean` +0.074, HotpotQA `ret_top1` +0.158).
4. **Hypothesis for the larger models, not yet a finding:** informativeness grows with model size. Mean `lp_mean` AUROC is 0.660 for
   Llama-8B against 0.625 for Qwen-1.5B (the control signals: 0.648 vs 0.652); the larger model is higher in 4 of 5 domains (CUAD 0.702 vs
   0.639, FinanceBench 0.657 vs 0.576, PubMedQA 0.618 vs 0.530, HotpotQA 0.667 vs 0.658; SQuAD the other way, 0.658 vs 0.723). With two
   sizes this is a trend, not evidence; Qwen-14B and a 70B-class model are the test (`paper/findings/analysis_plan_larger_models.md`).

**Correction.** An earlier exploratory sweep in this document counted 139 of 2,520 (signal, cell, lambda) tests as wins and "7 wins at
wrong-answer cost 2 and 4". It tested 16 signals per cell with no multiplicity correction and a split-share criterion that is not a
p-value; under the procedure above 1 of 60 survives at cost >= 2. That count is withdrawn. What can be claimed: log-probs are a free,
informative confidence signal (AUROC above chance in 24 of 30 cells; +8.8 accuracy points when answering only the 50% most confident,
the best single signal for that purpose), complementary to retrieval-based signals, and with no demonstrated value at high error cost.

## 7. Stage 2 Part B: the supervised RAGTruth checker does not help on our data (run 26 Oct 2026, ingested)
**The run was sound.** Total 2,172 s (36 min), shorter than the notebook's "2 h" guess because RAGTruth QA is small: training
took 14 min 53 s (849 steps over 4,531 examples), the zero-shot DeBERTa-large job ran in parallel (10 min 41 s for the 900
test responses), and scoring 20,200 of our answers took the rest. All 20,200 checker scores are present, with no duplicates or
NaN. The reported RAGTruth numbers reproduce exactly from the saved test predictions. The logged training loss (2.5 -> 1.3) looks
alarming but is most likely the Trainer's gradient-accumulation logging factor: divided by 4 it is 0.63 -> 0.32, against a
no-skill 0.62 for a 30.9%-positive label (consistent with this, not proven).

**On RAGTruth itself** (QA test split, 900 responses, 160 hallucinated = 17.8%; train split 30.9% hallucinated):

| Method | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|
| ModernBERT-base fine-tuned on RAGTruth QA (ours, supervised, dev-tuned threshold 0.197) | 38.7 | 76.3 | **51.4** [45.3, 56.6] | 0.825 |
| same, threshold tuned on the test set (optimistic) | | | 56.2 | |
| zero-shot DeBERTa-v3-large NLI, chunk / set / window / similarity (ours) | | | 43.0 / 41.5 / 38.3 / 35.2 | 0.762 / 0.736 / 0.721 / 0.620 |
| zero-shot DeBERTa-v3-base NLI, chunk / window / set (mid-sem) | | | 40.3 / 42.8 / 43.6 | |
| published: GPT-4-turbo prompt / SelfCheckGPT / fine-tuned Llama-2-13B | | | 45.6 / 43.7 / **68.2** | |

The checker is above the zero-shot methods and GPT-4 prompting and **well below the 68.2 target** (a fine-tuned Llama-2-13B,
about 90x more parameters than ModernBERT-base).

**Paired test against the mid-sem detector (27 Oct; `python -m scripts.compare_ragtruth`, saved scores, nothing re-run).** One primary test fixed
in advance: checker vs mid-sem zero-shot `nli_set` (DeBERTa-v3-base), same 900 test responses, each at its own pre-chosen threshold, paired bootstrap
(10,000 resamples). F1 51.4 vs 43.6: **+7.8 [3.3, 12.2], p = 0.001**; threshold-free AUROC 0.825 vs 0.760: **+0.065 [0.020, 0.109], p = 0.002**.
Secondary comparisons (uncorrected) point the same way: +8.3 to +16.2 F1 over every other zero-shot method, base or large. So the improvement
over the mid-sem detector is real on this benchmark. Limits: one fine-tuning run (seed variance of the checker unmeasured); the checker was trained on
RAGTruth's own training split (in-distribution advantage the zero-shot methods do not have); the GPT-4-turbo (45.6) comparison is a point estimate only,
since its per-response predictions are not published. Output: `experiments/results/ragtruth/paired_comparison.csv`. A 3x larger NLI model gives no zero-shot gain (best 43.0 against 43.6 for the base model). The dev threshold sits
too low for the test prevalence (30.9% vs 17.8%), which costs about 5 F1 points.

**On our data it is not useful**, by every test we have:
- Predictions are saturated: its mean P(hallucinated) on our answers is 0.73-0.88 by system (0.24 on RAGTruth test).
- **Correctness (Part A runs, 30 model-system-domain cells):** mean AUROC per domain 0.463 (PubMedQA), 0.505 (SQuAD), 0.540 (CUAD),
  0.548 (HotpotQA), 0.631 (FinanceBench); never at or above the best earlier signal (0 of 30 cells), below chance in 8 of 30.
  Adding it to the logistic regression does not help (mean AUROC 0.56-0.65, below the best single earlier signal).
- **Evidence removal (the task it should be best at, since removed-evidence answers are unsupported by construction):** last of
  12 signals for both models, mean AUROC 0.735 (Llama) against 0.810 for the best signal, and 0.602 (Qwen) against 0.747; a
  threshold fitted on the general domains loses up to 0.17 balanced accuracy (PubMedQA 0.17, CUAD 0.16, Llama).
- **Cost sweep (exploratory, uncorrected criterion; section 6 explains why the count of 'wins' was withdrawn):** the checker never wins at
  wrong-answer cost 2 or 4, and its mean held-out gain
  -0.034 (cost 1) and -0.036 (cost 2), the worst of all signals (token log-probabilities: +0.004 and -0.012). The checker alone wins
  once (Llama vanilla CUAD, cost 0.5), the logistic regression containing it twice, in cells where 7-9 other signals also win.

**Why (hypotheses, not tested):** RAGTruth QA inputs are short (median about 480 tokens, none over 811 words) while ours are about
1,300 tokens; RAGTruth responses come from older generators and general-knowledge questions; its labels mark span-level
contradictions in free-form answers, not the correctness of short answers in legal, medical and financial text.

**Consequence.** An off-the-shelf supervised hallucination detector trained on RAGTruth does not transfer to these domains, and
more NLI capacity does not fix the zero-shot gap. This is a reportable negative result. Token log-probabilities remain the most useful
confidence signal we have. The files are in `experiments/runs/*/checker_ModernBERT-base.jsonl`; the model weights stayed on Kaggle.

## 8. Attribution default: similarity with a 20-label domain-calibrated NLI mix (implemented 26 Oct 2026)
**Decision (team, after sections 2 and 3):** `sim_sent` is the default for attribution (WHICH passage a sentence comes from); NLI stays
for verification (is the claim supported), where similarity is much worse (RAGTruth F1 35.2, AUROC 0.62 vs 41-43, 0.74-0.76). Because
similarity fails in FinanceBench (0.33-0.35 vs 0.51-0.53 for evidence-set NLI with gold guaranteed), each domain gets a weight
w in {0, 0.1, ..., 1} for `w * sim_sent + (1 - w) * nli_set`, picked from 20 labelled questions (`src/attribution/fusion.py`,
`scripts/fit_attribution_weights.py`, weights in `configs/attribution_weights.json`, used by `explain.py` and the dashboard; unseen
domains get 0.9). Tie-break rule: equal accuracy goes to the larger w (cheaper, no NLI model needed).

Fitted on 20 Llama oracle-context questions per domain (seed 0): SQuAD 1.0, HotpotQA 1.0, PubMedQA 1.0, CUAD 0.7, FinanceBench 0.0.
- **Controlled evaluation (oracle context, held-out):** macro 0.739 vs 0.705 (`sim_sent`) and 0.681 (`nli_set`); worst domain 0.501 vs
  0.352 and 0.501; averaged over 200 random label draws 0.737 / worst 0.483. Same weights applied unchanged to Qwen-1.5B: macro 0.751 vs
  0.707 and 0.682, worst 0.528 vs 0.322: **they transfer across generators**. A mix chosen from 50 labels: 0.743 / 0.748 macro.
- **Natural retrieval order (closer to use):** the mix is about equal to plain `sim_sent` on average: improved-retrieval runs, macro
  0.746 vs 0.740 (Llama vanilla), 0.742 vs 0.744 (Qwen vanilla), 0.730 vs 0.719 and 0.743 vs 0.720 (trust prompt); the weakest domain
  improves in 3 of 4 cases (FinanceBench n = 35-46, so wide intervals). With the base retrieval (FinanceBench n = 28) it is -0.021 (Llama) and
  +0.004 (Qwen). So: clearly better under controlled evaluation, neutral to slightly positive in natural order, never meaningfully worse.
- **Caveats:** the weights come from automatic gold-chunk labels of one generator and one seed (the mode over draws differs for CUAD:
  0.7 vs 1.0); the 20-label recipe assumes 20 labelled questions per domain; a mix weight tuned only on the two general domains is 0.9,
  which would not have fixed FinanceBench (even 10% similarity drops it from 0.52 to 0.43).

## What this changes for the paper
- Usable, defensible new results: improved retrieval gains and the retrieval-free oracle decomposition (section 1-2); the
  oracle-condition attribution gap with n = 149 for FinanceBench; the scale dependence of prompted refusal and its
  cross-domain stability; the evidence-removal task as ground-truth unanswerables in every domain; LOO showing correct
  attributions are faithful.
- Must fix or caveat: nli_set tie rule (section 3); the 3-point re-run noise floor (section 0); small-n sections (5, FinanceBench
  natural pairing); the topic-absent character of the removal negatives.
- Log-probabilities (section 6, pre-specified test): an informative, free confidence signal and the best single signal for
  "answer only what you are most sure of" (+8.8 accuracy points at 50% coverage), complementary to retrieval signals, but with
  no demonstrated value at high error cost (1 of 60 tests) and not generally better than the best retrieval/NLI signal. The earlier
  claim of wins at cost 2-4 is withdrawn. The supervised RAGTruth checker (Part B) did not add a signal in any domain (section 7).
  The scale trend (larger model, more informative log-probs) is the hypothesis the Qwen-14B and 70B runs are meant to test.
  Still open: whether any of this survives human-validated labels.
- Attribution (section 8): similarity default with a 20-label per-domain NLI mix; clearly better than either single method in
  controlled evaluation, neutral in natural order.
- Part B has now been run: the supervised RAGTruth checker does not help (section 7).

## Part B preparation (dry-check done before the run; the run itself is section 7)
Found and fixed before it cost GPU time: (1) `train_checker.py:45` and `explain.py:117` had a backslash inside an f-string
expression, a SyntaxError on Python < 3.12 (Kaggle's version is not recorded in the logs), which would have killed Part B at import;
(2) ModernBERT now loads with `reference_compile=False` (no torch.compile on a T4); (3) the notebook applied the trained
checker only to vanilla/trust/improved runs, not to `oracle_vanilla`, `removed_vanilla`, `removed_trust`, which are exactly the
cases a hallucination checker should be tested on; (4) the notebook zip would have included the ~600 MB model weights and a
duplicate of Part A's runs, now it returns only the small `checker_*.jsonl` score files, the RAGTruth result CSVs and logs.
The analysis now reads `checker_*.jsonl` automatically (signal `checker` = 1 - P(hallucinated)), in the selective evaluation,
the cost sweep and the evidence-removal task, with tests. Needs the rebuilt notebook and bundle uploaded (see below).
