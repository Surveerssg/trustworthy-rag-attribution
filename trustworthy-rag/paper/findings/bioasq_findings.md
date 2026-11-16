# Second medical dataset (BioASQ, Part G): results (1 Nov 2026)

Plan: `analysis_plan_bioasq.md` (written before any answer existed). Script: `python -m scripts.analyze_bioasq`; log `experiments/logs/analyze_bioasq.log`;
tables `experiments/results/bioasq/`. Runs: `experiments/kaggle_raw/trag_results_stage2E_G{1,2}.zip`, all 7 models x 4 systems x 200 questions
complete and scored (5,600 answers; checked: real model, timings 1.8-25 s per answer by size, log-probs present, none equals the test-server pattern).

## Pre-registered verdicts
| | Hypothesis | Result | Verdict |
|---|---|---|---|
| B1 | PubMedQA's whole-abstract gold inflated medical attribution | BioASQ sim_sent .62-.71 vs PubMedQA .69-.89, 7/7 models lower | **supported** |
| B2 | sim_sent > nli_set on BioASQ | +.023 [-.013, .059] | not supported (no clear winner) |
| B3 | number-aware scorer harmless off finance | +.004 [-.009, .018], inside +-.05 | **supported** |
| B4 | retrieval loss >= .05 | +.037 [.000, .076], Holm p .105 | not supported |
| B5 | refusal balanced accuracy >= .70 for every model >= 8B, < .60 at 1.5B | .75-.81 for 6/6; 1.5B .52 | **supported** |

## Per model
| Model | sim / NLI (BioASQ) | sim (PubMedQA) | refuse removed | over-refusal | balanced acc. |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | .713 / .682 | .883 | .054 | .006 | .524 |
| Llama-3.1-8B | .636 / .635 | .741 | .665 | .066 | .799 |
| Qwen2.5-14B | .700 / .671 | .893 | .545 | .048 | .749 |
| Phi-4-14B | .662 / .635 | .795 | .605 | .048 | .778 |
| Mistral-Small-24B | .668 / .639 | .803 | .629 | .018 | .805 |
| Gemma-3-27B | .624 / .593 | .685 | .605 | .018 | .793 |
| Qwen2.5-32B | .674 / .659 | .886 | .605 | .054 | .775 |

Accuracy with gold guaranteed: factoid .53-.61, yes/no .64-.86; natural retrieval finds a gold chunk for 80% (factoid) / 87% (yes/no) of questions.
167 removal pairs per model; the gold answer string is still in 8.4% of the removed contexts (other abstracts).

## Reading
- With sentence-level evidence, medical attribution is clearly harder than PubMedQA suggested (about .15 lower on average): medicine sits between
  general text (.83-.93) and finance (.32-.39), like legal text.
- No finance-style failure and no method reversal: similarity and NLI are about equal; number matching changes nothing (as designed).
- Refusal transfers to a fourth domain, and over-refusal is low in medicine (.02-.07), unlike HotpotQA, CUAD and FinanceBench.
- Retrieval costs less than in law and finance (+.04, mostly factoid questions).
