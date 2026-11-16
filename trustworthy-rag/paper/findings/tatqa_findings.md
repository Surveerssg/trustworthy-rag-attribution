# TAT-QA and the two new model families: results (31 Oct 2026)

Pre-registration: `analysis_plan_tatqa.md` (written 28 Oct, before any answer existed; T7 added the same day, still before the runs).
Runs: Kaggle Part E, two sessions, all 9 jobs complete (`experiments/kaggle_raw/trag_results_stage2E_{A,B}.zip`; summary screenshots in
`experiments/kaggle_logs/partE_session_{A,B}_summary.jpeg`). Session A 5.8 h (Mistral-24B Part D subset 3.5 h; TAT-QA Mistral 1.0 h, Qwen-14B
0.7 h, Qwen-1.5B 0.2 h); session B 7.5 h (Phi-4 Part D subset 2.5 h; TAT-QA Qwen-32B 1.4 h, Llama-8B 0.4 h, Phi-4 0.7 h, Gemma-27B 1.9 h).
Every run has token log-probs; the longest prompt was 2,096 tokens (limit 4,096). Tables: `experiments/results/tatqa/`.

## Pre-registered verdicts
| | Hypothesis | Result | Verdict |
|---|---|---|---|
| T1 | TAT-QA attribution below general text, >= 6 of 7 models, both methods | sim 0.47-0.54 vs 0.86-0.91; NLI 0.56-0.68 vs 0.81-0.86; 7/7 both | **supported** |
| T2 | table-sourced < text-sourced | sim -0.358 [-0.479, -0.233], Holm p < .001; NLI -0.126, Holm p = .058 | sim **supported**, NLI not |
| T3 | computed < quoted (table-sourced) | NLI -0.251 [-0.340, -0.160], Holm p < .001; sim -0.026, n.s. | NLI **supported**, sim not |
| T4 | nli_set > sim_sent | +0.116 [0.062, 0.171], Holm p < .001 | **supported** |
| T5 | FinanceBench weight (w = 0) >= default (w = 0.9) | 0.56-0.68 vs 0.50-0.59, 7/7 models; 20 TAT-QA labels pick w = 0 or 0.1 | **supported** |
| T6 | refusal (evidence removed) rises with size; over-refusal does not fall | Spearman 0.67, one-sided p = 0.051; over-refusal rho +0.54 | first part **not supported** (narrowly); second supported |
| T7a | number match + sim_sent > sim_sent | 0.748 vs 0.506, +0.242 [0.197, 0.290], Holm p < .001 | **supported** |
| T7b | number match + nli_set > nli_set | 0.759 vs 0.622, +0.137 [0.110, 0.166], Holm p < .001 | **supported** |

## Per model (oracle accuracy; attribution sim / NLI / num+sim / num+NLI; refusal removed / over-refusal)
| Model | Acc | sim | NLI | num+sim | num+NLI | refuse removed | over-refusal |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | .450 | .501 | .613 | .721 | .761 | .045 | .000 |
| Llama-3.1-8B | .565 | .503 | .568 | .758 | .759 | .580 | .120 |
| Qwen2.5-14B | .595 | .521 | .684 | .817 | .815 | .590 | .090 |
| Phi-4-14B | .645 | .537 | .601 | .711 | .726 | .450 | .035 |
| Mistral-Small-24B | .605 | .494 | .682 | .759 | .774 | .675 | .190 |
| Gemma-3-27B | .670 | .473 | .558 | .686 | .686 | .475 | .045 |
| Qwen2.5-32B | .660 | .509 | .645 | .783 | .790 | .810 | .175 |

By answer source (pooled): sim table .387 / table-text .457 / text .744; NLI .561 / .636 / .687; with number match .707-.740 / .787-.791 /
.740-.751. By kind: NLI computed .491 vs quoted .729.

## Reading
- The FinanceBench attribution collapse is not a FinanceBench quirk: it replicates on cleaner, shorter financial excerpts, milder in size.
- Two separate mechanisms: similarity fails on tables (T2), entailment fails on computed figures (T3).
- The number-aware scorer, designed on FinanceBench, is confirmed on held-out data for every model and closes most of the table gap.

## Phi-4-14B and Mistral-Small-24B on the five original datasets (Part D subset)
Same picture as the other five models: FinanceBench sim .39 / .38 (general .83-.89); mixture worst .549 / .510; refusal balanced accuracy .769 /
.827 with over-refusal .110 / .147; lp_mean AUROC .674 / .667 (no size trend); oracle FinanceBench accuracy .576 / .475. With all 7 models: H1 still
not supported, H2 still not supported (2 of 140 tests at cost >= 2), the FinanceBench diagnosis and the number-aware gains hold for the two new
models, which were not used to design the scorer (`finance_diagnosis_findings.md`).
