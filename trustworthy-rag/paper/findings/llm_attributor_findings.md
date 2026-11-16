# LLM-as-attributor baseline (Part F): results (final, 1 Nov 2026)

Plan: `analysis_plan_llm_attributor.md` (with Amendment 1, written before any accuracy was computed: the judge's pick is the number written in its
reply). Script: `python -m scripts.analyze_llm_attributor`; log `experiments/logs/analyze_llm_attributor_final.log`; tables
`experiments/results/llm_attributor/`.

## Runs (all complete and checked)
- Llama-3.1-8B judge (F2): 12,008/12,008 sentences, about 3.5 GPU-hours on 2x T4. Writes `[n]` (hence Amendment 1).
- Qwen2.5-32B judge: 12,008/12,008 over three sessions (F1: FinanceBench, TAT-QA, CUAD and 767 PubMedQA sentences, after a parser-bug restart;
  F1med: the rest of PubMedQA, reusing F1's output; F1gen: HotpotQA and SQuAD), about 14 GPU-hours. No duplicates; every id valid.
- Qwen-32B declined to name any passage for 808 sentences (6.7%; PubMedQA 16%, FinanceBench 10%): "None of the passages ...",
  "The sentence 'Maybe' ...". Amendment 1 counts these as misses; the literal first-token rule guesses from leftover probability (~0.7%).

## Attribution accuracy (gold guaranteed, records pooled over the 7 generators)
| Domain | sim_sent | mixture | num + sim | num + mixture | Llama-8B judge | Qwen-32B judge | 32B judge + numbers |
|---|---|---|---|---|---|---|---|
| FinanceBench | .352 | .528 | .579 | .645 | .731 | .692 (literal .732) | .719 |
| TAT-QA | .506 | .560 | .748 | .753 | .762 | .789 (.797) | .797 |
| CUAD | .677 | .701 | .680 | .701 | .701 | .771 (.800) | .766 |
| PubMedQA | .820 | .820 | .815 | .815 | .846 | .791 (.843) | .817 |
| HotpotQA | .888 | .888 | .889 | .889 | .940 | .943 (.967) | .951 |
| SQuAD 2.0 | .881 | .881 | .882 | .882 | .908 | .908 (.910) | .906 |

## Pre-registered verdicts (primary reading)
| | Hypothesis | Result | Verdict |
|---|---|---|---|
| L1 | 32B judge > sim_sent in each domain | FinanceBench +.340, TAT-QA +.284, CUAD +.094, HotpotQA +.055 (all Holm p < .01); SQuAD +.027 (< .05); PubMedQA -.029 | supported in 4 of 6 |
| L2 | num_sim within .05 of the 32B judge (finance) | FinanceBench judge +.113 [.062, .162]; TAT-QA +.041 [.020, .062] | not supported (TAT-QA narrowly) |
| L3 | numbers + judge > judge (finance) | FinanceBench +.027 (Holm p .048, below .05 in size); TAT-QA +.008 | not supported |
| L4 | 32B judge > 8B judge | +.003 [-.010, .015] | not supported: the 8B judge is as good |

Sensitivity: under the literal rule every verdict is unchanged (L1 4/6: FinanceBench +.380, TAT-QA +.291, CUAD +.123, HotpotQA +.079; L2 gap +.153
[upper .204] and +.049 [upper .070]); only L4 moves: 32B - 8B = +.041 [.028, .055], significant but below the .05 threshold, so still not supported. With the 20-label mixture added
(num + mixture), the gap to the 32B judge is +.047 [-.001, .094] on FinanceBench and +.036 [.016, .055] on TAT-QA. Without the 32B judge's own
answers the numbers move by under .01.

## Reading
- LLM judges are the most accurate attributors and do not collapse on financial text (.69-.80): the collapse belongs to similarity and NLI scoring.
- A 4x larger judge is not better (L4): an 8B judge is enough.
- The free number-aware scorer is not equivalent to an LLM judge (L2 fails): of the gap from similarity to the 32B judge it closes 67% (num + sim)
  to 86% (num + mixture) on FinanceBench and 85-87% on TAT-QA, at zero GPU cost (judges: about 1 GPU-second per sentence at 8B, 4 at 32B, on 2x T4).
- Number matching does not meaningfully help a judge (L3).
- The 32B judge refuses to pick a passage for 6.7% of sentences, most often PubMedQA sentences that only state a label ("Maybe") — arguably
  correct behaviour that a forced-choice metric scores as a miss.
