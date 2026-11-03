# How many labels does the attribution mix need? (label curve, 28 Oct 2026, exploratory)

**Update 31 Oct 2026:** re-run with 7 models (CSVs and the paper use these): FinanceBench 0 labels .367, 20 labels .508 (87% of the gain),
50 labels .525, ceiling .529; other four domains 0 labels .822, ceiling .831. Conclusions unchanged. The table below is the original 5-model run.

Script: `python -m scripts.label_curve` (CPU, existing oracle runs, 500 draws). Tables: `experiments/results/label_curve/`.
Exploratory: the 20-label recipe was fixed on 26 Oct (stage2a_findings section 8); this analysis does not change it.

**Setup.** Gold passage guaranteed (`oracle_vanilla`), 5 models. Each draw: the test set = all usable questions except a pool of 50;
the weight w of `w * sim_sent + (1 - w) * nli_set` is picked from the first n of the pool (n = 0 means the default w = 0.9). The test
set is the same for every n within a draw. Gemma-27B and Qwen-32B have too few questions in SQuAD and CUAD (100-question subset) and
are left out there, so the per-domain rows average 3 to 5 models.

## Result (mean over models and draws)
| Domain | sim only | NLI only | 0 labels | 2 | 5 | 10 | 20 | 50 | ceiling |
|---|---|---|---|---|---|---|---|---|---|
| SQuAD 2.0 | .890 | .879 | **.905** | .891 | .892 | .894 | .897 | .902 | .914 |
| HotpotQA | .891 | .814 | **.896** | .884 | .883 | .882 | .885 | .889 | .903 |
| PubMedQA | .818 | .606 | .817 | .808 | .811 | .814 | .816 | .818 | .825 |
| CUAD | .658 | .613 | **.681** | .657 | .660 | .666 | .669 | .672 | .688 |
| FinanceBench | .341 | .527 | .356 | .423 | .460 | .486 | **.507** | .524 | .527 |

Unlucky draw (10th percentile), FinanceBench: 5 labels .337, 10 labels .370, 20 labels .430, 50 labels .487.

## What it says
1. **Labels only matter where the domain behaves differently (FinanceBench).** In the four other domains the default weight with NO labels
   is as good as or better than any label budget. A few labels slightly hurt there (.01-.02, within noise) because a small sample sometimes
   picks the wrong weight.
2. **In FinanceBench the curve is steep and then flat.** 0 → 20 labels recovers 88% of the possible gain (.356 → .507 of .527); 10 labels
   76%, 5 labels 61%. 50 labels reach the ceiling. With 5-10 labels an unlucky draw gains almost nothing; with 20 it reliably gains.
3. **Honest framing for the paper.** The mix does not beat the best single method within a domain: its value is choosing the right
   method per domain (similarity on general/medical/legal text, NLI on financial filings) from a handful of labels, instead of one method
   everywhere. The FinanceBench ceiling equals pure NLI (.527).

## Suggested sentence for the paper
"A label-budget curve (5 models, 500 draws) shows that the default weight needs no labels on general, medical and legal text, while on
financial filings 20 labelled questions recover 88% of the attainable gain (attribution .36 → .51; ceiling .53) and 50 reach it."

## Caveats
- Labels are the automatic gold chunks, not human labels.
- The per-domain rows mix 3-5 models (see Setup); per-model numbers are in `label_curve_by_model_domain.csv`.
- Whether 20 labels also suffice on a second financial dataset will be tested on TAT-QA (T5 in `analysis_plan_tatqa.md`).
