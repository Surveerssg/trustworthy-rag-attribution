# Phase 0 findings: honest selective-answering evaluation (25 Oct 2026)

Reproduce (about 20 s per run, CPU only, reads stored `.scored.jsonl`):
```bash
python -m scripts.eval_selective --run main_llama8b
python -m scripts.eval_selective --run main_qwen1.5b
python -m scripts.eval_selective --run main_llama8b --cost-sweep      # wrong-answer cost sensitivity, about 3 min
python -m pytest tests/test_selective.py -q
```
Outputs: `experiments/results/<run>/selective_{vanilla_rag,trust_rag,all}.csv`, `selective_reported_vs_trivial_*.csv` and
`selective_cost_sweep_*.csv`.

## Why this was needed
The reliability utility (+1 correct answer or correct refusal, 0 refusal of an answerable question, -1 wrong answer) is
maximised by **refusing everything** when accuracy is low, and plain RAG never refuses. Gains "over plain RAG" therefore
mostly measure how much a system refuses. A gate is worth something only if, on held-out questions, it beats the best of
{always answer, always refuse}, and ranks answers better than random abstention does.

## Protocol (per run x system x domain x signal)
- Trivial policies: always-answer, always-refuse, oracle selector (perfect confidence).
- AUROC (with bootstrap CI) and E-AURC (0 = random abstention, 1 = oracle) for "this answer is good".
- `gain@cal n`: threshold picked from n labelled in-domain questions (endpoints allowed), utility on the remaining
  questions minus the best trivial policy on the same questions; 200 random draws; P(gain<=0).
- Coverage at P(answered and wrong) <= 0.2, signal vs random abstention (threshold tuned on the test set, so optimistic).
- Signals: v1 trust score, trust with whole-chunk contradiction, trust without contradiction, ret_top1, sim_cov_mean,
  nli_cov_mean, nli_cov_min, logistic regression trained on the other general domain (`lr_transfer`), and a 5-fold
  in-domain CV logistic regression (`lr_indomain_cv`, optimistic upper bound that uses many in-domain labels).

## Results (2 generators x 2 prompts x 5 domains = 20 cells)
1. **At the mid-sem utility (wrong-answer cost 1), only 1 of 20 cells has a deployable signal that beats the best trivial
   policy on held-out questions** (gain > 0.01 and P(gain<=0) < 0.1): Llama-3.1-8B, vanilla prompt, CUAD, `sim_cov_mean`,
   +0.057, P = 0.03. In-domain CV logistic regression adds one more borderline cell (CUAD, +0.04 Llama and Qwen, P = 0.06
   and 0.17). Result 6 shows how this changes with the cost.
2. **The previously reported systems do not beat the best trivial policy.** "full" is built on the trust prompt, so its
   fair comparators are plain trust-prompt answers and refuse-all (the first version of this note used the vanilla-prompt
   baseline, which flattered it). v1 "full" minus the better of the two (explicit refusals only, like-for-like with "full"),
   Llama: SQuAD +0.03, HotpotQA -0.19, PubMedQA -0.07, CUAD -0.05, FinanceBench -0.01; Qwen: -0.04, +0.015, +0.02, -0.02,
   -0.01 (counting implicit refusals in the baseline makes Llama CUAD -0.07 and does not change any sign). So on HotpotQA,
   PubMedQA and CUAD (Llama) the gate and filter make the trust prompt worse. v2 "safe" (Llama, vanilla answers) minus best trivial: -0.04 to -0.08 SQuAD, -0.01 to -0.03
   HotpotQA/PubMedQA, -0.04 to -0.05 CUAD, -0.02 to -0.06 FinanceBench. The mid-sem "+0.68 over plain RAG" is the distance
   between "never refuse" and "refuse almost everything".
3. **The confidence signals are informative but weak.** Best deployable AUROC per cell is 0.57 to 0.79 (SQuAD 0.68-0.79,
   CUAD 0.71-0.73, FinanceBench 0.65-0.71, HotpotQA 0.57-0.65, PubMedQA 0.57-0.58). Even the in-domain CV logistic
   regression over all features stays at 0.51-0.77, so more calibration will not help: the features lack the information.
4. **Against random abstention the signals do help a little**: coverage at risk <= 0.2 is 0 to 18 points above random
   (typically about 5; SQuAD/Qwen 0.50 vs 0.33). Selective prediction with a risk guarantee is where the current signals
   have value, not the utility metric.
5. **The contradiction term adds nothing.** Mean AUROC (over models and prompts): v1 0.531-0.689, whole-chunk contradiction
   0.530-0.719, no contradiction at all 0.524-0.708 across domains; removing the term beats v1 in 4 of 5 domains. Fixing it properly (window-level re-scoring) is not
   worth Kaggle time; dropping it is as good. This revises HANDOFF 11.1, which expected the fix to rescue HotpotQA/PubMedQA.
6. **Sensitivity to the cost of a wrong answer** (utility = +1 correct answer or correct refusal, 0 refusal of an answerable
   question, minus lambda for a wrong answer or for answering an unanswerable question; lambda = 0.1 ... 4; 4 run-system
   combinations x 5 domains = 20 cells per lambda; about 8 deployable signals per cell; same splits for every lambda):

   | lambda | cells with a significant win (of 20) | where |
   |---|---|---|
   | 0.1 | 6 | SQuAD, CUAD |
   | 0.25 | 6 | SQuAD, CUAD |
   | 0.5 | 4 | SQuAD, CUAD |
   | 1 | 1 | CUAD |
   | 2 and 4 | 0 | none |

   - **Signals only ever win in the two domains that contain unanswerable questions (SQuAD, CUAD), and only when a wrong
     answer is cheap (lambda <= 0.5, once at 1).** In HotpotQA, PubMedQA and FinanceBench no signal wins at any lambda (best
     gain +0.039, smallest P 0.19). This is consistent with the AUROC findings: the signals partly separate answerable from
     unanswerable questions (where the +1 for refusing sits) but barely separate right from wrong answers among answerable
     ones.
   - **In the regime that motivates trustworthy RAG (lambda >= 2) nothing beats the trivial policies.** There refuse-all is
     the best trivial policy in nearly every cell, while a perfect selector would add 0.1 to 0.6. At lambda >= 2 the best
     signal captures none of it. The most it captures at any lambda is about 18% of the ceiling (SQuAD, lambda 0.25,
     averaged over the four combinations; CUAD 14%) and at most 4% in the answerable-only domains at lambda 0.25 or 1.
   - **Break-even cost** (answer-all = refuse-all): HotpotQA 1.0-2.2, PubMedQA 0.7-1.2, CUAD 0.13-1.7, FinanceBench
     0.29-0.39; in SQuAD refuse-all wins at every lambda for 3 of 4 combinations (0.29 for Llama trust) because half of its
     questions are unanswerable and a correct refusal earns the same +1 as a correct answer. Above lambda of about 2.2, refusing
     everything beats answering everything in every domain.
   - In SQuAD and CUAD several different signals win in the same cell (6 distinct signals at lambda 0.1), so these wins are
     not one lucky signal. Caveat: the table counts a cell as a win if the best of about 8 signals passes, with no
     multiplicity correction, and P(gain<=0) is the share of 150 random calibration splits with no gain, not a hypothesis-test
     p-value.

## Consequences for the paper
- Do not headline the full / v2 system as an improvement. Report it with the trivial baselines, or drop the utility metric.
- The defensible claims are the testbed, the retrieval-vs-attribution decomposition, the per-domain failure taxonomy, and
  calibration transfer (risk control breaks across domains) reported with coverage and against random abstention.
- To get a real system result the confidence signal must improve: RAGTruth-trained checker, token log-probabilities,
  a context-sufficiency signal, a stronger generator (higher base accuracy gives selective answering something to select).
- Report the whole cost curve (Result 6), not a single utility with lambda = 1, and justify the lambda range from the
  application: selective answering is worth something only when errors are cheap, and the current signals do nothing
  when errors are expensive, which is the case the paper's motivation is about. A stronger signal (Part A log-probs,
  Part B checker) must be judged on exactly this curve, at lambda >= 2.
- Treat SQuAD and CUAD separately from the answerable-only domains: with half or a third unanswerable questions, refuse-all
  is a strong baseline by construction. Answerable-only domains need a different headline metric (selective accuracy at
  fixed coverage, risk-coverage), which is also where Part A's evidence-removal test gives real unanswerable questions.

## Caveats
Correctness labels are automatic (not human validated). FinanceBench has 150 questions (110 answered by Llama), so its CIs
are wide. `trust_chunk_contra` uses whole-chunk NLI contradiction, a proxy for the window-level fix.
