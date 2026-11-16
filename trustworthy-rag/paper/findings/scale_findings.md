# Parts C and D: five-model results and the pre-registered hypotheses (analysed 27 Oct 2026)

Runs (all on Kaggle, 4-bit Ollama, log-probs for every answer): `v2_qwen14b` (Part C, all 1,350 questions, 7 systems, 8,750 answers, ~12 s per request),
`v2_qwen32b` and `v2_gemma27b` (Part D, 100 questions per domain, 6 systems, 2,834 answers each, ~26 s per request with 3 in parallel). All files complete.
Reproduce:
```bash
python -m scripts.ingest_kaggle experiments/kaggle_raw/trag_results_stage2C.zip --no-analyses        # and the two Part D zips
python -m scripts.analyze_stage2 --model qwen14b                                                      # and qwen32b, gemma27b
python -m scripts.analyze_stage2 --model <m> --qids-from v2_qwen32b --skip-selective                  # shared-subset version, every model
python -m scripts.scale_tables                                                                        # -> experiments/results/scale/
python -m scripts.analyze_logprobs --runs v2_qwen1.5b v2_llama8b v2_qwen14b v2_gemma27b v2_qwen32b --systems vanilla_rag improved_vanilla --out logprob_scale_full
python -m scripts.analyze_logprobs ... --qids-from v2_qwen32b --out logprob_scale_subset              # paired, shared subset
```
Logs: `experiments/logs/analyze_stage2_*.log`, `scale_tables.log`, `lp_full.log`, `lp_subset.log`, `lp_3sys.log`.

## Pre-registered hypotheses (paper/findings/analysis_plan_larger_models.md)
- **H1 (log-prob informativeness grows with size): NOT supported.** Mean `lp_mean` AUROC (vanilla + improved_vanilla): 1.5B .639, 8B .674, 14B .676,
  27B .650, 32B .647 (control .656/.651/.640/.646/.630). Shared subset: 1.5B .672 is highest. The 3-system statistic for 1.5B/8B/14B: .625/.660/.664.
- **H2 (useful at high error cost): NOT supported.** At cost >= 2: 1 of 100 discoveries (full), 0 of 100 (subset), 1 of 90 (3-system run).
- **H3 (log-probs win in HotpotQA/PubMedQA, retrieval in CUAD/FinanceBench): half supported.** Log-probs win in all five domains with five models
  (selective lift on the subset: CUAD +.115 vs ret_top1 +.090, FinanceBench +.110 vs +.013).
- **H4 (refusal improves with size; over-refusal does not grow): first part supported, second part NOT.** Trust-prompt balanced accuracy, subset
  means: .544 / .755 / .767 / .798 / .805 (1.5B/8B/14B/27B/32B); Qwen 1.5B->14B up in 5/5 domains, 14B->32B in 4/5. Over-refusal (subset mean):
  .010 / .183 / .180 / .080 / .247; Qwen-32B is the highest; Gemma-27B combines high refusal accuracy with low over-refusal.

## Other results
- Oracle retrieval loss: FinanceBench about +.29 for every model >= 8B; CUAD +.21 to +.30; PubMedQA ~0 (generator limit, oracle accuracy .52-.57 even at 32B).
  Base-retrieval FinanceBench accuracy .24-.26 for all five models.
- Attribution with gold guaranteed: `sim_sent` on FinanceBench .32-.36 for every model (general text .84-.93); `nli_set` .51-.55. The 20-label mixture
  (Llama weights, unchanged) raises the worst domain to .50-.55 and the macro to .74-.81 for all five models.
- Improved retrieval, Qwen-14B: SQuAD +.092, HotpotQA +.070, CUAD +.105 (CIs exclude zero), FinanceBench +.067 (n.s.).
- Selective lift (answer the top 50% by `lp_mean`): +.088 / +.087 / +.116 / +.136 / +.060 by model; `ret_top1` +.049 / +.054 / +.056 / +.039 / +.005.
