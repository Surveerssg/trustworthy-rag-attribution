# Larger models: analysis plan (written 26 Oct 2026, BEFORE the Qwen-14B and API runs are looked at) and run guide

Why: the study rests on two light models (Qwen2.5-1.5B, Llama-3.1-8B). Two sizes cannot show a scale trend and reviewers will say
"models too small". The plan adds a third open size (Qwen2.5-14B, Kaggle, free), a 70B-class open model through a hosted API (cheap, and it
returns token log-probabilities), and optionally one frontier model as a reference for the findings that do not need log-probabilities.

## 0. Decision 27 Oct 2026: FREE ONLY (no paid API). This section supersedes the paid parts below.
The team will not pay for models. The DeepInfra configs (`api_*_deepinfra.yaml`, about $5 per model) stay in the repo but are NOT part of the plan.
**Free ladder (everything runs on Kaggle's free T4 x 2, with token log-probabilities):**

| Model | Role | Where | Status |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | small | Kaggle (done) | done |
| Llama-3.1-8B-Instruct | small, second family | Kaggle (done) | done |
| **Qwen2.5-14B-Instruct** (same family as the 1.5B, about 9x the parameters; NOT a re-run) | medium | Kaggle Part C, no API key | configs ready, to run |
| **Qwen2.5-32B-Instruct** | large, completes the Qwen2.5 ladder 1.5B -> 14B -> 32B | Kaggle, ONE server split over both GPUs (about 20 GB at 4-bit), reduced suite | **Part D, built 27 Oct (section 5)** |
| **Gemma-3-27B-it** (Google's open-weight sibling of Gemini) | large, second family, "Gemini family" coverage | same as above (about 17 GB at 4-bit) | Part D, built; optional |
| gpt-oss-20b (OpenAI's open-weight model) | optional "GPT family" coverage | Kaggle; a reasoning model, check that log-probs behave | optional |

Part D design (not yet written): one Ollama server with both GPUs visible (`CUDA_VISIBLE_DEVICES=0,1`, Ollama splits the layers), the
config's `systems` reduced to [vanilla_rag, oracle_vanilla, removed_vanilla, improved_vanilla] (5,050 answers), `workers` 3-4,
`OLLAMA_NUM_PARALLEL` 2-4; estimated 5-8 h (unmeasured: time the first 100 answers and extrapolate); resumable across sessions
(re-run with the previous output attached). Then scoring as in Part C.

**Frontier models (Gemini / GPT / Grok): what is actually free** (provider help pages and community sources, 26-27 Oct 2026; excerpts, not full reads; none of them returns log-probs):
- *Gemini API:* Pro models are paid-only; the free tier covers only small Flash / Flash-Lite-class models with small daily request limits that have been cut
  repeatedly (a forum post reports Flash going from 250 to 20 requests/day in Dec 2025; limits are per project; check AI Studio's rate-limit page).
  Usable for a **small stratified sample** (for example 60 questions per domain, a few systems) spread over several days, as a log-prob-free reference.
- *OpenAI:* an official help page describes **complimentary daily tokens for organisations that opt in to share their API traffic**: a 1M tokens/day group
  (gpt-6.1-sol, gpt-6-astra, gpt-6-sol, gpt-5.6-sol, gpt-5.4) and a 10M tokens/day group (gpt-5.6-terra, gpt-5.6-luna, gpt-5.4-mini, gpt-5.4-nano), at
  one quarter of those quotas on the "Build" tier; the free tokens apply automatically but forum posts report they were not granted. It may need a
  payment method on file, and your prompts and outputs go to OpenAI (our corpora are public benchmark texts). Only worth trying if the account qualifies and a
  hard spending limit can be set; no log-probs for these models.
- *xAI Grok:* the credits are one-time ($25 at signup, per third-party sources) and the "$150 a month" route needs $5 of prior spend: not free. Skip.
- *Groq, OpenRouter, NVIDIA NIM:* free tiers exist but are far too small for a full run (Groq about 200K tokens/day, which is roughly 130 of our answers; OpenRouter `:free`
  models 50 requests/day without a purchase; NIM rate-limit based, conflicting reports). At most a tiny sample.
- *Consumer chat apps* (free ChatGPT / Gemini / Grok): fine for a manual look at a few cases; not reproducible, and automating them breaks their terms. Not for the paper.
So "Gemini, GPT, Grok" in the paper means open-weight siblings run free on Kaggle (Gemma, gpt-oss) plus, if the team wants one, a small log-prob-free API sample.

## 1. Which models can feed the log-probability analyses
Checked on 26 Oct 2026 from the providers' documentation and developer forums (search excerpts, not a full read: **confirm with
`scripts/probe_llm.py` before spending**).

| Model family | Token log-probs via API? | Consequence |
|---|---|---|
| Qwen2.5-14B on Kaggle (Ollama) | yes (verified in Part A for 1.5B and 8B) | full suite |
| Open-weight instruct models on DeepInfra, Together AI, Fireworks (e.g. Llama-3.3-70B-Instruct) | documented: `logprobs` / `top_logprobs` on chat completions | full suite |
| xAI Grok 4.20 and newer (incl. grok-4.7) | **no**: "logprobs and top_logprobs are not supported ... silently ignored" | reference only |
| OpenAI GPT-5-class reasoning models | **no** / reported removed ("not supported for reasoning output", "GPT 5.2 LogProbs support removed?") | reference only |
| Google Gemini 3.x Pro and newest Flash | **reported missing** (forum threads for 3.1 Pro, 3.6 Flash; 2.5 Flash also reported) | reference only |
| Anthropic Claude | no log-probs in the API | reference only |

"Reference only" models still support everything that does not need log-probabilities: retrieval/oracle decomposition, attribution accuracy,
accuracy, refusal behaviour (trust prompt, evidence removal). Use `logprobs: false` and the reduced system list in
`configs/api_openai_compat_template.yaml`. Reasoning models bill hidden thinking as output tokens: set `reasoning_effort` or a
non-reasoning variant and watch `reasoning_tokens` in the probe.

## 2. Pre-specified hypotheses and tests (same procedure for every model; `scripts/analyze_logprobs.py`, `scripts/analyze_stage2.py`)
- **H1 (scale):** token log-probabilities become more informative with model size. Primary signal `lp_mean`; statistic = mean AUROC for
  "the answer is good" over 3 systems x 5 domains per model; test = the sign of the trend over the sizes 1.5B < 8B < 14B < 70B (current:
  0.625, 0.660), and in how many of 5 domains the larger of each adjacent pair is higher. Not supported if the mean AUROC does not rise
  from 8B to 14B and again to 70B. Control: the best non-log-prob signal should show no such trend (current: 0.652, 0.648).
- **H2 (value at high error cost):** the number of BH discoveries (FDR 10%, 60 tests per model at wrong-answer cost >= 2) for `lp_mean` is larger
  for the larger models. Current: 1 of 60 across the two small models.
- **H3 (complementarity):** answering only the 50% most confident improves accuracy more with `lp_mean` than with the best retrieval/NLI
  signal in PubMedQA and HotpotQA, and less in CUAD and FinanceBench, for every model.
- **H4 (refusal vs scale):** trust-prompt balanced accuracy on evidence removal increases with size (1.5B: 0.52-0.60, 8B: 0.74-0.78) and
  the over-refusal on questions that had evidence (8B: 14-34%) does not grow.
- **Sensitivity signals (reported, not tested):** `lp_first_sent` (mean log-prob of the first sentence, which carries the direct answer),
  `lp_tok1` (first token), `lp_nolow`, `lp_margin`. Raw per-token log-probabilities (`lp_raw`) are stored for every new run, so any further
  feature can be built offline without regenerating.
- **Reported regardless of outcome:** a null or reversed result is a result. The earlier exploratory count of wins at high error cost was
  withdrawn after the corrected test (section 6 of `stage2a_findings.md`); nothing here may be reported on an uncorrected criterion.

## 3. Cost (ONLY relevant if the free-only decision is reversed; orders of magnitude, the probe replaces them with measured numbers)
Per answer about 1,400 input and 120 output tokens; the full 7-system suite is 8,750 answers (12.3 M input, 1.1 M output tokens).
Prices per million tokens (input/output) from the providers' pages on 26 Oct 2026: Gemini 3.1 Pro Preview $2/$12 (<= 200K prompt),
Gemini 4 Argon $2/$10 until 30 Nov, then $4/$20; Grok 4.7 $2/$6; GPT-6.1 Sol $2/$10; GPT-6 Astra $10/$50; hosted open 70B-class models roughly
$0.1-$1 (Together lists Llama-3.3-70B around $0.88).
- 70B open model, full suite: about **$10-15**.
- Frontier reference, reduced suite (vanilla_rag, trust_rag, oracle_vanilla, removed_vanilla, removed_trust = 6,050 answers): about
  **$25-40 for visible tokens, plus any reasoning tokens**, which can dominate (500-2,000 hidden tokens per answer at $6-12/M adds roughly $20-$150).
  Pro-tier plans (GPT-6 Astra) would be several times more. Keep reasoning off or minimal, and run the probe first.
- Qwen-14B on Kaggle: free, about 5-6 GPU-hours (estimate: Llama-8B took about 2.0 s per answer on one T4; 14B roughly 1.8x; 8,750 answers
  over two GPUs plus scoring).

## 4. Run guide (the API part is optional/paid: skip it under the free-only decision)
**Qwen2.5-14B (Kaggle, free, needs NO API key):** upload the rebuilt `dist/trag_bundle.zip` as a NEW VERSION of your private bundle dataset; import
`kaggle/trag_kaggle_stage2.ipynb` as a new notebook; inputs: ONLY the bundle dataset; GPU T4 x2, Internet on, Private; first cell
`PART = "C"`; Save & Run All (Commit). First lines to expect: `GPUs: 2`, `qwen14b_a -> OK | logprobs ok`, `qwen14b_b -> OK | logprobs ok`. Both servers load the
model: if you see `CUDA out of memory` or Ollama "failed to allocate", stop and send the log (the fix is fewer parallel requests).
Output `trag_results_stage2C.zip` ->
`python -m scripts.ingest_kaggle trag_results_stage2C.zip --no-analyses` -> `python -m scripts.analyze_stage2 --model qwen14b`.

**API model (generation on this laptop, scoring on Kaggle):**
1. `.env`: add the provider key (see `.env.example`); never paste it into chat, code or a URL.
2. `pip install tqdm requests pyyaml` (enough for generation; the full environment is not needed here).
3. `python -m scripts.probe_llm --config configs/api_llama70b_deepinfra.yaml --price-in <$/M> --price-out <$/M>`: confirms the model id, that log-probs
   come back, the token use of OUR prompts, the cost and time of the whole run.
4. `python -m scripts.run_generation --config configs/api_llama70b_deepinfra.yaml` (resumable, cached; Ctrl-C and rerun is safe).
5. `python -m scripts.make_run_bundle --run v2_llama70b` -> upload `dist/v2_llama70b_raw.zip` as a private Kaggle dataset.
6. Kaggle: the same notebook, `PART = "S"`, `SCORE_RUNS = ["v2_llama70b"]`, inputs = the bundle dataset + the raw-run dataset, GPU on, Internet on (about 1 h).
7. `python -m scripts.ingest_kaggle trag_results_stage2S.zip --no-analyses`, then `python -m scripts.analyze_stage2 --model llama70b` and
   `python -m scripts.analyze_logprobs --runs v2_llama8b v2_qwen1.5b v2_qwen14b v2_llama70b`.
   (Part S and the API configs were removed on 27 Oct under the free-only decision; this item is kept only as a record.)

## 5. Amendment for Part D (written 27 Oct 2026, while Part C was running and BEFORE any Part C or Part D output was seen)
Part D runs one large model per Kaggle session (`configs/kaggle_v2_qwen32b.yaml`, `configs/kaggle_v2_gemma27b.yaml`; run guide in `kaggle/README.md`).
To fit a 12 h session it differs from Parts A and C in two pre-specified ways, and the tests are adapted as follows (no other change):
- **Sample:** 100 questions per domain (`--limit 100`), a seeded subsample that keeps the answerable/unanswerable balance, is identical for both
  Part D models and is nested (a later run with a larger limit only adds questions). Every cross-model comparison that involves a Part D model is ALSO
  reported on this same 500-question subset for all models (paired), next to the full-set numbers of the smaller models.
- **Systems** (in priority order): vanilla_rag, trust_rag, removed_trust, oracle_vanilla, improved_vanilla, removed_vanilla. improved_trust is not run.
- **H1** is tested on the two systems every model has, vanilla_rag and improved_vanilla (`analyze_logprobs --systems vanilla_rag improved_vanilla`);
  the 3-system statistic is still reported for the models that have it. **H2/H3** use the same two systems.
- **H4** uses the trust_rag (evidence present, natural retrieval) / removed_trust (evidence removed) pair, as Part A did with the mid-sem trust_rag run.
- If the time budget stops a run before all six systems finish, the analysis uses what is complete and says which systems are missing; nothing is
  extrapolated. If Gemma returns no log-probabilities, Gemma is excluded from H1-H3 only.
