# Spot-check of the LLM judge's declines: plan (written 2 Nov 2026, before any label was seen)

**Status: not run, kept in reserve (decided 3 Nov 2026)** for a reviewer question such as "why count declines as misses?". Neither outcome
would change a pre-registered verdict or a table, so it was not done before submission. The sheet is built
(`paper/human_eval/judge_declines/`, kept out of the public repository); no labels exist.

**Exploratory.** This check was decided after the Part F results (the decline rate was seen); it is not part of the pre-registered
L1-L4 and must be reported as exploratory, with its sample size.

## Question
The Qwen2.5-32B judge declined to name any passage for about 800 of 12,008 sentences (6.7%; PubMedQA 16%, FinanceBench 10%), although the gold
passage was always in the context. Amendment 1 counts these as misses. Are these sentences mostly unsupported (the judge is flagging
unsupported content, a useful property), or mostly supported (the judge is simply failing)?

## Sample (`python -m scripts.judge_declines sheet`, seed 0)
- 30 declined sentences, 5 per domain (FinanceBench, TAT-QA, CUAD, PubMedQA, HotpotQA, SQuAD 2.0).
- 15 control sentences where the judge named a passage (3 each for FinanceBench, TAT-QA, PubMedQA; 2 each for the rest).
- Shuffled together; the labeller sees no judge output, no gold passage and no hint which items are declines.

## Labels (one labeller, a team member; from scratch, no proposed label)
Supported / Partially supported / Not supported / No factual claim, plus the supporting passage number(s).

## What will be reported (`python -m scripts.judge_declines analyze --sheet ...`)
1. Share of "Not supported" + "No factual claim" among declines vs controls, with a Fisher exact test (descriptive; 30 vs 15 items).
2. The same share among declines weighted by each domain's number of declines (the sample is balanced per domain, the declines are not).
3. Declined sentences labelled Supported by the gold passage (= the judge's genuine misses).

## Reading, fixed in advance
- Declines unsupported/no-claim >= 0.60 and at least 0.30 above the controls: "the judge's declines mostly mark unsupported sentences"
  (a small extra finding; one sentence in Section V on LLM judges and a line in the limitations).
- Otherwise: "the declines are mostly genuine misses", which supports counting them as misses (Amendment 1). Either result is reported.
- With one labeller and 45 items, this is a spot-check, not a validated measurement; the paper says so.
