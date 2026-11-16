# IEEE paper draft

- `main.tex`: IEEEtran, conference layout with the standard IEEE author blocks (`\documentclass[conference]{IEEEtran}`); `references.bib`; `figures/`.
- Compile on Overleaf (no LaTeX on the laptop): New Project → Upload Project → choose `deliverables/paper/ieee_overleaf.zip` (it holds main.tex,
  references.bib and figures/architecture.png in the right places) → Compiler pdfLaTeX. Uploading only main.tex gives "citation undefined"
  and "file not found" errors. Rebuild the zip after editing. There is no local PDF: the 28 Oct one was outdated and removed (3 Nov).
- All experiments are in (Parts A-G: seven generators, TAT-QA, FinanceBench diagnosis, LLM judges, BioASQ). The only open text is Section V-J
  (Human Validation): after the verifiers finish (at least items S001-S050, all 100 if time allows), replace the sentence there with the matching
  replacement (50 or 100) from the comment, using `trustworthy-rag/paper/human_eval/agreement.csv` and `method_validation*.csv`.
- The code link (https://github.com/Surveerssg/trustworthy-rag-attribution) appears after the contributions list and in the conclusion. For a
  double-blind venue, swap it for an anonymous mirror (anonymous.4open.science) in the submitted version.
- Bib entries marked `VERIFY` were not checked against the publisher page; check them before submission.

## Caption rule (the mid-sem report got this wrong)
IEEE Editorial Style Manual: **table captions go above the table** (printed as "TABLE I", Roman numerals) and **figure captions go below the
figure** (printed as "Fig. 1."). In LaTeX that means `\caption{...}` comes *before* the `tabular` inside `table`, and *after* `\includegraphics`
inside `figure`; IEEEtran then formats the labels. The mid-sem report (`deliverables/midsem/Midsem Report (final).docx`) placed all three table
captions below the tables and wrote "Table 1." / "Figure 1."; figures were placed correctly. The mid-sem files are frozen and were not edited.

## Numbers
All numbers come from `trustworthy-rag/paper/findings/*_findings.md` (stage2a, phase0, scale, tatqa, finance_diagnosis, label_curve,
llm_attributor, bioasq), `trustworthy-rag/paper/findings/midsem_results.md` (conformal result) and `trustworthy-rag/experiments/results/*`.
The mid-sem claims that were later corrected (nli_set order artefact, "full system wins") are reported in their corrected form (Section V-K).
