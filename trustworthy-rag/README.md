# Does Evidence Attribution Travel? A Controlled Cross-Domain Study of RAG Attribution, Hallucination Detection and Refusal

Code, configurations, pre-registered analysis plans and aggregate results for the paper of the same name.

Utkarsh Kumar Jha, Surveer Singh Guglani, Prajjwal Joshi, Savita Yadav. Department of Computer Science and Engineering, Netaji Subhas University
of Technology (NSUT), New Delhi.

## What this is
One fixed retrieval-augmented generation pipeline (same chunker, embedder, retriever, prompts and attribution methods) applied to seven
question-answering datasets in four domains:

| Domain | Datasets |
|---|---|
| General | HotpotQA, SQuAD 2.0 |
| Medical | PubMedQA, BioASQ (sentence-level evidence) |
| Legal | CUAD |
| Financial | FinanceBench, TAT-QA |

Retrieval is precomputed once, so all seven open-weight generators see exactly the same passages: Qwen2.5 1.5B/14B/32B, Llama-3.1-8B, Phi-4 14B,
Mistral-Small-24B and Gemma-3-27B, all run in 4-bit through Ollama on Kaggle's free 2x T4. Every question is also asked with the gold passage
guaranteed (oracle) and with it removed, which separates retrieval errors from generation errors and tests refusal.

## Main findings
1. **Retrieval is the main loss in legal and financial text** (FinanceBench Hit@5 0.19; guaranteeing the gold passage adds 0.21 to 0.30 accuracy
   in FinanceBench and CUAD for models of 8B and above).
2. **Sentence-to-passage attribution collapses on financial documents** for all seven models: with the gold passage guaranteed, similarity-based
   attribution is 0.32 to 0.39 on FinanceBench against 0.83 to 0.93 on general text. Replicated on TAT-QA.
3. **Cause and a free fix.** PDF tables give the similarity scorer one-cell fragments, and NLI cannot verify computed figures. A label-free,
   model-free **number-aware scorer** raises FinanceBench attribution from 0.35 to 0.58 (0.53 to 0.65 combined with the per-domain mixture) and is
   confirmed on held-out TAT-QA (0.51 to 0.75), with no change on other domains.
4. **LLM judges** (Llama-3.1-8B, Qwen2.5-32B) are the most accurate attributors (0.69 to 0.80 on finance); the 32B judge is no better than
   the 8B one. The free scorer does not match them but closes 67 to 87% of the gap at no cost.
5. **Sentence-level medical evidence (BioASQ)** gives lower attribution than PubMedQA for all seven models (0.62 to 0.71); refusal transfers.
6. **Refusal** when evidence is removed improves with model size (balanced accuracy 0.54 at 1.5B, 0.77 to 0.83 at 14-32B); over-refusal does not.
7. **Negative results are reported**: a supervised RAGTruth checker does not transfer across domains, a "trust layer" does not beat trivial
   policies, and token log-probabilities do not improve with model size.

## Pre-registration
Hypotheses were written down before the corresponding runs or analyses, in `paper/findings/analysis_plan_*.md` (H1-H4 larger models, T1-T7
TAT-QA, L1-L4 LLM judges, B1-B5 BioASQ; each file is dated, and amendments are dated and explained). Every verdict, including the unsupported
ones, is in the matching `*_findings.md`. This repository was created after those plans were written, so the commit history does not itself
timestamp them.

## Layout
| Path | Contents |
|---|---|
| `src/pipeline/` | data preparation, chunking, index, retrieval, prompts, LLM client (Ollama) |
| `src/attribution/` | attribution methods (similarity, NLI, evidence-set NLI), per-domain mixture, explanations |
| `src/evaluation/` | correctness and attribution metrics |
| `scripts/` | every runnable step; the docstring at the top of each file says how to run it |
| `configs/` | one YAML per run (model, datasets, conditions); `attribution_weights.json` = the per-domain mixture weights |
| `kaggle/` | the Kaggle notebooks and how to run every stage (`kaggle/README.md`) |
| `tests/` | unit tests |
| `paper/findings/` | analysis plans (pre-registered hypotheses), detailed findings, case studies |
| `paper/figures/`, `paper/tables/` | figures and tables |
| `experiments/results/` | aggregate result tables (CSV) behind every number in the paper |

Not included: the datasets (download them from their sources below), the raw model answers (about 0.8 GB, available from the authors on request),
and the human-verification sheets (added once verification is complete).

## Reproducing
Python 3.12. Laptop analysis needs only numpy, pandas, scipy, scikit-learn, matplotlib, openpyxl and pytest; generation, embeddings and NLI
scoring need a GPU (`pip install -r requirements.txt`; we used Kaggle, see `kaggle/README.md`).

1. Put the datasets in `data/raw/`:

   | Dataset | Source | File(s) in `data/raw/` |
   |---|---|---|
   | HotpotQA | Hugging Face `hotpotqa/hotpot_qa` (distractor, validation), downloaded automatically | none |
   | SQuAD 2.0 | https://rajpurkar.github.io/SQuAD-explorer/ (dev v2.0) | `squad2_dev.json` |
   | PubMedQA | https://github.com/pubmedqa/pubmedqa (`data/ori_pqal.json`) | `pubmedqa_ori_pqal.json` |
   | CUAD | https://github.com/TheAtticusProject/cuad (`data.zip`) | `cuad/CUADv1.json` |
   | FinanceBench | https://github.com/patronus-ai/financebench (questions, document information, PDFs) | `financebench_open_source.jsonl`, `financebench_document_information.jsonl`, `financebench_pdfs/` |
   | TAT-QA | https://github.com/NExTplusplus/TAT-QA (`dataset_raw/tatqa_dataset_dev.json`) | `tatqa/tatqa_dataset_dev.json` |
   | BioASQ | registered download (Task B training set 13b) from https://participants-area.bioasq.org; abstracts are fetched from PubMed | `bioasq/BioASQ-training13b/training13b.json` |

2. Build the corpora, chunks, index and retrieval plans:
   ```bash
   python -m scripts.prepare
   python -m scripts.export_plan --config configs/main_qwen1.5b.yaml
   python -m scripts.prepare_tatqa
   python -m scripts.prepare_bioasq
   ```
3. Generate and score (GPU): `kaggle/README.md` gives the exact notebook settings for each stage. Locally the same steps are
   `python -m scripts.run_generation --config configs/<config>.yaml` and `python -m scripts.score_run --run <run>`.
4. Analyse (CPU):
   ```bash
   python -m scripts.analyze_stage2 --model <qwen1.5b|llama8b|qwen14b|phi4|mistral24b|gemma27b|qwen32b>
   python -m scripts.scale_tables
   python -m scripts.analyze_logprobs --runs v2_qwen1.5b v2_llama8b v2_qwen14b v2_phi4 v2_mistral24b v2_gemma27b v2_qwen32b --systems vanilla_rag improved_vanilla
   python -m scripts.analyze_tatqa
   python -m scripts.finance_attribution_diagnosis diagnose
   python -m scripts.label_curve
   python -m scripts.analyze_llm_attributor
   python -m scripts.analyze_bioasq
   ```
5. Tests: `python -m pytest tests -q --ignore=tests/test_improvements.py -k "not evidence_set_nli_recognises"` (the two excluded tests need
   sentence-transformers and the NLI model).

## Design decisions
| Decision | Reason |
|---|---|
| No LangChain/LlamaIndex | Frameworks add hidden prompts and splitters; a controlled cross-domain study needs every step visible. |
| Chunks of 180 words, one-sentence overlap | all-MiniLM-L6-v2 truncates at 256 word pieces; longer chunks would be silently cut at embedding time. |
| Retrieval precomputed once (`data/plans/`) | Every model sees identical contexts, so differences come from the generator. |
| CUAD questions rewritten to name the contract | In a pooled corpus of 60 contracts, "this contract" is ambiguous. |
| CUAD unanswerables = CUAD's own empty annotations | A clause type that lawyers annotated as absent gives a gold "should refuse" without manual labelling. |
| FinanceBench corpus = every page of all 84 filings | Evidence-page-only corpora make retrieval trivial and would confound the cross-domain comparison. |
| PubMedQA abstracts exclude the conclusion | The answer must be inferred from the results. |
| Correctness: exact match (short), label accuracy (PubMedQA), span token recall >= 0.5 (CUAD), numeric match within 1% else NLI (FinanceBench) | Automatic and reproducible. |
| Open-weight models only, run free on Kaggle | No paid APIs; open models return token log-probabilities, which the confidence analyses need. |

## Licence
Code: MIT (see `LICENSE`). The datasets keep their own licences and terms; none is redistributed here. BioASQ requires registration, and its
data must be obtained from BioASQ directly.

## Citation
If you use this code, please cite the paper (citation to be added on publication).
