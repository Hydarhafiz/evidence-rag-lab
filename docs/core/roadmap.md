# Evidence RAG Lab roadmap

Approved direction: 10 November 2026. Target: interview-ready by 29 November 2026. M1 is the only active milestone. M1-F01 has accepted the FEVER dataset gate; M1-F02 is next.

## Purpose and boundaries

Build a small, reproducible Wikipedia-based claim-verification RAG case study that a Python backend engineer can explain and defend in Applied AI interviews. Prefer measurable engineering evidence, deterministic experiments, and failure analysis over breadth. Negative results are acceptable; no arbitrary accuracy target is imposed. Protect job applications, networking, and interview practice.

The accepted source is official FEVER (2018), its claims and evidence annotations, and its preprocessed June 2017 Wikipedia corpus. The mini-corpus and use conditions are recorded in `milestone.md` and `planning-sources.md`. SciFact is superseded because its rights remained unresolved; the earlier audit accepted its technical SciFact/BEIR compatibility, so it is not recorded as technically invalid.

## Approved MVP design

A local Python CLI accepts a claim, searches the same fixed passage corpus through BM25 or one dense retriever, sends selected passages through a thin LangChain structured-output boundary, validates the verdict and citations, and renders source page titles and original FEVER sentence IDs. Outcomes are SUPPORTS, REFUTES, and NOT ENOUGH INFO. Empty or insufficient evidence abstains; provider errors remain operational failures. Conflicting evidence is shown explicitly.

Start with `rank_bm25`, one Sentence Transformers model (`BAAI/bge-small-en-v1.5`), persistent local Qdrant, and ordinary Python orchestration. The initial LLM candidate is `mistralai/mistral-small-3.2-24b-instruct` through DeepInfra via OpenRouter. Pin exact model/provider identity, prices, and privacy settings before any authorized call. The planned RM50 usage ceiling is not authorization to spend; RM100 requires a separate decision.

The runtime/index path contains only source passage text and citation metadata. The evaluation path owns claim IDs, labels, evidence alternatives, and split membership. No label, gold evidence annotation, split marker, or gold document hint enters normal retrieval payloads or generation prompts.

## Milestones

| Milestone | Window | Status | Exit |
| --- | --- | --- | --- |
| M1 — Accepted dataset and minimal functioning RAG | 9–15 Nov | Active | FEVER gate accepted; deterministic mini-corpus, BM25/dense retrieval, and cited three-outcome CLI work with focused automated checks. |
| M2 — Controlled retrieval and LLM evaluation | 16–22 Nov | Planned; not active | Fixed evaluation protocol completed; retrieval and generation quality, latency, tokens/cost, failures, and limitations recorded reproducibly. |
| M3 — Failure analysis, reproducibility, case study, and interview preparation | 23–29 Nov | Planned; not active | Reproduction instructions and technical case study are complete; traced failures/successes support a clear engineering explanation and interview practice. |

M1 has three features: F01 dataset approval (complete), F02 corpus and retrieval (next), and F03 cited verification CLI (after F02 and separate spending authorization). M2 and M3 require their own activation. Do not create another milestone for optional research.

## Initial evaluation contract

- Use a fixed development sample of 30 FEVER train claims (10 per class) and a fixed held-out sample of 60 labeled shared-task development claims (20 per class). Exclude normalized train/dev claim-text duplicates and claims whose gold page IDs cannot be matched exactly to an unambiguous corpus page and sentence ID. Record this source-coverage filter.
- Corpus construction uses the selected claims’ gold evidence pages and labels for class balance. It is explicitly gold-informed. Keep all selected evidence references outside runtime. Report selected-claim evidence coverage, class counts, the page-ID filter, selection bias, and sample-size limits.
- Compare BM25 and dense retrieval on the same approximately 5,000 passages. Report custom page/sentence Recall@k and complete-evidence-group coverage. Do not call these BEIR qrel metrics.
- Score generated verdicts separately with per-class results, macro-F1, abstention/NEI behavior, citation validity, and the unchanged official FEVER scorer where its contract applies. Label every result as a curated FEVER subset; it is not full-benchmark performance.
- Freeze model, prompt, retrieval settings, and evaluation sample before final held-out runs. Record missing predictions and errors, latency, token use, dated costs, and failure examples.

## Optional engineering experiments — not active

Consider these only after core project acceptance or when real spare capacity exists without threatening required evaluation, interview preparation, or the deadline:

- Hybrid retrieval compared with BM25 and dense retrieval.
- Reranking quality versus added latency and cost.
- Nested corpora of approximately 5K, 25K, and 100K passages.
- LangGraph versus ordinary Python/LangChain only if a measurable workflow requirement appears.

These are optional backlog items, not milestones, MVP requirements, or committed work. Scaling must retain the fixed evaluation claims and evidence, add distractors through a documented deterministic protocol, and disclose gold-informed sampling and subset-benchmark limitations.

## Deadline and scope protection

Required work is the FEVER-based 5K mini-corpus, BM25/dense comparison, cited three-outcome CLI, focused tests, separate evaluations, reproducible results, failure analysis, and interview preparation. No frontend, cloud deployment, public hosting, training/fine-tuning, MLOps, mandatory authentication, reranking, hybrid retrieval, or LangGraph is in the MVP. Do not publish corpus-derived text or indexes as part of the MVP.

Cut optional experiments and presentation polish before extending the 29 November target. The precise source, license, sample, provider, and privacy conditions belong in `planning-sources.md`; raw corpus files, model weights, indexes, and generated bulk output stay out of Git. Reuse this roadmap and `milestone.md`; keep `.sdd/` local and ignored.
