# Evidence RAG Lab roadmap

Approved direction: 10 November 2026; M2 plan, F04 → F05 sequence and three-phase authorization procedure approved 13 November 2026. Target: interview-ready by 29 November 2026. M1 completed on 12 November. M2 is active; planning is complete and F04 has not started. M3 remains planned and inactive.

## Purpose and boundaries

Build a small, reproducible Wikipedia-based claim-verification RAG case study that a Python backend engineer can explain and defend in Applied AI interviews. Prefer measurable engineering evidence, deterministic experiments, and failure analysis over breadth. Negative results are acceptable; no arbitrary accuracy target is imposed. Protect job applications, networking, and interview practice.

The accepted source is official FEVER (2018), its claims and evidence annotations, and its preprocessed June 2017 Wikipedia corpus. The mini-corpus and use conditions are recorded in `milestone.md` and `planning-sources.md`. SciFact is superseded because its rights remained unresolved; the earlier audit accepted its technical SciFact/BEIR compatibility, so it is not recorded as technically invalid.

## Approved MVP design

A local Python CLI accepts a claim, searches the same fixed passage corpus through BM25 or one dense retriever, sends selected passages through a thin LangChain structured-output boundary, validates the verdict and citations, and renders source page titles and original FEVER sentence IDs. Outcomes are SUPPORTS, REFUTES, and NOT ENOUGH INFO. Empty or insufficient evidence abstains; provider errors remain operational failures. Conflicting evidence is shown explicitly.

The implementation uses `rank_bm25`, one Sentence Transformers model (`BAAI/bge-small-en-v1.5`), persistent local Qdrant, and ordinary Python orchestration. Its verification CLI uses `mistralai/mistral-small-3.2-24b-instruct` through DeepInfra via OpenRouter; exact settings, privacy routing, prompt version, tokenizer identity, and dated prices are pinned in the project configuration and source record. One M1 live call was separately authorized. For M2, the user reports account privacy controls and a dedicated spending limit are configured; RM50 is the initial project ceiling, not authorization for inference. Each of the three paid phases requires separate explicit approval; RM100 requires a separate decision.

The runtime/index path contains only source passage text and citation metadata. The evaluation path owns claim IDs, labels, evidence alternatives, and split membership. No label, gold evidence annotation, split marker, or gold document hint enters normal retrieval payloads or generation prompts.

## Milestones

| Milestone | Window | Status | Exit |
| --- | --- | --- | --- |
| M1 — Accepted dataset and minimal functioning RAG | 9–15 Nov | Complete — 12 Nov | FEVER gate accepted; 5,000-passage gold-informed corpus, BM25/dense retrieval, and cited three-outcome CLI verified with focused automated checks and one bounded live example. |
| M2 — Controlled retrieval and LLM evaluation | 16–22 Nov | Active; plan approved, F04 not started | Fixed evaluation protocol completed; retrieval and generation quality, latency, tokens/cost, failures, and limitations recorded reproducibly. |
| M3 — Failure analysis, reproducibility, case study, and interview preparation | 23–29 Nov | Planned; not active | Reproduction instructions and technical case study are complete; traced failures/successes support a clear engineering explanation and interview practice. |

## M1 completion — 12 November 2026

| Feature | Verified outcome | Feature commit |
| --- | --- | --- |
| M1-F01 — FEVER source and use gate | Official FEVER source, private/local use conditions, attribution, and evaluation/runtime boundary accepted. | `32dd19ebd5690d542931970782c5fa6ed3a08e25` |
| M1-F02 — Corpus and retrieval | Deterministic 5,000-passage gold-informed corpus; BM25 and BGE-small dense retrieval with persistent local Qdrant. Train-only top-5 smoke: page/sentence Recall@5 0.88125 BM25 and 0.89375 dense; complete-evidence-group hit 0.85 and 0.90. | `7c7f902cce8dac6dd56824b36311f18c654a8e6b` |
| M1-F03 — Cited verification CLI | BM25/dense evidence retrieval, bounded context packing, OpenRouter/DeepInfra structured verdicts, citation validation, and run measurements. Forty-four tests passed; one authorized train-only live check returned REFUTES and matched its annotated source. | `5ef862f1e4cc7912c39769d8d37d02b4652ec6d2` |

The one live call reported 1,009 input and 93 output tokens. The application estimate was USD 0.000094275; the provider activity screen displayed USD 0.0000943 with usage type Credits (rounded). This single training-example check does not establish system-wide accuracy or reliability. The 60-claim held-out split has not been evaluated, and the retrieval smoke figures are custom train-only curated-subset measurements, not full FEVER benchmark scores. The full M1 contract and acceptance record remain in closeout commit `15d3bf13f15691e1bba027a03782add8b15e7edd` (`git show 15d3bf1:docs/core/milestone.md`); source identities/use terms and pipeline settings remain in `planning-sources.md` and `config/verification.toml`.

## M2 approved direction — 13 November 2026

The reusable [milestone contract](milestone.md) contains the approved M2 methodology. F04 builds and verifies offline retrieval evaluation on the 30 selected train claims. F05 adds bounded generation experiments, then freezes all scientific settings before running both retrieval and generation on the existing 60 held-out claims. This ordering prevents held-out retrieval observations from influencing generation selection. Both features use a read-only methodology/boundary audit followed by implementation and verification, with one commit per completed feature.

Compare both retrievers at passage k=1/5/10 offline; compare generation at k=1/5 on train and one common development-selected k held-out. Separate corpus coverage, retrieval recovery, packing retention, citation validity, annotated evidence correctness and human semantic grounding. Controls/repeats bring the planned total to 272 logical inference requests, at most 544 attempts in 28 bounded internal chunks across three separately authorized phases; empty-context abstention is tested locally without inference. Detailed token/cost assumptions, stop rules and approval gates live only in `milestone.md` to avoid competing budget specifications.

The user authorized a standalone commit of the approved plan and instructed work to stop before starting F04. F04/F05 remain unimplemented; resume F04 only on a separate instruction. The approved three-phase procedure retains the 28 internal ten-request chunks and existing caps; each actual paid phase needs separate authorization. Final M2 closeout follows verified F04/F05 evidence. Do not activate M3 or create another milestone for optional research.

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
