# M1 — Accepted dataset and minimal functioning RAG

Status: ACTIVE. Target window: 9–15 November 2026; overall target remains 29 November 2026. M2 and M3 are planned but not active.

Dataset gate: ACCEPTED on 10 November 2026. The official FEVER (2018) claims and June 2017 Wikipedia corpus are the project’s primary source. Detailed artifact identity, licensing, access, evaluation evidence, and the accepted use matrix are in [`planning-sources.md`](planning-sources.md).

## Outcome and boundaries

Build the smallest useful Wikipedia-based claim-verification RAG: deterministic approximately 5,000-passage corpus, BM25 and one dense retrieval path, local Qdrant, and a Python CLI that returns SUPPORTS, REFUTES, or NOT ENOUGH INFO with validated page-title/sentence-ID citations and abstention behavior.

Use FEVER’s official license conditions for private local indexing, embeddings, evaluation, and demonstrations. GitHub publication is limited to code, source identity, protocol, and aggregate results. Raw article text, claim/evidence subsets, passage indexes, Qdrant data, and embeddings are excluded from Git. External inference is the intended path through OpenRouter/DeepInfra with the privacy controls recorded in `planning-sources.md`; spending is still unauthorized. Public hosting is excluded and is not an M1 prerequisite.

No reranking, hybrid retrieval, LangGraph, model training/fine-tuning, frontend, cloud deployment, or MLOps. Optional experiments are documented separately in the roadmap and are not M1 work.

## M1-F01 — FEVER access, rights, and evaluation gate

- Type: research. Status: COMPLETE; accepted for the defined scope.
- Outcome: select an accessible, properly licensed source and establish a reproducible evaluation boundary before product implementation.
- Route: `contract-auditor -> architect-planner`.
- Completion record: the approved read-only audit and user decision are promoted to `planning-sources.md`; this feature owns one focused research commit containing the canonical contract. It does not authorize API calls, spending, or public dataset redistribution.

| Acceptance ID | Required condition | Result |
| --- | --- | --- |
| M1-F01-01 | Official FEVER train, labeled development, and June 2017 Wikipedia artifacts actually accessible; URLs, sizes, and identities recorded. | PASS — direct publisher downloads and hashes recorded. |
| M1-F01-02 | Archive integrity, schemas, page keys, labels, evidence groups, sentence references, and source coverage verified. | PASS — raw title-key differences are documented; fixed sample uses exact source keys and complete references. |
| M1-F01-03 | Publisher checksums checked where available; local hashes distinguished from publisher identities. | PASS — no publisher digest found; all local SHA-256 values recorded. |
| M1-F01-04 | Actual FEVER/Wikipedia license terms, attribution, ShareAlike, local processing, and redistribution conditions assessed. | PASS for accepted private experiment scope; public data-derived artifacts remain excluded. |
| M1-F01-05 | External inference provider, training/retention policy, routing, and spending controls reviewed for the selected candidate. | PASS as a conditional future path; account settings and spend remain unconfigured and unauthorized. |
| M1-F01-06 | Fixed evaluation claims/evidence, retrieval and verdict metrics, official scorer semantics, sample limitations, and gold-informed selection disclosed. | PASS — contract recorded; no full-benchmark claim. |
| M1-F01-07 | Define the evaluation/runtime boundary: labels, evidence groups, claim IDs, and split metadata stay out of runtime retrieval and generation. | PASS — F02/F03 contracts and focused tests enforce this boundary. |
| M1-F01-08 | Human accepts the source and scope with no unresolved material blocker for the required private experiments. | PASS — user approved FEVER and the audit evidence on 10 November 2026. |

The record does not infer a license from a mirror or claim that all historical article-page exceptions were separately inspected. FEVER’s publisher terms govern; use known page-specific terms and the stated CC BY-SA 3.0 fallback. SciFact is superseded due to unresolved rights. Its technical SciFact/BEIR compatibility was accepted previously and remains a historical finding, not a failure reason.

## M1-F02 — Deterministic mini-corpus and retrieval

- Type: build. Status: READY; next feature, not started. Budget: about 5 hours.
- Route: `builder-executor -> tdd-loop`.
- Entry: F01 complete; use only the official artifacts and processing scope in `planning-sources.md`.
- Outcome: build the fixed FEVER mini-corpus and repeatable BM25/dense retrieval over the same passages.

### Data and sampling protocol

1. Read the official FEVER claim files and `wiki-pages.zip` as data. Verify local input hashes against `planning-sources.md`; do not run archive scripts or third-party dataset loaders.
2. Select 30 development claims from train, 10 per label, and 60 held-out claims from `shared_task_dev.jsonl`, 20 per label. Within each source split and label, rank by SHA-256 of UTF-8 `FEVER-M1-F01-v1|{split}|{claim_id}` and take the first eligible IDs. Split keys are `train` and `dev`. Before selecting, filter the three normalized train/dev claim-text overlaps, any claim whose exact page title or cited sentence position is absent/ambiguous, and any claim referencing a duplicate sentence ID. Preserve the selected claim IDs and split map in evaluation-only data.
3. For each selected SUPPORTS/REFUTES claim, retain all alternative gold evidence groups offline and include every exact source page needed by those groups. The audit sample contains 107 evidence groups, 72 exact-match pages, and 103 unique page/sentence pairs, all present in the archive. The evidence-page full-window estimate is about 389 passages at four sentence lines with stride three; calculate the final count after dropping empty sentence text.
4. Parse each `lines` row as sentence ID, sentence text, and optional trailing entity-link fields. Use only the first two fields; ignore nonnumeric rows and trailing links. Preserve exact FEVER page-title keys and original sentence IDs in passage metadata. Do not add numeric MediaWiki IDs absent from the source.
5. Fill the approximately 5,000-passage target with background windows from pages outside the selected evidence pages and the duplicate-sentence-ID page. Sort candidates by SHA-256 of UTF-8 `FEVER-M1-F01-v1|background|{page_id}|{first_sentence_id}` and take enough to reach the target. This background draw is deterministic, independent of system results, and is not described as representative or as a curated hard-negative set.
6. Create windows over ordered nonempty numeric sentence records: at most four consecutive records, stride three. Enforce the selected encoder’s documented token limit; if a long sentence must split, retain its original sentence ID on each fragment. Multi-sentence and multi-page evidence remains a set of original references reconstructed across retrieved passages.

The evaluation sample and source pages are selected using gold labels/evidence, including for held-out claims. This is gold-informed corpus construction. It supports a fixed, curated subset experiment only; it does not represent independent full-corpus FEVER retrieval. The three normalized claim overlaps and 170 Unicode title-key differences are disclosed in the source record. Exact page-key filtering preserves unchanged official scorer behavior for the fixed sample.

### Retrieval, tests, and evidence

- Use `rank_bm25` and one dense model, `BAAI/bge-small-en-v1.5`, with exact model revision and tokenizer identity recorded at implementation time.
- Store dense vectors in persistent local Qdrant. Keep the same passage IDs, text, sentence IDs, and corpus across both retrieval methods.
- Keep labels, evidence groups, claim IDs, and split markers outside runtime documents and query inputs. Offline construction/evaluation may use them only for the declared gold-informed selection and scoring.
- Add focused automated tests for source schemas, hash-stable selection, exact page/sentence mapping, evidence coverage, multi-page groups, duplicate/empty/nonnumeric lines, split overlap filtering, label isolation, deterministic rebuilds, and stable retrieval outputs. Use independent small fixtures rather than committing FEVER claim text.
- Record the final passage count, input hashes, selected claim IDs in local evaluation-only data, per-class counts, evidence-page coverage, tokenizer lengths/truncation, index rebuild instructions, and BM25/dense Recall@k plus complete-evidence-group hits. State that the latter are custom FEVER subset retrieval metrics, not BEIR qrel scores.

Non-goals: any larger corpus, BEIR conversion, query expansion, hybrid retrieval, reranking, extra embedding models, Qdrant server, or public index release. Completion: one feature commit with implementation, focused regression tests, reproducibility evidence, required docs, and the milestone update.

## M1-F03 — Evidence-grounded verification CLI

- Type: build. Status: GATED BY M1-F02 and separate spending authorization. Budget: about 3 hours.
- Route: `builder-executor -> tdd-loop`.
- Entry: F02 passes; external processing settings are rechecked against current provider disclosures; the user authorizes any API spend. Planned initial ceiling is RM50. RM100 requires a separate decision.
- Model/provider candidate: `mistralai/mistral-small-3.2-24b-instruct` through DeepInfra via OpenRouter. Use provider-only routing, no fallback, ZDR-only account policy, request `zdr: true`, `data_collection: deny`, logging off, and a single dedicated key with a non-resetting USD limit corresponding conservatively to the authorized RM ceiling. No call occurs until authorization.
- Scope: Python pipeline with thin LangChain structured-output integration; one selected retrieval method; token-bounded passages; one verdict, concise evidence-grounded explanation, and source citation set.
- Acceptance: exact three labels; every citation maps to returned FEVER page title and sentence ID; invalid references rejected; empty/insufficient context yields NOT ENOUGH INFO; conflicts represented honestly; timeouts, provider errors, malformed output, and retries remain distinguishable and counted. Record model/version, provider, prompt/settings, latency, tokens, actual cost, and failures.
- Evaluation: 30 train development claims may guide prompt/threshold choices. Freeze them before the 60 held-out shared-task-dev claims. Keep labels/evidence unavailable to runtime. Report FEVER scorer and custom retrieval metrics as fixed-subset results. Manually inspect preselected outputs and retain traced failure cases for M3.
- Non-goals: multiple generation models, agents, browser/UI/API service, hosting, or second SDK implementation. Completion: one focused feature commit with tests, evaluation evidence, docs, and accurate M1 status.

## Milestone completion and retention

M1 exits after F01–F03 acceptance and verification. Keep the source/license/use record and small reproducibility metadata in Git. Keep raw corpus downloads, weights, indexes, API keys, claim-bearing fixtures, and generated bulk output outside Git. Purge scratch downloads and ignored handoffs when their useful findings have been promoted. Preserve unrelated changes and stage only M1-owned files. M1 completion updates the roadmap and does not activate M2 automatically.
