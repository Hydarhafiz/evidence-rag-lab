# M1 — Accepted dataset and minimal functioning RAG

Status: ACTIVE. Target window: 9–15 November 2026; overall target remains 29 November 2026. M2 and M3 are planned but not active.

Dataset gate: ACCEPTED on 10 November 2026. The official FEVER (2018) claims and June 2017 Wikipedia corpus are the project’s primary source. Detailed artifact identity, licensing, access, evaluation evidence, and the accepted use matrix are in [`planning-sources.md`](planning-sources.md).

## Outcome and boundaries

Build the smallest useful Wikipedia-based claim-verification RAG: deterministic approximately 5,000-passage corpus, BM25 and one dense retrieval path, local Qdrant, and a Python CLI that returns SUPPORTS, REFUTES, or NOT ENOUGH INFO with validated page-title/sentence-ID citations and abstention behavior.

Use FEVER’s official license conditions for private local indexing, embeddings, evaluation, and demonstrations. GitHub publication is limited to code, source identity, protocol, and aggregate results. Raw article text, claim/evidence subsets, passage indexes, Qdrant data, and embeddings are excluded from Git. External inference uses OpenRouter/DeepInfra with the privacy controls recorded in `planning-sources.md`; only the one bounded M1-F03 test below was separately authorized, and no general or ongoing spend is authorized. Public hosting is excluded and is not an M1 prerequisite.

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
| M1-F01-05 | External inference provider, training/retention policy, routing, and spending controls reviewed for the selected candidate. | PASS as a conditional future path; at F01 completion account settings and spend remained unconfigured. F03 later records one separately authorized bounded check and an operator attestation of account controls. |
| M1-F01-06 | Fixed evaluation claims/evidence, retrieval and verdict metrics, official scorer semantics, sample limitations, and gold-informed selection disclosed. | PASS — contract recorded; no full-benchmark claim. |
| M1-F01-07 | Define the evaluation/runtime boundary: labels, evidence groups, claim IDs, and split metadata stay out of runtime retrieval and generation. | PASS — F02/F03 contracts and focused tests enforce this boundary. |
| M1-F01-08 | Human accepts the source and scope with no unresolved material blocker for the required private experiments. | PASS — user approved FEVER and the audit evidence on 10 November 2026. |

The record does not infer a license from a mirror or claim that all historical article-page exceptions were separately inspected. FEVER’s publisher terms govern; use known page-specific terms and the stated CC BY-SA 3.0 fallback. SciFact is superseded due to unresolved rights. Its technical SciFact/BEIR compatibility was accepted previously and remains a historical finding, not a failure reason.

## M1-F02 — Deterministic mini-corpus and retrieval

- Type: build. Status: COMPLETE; verified 10 November 2026.
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

### F02 implementation and verification record

The three official source files were restored locally because they were absent from the checkout. `fever-rag prepare` checked all three F01 SHA-256 values before parsing and again after construction; all matched. The wiki archive contains 224 members and 109 `wiki-pages/wiki-###.jsonl` shards. It also contains macOS sidecars, which ingestion excludes by path. The scan saw 5,416,537 page records, 5,416,536 non-empty unique page-title IDs, one empty page ID, one duplicate sentence ID on one page, 518 nonnumeric line rows, and 16,793,196 rows with empty sentence text. The empty ID is skipped; duplicate sentence IDs are removed from source passages, and the page with the duplicate ID is excluded from background selection.

The deterministic selection produced 30 train development claims (10 per label) and 60 held-out development claims (20 per label), stored in separate ignored `evaluation_samples_train.jsonl` and `evaluation_samples_dev.jsonl` files. The smoke loader accepts only the train file. The selected claims contain 107 evidence groups, 72 exact source pages, and 103 unique page/sentence pairs. The corpus includes all 103 pairs. The 72 evidence pages produced 291 four-sentence, stride-three passages after empty rows were removed. Background selection considered 10,410,277 windows and added 4,709 to reach exactly 5,000 passages. The corpus is gold-informed because both splits’ evidence pages were used in construction.

Runtime passage records expose only `passage_id`, `page_title`, `sentence_ids`, `text`, `source_artifact`, and `source_sha256`. The JSONL corpus SHA-256 is `be6bdf115415a42dbb2a55585c160d157b2528875f14d70f990cdb1f6a7ceebe`. Stable passage IDs use UUID5 over the exact page title, sentence references, and text. The longest built passage used 424 of the model’s 512 tokens; no truncation or sentence splitting occurred for this corpus. Synthetic tests cover over-limit sentence splitting while retaining the original sentence ID.

The dense encoder is `BAAI/bge-small-en-v1.5` at immutable revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. It emits 384-dimensional normalized vectors, uses cosine similarity, has a 512-token limit, and is kept in an ignored local model cache. Passage input is title, newline, and sentence text; query input uses `Represent this sentence for searching relevant passages: ` before the claim. Local Qdrant stores all 5,000 vectors persistently under a collection name derived from the corpus hash. The Qdrant build took 190,207 ms on this CPU environment. BM25 is rebuilt in memory from the same passage file and took about 100 ms. Both methods break score ties by ascending stable passage ID.

The final F02 development-only retrieval smoke evaluated the 20 verifiable claims in the 30-claim train sample at top 5. BM25 page/sentence Recall@5 was 0.88125, complete-evidence-group hit rate was 0.85, mean query latency was 15.39 ms, and median latency was 6.26 ms. Dense Recall@5 was 0.89375, complete-group hit rate was 0.90, mean query latency was 154.04 ms, and median latency was 110.08 ms. BM25 index construction took 104.97 ms in this run; the persisted Qdrant index was built from the same SHA-256-identical corpus and took 190,207 ms on the CPU environment. These custom curated-subset smoke observations are not BEIR qrel scores or final M2 evaluation results. No held-out retrieval or generation outcomes were inspected, and no retrieval settings were tuned against the 60 held-out claims.

Verification environment: Python 3.12.3, NumPy 2.5.4, `rank-bm25` 0.2.2, Sentence Transformers 6.2.0, PyTorch 2.15.0+cpu, Transformers 5.22.2, `qdrant-client` 1.20.0, pytest 9.2.0, and Ruff 0.17.4. Direct project dependencies and the test/lint tools are pinned in `pyproject.toml`.

Verification passed: `pytest -q` (14 passed), `ruff format --check src tests`, `ruff check src tests`, and `python -m compileall -q src`. The real local 5K index build, persistence across a separate CLI process, and development-only `fever-rag smoke --top-k 5` also completed. No manual product or external-provider check was required; no OpenRouter call was made.

Limitations: the corpus is a small gold-informed subset with a deterministic but nonrepresentative background draw. The smoke metrics cover only the train development sample and are not verdict metrics. Local Qdrant performs exact brute-force search; retrieving all 5,000 scores and sorting them by passage ID makes tie behavior deterministic but may increase latency as the corpus grows. Raw FEVER files, the selected evaluation sample, model weights, dense index, and generated manifests remain outside Git.

## M1-F03 — Evidence-grounded verification CLI

- Type: build. Status: COMPLETE; LIVE TEST AND PROVIDER ACTIVITY COST RECORDED. Budget: about 3 hours.
- Route: `builder-executor -> tdd-loop`.
- Entry: F02 passes; external processing settings are rechecked against current provider disclosures; the user authorizes any API spend. Planned initial ceiling is RM50. RM100 requires a separate decision.
- Model/provider candidate: `mistralai/mistral-small-3.2-24b-instruct` through DeepInfra via OpenRouter. Use provider-only routing, no fallback, ZDR-only account policy, request `zdr: true`, `data_collection: deny`, logging off, and a single dedicated key with a non-resetting USD limit corresponding conservatively to the authorized RM ceiling. The implementation is bounded to one claim and at most two provider attempts; the CLI also requires an operator attestation that account controls are configured. One separately authorized live check completed on an approved train-only example; no held-out generation evaluation was run.
- Scope: Python pipeline with thin LangChain structured-output integration; one selected retrieval method; token-bounded passages; one verdict, concise evidence-grounded explanation, and source citation set.
- Acceptance: exact three labels; every citation maps to returned FEVER page title and sentence ID; invalid references rejected; empty/insufficient context yields NOT ENOUGH INFO; conflicts represented honestly; timeouts, provider errors, malformed output, and retries remain distinguishable and counted. Record model/version, provider, prompt/settings, latency, tokens, actual cost, and failures. The application estimated USD 0.000094275 from provider-reported tokens and dated rates. A user-supplied provider activity screenshot for the matching model/provider and 1,009 input/93 output tokens displays cost USD 0.0000943 with usage type `Credits`; this is the provider-dashboard activity cost, rounded as displayed, and is distinct from the application estimate. No invoice or card-settlement record is asserted. Implementation pins DeepInfra only, disables fallbacks, requests ZDR/data denial/required parameters, and requests strict JSON Schema. On 11 November 2026, `pytest -q` passed 44 tests (30 F03 tests and 14 earlier tests); Ruff format/check, compilation, `pip check`, CLI help, and SDD preflight passed. The approved manual train-only check returned REFUTES on the first attempt, cited the annotated FEVER source sentence, and matched the train annotation. The run used corpus SHA-256 `be6bdf115415a42dbb2a55585c160d157b2528875f14d70f990cdb1f6a7ceebe`, configuration SHA-256 `f87b6e8762df9c54168fa0e7ee443a06a066ef5a3f0ecbd523f631f74304bd5a`, and prompt `m1-f03-v1`. The provider reported 1,009 input and 93 output tokens; local message tokenization counted 1,002, a seven-token framing difference for this request. End-to-end latency was 9.048 seconds. The CLI output attested that account controls were confirmed by the operator; they were not independently inspected. No held-out generation outcome was read. The pinned tokenizer hash was verified and local packing included five of five passages at 1,002/4,096 tokens with no omissions.
- Evaluation: 30 train development claims may guide prompt/threshold choices. Freeze them before the 60 held-out shared-task-dev claims. Keep labels/evidence unavailable to runtime. Report FEVER scorer and custom retrieval metrics as fixed-subset results. Manually inspect preselected outputs and retain traced failure cases for M3.
- Non-goals: multiple generation models, agents, browser/UI/API service, hosting, or second SDK implementation. Completion: one focused feature commit with tests, evaluation evidence, docs, and accurate M1 status.

## Milestone completion and retention

M1 exits after F01–F03 acceptance and verification. Keep the source/license/use record and small reproducibility metadata in Git. Keep raw corpus downloads, weights, indexes, API keys, claim-bearing fixtures, and generated bulk output outside Git. Purge scratch downloads and ignored handoffs when their useful findings have been promoted. Preserve unrelated changes and stage only M1-owned files. M1 completion updates the roadmap and does not activate M2 automatically.
