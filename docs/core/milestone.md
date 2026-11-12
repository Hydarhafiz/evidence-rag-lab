# Evidence RAG Lab — Milestone 2: Controlled Retrieval and LLM Evaluation

Status: **ACTIVE — plan and three-phase authorization procedure APPROVED on 13 November 2026.** Prepared 12 November; approval recorded 13 November. Execution window: 16–22 November; interview-ready target: 29 November 2026. Only M2 is active. M3 remains planned. F04 and F05 implementation have not started. Planning is a standalone completed deliverable; the user authorized its commit and instructed work to stop before F04. Each paid phase requires separate explicit authorization; no paid execution is authorized by plan approval.

## Entry evidence and outcome

M1 closeout is `15d3bf13f15691e1bba027a03782add8b15e7edd` (`docs(m1): close milestone 1`), following completed F01–F03. Recover the full M1 contract with `git show 15d3bf1:docs/core/milestone.md`; its concise completion summary remains in [roadmap.md](roadmap.md). Entry checks found a clean worktree, matching corpus/configuration identities and index manifest, and 44 passing offline tests. No held-out claim records or outcomes were inspected; the held-out file was hashed as opaque bytes only. No inference, secrets inspection or account inspection occurred.

Produce paired, reproducible observations explaining lexical versus semantic retrieval, evidence availability and context size versus verdict quality, and quality/latency/cost/complexity trade-offs. Negative results satisfy acceptance when measured and explained honestly; no accuracy target is imposed. Every result states **curated, gold-informed 5,000-passage FEVER subset; not full-FEVER benchmark performance**.

Keep the Python CLI, BM25, pinned BGE-small, persistent local Qdrant, ordinary orchestration and thin LangChain/OpenRouter boundary. Hybrid search, reranking, larger corpora, LangGraph, extra models, frontend, hosting, training and MLOps remain outside M2.

## Existing interfaces and minimal extension

`corpus.py` owns deterministic selection, four-sentence windows/stride three, source references and runtime/evaluation separation. `retrieval.py` owns `Retriever.search(claim, top_k)`, ranks and timing. `evaluation.py` currently provides only a train-only smoke loader, union-of-gold-sentences recall and complete-group hit at one k. `verification.py` owns the prompt, packing, retries, citation validation and run measurements; `openrouter.py` owns provider/schema controls. `cli.py` is the operator entry point.

Extend evaluation and CLI orchestration around these interfaces. Join IDs, labels, evidence groups and split membership offline after runtime processing. Preserve smoke's train-only guard and normal verification behavior; introduce an explicitly freeze-gated evaluation entry point for held-out work. Both retrievers remain supported; no path is being retired. Do not build an experiment framework or add dependencies except a justified minimal official-scorer dependency.

F05 must preserve safe parsed candidate verdict/citation diagnostics before rejection, without accepting invalid answers or enabling raw prompt/response logging. Currently rejected answers can lose those diagnostics, failed-attempt billing may be unknown, and local token estimates omit provider framing. Record response/request/model/provider metadata where supplied; disclose unknowns. The hosted model alias cannot prove immutable backend weights.

## M2-F04 — Retrieval evaluation

- Type: `build`. Status: APPROVED; NOT STARTED. Await a separate instruction to begin F04.
- Outcome: offline evaluator and train-development comparison, with a synthetically tested, gated held-out path.
- Scope: same-corpus BM25/dense at k=1,5,10; group-preserving metrics, per-claim ranks, failures, latency and available build/storage observations. No retriever tuning or corpus rebuilding.
- Entry: M1 closeout and identities verified; protocol approved. Read-only audit checks new numerical methodology, authoritative sources and split isolation.
- Route: `contract-auditor -> builder-executor -> tdd-loop`.
- Exit: F04-01–05 pass; exactly one completed feature commit contains evaluator, durable regressions, train aggregate evidence and necessary documentation. F05 owns the actual final held-out run after its full freeze; F04 completion must not imply held-out evidence exists.

| Acceptance | Required behavior and durable evidence |
| --- | --- |
| F04-01 | Check corpus/model/sample hashes, exact membership/counts, unique IDs and split isolation; reject malformed verifiable groups and nonfinite scores. Use synthetic held-out fixtures to prove gates without opening real held-out rows. |
| F04-02 | Implement the metric definitions below at all three k values. Independently calculated synthetic cases cover alternative/multi-page/multi-sentence groups, overlap deduplication, ties/ranks, empty results, errors and NEI denominators. Preserve groups and original references offline. |
| F04-03 | Evaluate only the existing 30 train claims during F04. Produce paired overall/label-specific results, recovery/ranking traces, actual denominators, failures and cause hypotheses. Verify ranking repeatability; do not relabel smoke observations as final results. |
| F04-04 | Report warm query p50/p95/counts, failures, environment/timing boundaries, startup, available indexing time and persistent storage characteristics. Explicitly mark unavailable/incomparable values and reasons. |
| F04-05 | Deliver reusable final-evaluation interface and compact train report; prove no gold leakage or accidental held-out access. Document actual implemented commands at completion, not speculative CLI capabilities. |

## M2-F05 — End-to-end generation evaluation

- Type: `build`. Status: APPROVED DIRECTION; NOT STARTED, depends on committed F04.
- Outcome: bounded runner, development experiments, frozen paired held-out retrieval/generation comparison and evidence ready for M3 interpretation.
- Scope: E2–E6 below, official scorer adapter, isolated annotation-selected control, packing/citation diagnostics, authorization/reservation, resumable records and bounded manual review. Reuse the existing verification/provider path.
- Non-goals: prompt/model sweeps, held-out tuning, production oracle retrieval, paid empty-context LLM baseline or evaluation service.
- Entry: F04 complete; read-only audit confirms official scorer identity/adapter, denominators, control/diagnostic isolation, rates and privacy authority. Offline/mocked gates precede paid execution.
- Route: `contract-auditor -> builder-executor -> tdd-loop`, with separate human authorization for each paid phase and pre-held-out freeze review. Authorization pauses do not create intermediate commits.
- Exit: F05-01–07 pass; exactly one completed feature commit includes runner, durable regressions, protocol/freeze identity, aggregate evidence and milestone update. Measured failures are evidence; omitted required experiments are incomplete work.

| Acceptance | Required behavior and durable evidence |
| --- | --- |
| F05-01 | Mocked tests prove claim/source-only requests, provider/privacy/schema controls, retry/spend bounds, no live default, freeze/split guards, pairing, resumability and duplicate prevention. Preserve no-call empty-context abstention and distinct operational failures. |
| F05-02 | Complete 120 development requests, 20 annotation-selected controls, 30 no-call empty controls and 12 extra repeatability requests under separately authorized phases with bounded internal chunks. Preserve failures and selection rationale; no extra sweep. |
| F05-03 | Materialize/review the full freeze below before any real held-out retrieval or generation. Select one common k for both methods and nominate an operational method using development only. |
| F05-04 | Run F04's evaluator on all 60 held-out claims at k=1,5,10, then 120 paired generation requests at the selected common k. No tuning, exclusions or winner-driven reruns. |
| F05-05 | Report all-label accuracy/macro-F1/confusion, applicable official versus custom metrics, citation validity/annotated correctness/manual grounding separately, abstention and unsupported definitive answers, and retrieval→packing→citation attribution. Put denominators/limitations beside results. |
| F05-06 | Report attempts/retries/errors, stage/end-to-end p50/p95 and counts, token completeness, dated rates, known versus unknown/reserved cost, and operator-supplied activity reconciliation where available. Failed requests never disappear. |
| F05-07 | Complete bounded human grounding review and repeatability comparison. Retain safe aggregate findings and locally traceable cases for M3; explain each experiment's result, confounds, engineering implication and interview explanation. Pass verification and retention gates. |

## Fixed data and pipeline identities

Official artifact URLs/hashes, rights and selection protocol remain in [planning-sources.md](planning-sources.md). Keep seed `FEVER-M1-F01-v1`, existing membership and exact raw page keys; do not reselect/rebalance. Development is 30 selected **train** claims, 10/class. Held-out is 60 selected **shared-task-dev** claims, 20/class, not FEVER's blind test set. Keep exact ordered ID/split lists in the local evaluation freeze; retain their hashes in safe metadata. Full sample hashes already bind IDs, text, labels and evidence groups.

| Identity | Accepted value |
| --- | --- |
| `data/processed/passages.jsonl` SHA-256 | `be6bdf115415a42dbb2a55585c160d157b2528875f14d70f990cdb1f6a7ceebe` |
| `evaluation_samples_train.jsonl` SHA-256 | `01f162c4db7a258c57b8f5275f4d4968239c244e3551dd5cb8c69a41a92f1685` |
| `evaluation_samples_dev.jsonl` SHA-256 | `ab3111841d8d6774389c13dcec63467654133e727f4d329582c2c7c47dd996a0` |
| Corpus coverage | 5,000 passages: 291 evidence-page + 4,709 background; accepted construction manifest records 103/103 selected unique gold pairs across 72 evidence pages. Verify per-split coverage at the permitted evaluation stage. |
| Encoder | `BAAI/bge-small-en-v1.5`, revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; CPU/eval, normalized 384 dimensions, cosine, 512-token limit; unchanged query instruction and title-prefixed passage input. |
| BM25 | `rank-bm25==0.2.2`, BM25Okapi defaults k1=1.5, b=0.75, epsilon=0.25; casefold/Unicode word tokens, no stemming/stopword changes. Both methods break ties by ascending stable passage ID. |
| Dense execution | Persistent local Qdrant exact search returns all 5,000 scores before sorting. Include query embedding, payload decoding and sorting in query latency; no ANN claim. |
| Generation | `mistralai/mistral-small-3.2-24b-instruct`, DeepInfra only via OpenRouter; temperature 0, output limit 512, timeout 30 s, max 2 attempts, backoff 0.25 s, SDK retries disabled. |
| Prompt/packing | `m1-f03-v1`; freeze exact prompt/template/schema hashes. 4,096 local message tokens including prompt/claim, whole passages in rank order, skip oversized passage then try later ones; no truncation, reranking, gold-aware rule or alternative packing policy. |
| Tokenizer | Tekken revision `9151ac4804f11eac6d7cd3b963eaf9ea1bc56839`, artifact SHA-256 `6e2501687ccd0e1f30f36319eaf2b46958b897811e246cd8eb5d385b9e3de7d1`. |
| M1 config baseline | `config/verification.toml` SHA-256 `f87b6e8762df9c54168fa0e7ee443a06a066ef5a3f0ecbd523f631f74304bd5a`; record explicit k variants and refreshed pricing independently, preserving historical identity. |

All labels, groups and split metadata stay evaluation-only. Runtime gets only claim/source text and citation coordinates. Construction used gold labels/evidence, including held-out evidence pages: this is an intentional **gold-informed corpus** limitation, not independent full-corpus retrieval. Disclose exact-title/source-coverage filters, balanced small samples, background selection and selection bias. Do not claim full-FEVER comparability or production reliability.

## Metric definitions and denominators

For verifiable claim c, G(c) contains its nonempty alternative gold groups of exact `(page_title, original_sentence_id)` pairs; U is their union. C is corpus references, Rk the union from first k **passages**, Pk the packed references, S the ordered distinct references actually cited by an accepted answer. Deduplicate pairs across overlapping windows; preserve group alternatives. This corpus has no fragmented sentences; fragment coverage would require a separately approved definition.

- **Corpus pair coverage:** `|U∩C|/|U|`; corpus complete-group coverage is `any(g⊆C)`. Report per-claim/macro verifiable results and unique-pair counts separately. Do not condition retrieval denominators on corpus availability silently.
- **Gold-union sentence Recall@k:** `|U∩Rk|/|U|`, macro mean per verifiable claim at 1,5,10. This continues M1's custom definition and penalizes missing alternatives even if one sufficient group is found. It is neither passage relevance recall nor BEIR qrel recall.
- **Complete evidence-group hit@k:** `1 if any(g⊆Rk) else 0`, macro mean; primary retrieval sufficiency metric. Cross-page and multi-sentence groups require all members of at least one alternative.
- **Ranking trace:** rank/score/passage ID/references, first rank recovering each pair and earliest complete-group rank within top 10. Missing means `>10/not recovered`, not absent below top 10. A single top-10 response yields nested k prefixes, not three timing samples.
- **Packing:** gold recall `|U∩Pk|/|U|`, complete-group hit after packing, unique-reference retention `|Pk|/|Rk|`, gold retention `|U∩Pk|/|U∩Rk|`, and lost complete groups. Zero conditional denominators are N/A plus counts. Preserve passage/token counts and omissions. These are distinct from verdict correctness.
- **NEI:** gold recall/group recovery/gold retention are N/A, not 0 or 1. Include NEI in query timings, context size, verdicts, abstention and citation validation. Gold-recovery denominators: 20 train / 40 held-out; verdict denominators per arm: 30 / 60.
- **Verdicts:** accepted correct labels / all scheduled claims; per-class precision/recall/F1, zero-division=0, unweighted macro-F1 across all three labels. Failed/missing/rejected outcomes have an ERROR confusion-table column, count as false negatives for their gold class and never become NEI. Report valid-response coverage and explicitly conditional success-only accuracy secondarily.
- **Abstention:** NEI prediction rate, gold-NEI recall, false abstention on SUPPORTS/REFUTES, definitive answers on gold NEI, with counts/denominators. Separate operational failure. Definitive answers lacking a complete cited gold group are **annotation-unsupported**, not automatically semantically wrong; human review assesses grounding.
- **Citation validity:** exact membership in packed supplied sources, source/ID validity, duplicates and missing required citations. Preserve safe rejected candidate-reference counts. Report valid/distinct candidate-reference fraction and fully valid-answer fraction over parseable candidates, plus parseability/accepted coverage over all scheduled claims. Empty citation sets are not 100% quality; report zero/missing/unparseable counts. Do not evaluate fabrication only on already accepted outputs.
- **Evidence correctness:** S versus U precision and complete-group hit for verifiable claims, separately from membership validity and human entailment. NEI has no gold evidence precision oracle. An annotation mismatch can be a valid alternative; never automatically call it semantic error.

For verifiable claims, cross-tabulate verdict correctness/errors against complete-group availability in corpus, retrieved passages, packed context and citations, with counts at each transition. This locates losses without pretending observational strata establish causality. Keep NEI analyses separate. Repeatability reports 12 scheduled pairs, counts of pairs with two valid responses, conditional verdict/citation agreement on those pairs, and one/both-failed counts; matching failures are not label agreement.

### Official scorer

Use the unchanged [official FEVER scorer](https://github.com/sheffieldnlp/fever-scorer), pinning commit/source hash/license before acceptance. An evaluation-only adapter supplies exact original gold annotations and aligned IDs. Flatten response citations in source/sentence order, deduplicate first occurrences, and use `max_evidence=5` **sentences**, independently of passage k. Report its strict score, label accuracy and evidence precision/recall/F1 with its native conventions. Test alignment and independent examples; never patch scorer formulas.

Official outputs require valid aligned responses. If some responses fail, report official outputs on the valid subset with coverage, plus custom **all-scheduled strict success rate** (strictly correct valid answers / scheduled claims). Do not call sentinel/error padding an official score. All-scheduled accuracy/macro-F1 remain primary operational measures. Preserve scorer exceptions or undefined arithmetic as unavailable, not invented values.

## Experiments and engineering interpretation

Reports must cover question, changed/constant variables, sample, primary/secondary metrics, confounds, observed result, implication and interview explanation. Use paired counts; no population/significance claims from these small curated samples.

| Experiment/question | Changed; held constant; sample | Primary; secondary metrics | Confounds, possible results and interview explanation |
| --- | --- | --- | --- |
| E1: Does semantic matching recover evidence missed lexically, at what operating cost? | BM25/dense, k=1,5,10 prefixes; same corpus/claims/ranking/CPU environment. Train 30, then frozen held-out 60. | Complete-group hit, gold-union recall; ranks, labels, p50/p95, build/storage. | Entity overlap, paraphrase and multi-hop needs are trace-checked hypotheses. Dense gains plus latency support a quality/complexity trade-off; BM25 parity/faster operation supports simpler defaults; mixed wins describe limitations, not admission of hybrid search. |
| E2: Does retriever evidence quality affect LLM verdicts at equal k? | Method; fixed LLM/provider/prompt/packing/claim/k. All 30 train claims at each k within 120-call factorial. | All-scheduled macro-F1/accuracy; cited complete groups, packing, grounding, latency/cost. | Service variance and conflicting/distracting evidence confound causality. Joint retrieval/verdict gain suggests propagation; retrieval-only gain points to packing/reasoning; verdict gain without gold recovery needs annotation/prior-knowledge review. Explains pipeline bottlenecks. |
| E3: Does k=5 help versus k=1 or add distractors/cost? | k within each method; same 30 train claims and generation settings, unchanged packer. | Macro-F1/paired correctness; group recovery before/after packing, tokens, latency/cost. | Additional evidence and distraction change together; packing may erase additions. Gain supports more context; loss suggests conflict/distraction; parity favors lower cost. Explains why more context need not help. |
| E4: Do development conclusions persist on untouched claims? | Both methods at one common development-selected k; everything else frozen. Held-out 60, 120 requests plus E1 retrieval. | All-scheduled macro-F1/paired correctness; applicable official strict score, citations/abstention/errors and operating costs. | Gold-informed/filter/balance bias and service drift limit generalization. Agreement supports a bounded choice; reversal is reported as instability, never retuning. Explains honest final comparison. |
| E5: Is the bottleneck missing evidence or using supplied evidence? | Train only: 20 verifiable annotation-selected contexts, 30 empty contexts; same claim/prompt/model/limits/validator. Compare selected-k development arms. | Verdict and complete evidence/citation retention; human grounding and application abstention. | Selected passages include neighbors and deliberate gold selection: diagnostic, not deployable retrieval or guaranteed upper bound. Improvement suggests retrieval bottlenecks; errors with complete supplied evidence suggest generation/citation issues. Empty context tests application abstention only, because it bypasses the LLM. |
| E6: Are fixed-setting outputs repeatable? | Six train IDs (2/class) chosen by seed before outputs; repeat both methods once at selected k, 12 extra requests versus 12 factorial originals; same cached contexts/messages/settings. | Verdict and citation-set agreement over 12 pairs; errors, tokens and latency differences. | Temperature 0/service alias is not deterministic; six claims cannot estimate reliability. Disagreement qualifies confidence and explains why identities plus observed variability matter. |

E5 is evaluation-only, never a normal retrieval mode. Choose the nonempty group with fewest distinct pairs, ties by lexicographic pair list. Greedily select fixed-corpus passages covering most uncovered chosen-group pairs, ties by passage ID, retaining selection order; use the same whole-passage packer. Supply only claim and ordinary source text/coordinates, no label, gold-group/oracle cue or split hint. Record neighbor text, passage count and packing loss offline. Keep all 20 scheduled verifiable cases even if complete evidence does not fit. NEI has no annotation-selected gold context. No held-out annotation control or paid empty-context control is proposed.

## Selection and pre-held-out freeze

1. After a separate instruction to resume, implement/verify F04 and then F05 offline controls. Pass independent formula/scorer and mock/privacy/authorization tests before requesting paid-phase authorization.
2. Obtain explicit authorization for the concrete development phase, then run its bounded factorial chunks. Prompt/model/packing stay fixed; only common k and operational-method nomination are selected. No threshold/prompt sweep.
3. Choose common k by higher mean all-scheduled macro-F1 across both methods on train; ties choose k=1. Nominate a method at that k by macro-F1, then accuracy, then fewer operational failures, then lower measured token cost, then BM25 if still tied/cost unavailable. Obtain separate authorization for the controls/repeatability phase using that selection; those diagnostics do not reopen the selection rule. Treat small differences cautiously; final evaluation retains both arms.
4. Before any held-out evaluation, materialize an immutable local freeze and compact text-free durable summary. Bind protocol `m2-evaluation-v1`; source/corpus/sample hashes; exact ordered ID/split lists and hashes; code revision plus working-tree source digest while F05 is uncommitted; environment/dependencies/hardware; both retrieval settings and encoder revision/instruction; k grid/selected k/selection rule/nomination; exact prompt/template/schema/version hashes; model/provider/observed endpoint; tokenizer; packing/token/output limits; retries/failure/denominator rules; scorer revision/hash; latency boundaries; run ordering; controls/review sampling; rates/FX assumptions and batch/cumulative caps.
5. Human reviews the complete freeze and separately authorizes the concrete held-out phase. Then the gated evaluator may open held-out records, run offline retrieval and final generation. An integrity-only adapter may record held-out IDs before the gate without exposing text/labels/evidence to runtime or evaluating them; file hash already binds membership. No retrieval/prediction precedes the gate.
6. Process stable seed-hash claim order, interleave method pairs and alternate first method by claim position. Alternate development k-block order by position too. No held-out-driven settings, substitutions, successful-answer retries, sample expansion or better-score reruns. Preserve originals; incomplete batches are provisional, never used to stop for quality. A compatibility change requires a separate human decision and distinct protocol; price-only refreshes are dated, not silent scientific changes.

This is a **freeze specification**, not a claim that final code/scorer hashes or selected k exist in this planning draft.

## Latency, storage and failure methodology

Retrieval: one fixed synthetic warm-up per method; exclude it. Reuse initialized retrievers, single concurrency, three top-10 queries/claim, score first rankings and check repeat ranking stability. Counts: 90 timed queries/method train, 180 held-out. Timings repeat queries, not independent quality observations. p50/p95 use nearest rank `ceil(p*n)`, one-based; report success/attempt/failure counts and range. NEI participates. Failed first queries contribute zero recovery for scheduled verifiable claims, not later-success replacements; report failed timing separately. No generation warm-ups.

Report startup/model loading/file reading separately. BM25 build excludes corpus read; historical dense build includes encode/upsert/count but excludes encoder load/read. Label historical dense 190,207 ms as an M1 observation, not a controlled contemporaneous rebuild. Record current corpus bytes, closed Qdrant directory bytes, count/dimensions and separate model-cache bytes where available. BM25 has no persisted index; do not imply zero memory. State CPU/OS/device/thread/package/cache/workload conditions. No unsolicited rebuild or optimization.

Generation: record retrieval/packing/provider attempts/validation/retry-backoff/end-to-end boundaries, UTC timestamps and logical request/config IDs. Primary online latency uses initialized resources; startup is separate. Report success-only p50/p95/counts plus failed-operation and all-attempt summaries. Missing timings are unknown. Record input/output usage for each attempt where provided, known cost and conservative unknown-billing reservation separately; operator activity reconciliation is not a card settlement claim.

Exactly one logical outcome per `(experiment, split, claim ID, method, k, repeat)` key. Persist dispatch/attempt status and reserve worst-case spend before sending. Serial calls, at most one existing transient retry; no malformed/citation/permanent retries. A crash after dispatch with no answer is unresolved/potentially billed: never auto-resend. Reconcile or retain missing status; reissue requires explicit approval as a distinct attempt/deviation. Resume skips dispatched work. Failed, missing, cancelled and queued slots remain visible; an interrupted run is incomplete, not a finished survivor score.

Trace categories: corpus absence, retrieval miss, packing group loss, wrong verdict despite complete supplied evidence, incomplete citation group, fabricated/missing/duplicate citation, conflict handling, schema/truncation, timeout/rate-limit/provider error, unknown usage. Mark inferred causes as hypotheses; correct labels alone do not prove grounding.

## Budget and paid-phase authorization

The user reports account privacy controls and a dedicated spending limit are configured. Do not inspect account state or secrets. Reconfirm operator attestation/current endpoint disclosures before authorizing each paid phase: DeepInfra only, no fallback, account ZDR-only excluding Safety Retention, request `zdr=true`, `data_collection=deny`, `require_parameters=true`, logging/training opt-ins off. Internal chunk checks below enforce the same restrictions throughout the phase. [Sources and caveats](planning-sources.md#m2-evaluation-sources-and-authority-gaps) remain binding.

Checked 12 November 2026: the [public endpoint listing](https://openrouter.ai/api/v1/models/mistralai/mistral-small-3.2-24b-instruct/endpoints) lists DeepInfra FP8 with structured outputs at **USD0.075/M input, USD0.20/M output tokens**. This metadata read is not inference or assurance of future availability.

| Experiment | Logical inference requests | Maximum attempts | Internal chunks ≤10 logical requests |
| --- | ---: | ---: | ---: |
| Development: 30 claims × 2 methods × 2 k | 120 | 240 | 12 |
| Annotation-selected train controls | 20 | 40 | 2 |
| Repeatability: 6 claims × 2 methods × 1 extra repeat | 12 | 24 | 2 (10 + 2) |
| Frozen held-out: 60 claims × 2 methods × selected k | 120 | 240 | 12 |
| Total | **272** | **544** | **28** |
| Empty-context control | 30 local outcomes, zero requests | 0 | Offline |

Working assumption: 1,500 input + 200 output tokens/request, above M1's single observed 1,009/93-token call, not a measured sample mean. Total **408,000 input + 54,400 output; USD0.04148** before retries. Formula: `(input*0.075 + output*0.20)/1,000,000` USD; no cache discount.

Conservative reservation: 4,608 provider input tokens (4,096 local budget + 512 framing/schema allowance) and 512 output tokens per attempt, all 544 attempts: **2,506,752 input + 278,528 output; USD0.243712**. Add 25% contingency: **USD0.30464, rounded USD0.31**. Input allowance is a planning assumption, not guaranteed billing; preflight serialized/schema overhead and check returned usage, stopping subsequent dispatch on overruns. Use **USD0.35 as the cumulative planning reservation** to cover per-chunk rounding, within **RM50 total project ceiling**. RM100 requires a separate decision.

Use **RM5/USD as a conservative planning assumption**; [BNM's 12 November display](https://www.bnm.gov.my/web/guest/exchange-rates) is approximately RM4.084/USD, not a card settlement rate. USD0.35 is RM1.75 at the planning conversion. RM50 implies at most USD10 including prior usage and funding/FX fees at that conversion; neither amount authorizes spending. Include M1's USD0.0000943 activity charge in project consumption; no top-up is authorized here.

Each internal chunk is limited to **10 logical requests / 20 attempts / USD0.012 reserved inference spend**. Normal paired chunks contain five claims × two methods. Chunk continuation is permitted only within the separately authorized phase manifest and remaining limits below. Offline tests/retrieval/packing/token counting make zero API calls.

Stop dispatch on exhausted request/cost reservation, three consecutive failed logical requests, provider/model/privacy/config mismatch, unexpected permanent auth/compatibility/schema-control error, or usage above reservation assumptions. Missing usage consumes full reservation until reconciled, never zero. Keep remaining slots incomplete. No fallback/top-up/new model/extra requests/RM100 expansion; no quality-based stop. At normal phase end request explicit authorization for the next concrete phase.

### Approved procedure — three phase approvals with bounded internal chunks

**APPROVED on 13 November 2026.** Three separately authorized phases replace the initial proposal's 28 approval prompts. Sample membership, experiments, metrics, privacy, scientific settings, ordering and request counts are unchanged. Approval of this procedure does not authorize inference.

Request three explicit phase approvals, each binding an exact local request-manifest hash, configurations, current rates/privacy attestation, maximum requests/attempts, and phase/cumulative spend reservations. Within an approved phase, run the predefined chunks serially without a new prompt for each chunk. “Phase” remains local orchestration of synchronous calls, not a provider batch API.

| Separately authorized phase | Logical requests | Maximum attempts | Internal chunks ≤10 requests | Phase reservation ceiling |
| --- | ---: | ---: | ---: | ---: |
| Development factorial | 120 | 240 | 12 | USD0.144 |
| Development-selected controls and repeatability | 32 (20 + 12) | 64 | 4 (10 + 10 controls; 10 + 2 repeats) | USD0.048 |
| Frozen held-out comparison | 120 | 240 | 12 | USD0.144 |
| Total | **272** | **544** | **28** | **USD0.336**, within the existing USD0.35 cumulative reservation and RM50 ceiling |

Each chunk retains the existing maximum of 10 logical requests, 20 attempts and USD0.012 reservation; the last small chunk is not permission to fill unused slots. Counters and reservations are cumulative across chunk boundaries and restarts. No transfer of unspent phase budget authorizes extra requests. Keep failed/unknown attempts, paired claim ordering and all-scheduled denominators exactly as specified. The three-consecutive-failure counter must not reset at a chunk boundary.

Before every internal chunk, verify remaining authorization, exact queued request/config identities, unchanged model/provider/privacy/rates, unresolved dispatch state and all spend/attempt caps. Checkpoint the ledger and emit a compact progress summary at each chunk boundary. Missing metadata consumes the conservative reservation; a successful prior authorization is never inferred from elapsed time or a CLI flag alone. Record the user's approval scope locally with the bound manifest; no new approval service, signature system or framework is needed.

An ordinary interruption may resume **the same approved phase only**, within its remaining manifest/caps, if identities are unchanged and the dispatch ledger proves which requests are safe to issue. Never automatically resend an uncertain dispatch. Explicit user stop/revocation, any safety/compatibility stop condition, identity/rate/privacy changes, scope changes, uncertain-dispatch reissue, exhausted caps, or a phase transition require a fresh concrete authorization. A phase approval never preauthorizes a later phase. Held-out phase approval still requires the full freeze review; development-selected diagnostic identities cannot be approved before selection exists.

Trade-off: a phase approval delegates more routine continuation than a ten-request approval. The fixed manifest, chunk/phase/cumulative caps, serial execution, stop conditions and persistent ledger preserve the technical controls; one concise approval per phase reduces prompts and bookkeeping. The user accepted this delegation as part of the milestone plan. Each actual paid phase still requires its own later authorization.

## Manual grounding, retention and exit

Preselect two IDs/class from train and two/class held-out by seed ordering before outputs; inspect both selected-k method outputs: at most **24 outputs**. Reuse the six train IDs for repeatability. Add at most **six diagnostic outputs**, first unused output per category in priority order: wrong definitive verdict, false NEI, annotation-unsupported definitive answer, citation rejection, packing loss, provider/schema failure; ties by stable claim/method order. Report missing/rejected outputs without substitution and fewer cases if categories are absent. Maximum review: 30 outputs.

Human records whether packed text supports the explanation, cited sentences justify the relation, contradictions are handled and NEI is warranted by available text; use supported/unsupported/unclear plus reason. Hide method names where practical. One reviewer's bounded sample is qualitative grounding evidence, distinct from automated label/annotation metrics. Codex can prepare traces/notes; human confirmation owns the manual judgments. Required before F05 completion.

Keep real claims/groups/source text, weights/indexes, detailed generated outputs and local freeze/dispatch ledgers in existing ignored data/model/generated directories. Keep safe structured local result records and hashes/regeneration instructions without enabling provider/raw prompt-response logging. Git retains code, capability-named synthetic regressions, compact text-free protocol/freeze metadata, source identities, aggregate reports and sanitized failure categories/ID trace references. No claim/evidence subset, article text or bulk inference output enters Git. Preserve accepted identities and irreplaceable observations locally; promote useful decisions, purge disposable scratch/logs and execution handoffs at completed feature boundaries.

Provenance: dataset/corpus/sample, encoder, model/provider, prompt/schema and normal runtime behavior **unchanged**; controlled evaluation methodology/freeze/report identity **newly introduced**; k configurations and dated price observations are explicit independent variants. Unexpected material identity changes require audit/decision, never silent edits.

Verification: focused independent capability cases plus `.venv/bin/pytest -q`, `.venv/bin/ruff format --check src tests`, `.venv/bin/ruff check src tests`, `.venv/bin/python -m compileall -q src`, `.venv/bin/python -m pip check`, applicable offline CLI help and `git diff --check`. No new formatter/type-checker/CI policy. `tdd-loop` owns verification, retention and one commit/feature. `sdd-router` prepares ignored untracked execution handoffs and runs preflight against approved scope. Durable planning decisions belong to this approved standalone plan; routine audit notes and handoffs stay ignored. Subsequent implementation features each own one completed-feature commit.

After F04/F05 completion, `release-manager-sync` reconciles actual M2 evidence/roadmap in one documentation-release closeout, as for M1. It cannot repair unfinished experiments or activate M3. M2 exits with reproducible comparisons, limitations, manual review, actual usage/error records and M3-ready interpretation material. Protect 23–29 November for case study/reproduction/interview preparation; cut optional experiments/polish before moving the target.

## Approval and remaining prerequisites

The user approved the methodology, F04→F05 direction and three-phase authorization procedure on 13 November 2026, preserving metric separation, frozen protocol, official FEVER scoring, gold-informed limitations and all-scheduled failure accounting. The user explicitly authorized one standalone planning commit and instructed work to stop before starting F04. F04/F05 implementation, paid inference and held-out evaluation remain unstarted. Planning is complete; implementation resumes only on a separate instruction.

No product-redesign/dataset-access blocker was found. Real future prerequisites: implemented/verified F04; F05 evaluator/authorization controls and offline tests; pinned official scorer and compatibility audit; development results selecting k; complete freeze review; current endpoint/privacy/rate attestation and separate authorization before **each** paid phase; bounded human grounding review. The next human checkpoint is an instruction to begin F04; no further milestone-plan approval is needed. Provider incompatibility returns for a separate decision while retaining the approved preference.
