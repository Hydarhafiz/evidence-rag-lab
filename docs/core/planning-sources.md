# Planning sources and accepted dataset scope

Record date: 10 November 2026. Owner: M1-F01. Dataset decision: **ACCEPTED for the defined private, local FEVER experiments.** The verification below promotes the read-only FEVER audit accepted by the user. It records source facts, licensing evidence, and use decisions separately; it is not a legal opinion or a blanket redistribution permission.

## Source register

| Source | Accepted relevance |
| --- | --- |
| [Official FEVER 2018 dataset page](https://fever.ai/dataset/fever.html) | Primary publisher page for training, labeled development, blind test, and preprocessed June 2017 Wikipedia page artifacts; dataset citation. |
| [FEVER license notice](https://fever.ai/download/fever/license.html) | Governs use of annotations incorporating Wikipedia material: applicable article-page terms, with CC BY-SA 3.0 where those terms are unavailable. |
| [FEVER paper](https://aclanthology.org/N18-1074/) and [official FEVER scorer](https://github.com/sheffieldnlp/fever-scorer) | Dataset provenance and shared-task scoring semantics. The scorer repository identifies its code as Apache-2.0; that code license is separate from dataset content terms. |
| [CC BY-SA 3.0 deed](https://creativecommons.org/licenses/by-sa/3.0/) and [legal code](https://creativecommons.org/licenses/by-sa/3.0/legalcode.en) | Fallback text license: copying and adaptation for any purpose, subject to attribution and ShareAlike for distributed adaptations. |
| [Wikipedia copyright policy](https://en.wikipedia.org/wiki/Wikipedia:Copyrights) | Current policy describes CC BY-SA 4.0 for much text. The FEVER archive’s own notice and explicit version 3.0 fallback control this historic dataset record; the current policy is not substituted for FEVER’s notice. |
| [OpenRouter model/provider page](https://openrouter.ai/mistralai/mistral-small-3.2-24b-instruct/providers) | Candidate `mistralai/mistral-small-3.2-24b-instruct`, structured JSON output, provider list, listed prices, and routed availability caveat. |
| [BAAI BGE-small model card](https://huggingface.co/BAAI/bge-small-en-v1.5) and [pinned model revision](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a) | F02 uses `BAAI/bge-small-en-v1.5` at revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. The model card identifies the released model as MIT licensed, with 384 dimensions and a 512-token sequence length. |
| [OpenRouter provider policies](https://openrouter.ai/providers), [data collection](https://openrouter.ai/docs/guides/privacy/data-collection), [ZDR](https://openrouter.ai/docs/guides/features/zdr), and [provider routing](https://openrouter.ai/docs/guides/routing/provider-selection) | Provider-level training/retention disclosures and required privacy/routing controls. |
| [DeepInfra terms](https://deepinfra.com/terms) and [privacy policy](https://deepinfra.com/privacy) | Current provider terms describe processing to deliver service, no training, and no routine content retention after processing, subject to operational, security, abuse, and legal exceptions. Verify the exact endpoint and terms again before a later call. |
| [OpenRouter API key controls](https://openrouter.ai/docs/api/api-reference/api-keys/create-keys) | Per-key USD spending limits and reset options; use a non-resetting cap for any separately authorized budget. |

## Verified official artifacts

The FEVER claim files and `wiki-pages.zip` were downloaded directly from the official release links on 10 November 2026. All returned HTTP 200 with no redirect. No publisher SHA-256/MD5 value was published on the page or response; the hashes below identify the inspected copies locally. The ZIP was streamed without executing archive content; CRC checks completed for all members.

| Artifact | Size | Rows / structure | Local SHA-256 |
| --- | ---: | --- | --- |
| `train.jsonl` | 33,024,303 bytes | 145,449 unique IDs; `claim`, `evidence`, `id`, `label`, `verifiable`; 80,035 SUPPORTS, 29,775 REFUTES, 35,639 NOT ENOUGH INFO | `eba7e8f87076753f8494718b9a857827af7bf73e76c9e4b75420207d26e588b6` |
| `shared_task_dev.jsonl` | 4,349,935 bytes | 19,998 unique IDs; same fields; 6,666 per label | `e89865bfe1b4dd054e03dd57d7241a6fde24862905f31117cf0cd719f7c78df7` |
| `wiki-pages.zip` | 1,713,485,474 bytes | 224 archive members; 109 `wiki-pages/wiki-###.jsonl` shards; 5,416,537 page records, of which 5,416,536 have non-empty unique page-title IDs; about 7.62 GB uncompressed | `4b06d95da6adf7fe02d2796176c670dacccb21348da89cba4c50676ab99665f2` |

The archive has macOS `__MACOSX` sidecar members in addition to the 109 wiki shards; F02 filters shard paths explicitly and never treats sidecars as data. The archive had no unsafe paths, and F01 CRC checks passed for all members. Page records have string fields `id`, `text`, and `lines`. `id` is the FEVER page-title key, not a numeric MediaWiki page ID. Sentence references are integer positions stored in `lines`. F02 source validation found one record with an empty `id`; it is counted and excluded from the passage corpus. This corrects the earlier shorthand “5,416,537 unique page-title IDs,” which counted page records rather than usable non-empty IDs.

Each claim evidence alternative is a group of tuples `[annotation_id, evidence_id, page_title, sentence_id]`. SUPPORTS and REFUTES have evidence groups; NOT ENOUGH INFO tuples have null page-title and sentence fields and are annotation metadata, not retrievable evidence. The train and development files have no claim-ID overlap. Three normalized claim texts occur in both; the fixed development selection excludes those duplicates.

In `lines`, the first tab-delimited field is the sentence ID, the second field is sentence text, and trailing fields contain linked-entity values. The index uses only the sentence text and retains the original ID; trailing entity fields and nonnumeric rows are excluded. One corpus page contains a duplicate sentence ID and is excluded from the mini-corpus candidate pool.

The pinned BGE tokenizer and sentence encoder loaded successfully in F02 with a 384-dimensional output and a 512-token limit. The implementation uses the model’s recommended search instruction on queries and no prefix on passages. Weights remain in the ignored local model cache; they are not committed.

FEVER has 170 reference title keys whose Unicode spelling differs from archive IDs. All 170 map to an NFC-equivalent source key, without collisions; all 356 affected sentence references exist under that key. To preserve the official scorer’s exact page-name behavior in M1, the selected claims must use exact raw page-title matches. The fixed sample verified below satisfies this; excluded claims are counted as source-coverage limitations rather than silently normalized in scoring.

## Mini-corpus and evaluation contract

Use seed `FEVER-M1-F01-v1`. Rank claims within each source split and class by SHA-256 of `FEVER-M1-F01-v1|{split}|{claim_id}` in UTF-8. The split keys are `train` and `dev`, where `dev` is `shared_task_dev.jsonl`. Select 10 claims per class from train for development and 20 per class from labeled development for held-out evaluation. Before selection, exclude normalized claim-text overlaps with train, claims with any non-exact/unavailable page key or sentence reference, and references to duplicate sentence IDs.

The audited selection contains 30 development and 60 held-out claims; 107 evidence groups, 72 exact-match evidence pages, and 103 unique page/sentence pairs. All selected references resolve. Four-sentence windows with stride three yield approximately 389 windows across the selected evidence pages; the builder recalculates the exact count after dropping empty sentence text. Fill to approximately 5,000 passages with background windows from other pages, ordered by SHA-256 of `FEVER-M1-F01-v1|background|{page_id}|{first_sentence_id}`. This is deterministic and independent of system results. Nearby evidence-page sentences and broad background pages provide context and distractors; no claim is made that this is a representative full-corpus sample or a curated hard-negative benchmark.

F02 measured 291 evidence-page passages after dropping empty and nonnumeric rows, followed by 4,709 background passages to reach exactly 5,000. The 103 selected evidence pairs all occur in the runtime corpus. The evidence-page count differs from the 389 full-window estimate because the estimate preceded filtering empty sentence text and applying the verified sample construction.

Keep the complete FEVER gold groups and verdicts in evaluation-only data. Runtime records contain only passage text, exact source page-title key, original sentence-ID list, and source attribution metadata. Multi-sentence and multi-page groups are reconstructed from retrieved original sentence references. Retrieval evaluation uses custom page/sentence Recall@k and complete-group hit metrics. End-to-end verdict evaluation uses the official FEVER scorer where applicable; it reports strict score, label accuracy, evidence precision/recall/F1, with the shared-task cap of five predicted evidence sentences. The 60-claim results are subset metrics, not full FEVER comparisons. The corpus is gold-informed because the selected development and held-out gold evidence determine the evidence pages included.

## Rights and per-use decisions

| Use | Decision and conditions |
| --- | --- |
| Private local parsing, indexing, embeddings, evaluation, and demonstration | **Accepted.** The official FEVER notice plus the applicable article terms/CC BY-SA 3.0 fallback support copying and adaptation for the planned experiments. Provide source attribution and license/change notices in local demonstration output and project documentation. Respect any page-specific terms. |
| External inference through OpenRouter | **Accepted as the intended conditional path, not exercised in F01.** Use only Mistral Small 3.2 24B through DeepInfra; set provider `only` to DeepInfra, `allow_fallbacks` false, `zdr` true, and `data_collection` to `deny`; use account setting ZDR-only so Safety Retention endpoints are excluded; leave input/output logging and training opt-ins off. Send only the claim and retrieved passages needed for the request. Confirm endpoint disclosure/terms at call time. The provider still receives request text; ZDR means no routine storage after processing, not local execution. |
| GitHub source and documentation | **Allowed with provenance.** Source code, source URLs/hashes, selection protocol, and aggregate metrics may be published with FEVER citation and notice. Do not place claims/evidence subsets, article text, passages, indexes, Qdrant data, or embeddings in Git for this MVP. Any later redistribution of claim/annotation or adapted text must satisfy applicable page terms, attribution, change indication, and ShareAlike requirements. |
| Hosted/public demo | **Excluded from current scope.** No hosting or public redistribution is required for F01 or M1. Revisit only if product scope changes; it is not a blocker for private local work. |
| OpenRouter spending | **Not authorized by this record.** RM50 is the planned initial ceiling; use one dedicated key with a conservative USD-equivalent, non-resetting limit only after separate approval. RM100 requires a separate decision. |

OpenRouter’s current model page lists $0.075/M input and $0.20/M output tokens, structured JSON-schema output, and routed model-wide availability that is not provider-specific. DeepInfra is listed as no-training and zero-retention; pinning it removes cross-provider fallback and means provider-specific availability must be observed during later authorized runs. Account privacy settings and key limits were not accessed in F01.

## F01 acceptance record

| Criterion | Result | Evidence |
| --- | --- | --- |
| M1-F01-01 — Official access and artifact identity | PASS | Direct publisher downloads, HTTP status, byte sizes, exact URLs, and local SHA-256 values above. |
| M1-F01-02 — Archive, schema, IDs, and references | PASS | Full ZIP member CRC and schema scan; unique IDs; exact title-key and sentence-position diagnostics above. Sample excludes raw-key mismatches and the duplicate-ID page. |
| M1-F01-03 — Checksums and provenance | PASS | No publisher digest found; local SHA-256 identities recorded and distinguished from publisher checksums. |
| M1-F01-04 — Source licensing | PASS for defined use | Official FEVER notice, CC BY-SA fallback, attribution/ShareAlike requirements, and per-page condition recorded. No blanket license claim is made. |
| M1-F01-05 — Local, API, and publication use matrix | PASS for private local scope and documented conditional API path | Local experiments accepted; external provider and privacy conditions specified; public text/index redistribution excluded. |
| M1-F01-06 — Evaluation compatibility | PASS | FEVER labels/evidence schema, sentence IDs, shared-task scorer, subset metric limits, exact-title sample filter, and corpus construction bias recorded. BEIR SciFact is not used. |
| M1-F01-07 — Split/runtime boundary | PASS | Fixed 30/60 sample, normalized duplicate filter, and evaluation-only labels/evidence contract recorded. |
| M1-F01-08 — Gate decision | PASS | User approved FEVER and accepted this evidence on 10 November 2026; private processing conditions have no remaining material blocker. |

**Superseded dataset:** SciFact remains technically compatible with the previously reviewed BEIR structure, but its rights remained unresolved. It is superseded for this project by the approved FEVER pivot, not rejected as technically invalid. No further SciFact, Ai2, or FEVER maintainer correspondence is planned.
