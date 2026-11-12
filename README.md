# Evidence RAG Lab

Evidence RAG Lab is a small Applied AI engineering case study for Wikipedia-based claim verification. Milestone 1 completed on 12 November 2026 with a gold-informed 5,000-passage FEVER corpus, BM25 and BGE-small dense retrieval, and a Python CLI that returns evidence-grounded SUPPORTS, REFUTES, or NOT ENOUGH INFO verdicts with validated source references. One separately authorized train-only live check passed and its provider activity cost was recorded; this is an observation, not evidence of system-wide accuracy or reliability. M2's plan and three-phase authorization procedure were approved on 13 November; planning is complete and F04 has not started. The held-out evaluation has not been run, and M3 remains planned.

## Dataset and evaluation boundary

The source is the official [FEVER 2018 release](https://fever.ai/dataset/fever.html) and its June 2017 Wikipedia corpus. The mini-corpus contains 5,000 four-sentence windows with stride three. It is gold-informed: selected train and held-out evidence pages determine which source pages enter the corpus. It is a curated subset experiment, not a full FEVER benchmark run.

Gold labels, evidence groups, claim IDs, and split membership live in separate local evaluation-only files for the 30-claim train sample and 60-claim held-out sample. The smoke command reads only the train file. Runtime passages contain only source text, exact FEVER page-title keys, original sentence IDs, and source provenance. The generated Qdrant index, raw data, model weights, and evaluation samples are ignored by Git.

## F02 implementation observations

The source scan confirmed 5,416,537 wiki page records, 5,416,536 non-empty unique page IDs, one empty page ID, and one page with a duplicate sentence ID. The empty ID and ambiguous sentence rows are not placed in retrieval. The selected sample contains 107 evidence groups across 72 pages and 103 distinct page/sentence pairs; all 103 pairs occur in the corpus. The 72 evidence pages produced 291 passages. A deterministic hash ordering selected 4,709 background passages from 10,410,277 candidate windows.

The encoder is `BAAI/bge-small-en-v1.5` at revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, with 384-dimensional normalized vectors and a 512-token limit. Passage input is the exact page title, a newline, and sentence text. Query input uses BGE’s search instruction followed by the raw claim. The longest passage in this corpus used 424 tokens; no passage was truncated or split.

Persistent local Qdrant and BM25 were built from the same passage file (`be6bdf115415a42dbb2a55585c160d157b2528875f14d70f990cdb1f6a7ceebe`). The final development-only top-5 smoke run covered the 20 verifiable claims from the 30-claim train sample. BM25 page/sentence Recall@5 was 0.88125 with a complete-evidence-group hit rate of 0.85; dense retrieval measured 0.89375 and 0.90. Mean query latency was 15.39 ms for BM25 and 154.04 ms for dense retrieval. These are preliminary retrieval observations, not M2 results; the 60-claim held-out split has not been evaluated.

## Retrieval mechanics

BM25 tokenizes passage text and claims by case-folding and splitting on non-word characters (`\w+`); it does no stemming or stop-word removal. `rank-bm25` `BM25Okapi` defaults are retained (`k1=1.5`, `b=0.75`, `epsilon=0.25`). Higher scores rank first; they are relevance scores, not probabilities. Ties resolve by ascending stable passage ID. The 5,000-document BM25 index is built in memory from `passages.jsonl` and took 104.97 ms in the final smoke process.

Dense retrieval embeds the exact passage text (page title, newline, then source sentences) and embeds each claim with BGE’s search instruction prefix. BGE returns 384-dimensional vectors, normalized before storage and query; Qdrant uses cosine similarity. Local persistent Qdrant stores the corpus vectors and passage payloads so a later CLI process can reopen the index without rebuilding the vectors. Search retrieves scores for the full 5,000-point collection and sorts ties by passage ID. The CPU index build took 190.21 seconds. Corpus preparation checks BGE token lengths before indexing and splits overlong source sentences while retaining their original sentence ID. The actual corpus’s longest passage used 424 of 512 tokens, so no passage was truncated.

The runtime dependencies each serve a narrow path: `rank-bm25` supplies the transparent lexical baseline, Sentence Transformers loads and encodes with the one pinned model, `qdrant-client` provides embedded persistent local storage, `transformers` supplies the tokenizer type/API used for token-boundary construction, `torch` is the Sentence Transformers model backend, and NumPy checks vector dimensions and normalization. For verification, `langchain-openrouter` and the OpenRouter SDK provide the thin structured chat boundary, Pydantic validates its output contract, `tiktoken` loads Mistral's pinned Tekken vocabulary for context counting, `httpx` configures the bounded client and mocked transport tests, and `python-dotenv` reads the API key only on the explicit live path. Setuptools provides the small `fever-rag` CLI package; pytest and Ruff are development-only tools.

The custom recall and complete-group metrics are separate from verdict evaluation and are not BEIR qrel scores. The verification command adds generation and citation validation; the official FEVER scorer workflow remains separate.

## M1-F03: cited claim verification

`fever-rag verify --claim "..." --method bm25 --live --confirm-account-controls` retrieves the configured top five passages from the existing runtime corpus, then greedily packs whole passages in retrieval rank order under the 4,096-token application-message budget. The versioned prompt and claim count against that budget. A passage is either included whole or recorded with its rank, page title, original sentence IDs, passage ID, and omission reason; source text and its citation mapping are never truncated. An empty packed context returns NOT ENOUGH INFO without making a provider request.

The live path requires both `--live` and `--confirm-account-controls`, reads `OPENROUTER_API_KEY` from the process environment or ignored `.env`, and makes at most two attempts for one claim. The second flag is an operator attestation: the CLI cannot inspect OpenRouter account settings, so configure account-level privacy, disabled logging, and a non-resetting spend cap before confirming. `config/verification.toml` pins `mistralai/mistral-small-3.2-24b-instruct`, DeepInfra only, temperature, output limit, timeouts, context budget, pricing date, and the ZDR, no-fallback, data-denial, and required-parameter routing fields. Copy `.env.example` to `.env` and set the key locally. Download only the tokenizer artifact used for local counting:

```sh
hf download mistralai/Mistral-Small-3.2-24B-Instruct-2506 tekken.json \
  --revision 9151ac4804f11eac6d7cd3b963eaf9ea1bc56839 \
  --local-dir models/hf/mistral-small-3.2-24b
```

The command verifies the tokenizer SHA-256 before counting. A normal run prints a machine-readable record containing retrieval and packing data, omissions, response, validated source references, latency, attempt outcomes, token usage when the provider reports it, and an estimated cost. It does not write raw prompts or responses to a log file. A missing required, fabricated, or otherwise invalid citation is an operational validation error, not NOT ENOUGH INFO. Provider errors, exhausted retries, malformed output, and timeout errors also remain operational failures and cause a nonzero exit status.

LangChain supplies the `ChatOpenRouter` model adapter and strict JSON Schema response-format request. Ordinary Python owns retrieval selection, whole-passage packing, request bounds, retry accounting, source-membership validation, and CLI presentation. The application independently checks the schema, exact verdict names, citation page-title/sentence-ID membership in packed evidence, duplicate citations, and the consistency rule that reported conflicting evidence uses NOT ENOUGH INFO. It then renders source text from the original stored passages rather than treating model-written quotations as evidence. These checks prove that a citation points into supplied text; they cannot prove that the passage supports the claim or that the explanation is factually correct.

NOT ENOUGH INFO is a verification result when supplied evidence is empty, insufficient, or reported as conflicting. Missing credentials, malformed model output, invalid citations, HTTP failures, timeouts, and exhausted retries are operational failures and never become that verdict. Raw output is not retained automatically. The local run record includes model, provider, prompt version, corpus hash, retrieval IDs, packed IDs, omissions, latency, attempt status, available provider token counts, and a dated price-based cost estimate. If provider usage metadata is missing, the estimate uses local message tokenization and the configured maximum output tokens for each attempt; message framing, failed-request billing, cache handling, taxes, and current provider billing can differ.

For the initial model, the dated price settings are $0.075 per million input tokens and $0.20 per million output tokens. At the configured message-text budget of 4,096 input tokens, 512 output tokens, and two maximum attempts, the estimated upper request usage is 8,192 input plus 1,024 output tokens, approximately $0.0008192 at those rates. The manual live run reported 1,009 input and 93 output tokens; the application estimated $0.000094275 using those token counts and the dated rates. The provider activity screenshot separately displays a cost of $0.0000943 with usage type “Credits” and matching model, provider, and token counts; this is the provider-reported activity cost, rounded in the dashboard, not an invoice or card-settlement record. The run's operator attestation recorded account controls as configured; the repository does not independently inspect account billing or privacy settings.

Learning checkpoint:

1. Retrieved runtime passages become a JSON evidence list in retrieval rank order; the system prompt, claim, and complete passage records are token-counted before the single structured request.
2. LangChain provides the OpenRouter chat adapter and structured-output request construction. Direct Python code keeps data selection, deterministic packing, citation validation, retry limits, measurements, and presentation visible.
3. Structured output constrains shape and allowed labels, but does not establish source membership or semantic correctness. The application therefore validates fields and every original page/sentence reference after parsing.
4. A citation can be valid while failing to support the explanation; source membership is a provenance check, not a factuality score.
5. NOT ENOUGH INFO describes what the retrieved evidence warrants. A timeout, HTTP error, missing key, malformed response, or rejected citation describes a failed operation and exits separately.
6. M2 should inspect retrieval misses, evidence dropped by context packing, conflicting evidence, citation omissions/fabrications, schema failures, provider latency and availability, missing usage metadata, and the gap between source validity and entailment.

## M1 closeout and current status

Release verification on 12 November 2026 passed: `pytest -q` (44 passed; 30 F03 cases and 14 earlier cases), `ruff format --check src tests`, `ruff check src tests`, `python -m compileall -q src`, `python -m pip check`, CLI help, SDD preflight, and `git diff --check`. The one live train-only result returned REFUTES and matched the annotated FEVER source; one successful claim does not establish system-wide reliability. No held-out retrieval or generation outcome was inspected. The corpus is gold-informed, and the reported retrieval smoke metrics cover only the train development sample, not the full FEVER benchmark. Validating that a source reference occurs in supplied text does not establish semantic correctness. The full M1 contract is preserved in closeout commit `15d3bf1`; the reusable [milestone file](docs/core/milestone.md) contains the approved M2 plan and three-phase authorization procedure. F04/F05 remain unimplemented; each actual paid phase requires separate authorization.

## Rebuild locally

Use Python 3.12. On CPU-only Linux, install the CPU PyTorch wheel before the project dependencies so pip does not select unnecessary CUDA libraries:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

Restore the three files from the official release URLs below. `fever-rag prepare` validates their SHA-256 values before reading them:

```sh
mkdir -p data/raw
curl --fail --location --output data/raw/train.jsonl https://fever.ai/download/fever/train.jsonl
curl --fail --location --output data/raw/shared_task_dev.jsonl https://fever.ai/download/fever/shared_task_dev.jsonl
curl --fail --location --output data/raw/wiki-pages.zip https://fever.ai/download/fever/wiki-pages.zip
```

Download the pinned public encoder snapshot into the ignored local cache. Corpus preparation then runs offline:

```sh
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="BAAI/bge-small-en-v1.5", revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", cache_dir="models/hf")'
fever-rag prepare
fever-rag index
```

`prepare` creates `data/processed/passages.jsonl`, separate `evaluation_samples_train.jsonl` and `evaluation_samples_dev.jsonl` files, and a local corpus manifest. `index` stores normalized 384-dimensional vectors in persistent local Qdrant at `data/indexes/qdrant`; no container or server is needed. BM25 builds in memory from the same passage file.

```sh
fever-rag search --method bm25 --query "claim text"
fever-rag search --method dense --query "claim text"
fever-rag smoke --top-k 5
fever-rag verify --claim "claim text" --method bm25 --live --confirm-account-controls
pytest
ruff format --check src tests
ruff check src tests
```

The smoke command reads `evaluation_samples_train.jsonl` only; its loader rejects any non-train sample. It does not open the held-out file or run generation. See the [current milestone](docs/core/milestone.md), [roadmap and M1 summary](docs/core/roadmap.md), and [source and use record](docs/core/planning-sources.md) for source hashes, license conditions, limitations, and milestone outcomes.

## Dataset use

FEVER’s [license notice](https://fever.ai/download/fever/license.html) refers to applicable Wikipedia article terms and specifies CC BY-SA 3.0 where those terms are unavailable. The accepted scope is private local parsing, indexing, embeddings, evaluation, and demonstration with source attribution and applicable page terms. Public redistribution of raw text, claim/evidence subsets, passage indexes, Qdrant data, or embeddings is outside this MVP. Code, source identity, and aggregate results may be published with provenance.
