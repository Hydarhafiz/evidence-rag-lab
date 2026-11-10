# Evidence RAG Lab

A small, reproducible Applied AI case study for Wikipedia-based claim verification. M1-F01 accepted the official FEVER source and local-use boundary. M1-F02 now prepares a fixed passage corpus and compares BM25 with one local dense retriever. M1-F03, the cited generation CLI, remains gated on separate spending authorization; no inference API has been called.

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

The direct runtime dependencies each serve this path: `rank-bm25` supplies the transparent lexical baseline, Sentence Transformers loads and encodes with the one pinned model, `qdrant-client` provides embedded persistent local storage, `transformers` supplies the tokenizer type/API used for token-boundary construction, `torch` is the Sentence Transformers model backend, and NumPy checks vector dimensions and normalization. Setuptools provides the small `fever-rag` CLI package; pytest and Ruff are development-only tools.

The custom recall and complete-group metrics are separate from verdict evaluation and are not BEIR qrel scores. M1-F03 still owns generation, citation validation, and the official FEVER scorer workflow.

## Rebuild locally

Use Python 3.12. On CPU-only Linux, install the CPU PyTorch wheel before the project dependencies so pip does not select unnecessary CUDA libraries:

```sh
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
pytest
ruff format --check src tests
ruff check src tests
```

The smoke command reads `evaluation_samples_train.jsonl` only; its loader rejects any non-train sample. It does not open the held-out file or run generation. See [the active milestone](docs/core/milestone.md), [roadmap](docs/core/roadmap.md), and [source and use record](docs/core/planning-sources.md) for source hashes, license conditions, limitations, and the remaining M1-F03 gate.

## Dataset use

FEVER’s [license notice](https://fever.ai/download/fever/license.html) refers to applicable Wikipedia article terms and specifies CC BY-SA 3.0 where those terms are unavailable. The accepted scope is private local parsing, indexing, embeddings, evaluation, and demonstration with source attribution and applicable page terms. Public redistribution of raw text, claim/evidence subsets, passage indexes, Qdrant data, or embeddings is outside this MVP. Code, source identity, and aggregate results may be published with provenance.
