# Evidence RAG Lab

A small, reproducible Applied AI engineering case study for Wikipedia-based claim verification. The system is planned as a Python CLI that retrieves passages, asks one model to classify a claim, and returns a verdict grounded in cited Wikipedia sentences.

## Current status

M1-F01 dataset approval is complete. M1-F02, the deterministic mini-corpus and retrieval implementation, is next and has not started. No product code, indexes, or evaluation results are claimed yet.

## Dataset and evaluation

The accepted source is the official [FEVER 2018 release](https://fever.ai/dataset/fever.html) and its [June 2017 Wikipedia corpus](https://fever.ai/download/fever/wiki-pages.zip). The initial index targets about 5,000 four-sentence passages. It uses a fixed 30-claim development sample from FEVER train and a fixed 60-claim held-out sample from labeled shared-task development, balanced across SUPPORTS, REFUTES, and NOT ENOUGH INFO.

Gold labels and evidence references inform which pages enter the mini-corpus. That makes the corpus gold-informed; results will be reported as a curated subset, not as full-benchmark FEVER performance. Runtime retrieval and prompts will not receive labels, evidence annotations, split membership, or gold document hints. Retrieval metrics are reported separately from generated-verdict metrics, with the [official FEVER scorer](https://github.com/sheffieldnlp/fever-scorer) used where its input contract applies.

## Planned MVP stack

- Python CLI with BM25 and one dense encoder.
- Persistent local Qdrant for dense search.
- Thin LangChain integration for structured generation.
- One inexpensive model through OpenRouter, with provider routing and privacy controls fixed before any authorized call.
- Citation validation, NEI abstention, latency/token/cost tracking, and failure analysis.

The initial model/provider candidate is Mistral Small 3.2 24B through DeepInfra. The planned RM50 ceiling is not spending authorization; any API use requires a separate approval and an enforced key limit. The RM100 envelope requires another decision.

## Dataset use

FEVER’s [license notice](https://fever.ai/download/fever/license.html) refers to the applicable Wikipedia article terms and specifies CC BY-SA 3.0 where those terms are unavailable. Private local experimentation is within the documented license basis when those terms are followed. Attribute article sources and identify modifications in demonstrations. Public redistribution of raw corpus text, passage indexes, Qdrant data, or embeddings is outside this MVP; code and aggregate results may be published with dataset citation and provenance.

See [the roadmap](docs/core/roadmap.md), [active milestone](docs/core/milestone.md), and [source and use record](docs/core/planning-sources.md) for the contract and limitations.
