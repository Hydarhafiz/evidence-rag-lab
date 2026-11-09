# Evidence RAG Lab — Codex Instructions

## Project purpose

Build a small, reproducible Applied AI engineering case study: a Wikipedia-based claim-verification RAG system with evidence citations and three verdicts: SUPPORTS, REFUTES, and NOT ENOUGH INFO.

The target is an interview-ready engineering case study by 29 November 2026. Demonstrate controlled experiments, measurable results, limitations, and failure analysis. Protect job applications and interview preparation; reduce optional work before extending the deadline.

## Working principles

- Use SDD Suite v2.1 for milestone and feature work. Keep one active milestone.
- Prefer simple, explainable Python architecture and deterministic evaluation.
- Record exact datasets, file hashes, model/provider versions, prompts, and settings.
- Keep FEVER labels, evidence annotations, and split metadata in evaluation-only inputs; never expose them to runtime retrieval or generation.
- Cite source page titles and original FEVER sentence IDs; validate every citation against retrieved text.
- Justify each dependency and abstraction. Do not add tools solely for resume appeal.
- Do not inspect secrets, credentials, or `.env` files. Preserve unrelated changes.

## Project constraints

- Official FEVER (2018) claims and its June 2017 Wikipedia corpus are the accepted dataset source. The initial searchable index is a deterministic, approximately 5,000-passage, gold-informed subset; it is not a full FEVER benchmark run.
- Compare BM25 with one dense embedding model. Use local Qdrant and a thin LangChain generation boundary in a Python CLI.
- Use evidence-grounded generation, citation validation, abstention, and separate retrieval and verdict evaluation.
- OpenRouter is the intended gateway. Mistral Small 3.2 24B through DeepInfra is the initial provider candidate, conditional on ZDR-only routing, data-collection denial, no fallback, logging disabled, and separate spending authorization.
- No frontend, public hosting, cloud deployment, model training/fine-tuning, or MLOps platform in the MVP.
- Hybrid retrieval, reranking, larger corpora, and LangGraph remain optional, non-active experiments documented in the roadmap. Do not start them without spare capacity after the MVP or a measurable requirement.

## SDD workflow

Canonical planning lives in `docs/core/roadmap.md`, `docs/core/milestone.md`, and `docs/core/planning-sources.md`. Use the installed SDD Suite v2.1 skills, ignored `.sdd/` handoffs, feature-level commits, and promote-or-purge retention rules. Discovery is read-only.
