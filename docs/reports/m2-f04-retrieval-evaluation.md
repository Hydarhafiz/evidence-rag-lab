# M2-F04 — Retrieval evaluation

**Run:** 14 November 2026 · **Protocol:** `m2-evaluation-v1` · **Scope:** curated, gold-informed 5,000-passage FEVER subset; not full-FEVER benchmark performance.

## Result

The evaluator compared BM25 with BGE-small dense retrieval over the same 5,000 passages and the same 30 selected FEVER train claims: 10 SUPPORTS, 10 REFUTES, and 10 NOT ENOUGH INFO. Evidence metrics use the 20 verifiable claims; NOT ENOUGH INFO remains N/A for gold recovery. Each method ran 90 timed top-10 queries (three per claim), after one excluded synthetic warm-up. The first timed ranking supplied the quality result; the two repeats measured latency and ranking stability.

| Method | Gold-union sentence Recall@1 / 5 / 10 | Complete-group hits@1 / 5 / 10 |
| --- | --- | --- |
| BM25 | 0.67708 / 0.88125 / 0.96875 | 14/20 (0.70) / 17/20 (0.85) / 19/20 (0.95) |
| BGE-small dense | 0.55833 / 0.89375 / 0.91875 | 12/20 (0.60) / 18/20 (0.90) / 18/20 (0.90) |

Recall is the macro mean of each claim's unique annotated sentence-pair union recovered in the first k passages. Complete-group hit is 1 when all references from at least one alternative group are present. Both metrics deduplicate overlapping passage references and retain multi-sentence and multi-page groups. The exact evidence denominator is 20 at every k; the retrieval/timing workload is 30 claims per method.

The paired complete-group results were mixed: at k=1 BM25 alone completed three claims, dense alone one, and 16 tied; at k=5 dense alone completed one, BM25 alone none, and 19 tied; at k=10 BM25 alone completed one, dense alone none, and 19 tied. Dense-minus-BM25 macro recall differences were −0.11875, +0.01250, and −0.05000 at k=1, 5, and 10. There is no single winner across the cutoffs.

Label-specific results show the same trade-off. At k=5, SUPPORTS recall was 0.9125 for BM25 and 0.9375 for dense, with both completing 9/10 groups. REFUTES recall was 0.85 for both, with 8/10 complete groups for BM25 and 9/10 for dense. At k=10, BM25 completed 10/10 REFUTES groups versus 9/10 for dense; both completed 9/10 SUPPORTS groups.

## Evidence availability and failure analysis

All 37 unique annotated sentence pairs across the 20 verifiable claims were present in the corpus, and every claim had at least one complete evidence group there. Corpus pair coverage and corpus complete-group coverage were therefore both 20/20 (macro 1.0). This is corpus availability, not retrieval success: the remaining misses were ranking losses.

At top 10, both methods partially recovered one claim's two-reference evidence group, recovering one reference each. Dense also missed a separate single-reference group that BM25 returned at rank 6. The traces distinguish these retrieval misses from corpus absence and record each reference's first rank, passage ID, score, and original source coordinates. Missing beyond rank 10 means unrecovered within the observed cutoff, not absent from the index.

Four claims had different complete-group outcomes at k=1. BM25 completed three at rank 1 while dense placed the corresponding groups at ranks 2–3; dense completed one at rank 1 while BM25 placed it at rank 8. Exact term overlap is a plausible contributor to some BM25 placements, and semantic matching is a plausible contributor to the dense placement. These are trace-based hypotheses from four disagreements, not causal findings. BM25 had tied scores in 3/90 timed rankings; passage-ID ordering remained stable. Dense had no tied scores. All 30 three-query ranking triplets were identical for both methods, with no retrieval errors or nonfinite scores.

## Latency and operations

Latency is from `Retriever.search()`, including query tokenization or encoding, retrieval, payload conversion, and sorting. Synthetic warm-ups are excluded. Percentiles use nearest rank `ceil(p*n)`, one-based; p50 and p95 use all 90 successful timed queries per method. Failed calls have a separate timing population; there were none.

| Method | Successful observations | p50 | p95 | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 90/90 | 6.319 ms | 11.024 ms | 4.139 ms | 214.234 ms |
| BGE-small dense | 90/90 | 120.679 ms | 332.335 ms | 112.327 ms | 345.036 ms |

In this CPU/WSL2 run, dense p50 was about 19.1× BM25 p50 and dense p95 about 30.1× BM25 p95. BM25's measured in-memory index build was 320.181 ms; the shared corpus read took 47.905 ms. Dense startup was 641.678 ms, including a 43.620 ms passage-file read, 235.641 ms encoder load, and 362.321 ms Qdrant open/count. The corpus file was 3,083,728 bytes; the closed Qdrant directory was 23,770,313 bytes; the model cache directory was 153,906,448 bytes. BM25 has no persistent index and its memory size was not measured.

The current Qdrant index was reused. Its manifest's 190,207.3 ms build observation is historical M1 evidence, not a contemporaneous rebuild measurement. The encoder remained `BAAI/bge-small-en-v1.5`, revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, 384 dimensions. Environment: Python 3.12.3, Linux WSL2 x86_64, 8 logical CPUs, 4 Torch CPU threads; `rank-bm25` 0.2.2, `sentence-transformers` 6.2.0, `qdrant-client` 1.20.0, Torch 2.15.0+cpu, and Transformers 5.22.2.

The BM25 configuration remained `BM25Okapi` defaults (k1=1.5, b=0.75, epsilon=0.25), casefolded Unicode word tokens, and score-descending/passage-ID-ascending ties. Dense retrieval reused the persistent normalized 384-dimensional cosine index and the existing BGE query instruction; exact Qdrant search returned all corpus scores before sorting. Accepted input identities were corpus SHA-256 `be6bdf115415a42dbb2a55585c160d157b2528875f14d70f990cdb1f6a7ceebe`, train sample SHA-256 `01f162c4db7a258c57b8f5275f4d4968239c244e3551dd5cb8c69a41a92f1685`, and held-out sample SHA-256 `ab3111841d8d6774389c13dcec63467654133e727f4d329582c2c7c47dd996a0`. The held-out digest was checked as opaque bytes only.

## Reproduction and limits

Run from the repository root:

```sh
.venv/bin/python -m evidence_rag.cli evaluate --split train \
  --output artifacts/generated/m2-f04/retrieval-evaluation-train.json
```

The ignored JSON contains claim-level ranks, attempts, and source coordinates for local diagnosis; it contains no claim or article text and is not committed. The compact report contains only aggregate metrics and source-free diagnostics. The existing train-only `smoke --top-k 5` path remains available. Its 20-verifiable-claim values (BM25 0.88125 / 0.85, dense 0.89375 / 0.90) match F04 at k=5 because the corpus, claims, and deterministic rankings are unchanged; that smoke does not provide F04's 30-claim timing workload, repeated-query stability, paired cutoffs, or traces.

Final verification on the completed code passed: `.venv/bin/pytest -q` (64 passed), `.venv/bin/ruff format --check src tests` (18 files), `.venv/bin/ruff check src tests`, `.venv/bin/python -m compileall -q src`, and `.venv/bin/python -m pip check` (no broken requirements). CLI help for `evaluate`, `smoke`, and `verify`, the existing `.venv/bin/python -m evidence_rag.cli smoke --top-k 5` command, SDD preflight, and `git diff --check` also passed.

This is a balanced 30-claim sample with only 20 evidence-bearing claims, drawn from a corpus intentionally selected using gold labels and evidence pages, including held-out evidence pages. These results are custom subset measurements, not population estimates, statistical significance, or full-FEVER performance. They do not assess verdict quality. Held-out records were checked only by opaque SHA-256; they were not parsed or evaluated. No provider calls, paid inference, index rebuild, or model download occurred.
