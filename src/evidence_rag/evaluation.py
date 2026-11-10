"""Offline retrieval metrics over the permitted 30-claim development sample."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from evidence_rag.models import Retriever, SearchResponse
from evidence_rag.retrieval import BM25Retriever, DenseRetriever, RetrievalError


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    split: str
    claim_id: int
    label: str
    claim: str
    evidence_groups: tuple[frozenset[tuple[str, int]], ...]


def load_development_samples(path: Path) -> tuple[EvaluationSample, ...]:
    samples: list[EvaluationSample] = []
    required_fields = {"split", "claim_id", "label", "claim", "evidence_groups"}
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, start=1):
            try:
                value = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise RetrievalError(
                    f"invalid evaluation JSON at line {line_number}: {error}"
                ) from error
            if not isinstance(value, dict) or set(value) != required_fields:
                raise RetrievalError(f"unexpected evaluation fields at line {line_number}")
            split = value["split"]
            claim_id = value["claim_id"]
            label = value["label"]
            claim = value["claim"]
            groups = value["evidence_groups"]
            if (
                split != "train"
                or isinstance(claim_id, bool)
                or not isinstance(claim_id, int)
                or not isinstance(label, str)
                or label not in {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
                or not isinstance(claim, str)
                or not isinstance(groups, list)
            ):
                raise RetrievalError(
                    "smoke evaluation accepts only train samples; "
                    f"malformed or non-train row at line {line_number}"
                )
            evidence_groups: list[frozenset[tuple[str, int]]] = []
            for group in groups:
                if not isinstance(group, list):
                    raise RetrievalError(f"malformed evidence group at line {line_number}")
                references: set[tuple[str, int]] = set()
                for reference in group:
                    if not isinstance(reference, dict) or set(reference) != {
                        "page_title",
                        "sentence_id",
                    }:
                        raise RetrievalError(f"malformed evidence reference at line {line_number}")
                    page_title = reference["page_title"]
                    sentence_id = reference["sentence_id"]
                    if (
                        not isinstance(page_title, str)
                        or not page_title
                        or isinstance(sentence_id, bool)
                        or not isinstance(sentence_id, int)
                        or sentence_id < 0
                    ):
                        raise RetrievalError(f"invalid evidence reference at line {line_number}")
                    references.add((page_title, sentence_id))
                evidence_groups.append(frozenset(references))
            samples.append(
                EvaluationSample(
                    split=split,
                    claim_id=claim_id,
                    label=label,
                    claim=claim,
                    evidence_groups=tuple(evidence_groups),
                )
            )
    return tuple(samples)


def _score_method(
    samples: tuple[EvaluationSample, ...],
    search: Retriever,
    top_k: int,
) -> dict[str, object]:
    recalls: list[float] = []
    group_hits: list[float] = []
    latencies: list[float] = []
    evaluated_claims = 0

    for sample in samples:
        if sample.split != "train" or sample.label == "NOT ENOUGH INFO":
            continue
        groups = tuple(group for group in sample.evidence_groups if group)
        if not groups:
            raise RetrievalError(f"development claim {sample.claim_id} has no evidence group")
        response: SearchResponse = search.search(sample.claim, top_k=top_k)
        latencies.append(response.elapsed_ms)
        retrieved = {
            (hit.page_title, sentence_id)
            for hit in response.hits
            for sentence_id in hit.sentence_ids
        }
        gold_union = set().union(*groups)
        recalls.append(len(retrieved & gold_union) / len(gold_union))
        group_hits.append(float(any(group <= retrieved for group in groups)))
        evaluated_claims += 1

    if evaluated_claims == 0:
        raise RetrievalError("the local development sample has no verifiable claims")
    return {
        "retrieval_method": search.method,
        "split": "train",
        "verifiable_claims": evaluated_claims,
        f"page_sentence_recall_at_{top_k}": statistics.fmean(recalls),
        f"complete_evidence_group_hit_rate_at_{top_k}": statistics.fmean(group_hits),
        "mean_retrieval_latency_ms": statistics.fmean(latencies),
        "median_retrieval_latency_ms": statistics.median(latencies),
    }


def run_development_smoke(
    passages_path: Path,
    evaluation_path: Path,
    qdrant_path: Path,
    cache_dir: Path,
    top_k: int = 5,
) -> dict[str, object]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    samples = load_development_samples(evaluation_path)
    training_samples = tuple(sample for sample in samples if sample.split == "train")
    if len(training_samples) != 30 or len(training_samples) != len(samples):
        raise RetrievalError(
            "expected exactly 30 train development samples, "
            f"found {len(training_samples)} training rows among {len(samples)} samples"
        )

    bm25, bm25_hash = BM25Retriever.from_path(passages_path)
    dense = DenseRetriever(passages_path, qdrant_path, cache_dir)
    try:
        if bm25_hash != dense.corpus_sha256:
            raise RetrievalError("BM25 and dense retrieval are not using the same passage corpus")
        return {
            "corpus_sha256": bm25_hash,
            "passage_count": len(bm25.passages),
            "top_k": top_k,
            "bm25_index_build_latency_ms": bm25.build_latency_ms,
            "bm25": _score_method(training_samples, bm25, top_k),
            "dense": _score_method(training_samples, dense, top_k),
            "held_out_split_accessed": False,
        }
    finally:
        dense.close()
