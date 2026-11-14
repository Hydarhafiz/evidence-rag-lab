"""Offline train-only retrieval metrics and a freeze-gated final evaluator."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Literal

from evidence_rag.corpus import MODEL_DIMENSION, MODEL_NAME, MODEL_REVISION
from evidence_rag.embeddings import QUERY_INSTRUCTION
from evidence_rag.models import Passage, RetrievalHit, Retriever, SearchResponse
from evidence_rag.retrieval import (
    INDEX_MANIFEST_NAME,
    BM25Retriever,
    DenseRetriever,
    RetrievalError,
    load_runtime_passages,
)

EvaluationSplit = Literal["train", "dev"]
Reference = tuple[str, int]
LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
VERIFIABLE_LABELS = ("SUPPORTS", "REFUTES")
RETRIEVAL_CUTOFFS = (1, 5, 10)
TIMED_REPEATS = 3
SYNTHETIC_WARMUP_QUERY = "fixed synthetic retrieval warm-up query"
PROTOCOL_ID = "m2-evaluation-v1"

ACCEPTED_CORPUS_SHA256 = "be6bdf115415a42dbb2a55585c160d157b2528875f14d70f990cdb1f6a7ceebe"
ACCEPTED_TRAIN_SAMPLE_SHA256 = "01f162c4db7a258c57b8f5275f4d4968239c244e3551dd5cb8c69a41a92f1685"
ACCEPTED_HELD_OUT_SAMPLE_SHA256 = "ab3111841d8d6774389c13dcec63467654133e727f4d329582c2c7c47dd996a0"


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    split: str
    claim_id: int
    label: str
    claim: str
    evidence_groups: tuple[frozenset[Reference], ...]


@dataclass(frozen=True, slots=True)
class EvaluationIdentity:
    """Accepted identities; overridable only to exercise synthetic fixtures."""

    corpus_sha256: str = ACCEPTED_CORPUS_SHA256
    train_sample_sha256: str = ACCEPTED_TRAIN_SAMPLE_SHA256
    held_out_sample_sha256: str = ACCEPTED_HELD_OUT_SAMPLE_SHA256


ACCEPTED_EVALUATION_IDENTITY = EvaluationIdentity()


@dataclass(frozen=True, slots=True)
class AttemptOutcome:
    status: str
    elapsed_ms: float | None
    response: SearchResponse | None
    failure_type: str | None = None
    failure_message: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_samples(path: Path, expected_split: EvaluationSplit) -> tuple[EvaluationSample, ...]:
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
            raw_groups = value["evidence_groups"]
            if (
                split != expected_split
                or isinstance(claim_id, bool)
                or not isinstance(claim_id, int)
                or claim_id < 0
                or not isinstance(label, str)
                or label not in LABELS
                or not isinstance(claim, str)
                or not claim.strip()
                or not isinstance(raw_groups, list)
            ):
                raise RetrievalError(
                    "smoke evaluation accepts only train samples; "
                    f"malformed or non-{expected_split} row at line {line_number}"
                )

            evidence_groups: list[frozenset[Reference]] = []
            for raw_group in raw_groups:
                if not isinstance(raw_group, list):
                    raise RetrievalError(f"malformed evidence group at line {line_number}")
                references: set[Reference] = set()
                for reference in raw_group:
                    if not isinstance(reference, dict) or set(reference) != {
                        "page_title",
                        "sentence_id",
                    }:
                        raise RetrievalError(f"malformed evidence reference at line {line_number}")
                    page_title = reference["page_title"]
                    sentence_id = reference["sentence_id"]
                    if (
                        not isinstance(page_title, str)
                        or not page_title.strip()
                        or isinstance(sentence_id, bool)
                        or not isinstance(sentence_id, int)
                        or sentence_id < 0
                    ):
                        raise RetrievalError(f"invalid evidence reference at line {line_number}")
                    references.add((page_title, sentence_id))
                evidence_groups.append(frozenset(references))

            nonempty_groups = [group for group in evidence_groups if group]
            if label in VERIFIABLE_LABELS and (
                not evidence_groups or len(nonempty_groups) != len(evidence_groups)
            ):
                raise RetrievalError(
                    f"verifiable claim {claim_id} has a missing or empty evidence group"
                )
            if label == "NOT ENOUGH INFO" and nonempty_groups:
                raise RetrievalError(
                    f"NOT ENOUGH INFO claim {claim_id} has annotated gold evidence"
                )

            samples.append(
                EvaluationSample(
                    split=expected_split,
                    claim_id=claim_id,
                    label=label,
                    claim=claim,
                    evidence_groups=tuple(evidence_groups),
                )
            )

    if not samples:
        raise RetrievalError(f"the {expected_split} evaluation sample is empty")
    identifiers = [sample.claim_id for sample in samples]
    if len(set(identifiers)) != len(identifiers):
        raise RetrievalError(f"the {expected_split} evaluation sample has duplicate claim IDs")
    return tuple(samples)


def load_development_samples(path: Path) -> tuple[EvaluationSample, ...]:
    """Keep the existing train-only loader used by the smoke command."""
    return _parse_samples(path, "train")


def _validate_sample_counts(samples: tuple[EvaluationSample, ...], split: EvaluationSplit) -> None:
    expected_count = 30 if split == "train" else 60
    expected_per_label = 10 if split == "train" else 20
    counts = Counter(sample.label for sample in samples)
    expected = {label: expected_per_label for label in LABELS}
    if len(samples) != expected_count or dict(counts) != expected:
        raise RetrievalError(
            f"expected {expected_count} {split} samples with {expected_per_label} per label; "
            f"found total={len(samples)}, counts={dict(counts)}"
        )


def _validate_freeze_gate(
    freeze_path: Path,
    confirmed_freeze_sha256: str | None,
    held_out_phase_authorized: bool,
    identity: EvaluationIdentity,
) -> str:
    """Require a reviewed freeze checksum and separate phase authorization before dev access."""
    if not held_out_phase_authorized:
        raise RetrievalError(
            "held-out evaluation requires separate authorization for the concrete phase"
        )
    if not confirmed_freeze_sha256:
        raise RetrievalError("held-out evaluation requires an explicit freeze SHA-256 confirmation")
    if not freeze_path.is_file():
        raise RetrievalError(f"held-out evaluation freeze is missing: {freeze_path}")

    observed_freeze_sha256 = _sha256_file(freeze_path)
    if observed_freeze_sha256 != confirmed_freeze_sha256.lower():
        raise RetrievalError("confirmed freeze SHA-256 does not match the freeze file")
    try:
        freeze = json.loads(freeze_path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RetrievalError(f"held-out evaluation freeze is invalid: {error}") from error
    if not isinstance(freeze, dict):
        raise RetrievalError("held-out evaluation freeze must be a JSON object")
    expected_freeze_fields = {
        "protocol": PROTOCOL_ID,
        "corpus_sha256": identity.corpus_sha256,
        "held_out_sample_sha256": identity.held_out_sample_sha256,
    }
    mismatches = [
        name for name, expected in expected_freeze_fields.items() if freeze.get(name) != expected
    ]
    if mismatches:
        raise RetrievalError(f"held-out evaluation freeze identity mismatch: {mismatches}")
    return observed_freeze_sha256


def validate_freeze_gate(
    freeze_path: Path,
    confirmed_freeze_sha256: str | None,
    held_out_phase_authorized: bool,
) -> str:
    """Check the fixed accepted identities before any held-out sample is read."""
    return _validate_freeze_gate(
        freeze_path,
        confirmed_freeze_sha256,
        held_out_phase_authorized,
        ACCEPTED_EVALUATION_IDENTITY,
    )


def _validate_search_response(
    response: SearchResponse,
    method: str,
    query: str,
    corpus_by_id: dict[str, Passage],
) -> None:
    if not isinstance(response, SearchResponse):
        raise RetrievalError("retriever returned an invalid response object")
    if response.retrieval_method != method or response.query != query:
        raise RetrievalError("retriever response method or query does not match the request")
    if (
        isinstance(response.elapsed_ms, bool)
        or not isinstance(response.elapsed_ms, (int, float))
        or not math.isfinite(response.elapsed_ms)
        or response.elapsed_ms < 0
    ):
        raise RetrievalError("retriever returned an invalid query latency")
    if len(response.hits) > 10:
        raise RetrievalError("retriever returned more than the requested top 10 passages")

    previous: RetrievalHit | None = None
    seen_passages: set[str] = set()
    for expected_rank, hit in enumerate(response.hits, start=1):
        if not isinstance(hit, RetrievalHit):
            raise RetrievalError("retriever returned an invalid hit object")
        if hit.rank != expected_rank or hit.retrieval_method != method:
            raise RetrievalError("retriever returned invalid rank or method metadata")
        if (
            isinstance(hit.score, bool)
            or not isinstance(hit.score, (int, float))
            or not math.isfinite(hit.score)
        ):
            raise RetrievalError("retriever returned a non-finite or invalid score")
        if hit.passage_id in seen_passages:
            raise RetrievalError("retriever returned a duplicate passage ID")
        seen_passages.add(hit.passage_id)
        source = corpus_by_id.get(hit.passage_id)
        if (
            source is None
            or source.page_title != hit.page_title
            or source.sentence_ids != hit.sentence_ids
            or source.text != hit.text
        ):
            raise RetrievalError("retriever passage payload does not match the accepted corpus")
        if previous is not None:
            if previous.score < hit.score or (
                previous.score == hit.score and previous.passage_id > hit.passage_id
            ):
                raise RetrievalError("retriever ranking violates score or stable tie ordering")
        previous = hit


def _run_attempt(
    retriever: Retriever,
    method: str,
    query: str,
    corpus_by_id: dict[str, Passage],
) -> AttemptOutcome:
    started = perf_counter()
    try:
        response = retriever.search(query, top_k=10)
        wall_elapsed_ms = (perf_counter() - started) * 1_000
        _validate_search_response(response, method, query, corpus_by_id)
        return AttemptOutcome("success", float(response.elapsed_ms), response)
    except Exception as error:
        wall_elapsed_ms = (perf_counter() - started) * 1_000
        return AttemptOutcome(
            "failure",
            wall_elapsed_ms,
            None,
            type(error).__name__,
            str(error)[:240],
        )


def _references_for_hit(hit: RetrievalHit) -> frozenset[Reference]:
    return frozenset((hit.page_title, sentence_id) for sentence_id in hit.sentence_ids)


def _claim_trace(
    sample: EvaluationSample,
    hits: tuple[RetrievalHit, ...],
    corpus_references: set[Reference],
    retrieval_status: str,
    failure: str | None,
) -> dict[str, object]:
    groups = tuple(group for group in sample.evidence_groups if group)
    if sample.label in VERIFIABLE_LABELS and not groups:
        raise RetrievalError(f"verifiable claim {sample.claim_id} has no evidence group")
    gold_union = set().union(*groups) if groups else set()
    first_rank: dict[Reference, int] = {}
    retrieved_passages: list[dict[str, object]] = []
    tie_positions: list[int] = []
    previous: RetrievalHit | None = None
    for hit in hits:
        references = _references_for_hit(hit)
        for reference in references:
            first_rank.setdefault(reference, hit.rank)
        retrieved_passages.append(
            {
                "rank": hit.rank,
                "passage_id": hit.passage_id,
                "score": float(hit.score),
                "source_references": [
                    {"page_title": page_title, "sentence_id": sentence_id}
                    for page_title, sentence_id in sorted(references)
                ],
            }
        )
        if previous is not None and previous.score == hit.score:
            tie_positions.append(previous.rank)
        previous = hit

    evidence_ranks: list[dict[str, object]] = []
    for reference in sorted(gold_union):
        rank = first_rank.get(reference)
        evidence_ranks.append(
            {
                "page_title": reference[0],
                "sentence_id": reference[1],
                "first_rank": rank,
                "recovered_within_top_10": rank is not None,
            }
        )

    group_ranks: list[int] = []
    group_traces: list[dict[str, object]] = []
    for group in groups:
        recovered = sorted(group & first_rank.keys())
        complete_rank = (
            max(first_rank[reference] for reference in group)
            if group <= first_rank.keys()
            else None
        )
        if complete_rank is not None:
            group_ranks.append(complete_rank)
        group_traces.append(
            {
                "references": [
                    {"page_title": page_title, "sentence_id": sentence_id}
                    for page_title, sentence_id in sorted(group)
                ],
                "recovered_reference_count": len(recovered),
                "reference_count": len(group),
                "complete_at_rank": complete_rank,
            }
        )

    metric_by_k: dict[str, object] = {}
    if sample.label in VERIFIABLE_LABELS:
        for cutoff in RETRIEVAL_CUTOFFS:
            retrieved = {
                reference
                for hit in hits
                if hit.rank <= cutoff
                for reference in _references_for_hit(hit)
            }
            recall = len(gold_union & retrieved) / len(gold_union)
            complete = any(group <= retrieved for group in groups)
            metric_by_k[str(cutoff)] = {
                "gold_union_sentence_recall": recall,
                "recovered_gold_pair_count": len(gold_union & retrieved),
                "gold_union_pair_denominator": len(gold_union),
                "complete_evidence_group_hit": complete,
            }
    else:
        metric_by_k = {
            str(cutoff): {
                "gold_union_sentence_recall": None,
                "recovered_gold_pair_count": None,
                "gold_union_pair_denominator": None,
                "complete_evidence_group_hit": None,
            }
            for cutoff in RETRIEVAL_CUTOFFS
        }

    corpus_pairs = gold_union & corpus_references
    corpus_group_hits = sum(group <= corpus_references for group in groups)
    earliest_complete_rank = min(group_ranks) if group_ranks else None
    return {
        "claim_id": sample.claim_id,
        "label": sample.label,
        "retrieval_status": retrieval_status,
        "failure": failure,
        "gold_pair_count": len(gold_union) if sample.label in VERIFIABLE_LABELS else None,
        "corpus_evidence": {
            "gold_pair_count": len(gold_union) if sample.label in VERIFIABLE_LABELS else None,
            "present_pair_count": len(corpus_pairs) if sample.label in VERIFIABLE_LABELS else None,
            "pair_coverage": (len(corpus_pairs) / len(gold_union) if gold_union else None),
            "complete_group_count": len(groups) if sample.label in VERIFIABLE_LABELS else None,
            "complete_groups_present": corpus_group_hits
            if sample.label in VERIFIABLE_LABELS
            else None,
            "any_complete_group_present": (
                any(group <= corpus_references for group in groups)
                if sample.label in VERIFIABLE_LABELS
                else None
            ),
            "missing_gold_references": [
                {"page_title": page_title, "sentence_id": sentence_id}
                for page_title, sentence_id in sorted(gold_union - corpus_references)
            ],
        },
        "retrieved_passages": retrieved_passages,
        "ranking_tie_after_ranks": tie_positions,
        "first_rank_by_gold_reference": evidence_ranks,
        "earliest_complete_evidence_group_rank": earliest_complete_rank,
        "evidence_groups": group_traces,
        "unrecovered_gold_references_within_top_10": [
            item for item in evidence_ranks if not item["recovered_within_top_10"]
        ],
        "at_k": metric_by_k,
    }


def _recovery_shape(
    selected_traces: list[dict[str, object]],
    cutoff: int,
) -> dict[str, object]:
    if not selected_traces:
        return {
            "claims_with_any_gold_pair_recovered": None,
            "claims_with_partial_gold_union_recovery": None,
            "claims_with_no_gold_pair_recovered": None,
            "claims_with_full_gold_union_recovery": None,
        }
    recovered_counts = [
        int(trace["at_k"][str(cutoff)]["recovered_gold_pair_count"]) for trace in selected_traces
    ]
    gold_counts = [
        int(trace["at_k"][str(cutoff)]["gold_union_pair_denominator"]) for trace in selected_traces
    ]
    return {
        "claims_with_any_gold_pair_recovered": sum(count > 0 for count in recovered_counts),
        "claims_with_partial_gold_union_recovery": sum(
            0 < recovered < gold
            for recovered, gold in zip(recovered_counts, gold_counts, strict=True)
        ),
        "claims_with_no_gold_pair_recovered": sum(count == 0 for count in recovered_counts),
        "claims_with_full_gold_union_recovery": sum(
            recovered == gold for recovered, gold in zip(recovered_counts, gold_counts, strict=True)
        ),
    }


def _score_rankings(
    samples: tuple[EvaluationSample, ...],
    hits_by_claim_id: dict[int, tuple[RetrievalHit, ...]],
    status_by_claim_id: dict[int, tuple[str, str | None]],
    corpus_references: set[Reference],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    traces: list[dict[str, object]] = []
    for sample in samples:
        status, failure = status_by_claim_id[sample.claim_id]
        traces.append(
            _claim_trace(
                sample,
                hits_by_claim_id[sample.claim_id],
                corpus_references,
                status,
                failure,
            )
        )

    verifiable = [sample for sample in samples if sample.label in VERIFIABLE_LABELS]
    at_k: dict[str, object] = {}
    for cutoff in RETRIEVAL_CUTOFFS:
        by_label: dict[str, object] = {}
        for label in LABELS:
            selected = [sample for sample in samples if sample.label == label]
            if label == "NOT ENOUGH INFO":
                by_label[label] = {
                    "sample_claims": len(selected),
                    "evidence_claim_denominator": 0,
                    "gold_union_sentence_recall_macro": None,
                    "complete_evidence_group_hits": None,
                    "complete_evidence_group_hit_rate_macro": None,
                    **_recovery_shape([], cutoff),
                }
                continue
            selected_traces = [trace for trace in traces if trace["label"] == label]
            metrics = [trace["at_k"][str(cutoff)] for trace in selected_traces]
            recall_values = [float(metric["gold_union_sentence_recall"]) for metric in metrics]
            group_hits = sum(bool(metric["complete_evidence_group_hit"]) for metric in metrics)
            by_label[label] = {
                "sample_claims": len(selected),
                "evidence_claim_denominator": len(selected),
                "gold_union_sentence_recall_macro": (
                    statistics.fmean(recall_values) if recall_values else None
                ),
                "complete_evidence_group_hits": group_hits,
                "complete_evidence_group_hit_rate_macro": (
                    group_hits / len(selected) if selected else None
                ),
                **_recovery_shape(selected_traces, cutoff),
            }

        selected_traces = [trace for trace in traces if trace["label"] in VERIFIABLE_LABELS]
        metrics = [trace["at_k"][str(cutoff)] for trace in selected_traces]
        recall_values = [float(metric["gold_union_sentence_recall"]) for metric in metrics]
        group_hits = sum(bool(metric["complete_evidence_group_hit"]) for metric in metrics)
        at_k[str(cutoff)] = {
            "gold_union_sentence_recall_macro": (
                statistics.fmean(recall_values) if recall_values else None
            ),
            "gold_union_sentence_recall_claims_denominator": len(verifiable),
            "complete_evidence_group_hits": group_hits,
            "complete_evidence_group_hit_rate_macro": (
                group_hits / len(verifiable) if verifiable else None
            ),
            "complete_evidence_group_claims_denominator": len(verifiable),
            **_recovery_shape(selected_traces, cutoff),
            "by_label": by_label,
        }

    corpus_traces = [trace for trace in traces if trace["label"] in VERIFIABLE_LABELS]
    corpus_pair_coverages = [
        float(trace["corpus_evidence"]["pair_coverage"]) for trace in corpus_traces
    ]
    corpus_group_hits = sum(
        bool(trace["corpus_evidence"]["any_complete_group_present"]) for trace in corpus_traces
    )
    unique_gold_pairs: set[Reference] = set()
    present_gold_pairs: set[Reference] = set()
    for sample in verifiable:
        gold = set().union(*(group for group in sample.evidence_groups if group))
        unique_gold_pairs.update(gold)
        present_gold_pairs.update(gold & corpus_references)

    corpus_coverage = {
        "verifiable_claim_denominator": len(verifiable),
        "gold_pair_coverage_macro": (
            statistics.fmean(corpus_pair_coverages) if corpus_pair_coverages else None
        ),
        "complete_group_hit_claims": corpus_group_hits,
        "complete_group_hit_rate_macro": (
            corpus_group_hits / len(verifiable) if verifiable else None
        ),
        "unique_gold_pairs_across_sample": len(unique_gold_pairs),
        "unique_gold_pairs_present_in_corpus": len(present_gold_pairs),
        "by_label": {
            label: {
                "claim_denominator": sum(sample.label == label for sample in verifiable),
                "gold_pair_coverage_macro": (
                    statistics.fmean(
                        float(trace["corpus_evidence"]["pair_coverage"])
                        for trace in corpus_traces
                        if trace["label"] == label
                    )
                    if any(trace["label"] == label for trace in corpus_traces)
                    else None
                ),
                "complete_group_hit_claims": sum(
                    bool(trace["corpus_evidence"]["any_complete_group_present"])
                    for trace in corpus_traces
                    if trace["label"] == label
                ),
            }
            for label in VERIFIABLE_LABELS
        },
    }
    return (
        {
            "total_claims": len(samples),
            "gold_recovery_claim_denominator": len(verifiable),
            "nei_claims_excluded_from_gold_recovery": len(samples) - len(verifiable),
            "corpus_evidence_coverage": corpus_coverage,
            "at_k": at_k,
        },
        traces,
    )


def _paired_comparison(
    samples: tuple[EvaluationSample, ...],
    method_traces: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    by_id = {
        method: {int(trace["claim_id"]): trace for trace in traces}
        for method, traces in method_traces.items()
    }
    verifiable = [sample for sample in samples if sample.label in VERIFIABLE_LABELS]
    comparison: dict[str, object] = {}
    for cutoff in RETRIEVAL_CUTOFFS:
        if not verifiable:
            comparison[str(cutoff)] = {
                "paired_verifiable_claims": 0,
                "dense_minus_bm25_gold_union_recall_macro": None,
                "complete_group_dense_only_claims": 0,
                "complete_group_bm25_only_claims": 0,
                "complete_group_tied_claims": 0,
            }
            continue
        recall_differences: list[float] = []
        bm25_only_groups = 0
        dense_only_groups = 0
        tied_groups = 0
        for sample in verifiable:
            bm25_metric = by_id["bm25"][sample.claim_id]["at_k"][str(cutoff)]
            dense_metric = by_id["dense"][sample.claim_id]["at_k"][str(cutoff)]
            recall_differences.append(
                float(dense_metric["gold_union_sentence_recall"])
                - float(bm25_metric["gold_union_sentence_recall"])
            )
            bm25_hit = bool(bm25_metric["complete_evidence_group_hit"])
            dense_hit = bool(dense_metric["complete_evidence_group_hit"])
            if bm25_hit and not dense_hit:
                bm25_only_groups += 1
            elif dense_hit and not bm25_hit:
                dense_only_groups += 1
            else:
                tied_groups += 1
        comparison[str(cutoff)] = {
            "paired_verifiable_claims": len(verifiable),
            "dense_minus_bm25_gold_union_recall_macro": statistics.fmean(recall_differences),
            "complete_group_dense_only_claims": dense_only_groups,
            "complete_group_bm25_only_claims": bm25_only_groups,
            "complete_group_tied_claims": tied_groups,
        }
    return comparison


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(percentile * len(ordered))
    return ordered[rank - 1]


def _latency_summary(
    outcomes: list[AttemptOutcome], scheduled: int, attempted: int
) -> dict[str, object]:
    successful = [
        outcome.elapsed_ms
        for outcome in outcomes
        if outcome.status == "success" and outcome.elapsed_ms is not None
    ]
    failed = [
        outcome.elapsed_ms
        for outcome in outcomes
        if outcome.status == "failure" and outcome.elapsed_ms is not None
    ]
    successes = len(successful)
    failures = sum(outcome.status == "failure" for outcome in outcomes)
    missing = scheduled - attempted
    return {
        "scheduled_attempts": scheduled,
        "attempted_queries": attempted,
        "successful_queries": successes,
        "failed_queries": failures + missing,
        "unattempted_queries": missing,
        "successful_warm_query_ms": {
            "observations": successes,
            "p50_nearest_rank": _nearest_rank(successful, 0.50),
            "p95_nearest_rank": _nearest_rank(successful, 0.95),
            "minimum": min(successful) if successful else None,
            "maximum": max(successful) if successful else None,
            "population": "successful timed top-10 calls after the fixed warm-up",
        },
        "failed_query_wall_time_ms": {
            "observations": len(failed),
            "p50_nearest_rank": _nearest_rank(failed, 0.50),
            "p95_nearest_rank": _nearest_rank(failed, 0.95),
            "minimum": min(failed) if failed else None,
            "maximum": max(failed) if failed else None,
            "population": "failed calls measured at the outer search boundary",
        },
    }


def _directory_bytes(path: Path) -> int | None:
    if not path.is_dir():
        return None
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return total


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _environment_record() -> dict[str, object]:
    try:
        import torch

        torch_threads: int | None = torch.get_num_threads()
    except ImportError:
        torch_threads = None
    return {
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "torch_cpu_threads": torch_threads,
        "device": "cpu",
        "packages": {
            "numpy": _package_version("numpy"),
            "rank-bm25": _package_version("rank-bm25"),
            "sentence-transformers": _package_version("sentence-transformers"),
            "qdrant-client": _package_version("qdrant-client"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
    }


def _warm_up(
    retriever: Retriever | None, method: str, corpus_by_id: dict[str, Passage]
) -> dict[str, object]:
    if retriever is None:
        return {"status": "not_attempted", "elapsed_ms": None, "failure": "initialization_failed"}
    started = perf_counter()
    try:
        response = retriever.search(SYNTHETIC_WARMUP_QUERY, top_k=10)
        _validate_search_response(response, method, SYNTHETIC_WARMUP_QUERY, corpus_by_id)
        return {
            "status": "success",
            "elapsed_ms": float(response.elapsed_ms),
            "wall_elapsed_ms": (perf_counter() - started) * 1_000,
        }
    except Exception as error:
        return {
            "status": "failure",
            "elapsed_ms": None,
            "wall_elapsed_ms": (perf_counter() - started) * 1_000,
            "failure_type": type(error).__name__,
            "failure_message": str(error)[:240],
        }


def _attempt_record(outcome: AttemptOutcome, attempt: int) -> dict[str, object]:
    return {
        "attempt": attempt,
        "status": outcome.status,
        "elapsed_ms": outcome.elapsed_ms,
        "failure_type": outcome.failure_type,
        "failure_message": outcome.failure_message,
    }


def _ranking_stability(claim_attempts: list[list[AttemptOutcome]]) -> dict[str, object]:
    complete_triplets = 0
    stable_triplets = 0
    compared_rankings = 0
    for attempts in claim_attempts:
        successful = [
            outcome.response.hits
            for outcome in attempts
            if outcome.status == "success" and outcome.response is not None
        ]
        compared_rankings += max(0, len(successful) - 1)
        if len(attempts) == TIMED_REPEATS and len(successful) == TIMED_REPEATS:
            complete_triplets += 1
            rankings = [tuple(hit.passage_id for hit in hits) for hits in successful]
            if rankings[0] == rankings[1] == rankings[2]:
                stable_triplets += 1
    return {
        "claims_with_three_successful_rankings": complete_triplets,
        "claims_stable_across_all_three_rankings": stable_triplets,
        "ranking_stability_denominator": complete_triplets,
        "matching_pairs_among_successful_rankings": compared_rankings,
    }


def _failure_and_tie_summary(
    samples: tuple[EvaluationSample, ...],
    attempts: dict[int, list[AttemptOutcome]],
) -> dict[str, object]:
    all_outcomes = [outcome for sample in samples for outcome in attempts[sample.claim_id]]
    failures_by_type = Counter(
        outcome.failure_type
        for outcome in all_outcomes
        if outcome.status != "success" and outcome.failure_type is not None
    )
    first_failures = sum(attempts[sample.claim_id][0].status != "success" for sample in samples)
    later_successes = sum(
        attempts[sample.claim_id][0].status != "success"
        and any(outcome.status == "success" for outcome in attempts[sample.claim_id][1:])
        for sample in samples
    )
    successful_rankings = [
        outcome.response.hits
        for outcome in all_outcomes
        if outcome.status == "success" and outcome.response is not None
    ]
    tied_rankings = [
        hits
        for hits in successful_rankings
        if any(left.score == right.score for left, right in zip(hits, hits[1:], strict=False))
    ]
    tied_pairs = sum(
        left.score == right.score
        for hits in successful_rankings
        for left, right in zip(hits, hits[1:], strict=False)
    )
    return {
        "failed_search_attempts": sum(outcome.status == "failure" for outcome in all_outcomes),
        "unattempted_scheduled_queries": sum(
            outcome.status == "not_attempted" for outcome in all_outcomes
        ),
        "failure_counts_by_type": dict(sorted(failures_by_type.items())),
        "first_quality_query_non_successes": first_failures,
        "claims_with_later_success_after_first_non_success": later_successes,
        "successful_rankings_with_score_ties": len(tied_rankings),
        "successful_ranking_observations": len(successful_rankings),
        "adjacent_tied_rank_pairs": tied_pairs,
    }


def _historical_index_observation(qdrant_path: Path) -> dict[str, object]:
    manifest_path = qdrant_path / INDEX_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {
            "build_latency_ms": None,
            "label": "historical M1 observation unavailable in the current manifest",
        }
    value = manifest.get("build_latency_ms") if isinstance(manifest, dict) else None
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        value = None
    return {
        "build_latency_ms": value,
        "label": "historical M1 index-manifest observation; no contemporaneous rebuild performed",
    }


def run_development_smoke(
    passages_path: Path,
    evaluation_path: Path,
    qdrant_path: Path,
    cache_dir: Path,
    top_k: int = 5,
) -> dict[str, object]:
    """Retain the existing train-only smoke workflow and its output contract."""
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


def run_retrieval_evaluation(
    passages_path: Path,
    evaluation_path: Path,
    qdrant_path: Path,
    cache_dir: Path,
    *,
    split: EvaluationSplit = "train",
    freeze_path: Path | None = None,
    confirmed_freeze_sha256: str | None = None,
    held_out_phase_authorized: bool = False,
) -> dict[str, object]:
    """Run paired top-10 retrieval; derive quality from attempt one only."""
    identity = ACCEPTED_EVALUATION_IDENTITY
    freeze_sha256: str | None = None
    if split == "dev":
        if freeze_path is None:
            raise RetrievalError("held-out evaluation requires a reviewed freeze file")
        freeze_sha256 = _validate_freeze_gate(
            freeze_path,
            confirmed_freeze_sha256,
            held_out_phase_authorized,
            identity,
        )

    expected_sample_sha256 = (
        identity.train_sample_sha256 if split == "train" else identity.held_out_sample_sha256
    )
    if _sha256_file(evaluation_path) != expected_sample_sha256:
        raise RetrievalError(
            f"{split} evaluation sample SHA-256 does not match the accepted identity"
        )
    samples = _parse_samples(evaluation_path, split)
    _validate_sample_counts(samples, split)

    held_out_identity_verified = False
    if split == "train":
        if _sha256_file(evaluation_path.parent / "evaluation_samples_dev.jsonl") != (
            identity.held_out_sample_sha256
        ):
            raise RetrievalError("held-out sample SHA-256 does not match the accepted identity")
        held_out_identity_verified = True

    corpus_read_started = perf_counter()
    passages, corpus_sha256 = load_runtime_passages(passages_path)
    corpus_read_ms = (perf_counter() - corpus_read_started) * 1_000
    if corpus_sha256 != identity.corpus_sha256:
        raise RetrievalError("passage corpus SHA-256 does not match the accepted identity")
    if len(passages) != 5_000:
        raise RetrievalError(f"expected 5,000 accepted passages, found {len(passages)}")
    corpus_by_id = {passage.passage_id: passage for passage in passages}
    corpus_references = {
        (passage.page_title, sentence_id)
        for passage in passages
        for sentence_id in passage.sentence_ids
    }

    method_order = ("bm25", "dense")
    retrievers: dict[str, Retriever | None] = {method: None for method in method_order}
    startup: dict[str, dict[str, object]] = {}
    initialization_failures: dict[str, str] = {}
    bm25_started = perf_counter()
    try:
        bm25 = BM25Retriever(passages)
        retrievers["bm25"] = bm25
        startup["bm25"] = {
            "startup_ms": (perf_counter() - bm25_started) * 1_000,
            "in_memory_index_build_ms": bm25.build_latency_ms,
            "corpus_file_read_ms": corpus_read_ms,
            "persistent_index_bytes": None,
            "persistent_index_note": "BM25 index exists only in process memory",
        }
    except Exception as error:
        initialization_failures["bm25"] = f"{type(error).__name__}: {str(error)[:240]}"
        startup["bm25"] = {"startup_ms": None, "in_memory_index_build_ms": None}

    dense_started = perf_counter()
    try:
        dense = DenseRetriever(passages_path, qdrant_path, cache_dir)
        if dense.corpus_sha256 != corpus_sha256:
            dense.close()
            raise RetrievalError("BM25 and dense retrieval are not using the same passage corpus")
        retrievers["dense"] = dense
        startup["dense"] = {
            "startup_ms": (perf_counter() - dense_started) * 1_000,
            "passage_file_read_ms": dense.passage_load_latency_ms,
            "encoder_model_load_ms": dense.encoder_load_latency_ms,
            "qdrant_open_and_count_ms": dense.qdrant_open_latency_ms,
            "dense_index_build_ms": None,
            "dense_index_build_note": "existing persistent index reused; no rebuild performed",
        }
    except Exception as error:
        initialization_failures["dense"] = f"{type(error).__name__}: {str(error)[:240]}"
        startup["dense"] = {"startup_ms": None, "encoder_model_load_ms": None}

    warmups = {
        method: _warm_up(retrievers[method], method, corpus_by_id) for method in method_order
    }
    attempts: dict[str, dict[int, list[AttemptOutcome]]] = {
        method: {sample.claim_id: [] for sample in samples} for method in method_order
    }
    attempt_records: dict[str, dict[int, list[dict[str, object]]]] = {
        method: {sample.claim_id: [] for sample in samples} for method in method_order
    }
    attempted_calls = Counter({method: 0 for method in method_order})

    for sample_position, sample in enumerate(samples):
        paired_order = method_order if sample_position % 2 == 0 else tuple(reversed(method_order))
        for method in paired_order:
            retriever = retrievers[method]
            for repeat in range(1, TIMED_REPEATS + 1):
                if retriever is None:
                    outcome = AttemptOutcome(
                        "not_attempted",
                        None,
                        None,
                        "InitializationError",
                        initialization_failures[method],
                    )
                else:
                    attempted_calls[method] += 1
                    outcome = _run_attempt(retriever, method, sample.claim, corpus_by_id)
                attempts[method][sample.claim_id].append(outcome)
                attempt_records[method][sample.claim_id].append(_attempt_record(outcome, repeat))

    metrics_by_method: dict[str, dict[str, object]] = {}
    traces_by_method: dict[str, list[dict[str, object]]] = {}
    latency_by_method: dict[str, dict[str, object]] = {}
    stability_by_method: dict[str, dict[str, object]] = {}
    failure_summary_by_method: dict[str, dict[str, object]] = {}
    for method in method_order:
        first_hits: dict[int, tuple[RetrievalHit, ...]] = {}
        first_status: dict[int, tuple[str, str | None]] = {}
        all_outcomes: list[AttemptOutcome] = []
        claim_attempts: list[list[AttemptOutcome]] = []
        for sample in samples:
            current = attempts[method][sample.claim_id]
            claim_attempts.append(current)
            all_outcomes.extend(current)
            first = current[0]
            first_hits[sample.claim_id] = (
                first.response.hits
                if first.status == "success" and first.response is not None
                else ()
            )
            first_status[sample.claim_id] = (
                first.status,
                first.failure_type if first.status == "failure" else None,
            )
        metrics, traces = _score_rankings(
            samples,
            first_hits,
            first_status,
            corpus_references,
        )
        metrics_by_method[method] = metrics
        traces_by_method[method] = traces
        latency_by_method[method] = _latency_summary(
            all_outcomes,
            scheduled=len(samples) * TIMED_REPEATS,
            attempted=attempted_calls[method],
        )
        stability_by_method[method] = _ranking_stability(claim_attempts)
        failure_summary_by_method[method] = _failure_and_tie_summary(
            samples,
            attempts[method],
        )
        trace_attempts = {int(trace["claim_id"]): trace for trace in traces}
        for sample in samples:
            trace_attempts[sample.claim_id]["attempts"] = attempt_records[method][sample.claim_id]

    for retriever in retrievers.values():
        if isinstance(retriever, DenseRetriever):
            retriever.close()

    qdrant_bytes = _directory_bytes(qdrant_path)
    cache_bytes = _directory_bytes(cache_dir)
    for method, value in startup.items():
        if method == "dense":
            value["persistent_index_bytes"] = qdrant_bytes
            value["model_cache_bytes"] = cache_bytes

    label_counts = dict(Counter(sample.label for sample in samples))
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "split": split,
        "dataset_scope": (
            "curated, gold-informed 5,000-passage FEVER subset; "
            "not full-FEVER benchmark performance"
        ),
        "dataset_identity": {
            "corpus_sha256": corpus_sha256,
            "corpus_passages": len(passages),
            "evaluation_sample_sha256": expected_sample_sha256,
            "evaluation_sample_claims": len(samples),
            "label_counts": label_counts,
            "held_out_sample_sha256_opaque_check": (
                identity.held_out_sample_sha256 if held_out_identity_verified else None
            ),
            "held_out_records_opened": split == "dev",
            "held_out_phase_authorized": held_out_phase_authorized if split == "dev" else False,
            "reviewed_freeze_sha256": freeze_sha256,
        },
        "retrieval_configuration": {
            "methods": list(method_order),
            "cutoffs_from_one_top_10_ranking": list(RETRIEVAL_CUTOFFS),
            "timed_repeats_per_claim_method": TIMED_REPEATS,
            "timed_queries_per_method_expected": len(samples) * TIMED_REPEATS,
            "concurrency": 1,
            "warmup": (
                "one fixed synthetic top-10 query per initialized method; excluded from timings"
            ),
            "quality_ranking": (
                "first timed top-10 query only; later repeats measure latency and stability"
            ),
            "timing_percentiles": "nearest rank, ceil(p*n), one-based",
            "query_boundary": (
                "Retriever.search including query encoding, retrieval, payload conversion "
                "and sorting"
            ),
            "bm25": {
                "implementation": "BM25Okapi",
                "k1": 1.5,
                "b": 0.75,
                "epsilon": 0.25,
                "tokenization": "casefolded Unicode word tokens",
                "tie_break": "descending score, ascending passage ID",
            },
            "dense": {
                "model": MODEL_NAME,
                "revision": MODEL_REVISION,
                "dimension": MODEL_DIMENSION,
                "query_instruction": QUERY_INSTRUCTION,
                "index": (
                    "existing persistent local Qdrant; exact cosine search returns "
                    "all corpus scores before sorting"
                ),
                "tie_break": "descending score, ascending passage ID",
            },
        },
        "environment": _environment_record(),
        "operations": {
            "corpus_file_bytes": passages_path.stat().st_size,
            "corpus_file_read_ms": corpus_read_ms,
            "qdrant_directory_bytes_after_close": qdrant_bytes,
            "encoder_cache_directory_bytes": cache_bytes,
            "historical_dense_build": _historical_index_observation(qdrant_path),
            "current_dense_rebuild_performed": False,
            "provider_calls": 0,
            "paid_inference": False,
        },
        "methods": {
            method: {
                "startup": startup[method],
                "initialization_failure": initialization_failures.get(method),
                "synthetic_warmup": warmups[method],
                "latency": latency_by_method[method],
                "ranking_stability": stability_by_method[method],
                "failure_and_tie_diagnostics": failure_summary_by_method[method],
                "metrics": metrics_by_method[method],
                "claim_traces": traces_by_method[method],
            }
            for method in method_order
        },
        "paired_comparison": _paired_comparison(samples, traces_by_method),
    }
