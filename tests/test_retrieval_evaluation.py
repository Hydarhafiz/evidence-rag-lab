"""Independent synthetic contracts for paired retrieval evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import cast

import pytest

from evidence_rag import evaluation
from evidence_rag.evaluation import (
    AttemptOutcome,
    EvaluationIdentity,
    EvaluationSample,
    _claim_trace,
    _latency_summary,
    _nearest_rank,
    _paired_comparison,
    _score_rankings,
    _validate_search_response,
    load_development_samples,
    run_retrieval_evaluation,
    validate_freeze_gate,
)
from evidence_rag.models import Passage, RetrievalHit, SearchResponse
from evidence_rag.retrieval import RetrievalError


def _sample(
    claim_id: int,
    label: str,
    groups: list[set[tuple[str, int]]],
    split: str = "train",
) -> EvaluationSample:
    return EvaluationSample(
        split=split,
        claim_id=claim_id,
        label=label,
        claim=f"synthetic claim {claim_id}",
        evidence_groups=tuple(frozenset(group) for group in groups),
    )


def _hit(
    rank: int,
    passage_id: str,
    page_title: str,
    sentence_ids: tuple[int, ...],
    score: float = 1.0,
    method: str = "bm25",
) -> RetrievalHit:
    return RetrievalHit(
        passage_id=passage_id,
        page_title=page_title,
        sentence_ids=sentence_ids,
        score=score,
        text="synthetic source text",
        rank=rank,
        retrieval_method=method,
    )


def _score(
    samples: tuple[EvaluationSample, ...],
    rankings: dict[int, tuple[RetrievalHit, ...]],
    corpus: set[tuple[str, int]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    statuses = {sample.claim_id: ("success", None) for sample in samples}
    return _score_rankings(samples, rankings, statuses, corpus)


def test_overlapping_pairs_and_alternative_multi_page_groups_are_deduplicated() -> None:
    sample = _sample(
        1,
        "SUPPORTS",
        [
            {("Page A", 1), ("Page B", 2), ("Page A", 1)},
            {("Page A", 1), ("Page C", 3)},
        ],
    )
    hits = (
        _hit(1, "p1", "Page A", (1,)),
        _hit(2, "p2", "Page B", (2,), score=0.8),
        _hit(3, "p3", "Page A", (1,), score=0.7),
        _hit(5, "p5", "Page C", (3,), score=0.5),
    )

    metrics, traces = _score((sample,), {1: hits}, {("Page A", 1), ("Page B", 2), ("Page C", 3)})

    trace = traces[0]
    assert trace["gold_pair_count"] == 3
    assert trace["earliest_complete_evidence_group_rank"] == 2
    assert [group["complete_at_rank"] for group in trace["evidence_groups"]] == [2, 5]
    assert trace["at_k"]["1"]["gold_union_sentence_recall"] == 1 / 3
    assert trace["at_k"]["1"]["complete_evidence_group_hit"] is False
    assert metrics["at_k"]["1"]["claims_with_partial_gold_union_recovery"] == 1
    assert trace["at_k"]["5"]["gold_union_sentence_recall"] == 1.0
    assert trace["at_k"]["5"]["complete_evidence_group_hit"] is True
    assert metrics["at_k"]["5"]["claims_with_full_gold_union_recovery"] == 1
    assert metrics["at_k"]["5"]["complete_evidence_group_hit_rate_macro"] == 1.0


def test_evidence_recovered_only_at_rank_five_or_ten_uses_nested_prefixes() -> None:
    sample = _sample(
        2,
        "REFUTES",
        [{("Page Five", 5)}, {("Page Ten", 10)}],
    )
    hits = tuple(
        [
            _hit(rank, f"p{rank:02}", f"Noise {rank}", (rank,), 1.0 - rank / 100)
            for rank in range(1, 5)
        ]
        + [
            _hit(5, "p05-evidence", "Page Five", (5,), 0.9),
            *[
                _hit(rank, f"p{rank:02}", f"Noise {rank}", (rank,), 0.8 - rank / 100)
                for rank in range(6, 10)
            ],
            _hit(10, "p10-evidence", "Page Ten", (10,), 0.1),
        ]
    )

    _, traces = _score((sample,), {2: hits}, {("Page Five", 5), ("Page Ten", 10)})

    trace = traces[0]
    assert trace["first_rank_by_gold_reference"] == [
        {
            "page_title": "Page Five",
            "sentence_id": 5,
            "first_rank": 5,
            "recovered_within_top_10": True,
        },
        {
            "page_title": "Page Ten",
            "sentence_id": 10,
            "first_rank": 10,
            "recovered_within_top_10": True,
        },
    ]
    assert trace["at_k"]["1"]["gold_union_sentence_recall"] == 0.0
    assert trace["at_k"]["5"]["gold_union_sentence_recall"] == 0.5
    assert trace["at_k"]["5"]["complete_evidence_group_hit"] is True
    assert trace["at_k"]["10"]["gold_union_sentence_recall"] == 1.0
    assert trace["earliest_complete_evidence_group_rank"] == 5


def test_empty_ranking_is_a_zero_retrieval_result_with_corpus_coverage_kept() -> None:
    sample = _sample(3, "SUPPORTS", [{("Page A", 7)}])

    metrics, traces = _score((sample,), {3: ()}, {("Page A", 7)})

    assert traces[0]["corpus_evidence"]["pair_coverage"] == 1.0
    assert traces[0]["unrecovered_gold_references_within_top_10"] == [
        {
            "page_title": "Page A",
            "sentence_id": 7,
            "first_rank": None,
            "recovered_within_top_10": False,
        }
    ]
    assert metrics["at_k"]["10"]["gold_union_sentence_recall_macro"] == 0.0
    assert metrics["at_k"]["10"]["complete_evidence_group_hits"] == 0


def test_nei_is_present_in_sample_but_outside_gold_recovery_denominators() -> None:
    samples = (
        _sample(1, "SUPPORTS", [{("Page A", 1)}]),
        _sample(2, "REFUTES", [{("Page B", 2)}]),
        _sample(3, "NOT ENOUGH INFO", []),
    )

    metrics, traces = _score(samples, {1: (), 2: (), 3: ()}, {("Page A", 1), ("Page B", 2)})

    at_k = metrics["at_k"]["1"]
    assert metrics["total_claims"] == 3
    assert metrics["gold_recovery_claim_denominator"] == 2
    assert metrics["nei_claims_excluded_from_gold_recovery"] == 1
    assert at_k["gold_union_sentence_recall_claims_denominator"] == 2
    assert at_k["by_label"]["NOT ENOUGH INFO"]["sample_claims"] == 1
    assert at_k["by_label"]["NOT ENOUGH INFO"]["evidence_claim_denominator"] == 0
    assert at_k["by_label"]["NOT ENOUGH INFO"]["gold_union_sentence_recall_macro"] is None
    assert traces[2]["at_k"]["1"]["complete_evidence_group_hit"] is None


def test_zero_verifiable_denominator_is_reported_as_na() -> None:
    sample = _sample(4, "NOT ENOUGH INFO", [])

    metrics, traces = _score((sample,), {4: ()}, set())
    comparison = _paired_comparison((sample,), {"bm25": traces, "dense": traces})

    assert metrics["gold_recovery_claim_denominator"] == 0
    assert metrics["corpus_evidence_coverage"]["gold_pair_coverage_macro"] is None
    assert metrics["at_k"]["1"]["gold_union_sentence_recall_macro"] is None
    assert comparison["1"]["dense_minus_bm25_gold_union_recall_macro"] is None


def test_corpus_absence_is_distinct_from_evidence_present_but_not_retrieved() -> None:
    samples = (
        _sample(1, "SUPPORTS", [{("Absent Page", 1)}]),
        _sample(2, "REFUTES", [{("Present Page", 2)}]),
    )

    metrics, traces = _score(samples, {1: (), 2: ()}, {("Present Page", 2)})

    assert traces[0]["corpus_evidence"]["pair_coverage"] == 0.0
    assert traces[0]["corpus_evidence"]["missing_gold_references"] == [
        {"page_title": "Absent Page", "sentence_id": 1}
    ]
    assert traces[1]["corpus_evidence"]["pair_coverage"] == 1.0
    assert traces[1]["unrecovered_gold_references_within_top_10"][0]["page_title"] == "Present Page"
    assert metrics["corpus_evidence_coverage"]["gold_pair_coverage_macro"] == 0.5
    assert metrics["corpus_evidence_coverage"]["unique_gold_pairs_present_in_corpus"] == 1


def test_stable_ties_use_ascending_passage_id_and_are_recorded() -> None:
    corpus = {
        "p-a": Passage("p-a", "Page A", (1,), "synthetic source text", "synthetic", "0" * 64),
        "p-b": Passage("p-b", "Page B", (2,), "synthetic source text", "synthetic", "0" * 64),
    }
    tied = SearchResponse(
        "bm25",
        "synthetic query",
        1.0,
        (_hit(1, "p-a", "Page A", (1,)), _hit(2, "p-b", "Page B", (2,))),
    )

    _validate_search_response(tied, "bm25", "synthetic query", corpus)
    trace = _claim_trace(
        _sample(1, "SUPPORTS", [{("Page A", 1)}]),
        tied.hits,
        {("Page A", 1), ("Page B", 2)},
        "success",
        None,
    )
    assert trace["ranking_tie_after_ranks"] == [1]

    reversed_tie = SearchResponse(
        "bm25",
        "synthetic query",
        1.0,
        (_hit(1, "p-b", "Page B", (2,)), _hit(2, "p-a", "Page A", (1,))),
    )
    with pytest.raises(RetrievalError, match="stable tie ordering"):
        _validate_search_response(reversed_tie, "bm25", "synthetic query", corpus)


def test_invalid_or_nonfinite_retrieval_scores_are_rejected() -> None:
    corpus = {"p-a": Passage("p-a", "Page A", (1,), "source", "synthetic", "0" * 64)}
    response = SearchResponse(
        "dense",
        "synthetic query",
        1.0,
        (_hit(1, "p-a", "Page A", (1,), score=float("nan"), method="dense"),),
    )

    with pytest.raises(RetrievalError, match="non-finite"):
        _validate_search_response(response, "dense", "synthetic query", corpus)


def test_failed_search_attempts_remain_in_the_scheduled_denominator() -> None:
    outcomes = [
        AttemptOutcome("success", 1.0, None),
        AttemptOutcome("failure", 3.0, None, "TimeoutError", "synthetic timeout"),
        AttemptOutcome("not_attempted", None, None, "InitializationError", "synthetic init"),
    ]

    latency = _latency_summary(outcomes, scheduled=4, attempted=2)

    assert latency["scheduled_attempts"] == 4
    assert latency["attempted_queries"] == 2
    assert latency["successful_queries"] == 1
    assert latency["failed_queries"] == 3
    assert latency["unattempted_queries"] == 2
    assert latency["successful_warm_query_ms"]["observations"] == 1
    assert latency["failed_query_wall_time_ms"]["observations"] == 1


def test_nearest_rank_percentiles_follow_ceil_rule() -> None:
    assert _nearest_rank([5.0, 1.0, 4.0, 2.0, 3.0], 0.50) == 3.0
    assert _nearest_rank([5.0, 1.0, 4.0, 2.0, 3.0], 0.95) == 5.0
    assert _nearest_rank([], 0.95) is None


def test_aggregate_results_are_reproducible_for_fixed_rankings() -> None:
    sample = _sample(8, "SUPPORTS", [{("Page A", 1), ("Page B", 2)}])
    hits = (_hit(1, "p1", "Page A", (1,)), _hit(2, "p2", "Page B", (2,), 0.5))
    corpus = {("Page A", 1), ("Page B", 2)}

    first = _score((sample,), {8: hits}, corpus)
    second = _score((sample,), {8: hits}, corpus)

    assert first == second


def test_sample_parser_deduplicates_exact_pairs_and_preserves_alternatives(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    row = {
        "split": "train",
        "claim_id": 1,
        "label": "SUPPORTS",
        "claim": "synthetic",
        "evidence_groups": [
            [
                {"page_title": "Page A", "sentence_id": 1},
                {"page_title": "Page B", "sentence_id": 2},
                {"page_title": "Page A", "sentence_id": 1},
            ],
            [{"page_title": "Page C", "sentence_id": 3}],
        ],
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")

    sample = load_development_samples(source)[0]

    assert len(sample.evidence_groups) == 2
    assert sample.evidence_groups[0] == frozenset({("Page A", 1), ("Page B", 2)})
    assert sample.evidence_groups[1] == frozenset({("Page C", 3)})


@pytest.mark.parametrize(
    "label,groups",
    [
        ("SUPPORTS", []),
        ("SUPPORTS", [[]]),
        ("REFUTES", [[{"page_title": "Page A", "sentence_id": -1}]]),
        ("NOT ENOUGH INFO", [[{"page_title": "Page A", "sentence_id": 1}]]),
    ],
)
def test_malformed_verifiable_groups_and_nei_gold_are_rejected(
    tmp_path: Path,
    label: str,
    groups: list[object],
) -> None:
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(
            {
                "split": "train",
                "claim_id": 1,
                "label": label,
                "claim": "synthetic",
                "evidence_groups": groups,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RetrievalError):
        load_development_samples(source)


def test_duplicate_sample_ids_and_heldout_rows_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    train_row = {
        "split": "train",
        "claim_id": 9,
        "label": "SUPPORTS",
        "claim": "synthetic",
        "evidence_groups": [[{"page_title": "Page A", "sentence_id": 1}]],
    }
    source.write_text(json.dumps(train_row) + "\n" + json.dumps(train_row) + "\n", encoding="utf-8")
    with pytest.raises(RetrievalError, match="duplicate claim IDs"):
        load_development_samples(source)

    train_row["split"] = "dev"
    source.write_text(json.dumps(train_row) + "\n", encoding="utf-8")
    with pytest.raises(RetrievalError, match="only train samples"):
        load_development_samples(source)


def test_freeze_gate_checks_identity_checksum_and_separate_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = EvaluationIdentity("c" * 64, "a" * 64, "b" * 64)
    monkeypatch.setattr(evaluation, "ACCEPTED_EVALUATION_IDENTITY", identity)
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(
        json.dumps(
            {
                "protocol": evaluation.PROTOCOL_ID,
                "corpus_sha256": identity.corpus_sha256,
                "held_out_sample_sha256": identity.held_out_sample_sha256,
            }
        ),
        encoding="utf-8",
    )
    freeze_hash = hashlib.sha256(freeze_path.read_bytes()).hexdigest()

    with pytest.raises(RetrievalError, match="separate authorization"):
        validate_freeze_gate(freeze_path, freeze_hash, False)
    with pytest.raises(RetrievalError, match="does not match"):
        validate_freeze_gate(freeze_path, "0" * 64, True)
    assert validate_freeze_gate(freeze_path, freeze_hash, True) == freeze_hash


def _synthetic_rows(split: str, per_label: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    claim_id = 100
    for label in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"):
        for _ in range(per_label):
            groups = (
                []
                if label == "NOT ENOUGH INFO"
                else [[{"page_title": f"Evidence {claim_id}", "sentence_id": claim_id}]]
            )
            rows.append(
                {
                    "split": split,
                    "claim_id": claim_id,
                    "label": label,
                    "claim": f"synthetic claim {claim_id}",
                    "evidence_groups": groups,
                }
            )
            claim_id += 1
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _SyntheticRetriever:
    def __init__(
        self, method: str, queries: list[tuple[str, str, int]], fail_once: str | None = None
    ) -> None:
        self.method = method
        self.queries = queries
        self.fail_once = fail_once
        self.failed = False

    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        self.queries.append((self.method, query, top_k))
        if query == self.fail_once and not self.failed:
            self.failed = True
            raise TimeoutError("synthetic timeout")
        return SearchResponse(self.method, query, 1.0, ())


def _install_synthetic_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    split: str,
) -> tuple[Path, Path, Path, Path, EvaluationIdentity, list[tuple[str, str, int]]]:
    per_label = 10 if split == "train" else 20
    rows = _synthetic_rows(split, per_label)
    evaluation_path = tmp_path / f"evaluation_samples_{split}.jsonl"
    selected_hash = _write_rows(evaluation_path, rows)
    held_out_path = tmp_path / "evaluation_samples_dev.jsonl"
    if split == "train":
        held_out_path.write_bytes(b"synthetic opaque fixture; do not parse\n")
        held_out_hash = hashlib.sha256(held_out_path.read_bytes()).hexdigest()
        selected_train_hash = selected_hash
    else:
        held_out_hash = selected_hash
        selected_train_hash = hashlib.sha256(b"synthetic train identity\n").hexdigest()
    corpus_hash = hashlib.sha256(b"synthetic corpus identity\n").hexdigest()
    identity = EvaluationIdentity(corpus_hash, selected_train_hash, held_out_hash)
    passages = [
        Passage(
            f"p-{index:04}",
            f"Evidence {index + 100}",
            (index + 100,),
            "synthetic source",
            "synthetic",
            "0" * 64,
        )
        for index in range(5_000)
    ]
    passage_path = tmp_path / "passages.jsonl"
    passage_path.write_text("synthetic runtime corpus marker\n", encoding="utf-8")
    qdrant_path = tmp_path / "qdrant"
    cache_path = tmp_path / "cache"
    query_calls: list[tuple[str, str, int]] = []
    fail_query = cast(str, rows[0]["claim"]) if split == "train" else None

    def fake_load_runtime_passages(_path: Path) -> tuple[tuple[Passage, ...], str]:
        return tuple(passages), corpus_hash

    class FakeBm25(_SyntheticRetriever):
        def __init__(self, _passages: tuple[Passage, ...]) -> None:
            super().__init__("bm25", query_calls, fail_once=fail_query)
            self.build_latency_ms = 2.0

    class FakeDense(_SyntheticRetriever):
        def __init__(self, _passages_path: Path, _qdrant_path: Path, _cache_path: Path) -> None:
            super().__init__("dense", query_calls)
            self.corpus_sha256 = corpus_hash
            self.passage_load_latency_ms = 1.0
            self.encoder_load_latency_ms = 2.0
            self.qdrant_open_latency_ms = 3.0

        def close(self) -> None:
            return None

    monkeypatch.setattr(evaluation, "load_runtime_passages", fake_load_runtime_passages)
    monkeypatch.setattr(evaluation, "BM25Retriever", FakeBm25)
    monkeypatch.setattr(evaluation, "DenseRetriever", FakeDense)
    monkeypatch.setattr(evaluation, "ACCEPTED_EVALUATION_IDENTITY", identity)
    return passage_path, evaluation_path, qdrant_path, cache_path, identity, query_calls


def test_actual_runner_accounts_repeats_failures_and_claim_only_retrieval_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    passages, training, qdrant, cache, identity, queries = _install_synthetic_runtime(
        monkeypatch, tmp_path, "train"
    )

    report = run_retrieval_evaluation(
        passages,
        training,
        qdrant,
        cache,
    )

    assert report["dataset_identity"]["evaluation_sample_claims"] == 30
    assert report["dataset_identity"]["held_out_records_opened"] is False
    assert report["dataset_identity"]["label_counts"] == {
        "SUPPORTS": 10,
        "REFUTES": 10,
        "NOT ENOUGH INFO": 10,
    }
    for method in ("bm25", "dense"):
        latency = report["methods"][method]["latency"]
        assert latency["scheduled_attempts"] == 90
        assert latency["attempted_queries"] == 90
        assert latency["successful_queries"] == (89 if method == "bm25" else 90)
        assert latency["failed_queries"] == (1 if method == "bm25" else 0)
        assert report["methods"][method]["metrics"]["gold_recovery_claim_denominator"] == 20
    bm25_trace = report["methods"]["bm25"]["claim_traces"][0]
    assert bm25_trace["attempts"][0]["status"] == "failure"
    assert bm25_trace["at_k"]["10"]["gold_union_sentence_recall"] == 0.0
    assert (
        report["methods"]["bm25"]["failure_and_tie_diagnostics"][
            "claims_with_later_success_after_first_non_success"
        ]
        == 1
    )
    assert report["methods"]["bm25"]["failure_and_tie_diagnostics"]["failure_counts_by_type"] == {
        "TimeoutError": 1
    }
    assert Counter(method for method, _query, _k in queries) == {"bm25": 91, "dense": 91}
    assert all(isinstance(query, str) for _method, query, _k in queries)
    assert all(
        query
        in {cast(str, row["claim"]) for row in _synthetic_rows("train", 10)}
        | {evaluation.SYNTHETIC_WARMUP_QUERY}
        for _method, query, _k in queries
    )
    assert all(
        "SUPPORTS" not in query and "evidence_groups" not in query for _method, query, _k in queries
    )
    assert report["operations"]["provider_calls"] == 0


def test_synthetic_heldout_run_is_blocked_before_sample_access_then_freeze_gated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    passages, heldout, qdrant, cache, identity, _queries = _install_synthetic_runtime(
        monkeypatch, tmp_path, "dev"
    )
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text("not a reviewed freeze\n", encoding="utf-8")

    with pytest.raises(RetrievalError, match="separate authorization"):
        run_retrieval_evaluation(
            passages,
            tmp_path / "missing-heldout-file.jsonl",
            qdrant,
            cache,
            split="dev",
            freeze_path=freeze_path,
            held_out_phase_authorized=False,
        )

    freeze_path.write_text(
        json.dumps(
            {
                "protocol": evaluation.PROTOCOL_ID,
                "corpus_sha256": identity.corpus_sha256,
                "held_out_sample_sha256": identity.held_out_sample_sha256,
            }
        ),
        encoding="utf-8",
    )
    freeze_hash = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    report = run_retrieval_evaluation(
        passages,
        heldout,
        qdrant,
        cache,
        split="dev",
        freeze_path=freeze_path,
        confirmed_freeze_sha256=freeze_hash,
        held_out_phase_authorized=True,
    )

    assert report["dataset_identity"]["held_out_records_opened"] is True
    assert report["dataset_identity"]["reviewed_freeze_sha256"] == freeze_hash
    assert report["dataset_identity"]["evaluation_sample_claims"] == 60
    assert report["dataset_identity"]["label_counts"] == {
        "SUPPORTS": 20,
        "REFUTES": 20,
        "NOT ENOUGH INFO": 20,
    }
