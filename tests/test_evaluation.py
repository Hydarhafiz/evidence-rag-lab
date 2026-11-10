"""Evaluation ownership and development-only retrieval metric tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidence_rag.evaluation import EvaluationSample, _score_method, load_development_samples
from evidence_rag.models import RetrievalHit, SearchResponse
from evidence_rag.retrieval import RetrievalError


def _sample(
    split: str,
    claim_id: int,
    label: str,
    claim: str,
    groups: list[list[dict[str, object]]],
) -> dict[str, object]:
    return {
        "split": split,
        "claim_id": claim_id,
        "label": label,
        "claim": claim,
        "evidence_groups": groups,
    }


class RecordingRetriever:
    method = "bm25"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        self.queries.append(query)
        if query == "train support":
            hits = (
                RetrievalHit(
                    passage_id="one",
                    page_title="Page A",
                    sentence_ids=(3,),
                    score=1.0,
                    text="source sentence",
                    rank=1,
                    retrieval_method="bm25",
                ),
            )
        else:
            hits = ()
        return SearchResponse("bm25", query, 1.25, hits[:top_k])


def test_evaluation_groups_are_separate_and_smoke_uses_only_train_claims(
    tmp_path: Path,
) -> None:
    source = tmp_path / "evaluation_samples_train.jsonl"
    training_rows = [
        _sample(
            "train",
            1,
            "SUPPORTS",
            "train support",
            [
                [
                    {"page_title": "Page A", "sentence_id": 1},
                    {"page_title": "Page B", "sentence_id": 2},
                ],
                [{"page_title": "Page A", "sentence_id": 3}],
            ],
        ),
        _sample(
            "train",
            2,
            "REFUTES",
            "train refute",
            [[{"page_title": "Page C", "sentence_id": 4}]],
        ),
        _sample("train", 3, "NOT ENOUGH INFO", "train nei", [[]]),
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in training_rows),
        encoding="utf-8",
    )

    samples = load_development_samples(source)
    held_out = EvaluationSample(
        split="dev",
        claim_id=4,
        label="SUPPORTS",
        claim="heldout claim",
        evidence_groups=(frozenset({("Page D", 5)}),),
    )
    retriever = RecordingRetriever()
    metrics = _score_method((*samples, held_out), retriever, top_k=5)

    assert len(samples[0].evidence_groups) == 2
    assert samples[0].evidence_groups[0] == frozenset({("Page A", 1), ("Page B", 2)})
    assert retriever.queries == ["train support", "train refute"]
    assert metrics["split"] == "train"
    assert metrics["verifiable_claims"] == 2
    assert metrics["page_sentence_recall_at_5"] == (1 / 3) / 2
    assert metrics["complete_evidence_group_hit_rate_at_5"] == 0.5

    held_out_source = tmp_path / "evaluation_samples_dev.jsonl"
    held_out_source.write_text(
        json.dumps(
            _sample(
                "dev",
                4,
                "SUPPORTS",
                "heldout claim",
                [[{"page_title": "Page D", "sentence_id": 5}]],
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RetrievalError, match="only train samples"):
        load_development_samples(held_out_source)


def test_evaluation_sample_type_errors_are_reported(tmp_path: Path) -> None:
    source = tmp_path / "malformed.jsonl"
    source.write_text(
        json.dumps(
            {
                "split": "train",
                "claim_id": 1,
                "label": [],
                "claim": "synthetic",
                "evidence_groups": [],
            }
        ),
        encoding="utf-8",
    )

    try:
        load_development_samples(source)
    except ValueError as error:
        assert "malformed or non-train row" in str(error)
    else:
        raise AssertionError("malformed evaluation label was accepted")
