"""Source validation, deterministic sampling, and bounded passage tests."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

from evidence_rag import corpus
from evidence_rag.corpus import (
    CorpusError,
    SentenceRecord,
    _parse_wiki_page,
    inventory_wikipedia_source,
    load_claims,
    parse_wiki_lines,
    passages_for_window,
    prepare_corpus,
    verify_source_hashes,
)
from tests.conftest import SyntheticSources, TinyTokenizer


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def test_source_hash_mismatch_stops_before_parsing(
    synthetic_sources: SyntheticSources,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(corpus.SOURCE_HASHES, "train.jsonl", "0" * 64)

    with pytest.raises(CorpusError, match="source hash mismatch"):
        verify_source_hashes(
            synthetic_sources.train_path,
            synthetic_sources.dev_path,
            synthetic_sources.wiki_path,
        )


def test_claim_schema_and_page_identifier_validation() -> None:
    with pytest.raises(CorpusError, match="unexpected claim fields"):
        corpus._parse_claim_row(
            {"id": 1, "claim": "fixture", "label": "SUPPORTS", "evidence": []},
            "train",
            1,
        )

    with pytest.raises(CorpusError, match="malformed page id"):
        _parse_wiki_page({"text": "fixture", "lines": "0\\tsentence"}, "fixture", 1)
    with pytest.raises(CorpusError, match="malformed page id"):
        _parse_wiki_page({"id": 17, "text": "fixture", "lines": "0\\tsentence"}, "fixture", 1)


def test_wiki_lines_keep_unique_numeric_ids_and_ignore_bad_rows() -> None:
    parsed = parse_wiki_lines(
        "header\n0\tfirst version\tlink\n1\t   \n0\tduplicate version\tlink\n2\tkept"
    )

    assert parsed.duplicate_ids == frozenset({0})
    assert [(record.sentence_id, record.text) for record in parsed.records] == [(2, "kept")]
    assert parsed.nonnumeric_rows == 1
    assert parsed.empty_text_rows == 1


def test_duplicate_wikipedia_page_identifier_is_rejected(
    synthetic_sources: SyntheticSources,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(corpus, "EXPECTED_WIKI_PAGES", 2)

    with pytest.raises(CorpusError, match="duplicate Wikipedia page id"):
        inventory_wikipedia_source(synthetic_sources.duplicate_wiki_path, set())


def test_long_sentence_is_split_without_losing_original_reference() -> None:
    original = SentenceRecord(23, "one two three four five six seven eight nine")

    passages, split_sentences, fragment_count, maximum_length = passages_for_window(
        "Article",
        (original,),
        "a" * 64,
        TinyTokenizer(),
        maximum_tokens=7,
    )

    assert split_sentences == 1
    assert fragment_count == len(passages) == 3
    assert all(passage.sentence_ids == (23,) for passage in passages)
    assert maximum_length <= 7
    assert "".join(passage.text.split("\n", maxsplit=1)[1] for passage in passages) == original.text


def test_selection_rebuild_and_runtime_isolation_are_deterministic(
    synthetic_sources: SyntheticSources,
) -> None:
    first = prepare_corpus(
        synthetic_sources.train_path,
        synthetic_sources.dev_path,
        synthetic_sources.wiki_path,
        synthetic_sources.output_one,
        TinyTokenizer(),
        maximum_tokens=32,
        target_passages=25,
    )
    second = prepare_corpus(
        synthetic_sources.train_path,
        synthetic_sources.dev_path,
        synthetic_sources.wiki_path,
        synthetic_sources.output_two,
        TinyTokenizer(),
        maximum_tokens=32,
        target_passages=25,
    )

    first_passage_path = synthetic_sources.output_one / "passages.jsonl"
    second_passage_path = synthetic_sources.output_two / "passages.jsonl"
    first_train_path = synthetic_sources.output_one / "evaluation_samples_train.jsonl"
    second_train_path = synthetic_sources.output_two / "evaluation_samples_train.jsonl"
    first_dev_path = synthetic_sources.output_one / "evaluation_samples_dev.jsonl"
    second_dev_path = synthetic_sources.output_two / "evaluation_samples_dev.jsonl"
    assert first_passage_path.read_bytes() == second_passage_path.read_bytes()
    assert first_train_path.read_bytes() == second_train_path.read_bytes()
    assert first_dev_path.read_bytes() == second_dev_path.read_bytes()
    assert first.manifest == second.manifest

    passages = _read_jsonl(first_passage_path)
    train_evaluation = _read_jsonl(first_train_path)
    dev_evaluation = _read_jsonl(first_dev_path)
    evaluation = train_evaluation + dev_evaluation
    assert len(passages) == 25
    assert len(train_evaluation) == 30
    assert len(dev_evaluation) == 60
    assert {row["split"] for row in train_evaluation} == {"train"}
    assert {row["split"] for row in dev_evaluation} == {"dev"}
    assert not (synthetic_sources.output_one / "evaluation_samples.jsonl").exists()
    assert Counter((row["split"], row["label"]) for row in evaluation) == Counter(
        {
            (split, label): count
            for split, count in (("train", 10), ("dev", 20))
            for label in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
        }
    )

    runtime_fields = {
        "passage_id",
        "page_title",
        "sentence_ids",
        "text",
        "source_artifact",
        "source_sha256",
    }
    assert all(set(row) == runtime_fields for row in passages)
    assert all(
        "claim" not in row and "label" not in row and "evidence_groups" not in row
        for row in passages
    )
    assert all(row["page_title"] not in {"", "duplicate-sentence-page"} for row in passages)
    assert all("linked-entity" not in row["text"] for row in passages)

    train_claims = load_claims(synthetic_sources.train_path, "train")
    dev_claims = load_claims(synthetic_sources.dev_path, "dev")
    cross_split_texts = {
        _normalize(train.claim)
        for train in train_claims
        if any(_normalize(train.claim) == _normalize(dev.claim) for dev in dev_claims)
    }
    expected_ids: list[tuple[str, str, int]] = []
    for split, rows, count in (
        ("train", train_claims, 10),
        ("dev", dev_claims, 20),
    ):
        for label in ("NOT ENOUGH INFO", "REFUTES", "SUPPORTS"):
            candidates = [
                claim
                for claim in rows
                if claim.label == label and _normalize(claim.claim) not in cross_split_texts
            ]
            candidates.sort(
                key=lambda claim: (
                    hashlib.sha256(
                        f"{corpus.SELECTION_SEED}|{split}|{claim.claim_id}".encode()
                    ).hexdigest(),
                    claim.claim_id,
                )
            )
            expected_ids.extend((split, label, claim.claim_id) for claim in candidates[:count])
    observed_ids = sorted(
        (str(row["split"]), str(row["label"]), int(row["claim_id"])) for row in evaluation
    )
    assert observed_ids == sorted(expected_ids)

    expected_background = sorted(
        synthetic_sources.expected_background_windows,
        key=lambda window: (
            hashlib.sha256(
                f"{corpus.SELECTION_SEED}|background|{window[0]}|{window[1][0]}".encode()
            ).hexdigest(),
            window[0],
            window[1][0],
        ),
    )[:22]
    observed_background = {
        (str(row["page_title"]), tuple(row["sentence_ids"]))
        for row in passages
        if row["page_title"] != "Evidence_A"
    }
    assert observed_background == set(expected_background)

    source_structure = first.manifest["source_structure"]
    assert source_structure["missing_page_id_rows"] == 1
    assert source_structure["duplicate_sentence_id_pages"] == 1
    assert first.manifest["passage_construction"]["text_truncation_count"] == 0
