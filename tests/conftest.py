"""Synthetic source fixtures for FEVER ingestion and passage construction."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


@dataclass(frozen=True)
class SyntheticSources:
    train_path: Path
    dev_path: Path
    wiki_path: Path
    duplicate_wiki_path: Path
    output_one: Path
    output_two: Path
    expected_background_windows: tuple[tuple[str, tuple[int, ...]], ...]


class TinyTokenizer:
    """Whitespace tokenizer that implements only the encoder calls used by corpus preparation."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        del truncation
        matches = list(re.finditer(r"\S+", text))
        token_ids = list(range(1, len(matches) + 1))
        result: dict[str, Any] = {"input_ids": token_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(match.start(), match.end()) for match in matches]
        if add_special_tokens:
            result["input_ids"] = [101, *token_ids, 102]
            if return_offsets_mapping:
                result["offset_mapping"] = [
                    (0, 0),
                    *result["offset_mapping"],
                    (0, 0),
                ]
        return result


def _claim_row(claim_id: int, label: str, text: str) -> dict[str, object]:
    is_nei = label == "NOT ENOUGH INFO"
    evidence = (
        [[[claim_id, None, None, None]]] if is_nei else [[[claim_id, claim_id, "Evidence_A", 0]]]
    )
    return {
        "id": claim_id,
        "label": label,
        "claim": text,
        "evidence": evidence,
        "verifiable": "NOT VERIFIABLE" if is_nei else "VERIFIABLE",
    }


def _wiki_page(page_id: str, sentences: tuple[str, ...]) -> dict[str, object]:
    return {
        "id": page_id,
        "text": page_id,
        "lines": "\n".join(
            f"{sentence_id}\t{text}\tlinked-entity" for sentence_id, text in enumerate(sentences)
        ),
    }


def _write_archive(path: Path, pages: tuple[dict[str, object], ...]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("license.html", "Synthetic fixture license metadata")
        archive.writestr("wiki-pages/", b"")
        archive.writestr("__MACOSX/", b"")
        archive.writestr("__MACOSX/wiki-pages/", b"")
        for shard_number in range(1, 110):
            name = f"wiki-pages/wiki-{shard_number:03}.jsonl"
            content = b""
            if shard_number == 1:
                content = b"".join(
                    json.dumps(page, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    + b"\n"
                    for page in pages
                )
            archive.writestr(name, content)
        for file_number in range(111):
            archive.writestr(f"__MACOSX/._fixture-{file_number:03}.txt", "sidecar")


@pytest.fixture
def synthetic_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SyntheticSources:
    from evidence_rag import corpus

    labels = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
    train_rows: list[dict[str, object]] = []
    dev_rows: list[dict[str, object]] = []
    identifier = 1
    for label in labels:
        for sample_number in range(10):
            train_rows.append(_claim_row(identifier, label, f"train {label} claim {sample_number}"))
            identifier += 1
        train_rows.append(_claim_row(identifier, label, f"Shared overlap {labels.index(label)}"))
        identifier += 1

    identifier = 10_000
    for label in labels:
        for sample_number in range(20):
            dev_rows.append(_claim_row(identifier, label, f"dev {label} claim {sample_number}"))
            identifier += 1
        dev_rows.append(
            _claim_row(identifier, label, f"  shared   OVERLAP {labels.index(label)}  ")
        )
        identifier += 1

    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "shared_task_dev.jsonl"
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train_rows),
        encoding="utf-8",
    )
    dev_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in dev_rows),
        encoding="utf-8",
    )

    background_pages = tuple(
        _wiki_page(
            f"background-{page_number:02}",
            tuple(
                f"background {page_number} sentence {sentence_number}"
                for sentence_number in range(12)
            ),
        )
        for page_number in range(20)
    )
    pages = (
        _wiki_page("Evidence_A", tuple(f"evidence sentence {sid}" for sid in range(9))),
        *background_pages,
        {"id": "duplicate-sentence-page", "text": "duplicate", "lines": "0\tfirst\n0\tsecond"},
        {"id": "", "text": "", "lines": ""},
    )
    wiki_path = tmp_path / "wiki-pages.zip"
    _write_archive(wiki_path, pages)

    duplicate_wiki_path = tmp_path / "duplicate-page-id.zip"
    _write_archive(
        duplicate_wiki_path,
        (
            _wiki_page("repeated-page", ("first source row",)),
            _wiki_page("repeated-page", ("second source row",)),
        ),
    )

    monkeypatch.setattr(corpus, "EXPECTED_WIKI_PAGES", len(pages))
    monkeypatch.setattr(
        corpus,
        "SOURCE_HASHES",
        {
            "train.jsonl": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            "shared_task_dev.jsonl": hashlib.sha256(dev_path.read_bytes()).hexdigest(),
            "wiki-pages.zip": hashlib.sha256(wiki_path.read_bytes()).hexdigest(),
        },
    )

    windows = tuple(
        (page_id, tuple(range(start, min(start + 4, 12))))
        for page_id in (f"background-{number:02}" for number in range(20))
        for start in range(0, 12, 3)
    )
    return SyntheticSources(
        train_path=train_path,
        dev_path=dev_path,
        wiki_path=wiki_path,
        duplicate_wiki_path=duplicate_wiki_path,
        output_one=tmp_path / "processed-one",
        output_two=tmp_path / "processed-two",
        expected_background_windows=windows,
    )
