"""Runtime passage and retrieval result contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

PASSAGE_FIELDS = {
    "passage_id",
    "page_title",
    "sentence_ids",
    "text",
    "source_artifact",
    "source_sha256",
}


@dataclass(frozen=True, slots=True)
class Passage:
    """A citation-ready source passage with no evaluation annotations."""

    passage_id: str
    page_title: str
    sentence_ids: tuple[int, ...]
    text: str
    source_artifact: str
    source_sha256: str

    def __post_init__(self) -> None:
        text_fields = (
            self.passage_id,
            self.page_title,
            self.text,
            self.source_artifact,
            self.source_sha256,
        )
        if any(not isinstance(value, str) for value in text_fields):
            raise ValueError("passage text and provenance fields must be strings")
        if not self.passage_id or not self.page_title or not self.text.strip():
            raise ValueError("passage id, page title, and text must be non-empty")
        if not self.sentence_ids or any(
            isinstance(sentence_id, bool) or not isinstance(sentence_id, int) or sentence_id < 0
            for sentence_id in self.sentence_ids
        ):
            raise ValueError("passage must contain non-negative sentence identifiers")
        if len(set(self.sentence_ids)) != len(self.sentence_ids):
            raise ValueError("passage sentence identifiers must be unique")
        if (
            not self.source_artifact
            or len(self.source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_sha256)
        ):
            raise ValueError("passage source provenance is incomplete")

    def to_runtime_payload(self) -> dict[str, object]:
        return {
            "passage_id": self.passage_id,
            "page_title": self.page_title,
            "sentence_ids": list(self.sentence_ids),
            "text": self.text,
            "source_artifact": self.source_artifact,
            "source_sha256": self.source_sha256,
        }

    @classmethod
    def from_runtime_payload(cls, value: object) -> Passage:
        if not isinstance(value, dict) or set(value) != PASSAGE_FIELDS:
            raise ValueError("runtime passage has an unexpected field set")

        sentence_ids = value["sentence_ids"]
        if not isinstance(sentence_ids, list) or any(
            isinstance(sentence_id, bool) or not isinstance(sentence_id, int)
            for sentence_id in sentence_ids
        ):
            raise ValueError("runtime passage sentence_ids must be a list of integers")

        text_fields = (
            "passage_id",
            "page_title",
            "text",
            "source_artifact",
            "source_sha256",
        )
        if any(not isinstance(value[field], str) for field in text_fields):
            raise ValueError("runtime passage text fields must be strings")

        return cls(
            passage_id=cast(str, value["passage_id"]),
            page_title=cast(str, value["page_title"]),
            sentence_ids=tuple(sentence_ids),
            text=cast(str, value["text"]),
            source_artifact=cast(str, value["source_artifact"]),
            source_sha256=cast(str, value["source_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    passage_id: str
    page_title: str
    sentence_ids: tuple[int, ...]
    score: float
    text: str
    rank: int
    retrieval_method: str


@dataclass(frozen=True, slots=True)
class SearchResponse:
    retrieval_method: str
    query: str
    elapsed_ms: float
    hits: tuple[RetrievalHit, ...]


class Retriever(Protocol):
    """Small common search surface shared by BM25 and dense retrieval."""

    method: str

    def search(self, query: str, top_k: int = 5) -> SearchResponse: ...
