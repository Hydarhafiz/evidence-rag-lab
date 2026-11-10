"""Shared runtime payload, ranking, and persistent local Qdrant tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evidence_rag import retrieval
from evidence_rag.models import Passage
from evidence_rag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    RetrievalError,
    build_dense_index,
    load_runtime_passages,
    tokenize,
)


def _passage(identifier: int, page_title: str, text: str) -> Passage:
    passage_id = f"00000000-0000-0000-0000-{identifier:012d}"
    return Passage(
        passage_id=passage_id,
        page_title=page_title,
        sentence_ids=(identifier,),
        text=text,
        source_artifact="synthetic fixture",
        source_sha256="a" * 64,
    )


def _write_passages(path: Path, passages: tuple[Passage, ...]) -> None:
    path.write_text(
        "".join(
            json.dumps(passage.to_runtime_payload(), separators=(",", ":")) + "\n"
            for passage in passages
        ),
        encoding="utf-8",
    )


def test_bm25_tokenization_and_ties_are_stable() -> None:
    passages = (
        _passage(2, "Page B", "common words here"),
        _passage(1, "Page A", "common words there"),
    )
    retriever = BM25Retriever(passages)

    assert tokenize("BGE-small, TEST!") == ["bge", "small", "test"]
    first = retriever.search("not-in-the-index", top_k=2)
    second = retriever.search("not-in-the-index", top_k=2)
    assert [hit.passage_id for hit in first.hits] == [hit.passage_id for hit in second.hits]
    assert [hit.passage_id for hit in first.hits] == sorted(
        passage.passage_id for passage in passages
    )
    assert [hit.rank for hit in first.hits] == [1, 2]
    assert all(hit.score == 0.0 for hit in first.hits)
    assert all(hit.retrieval_method == "bm25" for hit in first.hits)


def test_runtime_payload_rejects_evaluation_fields_and_duplicate_ids(tmp_path: Path) -> None:
    passage = _passage(1, "Page A", "source text")
    payload = passage.to_runtime_payload()
    payload["label"] = "SUPPORTS"

    with pytest.raises(ValueError, match="unexpected field set"):
        Passage.from_runtime_payload(payload)

    passage_path = tmp_path / "passages.jsonl"
    passage_payload = json.dumps(passage.to_runtime_payload())
    passage_path.write_text(f"{passage_payload}\n{passage_payload}\n", encoding="utf-8")
    with pytest.raises(RetrievalError, match="duplicate passage identifier"):
        load_runtime_passages(passage_path)


class FakeEncoder:
    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        del kwargs
        vectors = []
        for text in texts:
            normalized = text.casefold()
            if "alpha" in normalized:
                vectors.append([1.0, 0.0, 0.0])
            elif "beta" in normalized:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)

    def eval(self) -> None:
        return None


def test_dense_qdrant_uses_shared_corpus_and_persists_across_restarts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(retrieval, "MODEL_DIMENSION", 3)
    monkeypatch.setattr(retrieval, "load_encoder", lambda cache_dir: FakeEncoder())
    passages = (
        _passage(3, "Alpha One", "alpha evidence"),
        _passage(1, "Alpha Two", "alpha evidence"),
        _passage(2, "Beta", "beta evidence"),
    )
    passages_path = tmp_path / "passages.jsonl"
    _write_passages(passages_path, passages)
    qdrant_path = tmp_path / "qdrant"
    cache_dir = tmp_path / "models"

    bm25, bm25_hash = BM25Retriever.from_path(passages_path)
    build_result = build_dense_index(
        passages_path,
        qdrant_path,
        cache_dir,
        batch_size=2,
    )
    assert (
        build_result.corpus_sha256
        == bm25_hash
        == hashlib.sha256(passages_path.read_bytes()).hexdigest()
    )
    assert build_result.passage_count == len(bm25.passages) == 3
    assert build_result.embedding_dimension == 3

    first_client = DenseRetriever(passages_path, qdrant_path, cache_dir)
    first = first_client.search("alpha claim", top_k=3)
    first_client.close()
    restarted_client = DenseRetriever(passages_path, qdrant_path, cache_dir)
    second = restarted_client.search("alpha claim", top_k=3)
    restarted_client.close()

    assert [hit.passage_id for hit in first.hits] == [hit.passage_id for hit in second.hits]
    assert [hit.passage_id for hit in first.hits[:2]] == sorted(
        passage.passage_id for passage in passages if "alpha" in passage.text.casefold()
    )
    assert [hit.rank for hit in first.hits] == [1, 2, 3]
    assert all(hit.retrieval_method == "dense" for hit in first.hits)
    assert all(hit.page_title and hit.sentence_ids and hit.text for hit in first.hits)
    assert first.retrieval_method == "dense"
