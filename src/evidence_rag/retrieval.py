"""BM25 and persistent local Qdrant retrieval over one passage file."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from evidence_rag.corpus import MODEL_DIMENSION, MODEL_NAME, MODEL_REVISION
from evidence_rag.embeddings import QUERY_INSTRUCTION, load_encoder
from evidence_rag.models import Passage, RetrievalHit, SearchResponse

TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
COLLECTION_PREFIX = "fever_passages_"
INDEX_MANIFEST_NAME = "index_manifest.json"


class RetrievalError(ValueError):
    """Raised when passage data or its dense index cannot be used safely."""


@dataclass(frozen=True, slots=True)
class IndexBuildResult:
    collection_name: str
    corpus_sha256: str
    passage_count: int
    embedding_dimension: int
    build_latency_ms: float
    model_name: str
    model_revision: str
    qdrant_path: str


def tokenize(text: str) -> list[str]:
    """Case-fold and split on non-word characters; no stemming or stop-word removal."""
    return TOKEN_PATTERN.findall(text.casefold())


def load_runtime_passages(path: Path) -> tuple[tuple[Passage, ...], str]:
    digest = hashlib.sha256()
    passages: list[Passage] = []
    passage_ids: set[str] = set()
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, start=1):
            digest.update(raw)
            if not raw.strip():
                raise RetrievalError(f"blank passage row at line {line_number}")
            try:
                value = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise RetrievalError(
                    f"invalid passage JSON at line {line_number}: {error}"
                ) from error
            try:
                passage = Passage.from_runtime_payload(value)
            except ValueError as error:
                raise RetrievalError(
                    f"invalid runtime passage at line {line_number}: {error}"
                ) from error
            if passage.passage_id in passage_ids:
                raise RetrievalError(f"duplicate passage identifier: {passage.passage_id}")
            passage_ids.add(passage.passage_id)
            passages.append(passage)

    if not passages:
        raise RetrievalError("the accepted passage corpus is empty")
    return tuple(passages), digest.hexdigest()


def _rank_hits(
    passages: tuple[Passage, ...],
    scores: list[float],
    method: str,
    top_k: int,
) -> tuple[RetrievalHit, ...]:
    ranked = sorted(
        zip(scores, passages, strict=True),
        key=lambda pair: (-pair[0], pair[1].passage_id),
    )[:top_k]
    return tuple(
        RetrievalHit(
            passage_id=passage.passage_id,
            page_title=passage.page_title,
            sentence_ids=passage.sentence_ids,
            score=float(score),
            text=passage.text,
            rank=rank,
            retrieval_method=method,
        )
        for rank, (score, passage) in enumerate(ranked, start=1)
    )


class BM25Retriever:
    """Small in-memory BM25 index rebuilt from the shared runtime passage file."""

    method = "bm25"

    def __init__(self, passages: tuple[Passage, ...]) -> None:
        started = perf_counter()
        self.passages = passages
        self.index = BM25Okapi([tokenize(passage.text) for passage in passages])
        self.build_latency_ms = (perf_counter() - started) * 1_000

    @classmethod
    def from_path(cls, passages_path: Path) -> tuple[BM25Retriever, str]:
        passages, corpus_hash = load_runtime_passages(passages_path)
        return cls(passages), corpus_hash

    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        started = perf_counter()
        query_tokens = tokenize(query)
        if not query_tokens:
            hits: tuple[RetrievalHit, ...] = ()
        else:
            scores = [float(score) for score in self.index.get_scores(query_tokens)]
            hits = _rank_hits(self.passages, scores, self.method, top_k)
        return SearchResponse(
            retrieval_method=self.method,
            query=query,
            elapsed_ms=(perf_counter() - started) * 1_000,
            hits=hits,
        )


def _collection_name(corpus_sha256: str) -> str:
    return f"{COLLECTION_PREFIX}{corpus_sha256[:24]}"


def _write_index_manifest(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_dense_index(
    passages_path: Path,
    qdrant_path: Path,
    cache_dir: Path,
    batch_size: int = 64,
) -> IndexBuildResult:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    passages, corpus_hash = load_runtime_passages(passages_path)
    encoder = load_encoder(cache_dir)
    collection_name = _collection_name(corpus_hash)
    qdrant_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_path))
    started = perf_counter()
    try:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=MODEL_DIMENSION,
                distance=models.Distance.COSINE,
            ),
        )
        for start in range(0, len(passages), batch_size):
            batch = passages[start : start + batch_size]
            vectors = encoder.encode(
                [passage.text for passage in batch],
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if vectors.shape != (len(batch), MODEL_DIMENSION):
                raise RetrievalError(f"unexpected embedding batch shape: {vectors.shape}")
            norms = np.linalg.norm(vectors, axis=1)
            if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
                raise RetrievalError("embedding normalization check failed")
            points = [
                models.PointStruct(
                    id=passage.passage_id,
                    vector=vector.tolist(),
                    payload=passage.to_runtime_payload(),
                )
                for passage, vector in zip(batch, vectors, strict=True)
            ]
            client.upsert(collection_name=collection_name, points=points, wait=True)

        point_count = client.count(collection_name=collection_name, exact=True).count
        if point_count != len(passages):
            raise RetrievalError(
                f"Qdrant persisted {point_count} points for {len(passages)} source passages"
            )
        elapsed_ms = (perf_counter() - started) * 1_000
        result = IndexBuildResult(
            collection_name=collection_name,
            corpus_sha256=corpus_hash,
            passage_count=point_count,
            embedding_dimension=MODEL_DIMENSION,
            build_latency_ms=elapsed_ms,
            model_name=MODEL_NAME,
            model_revision=MODEL_REVISION,
            qdrant_path=str(qdrant_path),
        )
        manifest = {
            "format_version": 1,
            "collection_name": result.collection_name,
            "corpus_sha256": result.corpus_sha256,
            "passage_count": result.passage_count,
            "embedding_dimension": result.embedding_dimension,
            "model_name": result.model_name,
            "model_revision": result.model_revision,
            "similarity_metric": "cosine",
            "normalized_embeddings": True,
            "query_instruction": "Represent this sentence for searching relevant passages: ",
            "passage_input": "Page title, newline, then source sentence text.",
            "build_latency_ms": result.build_latency_ms,
        }
        _write_index_manifest(qdrant_path / INDEX_MANIFEST_NAME, manifest)
        return result
    finally:
        client.close()


class DenseRetriever:
    """Local Qdrant cosine search over the exact corpus named by its content hash."""

    method = "dense"

    def __init__(
        self,
        passages_path: Path,
        qdrant_path: Path,
        cache_dir: Path,
    ) -> None:
        startup_started = perf_counter()
        passage_load_started = perf_counter()
        self.passages, self.corpus_sha256 = load_runtime_passages(passages_path)
        self.passage_load_latency_ms = (perf_counter() - passage_load_started) * 1_000
        encoder_load_started = perf_counter()
        self.encoder: SentenceTransformer = load_encoder(cache_dir)
        self.encoder_load_latency_ms = (perf_counter() - encoder_load_started) * 1_000
        self.collection_name = _collection_name(self.corpus_sha256)
        manifest_path = qdrant_path / INDEX_MANIFEST_NAME
        if not manifest_path.is_file():
            raise RetrievalError(f"dense index manifest is missing: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RetrievalError(f"cannot read dense index manifest: {error}") from error
        expected_manifest = {
            "corpus_sha256": self.corpus_sha256,
            "collection_name": self.collection_name,
            "passage_count": len(self.passages),
            "embedding_dimension": MODEL_DIMENSION,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "similarity_metric": "cosine",
            "normalized_embeddings": True,
        }
        if not isinstance(manifest, dict) or any(
            manifest.get(key) != expected for key, expected in expected_manifest.items()
        ):
            raise RetrievalError("dense index manifest does not match the accepted corpus/model")

        qdrant_open_started = perf_counter()
        self.client = QdrantClient(path=str(qdrant_path))
        if not self.client.collection_exists(self.collection_name):
            self.client.close()
            raise RetrievalError(f"Qdrant collection is missing: {self.collection_name}")
        count = self.client.count(collection_name=self.collection_name, exact=True).count
        if count != len(self.passages):
            self.client.close()
            raise RetrievalError(
                f"Qdrant point count mismatch: expected {len(self.passages)}, got {count}"
            )
        self.qdrant_open_latency_ms = (perf_counter() - qdrant_open_started) * 1_000
        self.startup_latency_ms = (perf_counter() - startup_started) * 1_000

    def close(self) -> None:
        self.client.close()

    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        started = perf_counter()
        query_vector = self.encoder.encode(
            [QUERY_INSTRUCTION + query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        if query_vector.shape != (MODEL_DIMENSION,):
            raise RetrievalError(f"unexpected query embedding shape: {query_vector.shape}")
        points = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            query_filter=None,
            limit=len(self.passages),
            with_payload=True,
            with_vectors=False,
        ).points

        passages: list[Passage] = []
        scores: list[float] = []
        for point in points:
            if point.payload is None:
                raise RetrievalError(f"Qdrant point has no passage payload: {point.id}")
            try:
                passage = Passage.from_runtime_payload(cast(dict[str, object], point.payload))
            except ValueError as error:
                raise RetrievalError(f"invalid Qdrant passage payload: {error}") from error
            score = float(point.score or 0.0)
            if not math.isfinite(score):
                raise RetrievalError("Qdrant returned a non-finite similarity score")
            passages.append(passage)
            scores.append(score)
        if len(points) != len(self.passages):
            raise RetrievalError(
                f"Qdrant returned {len(points)} of {len(self.passages)} source passages"
            )
        hits = _rank_hits(tuple(passages), scores, self.method, top_k)
        return SearchResponse(
            retrieval_method=self.method,
            query=query,
            elapsed_ms=(perf_counter() - started) * 1_000,
            hits=hits,
        )
