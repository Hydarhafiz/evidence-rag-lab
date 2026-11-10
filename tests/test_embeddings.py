"""Pinned encoder configuration and dimension/token-boundary checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evidence_rag import embeddings
from evidence_rag.corpus import MODEL_DIMENSION, MODEL_MAX_TOKENS, MODEL_NAME, MODEL_REVISION


class FakeSentenceTransformer:
    def __init__(self, dimension: int, max_seq_length: int) -> None:
        self.dimension = dimension
        self.max_seq_length = max_seq_length
        self.kwargs: dict[str, Any] = {}

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def eval(self) -> None:
        return None


def test_encoder_uses_pinned_revision_local_cache_and_expected_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = FakeSentenceTransformer(MODEL_DIMENSION, MODEL_MAX_TOKENS)

    def build_encoder(model_name: str, **kwargs: Any) -> FakeSentenceTransformer:
        assert model_name == MODEL_NAME
        instance.kwargs = kwargs
        return instance

    monkeypatch.setattr(embeddings, "SentenceTransformer", build_encoder)
    cache_dir = tmp_path / "model-cache"

    observed = embeddings.load_encoder(cache_dir)

    assert observed is instance
    assert cache_dir.is_dir()
    assert instance.kwargs["revision"] == MODEL_REVISION
    assert instance.kwargs["device"] == "cpu"
    assert instance.kwargs["local_files_only"] is True
    assert instance.max_seq_length == 512


@pytest.mark.parametrize(
    ("dimension", "maximum_tokens", "message"),
    [
        (MODEL_DIMENSION - 1, MODEL_MAX_TOKENS, "embedding dimension"),
        (MODEL_DIMENSION, MODEL_MAX_TOKENS - 1, "tokenizer limit"),
    ],
)
def test_encoder_rejects_revision_configuration_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
    maximum_tokens: int,
    message: str,
) -> None:
    instance = FakeSentenceTransformer(dimension, maximum_tokens)
    monkeypatch.setattr(embeddings, "SentenceTransformer", lambda *args, **kwargs: instance)

    with pytest.raises(ValueError, match=message):
        embeddings.load_encoder(tmp_path / "model-cache")
