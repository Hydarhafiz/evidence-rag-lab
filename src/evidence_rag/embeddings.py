"""Pinned local embedding model configuration."""

from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer

from evidence_rag.corpus import MODEL_DIMENSION, MODEL_MAX_TOKENS, MODEL_NAME, MODEL_REVISION

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def load_encoder(cache_dir: Path) -> SentenceTransformer:
    cache_dir.mkdir(parents=True, exist_ok=True)
    encoder = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        cache_folder=str(cache_dir),
        device="cpu",
        local_files_only=True,
    )
    dimension = encoder.get_embedding_dimension()
    if dimension != MODEL_DIMENSION:
        raise ValueError(
            f"unexpected embedding dimension: expected {MODEL_DIMENSION}, got {dimension}"
        )
    if encoder.max_seq_length != MODEL_MAX_TOKENS:
        raise ValueError(
            f"unexpected tokenizer limit: expected {MODEL_MAX_TOKENS}, got {encoder.max_seq_length}"
        )
    encoder.eval()
    return encoder
