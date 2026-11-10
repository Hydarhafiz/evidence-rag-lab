"""Command-line entry points for corpus preparation and retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from evidence_rag.corpus import CorpusError, prepare_corpus
from evidence_rag.embeddings import load_encoder
from evidence_rag.evaluation import run_development_smoke
from evidence_rag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    RetrievalError,
    build_dense_index,
)

DEFAULT_TRAIN = Path("data/raw/train.jsonl")
DEFAULT_DEV = Path("data/raw/shared_task_dev.jsonl")
DEFAULT_WIKI = Path("data/raw/wiki-pages.zip")
DEFAULT_PASSAGES = Path("data/processed/passages.jsonl")
DEFAULT_EVALUATION = Path("data/processed/evaluation_samples_train.jsonl")
DEFAULT_QDRANT = Path("data/indexes/qdrant")
DEFAULT_MODEL_CACHE = Path("models/hf")


def _add_model_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_MODEL_CACHE)


def _add_retrieval_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--passages", type=Path, default=DEFAULT_PASSAGES)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT)
    _add_model_paths(parser)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fever-rag")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="validate FEVER inputs and build the 5K corpus")
    prepare.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    prepare.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    prepare.add_argument("--wiki-zip", type=Path, default=DEFAULT_WIKI)
    prepare.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    prepare.add_argument("--target-passages", type=int, default=5_000)
    _add_model_paths(prepare)

    index = commands.add_parser("index", help="build the persistent local dense Qdrant index")
    index.add_argument("--passages", type=Path, default=DEFAULT_PASSAGES)
    index.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT)
    index.add_argument("--batch-size", type=int, default=64)
    _add_model_paths(index)

    search = commands.add_parser(
        "search", help="retrieve source passages using BM25 or dense search"
    )
    search.add_argument("--method", choices=("bm25", "dense"), required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=5)
    _add_retrieval_paths(search)

    smoke = commands.add_parser(
        "smoke",
        help=(
            "compare retrieval on the 30-claim development sample without opening held-out results"
        ),
    )
    smoke.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    smoke.add_argument("--top-k", type=int, default=5)
    _add_retrieval_paths(smoke)
    return parser


def _prepare(args: argparse.Namespace) -> dict[str, object]:
    encoder = load_encoder(args.cache_dir)
    result = prepare_corpus(
        train_path=args.train,
        dev_path=args.dev,
        wiki_zip_path=args.wiki_zip,
        output_dir=args.output_dir,
        tokenizer=encoder.tokenizer,
        maximum_tokens=encoder.max_seq_length,
        target_passages=args.target_passages,
    )
    return {
        "elapsed_seconds": result.elapsed_seconds,
        "manifest": result.manifest,
    }


def _index(args: argparse.Namespace) -> dict[str, object]:
    bm25, bm25_hash = BM25Retriever.from_path(args.passages)
    result = build_dense_index(
        passages_path=args.passages,
        qdrant_path=args.qdrant_path,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
    )
    if result.corpus_sha256 != bm25_hash or result.passage_count != len(bm25.passages):
        raise RetrievalError("BM25 and Qdrant were built from different passage files")
    return {
        "passage_count": len(bm25.passages),
        "corpus_sha256": bm25_hash,
        "bm25_index_build_latency_ms": bm25.build_latency_ms,
        "dense_index": asdict(result),
    }


def _search(args: argparse.Namespace) -> dict[str, object]:
    if args.method == "bm25":
        retriever, _ = BM25Retriever.from_path(args.passages)
        response = retriever.search(args.query, args.top_k)
        index_build_ms = retriever.build_latency_ms
    else:
        retriever = DenseRetriever(args.passages, args.qdrant_path, args.cache_dir)
        try:
            response = retriever.search(args.query, args.top_k)
        finally:
            retriever.close()
        index_build_ms = None
    return {
        "index_build_latency_ms": index_build_ms,
        "response": asdict(response),
    }


def _smoke(args: argparse.Namespace) -> dict[str, object]:
    return run_development_smoke(
        passages_path=args.passages,
        evaluation_path=args.evaluation,
        qdrant_path=args.qdrant_path,
        cache_dir=args.cache_dir,
        top_k=args.top_k,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {
        "prepare": _prepare,
        "index": _index,
        "search": _search,
        "smoke": _smoke,
    }
    try:
        result = handlers[args.command](args)
    except (CorpusError, RetrievalError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
